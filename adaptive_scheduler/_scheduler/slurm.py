"""SLURM for Adaptive Scheduler."""
from __future__ import annotations

import getpass
import re
import subprocess
import textwrap
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING

from adaptive_scheduler._scheduler.base_scheduler import BaseScheduler
from adaptive_scheduler._scheduler.common import run_submit

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Literal


class SLURM(BaseScheduler):
    """Base object for a Scheduler.

    Parameters
    ----------
    cores : int | None
        Number of cores per job (so per learner.)
        Either use `cores` or `nodes` and `cores_per_node`.
    nodes : int | None
        Number of nodes per job (so per learner.)
        Either `nodes` and `cores_per_node` or use `cores`.
    cores_per_node: int | None
        Number of cores per node.
        Either `nodes` and `cores_per_node` or use `cores`.
    partition: str | None
        The SLURM partition to submit the job to.
    exclusive : bool
        Whether to use exclusive nodes (e.g., if SLURM it adds ``--exclusive`` as option).
    run_script : str
        Filename of the script that is run on the nodes. Inside this script we
        query the database and run the learner.
    python_executable : str, default: `sys.executable`
        The Python executable that should run the `run_script`. By default
        it uses the same Python as where this function is called.
    log_folder : str, default: ""
        The folder in which to put the log-files.
    mpiexec_executable : str, optional
        ``mpiexec`` executable. By default `mpiexec` will be
        used (so probably from ``conda``).
    executor_type : str, default: "mpi4py"
        The executor that is used, by default `mpi4py.futures.MPIPoolExecutor` is used.
        One can use ``"ipyparallel"``, ``"dask-mpi"``, ``"mpi4py"``, or ``"process-pool"``.
    num_threads : int, default 1
        ``MKL_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``, ``OMP_NUM_THREADS``, and
        ``NUMEXPR_NUM_THREADS`` will be set to this number.
    extra_scheduler : list, optional
        Extra ``#SLURM`` (depending on scheduler type)
        arguments, e.g. ``["--exclusive=user", "--time=1"]``.
    extra_env_vars : list, optional
        Extra environment variables that are exported in the job
        script. e.g. ``["TMPDIR='/scratch'", "PYTHONPATH='my_dir:$PYTHONPATH'"]``.
    extra_script : str, optional
        Extra script that will be executed after any environment variables are set,
        but before the main scheduler is run.
    """

    # Attributes that all schedulers need to have
    _ext = ".sbatch"
    _submit_cmd = "sbatch"
    _JOB_ID_VARIABLE = "${SLURM_JOB_ID}"
    _options_flag = "SBATCH"
    _cancel_cmd = "scancel"

    def __init__(
        self,
        *,
        cores: int | None = None,
        nodes: int | None = None,
        cores_per_node: int | None = None,
        partition: str | None = None,
        exclusive: bool = True,
        run_script: str | Path = "run_learner.py",
        python_executable: str | None = None,
        log_folder: str | Path = "",
        mpiexec_executable: str | None = None,
        executor_type: Literal[
            "ipyparallel",
            "dask-mpi",
            "mpi4py",
            "process-pool",
        ] = "mpi4py",
        num_threads: int = 1,
        extra_scheduler: list[str] | None = None,
        extra_env_vars: list[str] | None = None,
        extra_script: str | None = None,
    ) -> None:
        """Initialize the scheduler."""
        self._cores = cores
        self.nodes = nodes
        self.cores_per_node = cores_per_node
        self.partition = partition
        self.exclusive = exclusive
        self.__extra_scheduler = extra_scheduler

        msg = "Specify either `nodes` and `cores_per_node`, or only `cores`, not both."
        if cores is None:
            if nodes is None or cores_per_node is None:
                raise ValueError(msg)
        elif nodes is not None or cores_per_node is not None:
            raise ValueError(msg)

        if extra_scheduler is None:
            extra_scheduler = []

        if cores_per_node is not None:
            extra_scheduler.append(f"--ntasks-per-node={cores_per_node}")
            assert nodes is not None
            cores = nodes * cores_per_node

        if partition is not None:
            if partition not in self.partitions:
                msg = f"Invalid partition: {partition}, only {self.partitions} are available."
                raise ValueError(msg)
            extra_scheduler.append(f"--partition={partition}")

        if exclusive:
            extra_scheduler.append("--exclusive")
        assert cores is not None
        super().__init__(
            cores,
            run_script=run_script,
            python_executable=python_executable,
            log_folder=log_folder,
            mpiexec_executable=mpiexec_executable,
            executor_type=executor_type,
            num_threads=num_threads,
            extra_scheduler=extra_scheduler,
            extra_env_vars=extra_env_vars,
            extra_script=extra_script,
        )
        # SLURM specific
        self.mpiexec_executable = mpiexec_executable or "srun --mpi=pmi2"

    def __getstate__(self) -> dict[str, Any]:
        """Get the state of the SLURM scheduler."""
        state = super().__getstate__()
        state["cores"] = self._cores
        state["nodes"] = self.nodes
        state["cores_per_node"] = self.cores_per_node
        state["partition"] = self.partition
        state["exclusive"] = self.exclusive
        state["extra_scheduler"] = self.__extra_scheduler
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state of the SLURM scheduler."""
        self.__init__(**state)  # type: ignore[misc]

    def _ipyparallel(self, name: str) -> str:
        log_fname = self.log_fname(name)
        job_id = self._JOB_ID_VARIABLE
        profile = "${profile}"
        cores = self.cores - 1
        if self.nodes is not None and self.partition is not None and self.exclusive:
            max_cores_per_node = self.partitions[self.partition]
            tot_cores = self.nodes * max_cores_per_node
            cores = min(self.cores, tot_cores - 1)
        return textwrap.dedent(
            f"""\
            profile=adaptive_scheduler_{job_id}

            echo "Creating profile {profile}"
            ipython profile create {profile}

            echo "Launching controller"
            ipcontroller --ip="*" --profile={profile} --log-to-file &
            sleep 10

            echo "Launching engines"
            srun --ntasks {cores} ipengine --profile={profile} --cluster-id='' --log-to-file &

            echo "Starting the Python script"
            srun --ntasks 1 {self.python_executable} {self.run_script} --profile {profile} --n {cores} --log-fname {log_fname} --job-id {job_id} --name {name}
            """,  # noqa: E501
        )

    def job_script(self) -> str:
        """Get a jobscript in string form.

        Returns
        -------
        job_script : str
            A job script that can be submitted to SLURM.
        """
        job_script = textwrap.dedent(
            f"""\
            #!/bin/bash
            #SBATCH --ntasks {self.cores}
            #SBATCH --no-requeue
            {{extra_scheduler}}

            export MKL_NUM_THREADS={self.num_threads}
            export OPENBLAS_NUM_THREADS={self.num_threads}
            export OMP_NUM_THREADS={self.num_threads}
            export NUMEXPR_NUM_THREADS={self.num_threads}
            {{extra_env_vars}}

            {{extra_script}}

            {{executor_specific}}
            """,
        )

        job_script = job_script.format(
            extra_scheduler=self.extra_scheduler,
            extra_env_vars=self.extra_env_vars,
            extra_script=self.extra_script,
            executor_specific=self._executor_specific("${NAME}"),
        )
        return job_script

    def start_job(self, name: str) -> None:
        """Writes a job script and submits it to the scheduler."""
        name_prefix = name.rsplit("-", 1)[0]
        self.write_job_script(name_prefix)

        output_fname = str(self.output_fnames(name)[0]).replace(
            self._JOB_ID_VARIABLE,
            "%A",
        )
        output_opt = f"--output {output_fname}"
        name_opt = f"--job-name {name}"
        submit_cmd = (
            f"{self.submit_cmd} {name_opt} {output_opt} {self.batch_fname(name_prefix)}"
        )
        run_submit(submit_cmd, name)

    def queue(self, *, me_only: bool = True) -> dict[str, dict[str, str]]:
        """Get the queue of jobs."""
        python_format = {
            "jobid": 100,
            "name": 100,
            "state": 100,
            "numnodes": 100,
            "reasonlist": 4000,
        }  # (key -> length) mapping

        slurm_format = ",".join(f"{k}:{v}" for k, v in python_format.items())
        cmd = [
            "/usr/bin/squeue",
            rf'--Format=",{slurm_format},"',
            "--noheader",
            "--array",
        ]
        if me_only:
            username = getpass.getuser()
            cmd.append(f"--user={username}")
        proc = subprocess.run(cmd, text=True, capture_output=True)
        output = proc.stdout

        if (
            "squeue: error" in output
            or "slurm_load_jobs error" in output
            or proc.returncode != 0
        ):
            msg = "SLURM is not responding."
            raise RuntimeError(msg)

        def line_to_dict(line: str) -> dict[str, str]:
            chars = list(line)
            info = {}
            for k, v in python_format.items():
                info[k] = "".join(chars[:v]).strip()
                chars = chars[v:]
            return info

        squeue = [line_to_dict(line) for line in output.split("\n")]
        states = ("PENDING", "RUNNING", "CONFIGURING")
        squeue = [info for info in squeue if info["state"] in states]
        running = {info.pop("jobid"): info for info in squeue}
        for info in running.values():
            info["job_name"] = info.pop("name")
        return running

    @cached_property
    def partitions(self) -> dict[str, int]:
        """Get the partitions of the SLURM scheduler."""
        return slurm_partitions()  # type: ignore[return-value]


def _get_ncores(partition: str) -> int:
    numbers = re.findall(r"\d+", partition)
    return int(numbers[0])


@lru_cache(maxsize=1)
def slurm_partitions(
    *,
    timeout: int = 5,
    with_ncores: bool = True,
) -> list[str] | dict[str, int]:
    """Get the available slurm partitions, raises subprocess.TimeoutExpired after timeout."""
    output = subprocess.run(
        ["sinfo", "-ahO", "partition"],
        capture_output=True,
        timeout=timeout,
    )
    lines = output.stdout.decode("utf-8").split("\n")
    partitions = sorted(partition for line in lines if (partition := line.strip()))
    # Sort partitions alphabetically, but put the default partition first
    partitions = sorted(partitions, key=lambda s: ("*" not in s, s))
    # Remove asterisk, which is used for default partition
    partitions = [partition.replace("*", "") for partition in partitions]
    if not with_ncores:
        return partitions

    return {partition: _get_ncores(partition) for partition in partitions}
