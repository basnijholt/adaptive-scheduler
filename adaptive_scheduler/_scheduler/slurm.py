"""SLURM for Adaptive Scheduler."""
from __future__ import annotations

import copy
import getpass
import re
import subprocess
import textwrap
from distutils.spawn import find_executable
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, TypeVar

from adaptive_scheduler._scheduler.base_scheduler import BaseScheduler
from adaptive_scheduler._scheduler.common import run_submit

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from adaptive_scheduler.utils import EXECUTOR_TYPES


T = TypeVar("T")


def _maybe_as_tuple(
    x: T | tuple[T, ...] | None,
    n: int | None,
    *,
    check_type: type | None = None,
) -> tuple[T, ...] | T | None:
    if x is None:
        return x
    if check_type is not None and not isinstance(x, (check_type, tuple)):
        msg = f"Expected `{check_type}` or `tuple[{check_type}, ...]`, got `{type(x)}`"
        raise TypeError(msg)
    if n is None:
        return x
    if isinstance(x, tuple):
        assert len(x) == n
        return x
    return tuple(copy.deepcopy(x) for _ in range(n))


def _tuple_lengths(*maybe_tuple: tuple[Any, ...] | Any) -> int | None:
    """Get the length of the items that are in tuples."""
    length = None
    for y in maybe_tuple:
        if isinstance(y, tuple):
            if length is None:
                length = len(y)
            elif length != len(y):
                msg = "All tuples should have the same length."
                raise ValueError(msg)
    return length


class SLURM(BaseScheduler):
    """Base object for a Scheduler.

    ``cores``, ``nodes``, ``cores_per_node``, ``extra_scheduler`` and
    ``partition`` can be either a single value or a tuple of values.
    If a tuple is given, then the length of the tuple should be the same
    as the number of learners (jobs) that are run. This allows for
    different resources for different jobs.

    Parameters
    ----------
    cores
        Number of cores per job (so per learner.)
        Either use `cores` or `nodes` and `cores_per_node`.
    nodes
        Number of nodes per job (so per learner.)
        Either `nodes` and `cores_per_node` or use `cores`.
    cores_per_node
        Number of cores per node.
        Either `nodes` and `cores_per_node` or use `cores`.
    partition
        The SLURM partition to submit the job to.
    exclusive
        Whether to use exclusive nodes (e.g., if SLURM it adds ``--exclusive`` as option).
    log_folder
        The folder in which to put the log-files.
    mpiexec_executable
        ``mpiexec`` executable. By default `mpiexec` will be
        used (so probably from ``conda``).
    executor_type
        The executor that is used, by default `mpi4py.futures.MPIPoolExecutor` is used.
        One can use ``"ipyparallel"``, ``"dask-mpi"``, ``"mpi4py"``,
        ``"loky"``, or ``"process-pool"``.
    num_threads
        ``MKL_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``, ``OMP_NUM_THREADS``, and
        ``NUMEXPR_NUM_THREADS`` will be set to this number.
    extra_scheduler
        Extra ``#SLURM`` (depending on scheduler type)
        arguments, e.g. ``["--exclusive=user", "--time=1"]`` or a tuple of lists,
        e.g. ``(["--time=10"], ["--time=20"]])`` for two jobs.
    extra_env_vars
        Extra environment variables that are exported in the job
        script. e.g. ``["TMPDIR='/scratch'", "PYTHONPATH='my_dir:$PYTHONPATH'"]``.
    extra_script
        Extra script that will be executed after any environment variables are set,
        but before the main scheduler is run.
    """

    # Attributes that all schedulers need to have
    _ext = ".sbatch"
    _submit_cmd = "sbatch"
    _JOB_ID_VARIABLE = "${SLURM_JOB_ID}"
    _options_flag = "SBATCH"
    _cancel_cmd = "scancel"

    def __init__(  # noqa: PLR0912, PLR0915
        self,
        *,
        cores: int | tuple[int, ...] | None = None,
        nodes: int | tuple[int, ...] | None = None,
        cores_per_node: int | tuple[int, ...] | None = None,
        partition: str | tuple[str, ...] | None = None,
        exclusive: bool = True,
        python_executable: str | None = None,
        log_folder: str | Path = "",
        mpiexec_executable: str | None = None,
        executor_type: EXECUTOR_TYPES = "process-pool",
        num_threads: int = 1,
        extra_scheduler: list[str] | tuple[list[str], ...] | None = None,
        extra_env_vars: list[str] | None = None,
        extra_script: str | None = None,
        batch_folder: str | Path = "",
    ) -> None:
        """Initialize the scheduler."""
        self.exclusive = exclusive
        # Store the original values
        self._cores = cores
        self._nodes = nodes
        self._cores_per_node = cores_per_node
        self._partition = partition
        self.__extra_scheduler = extra_scheduler

        msg = "Specify either `nodes` and `cores_per_node`, or only `cores`, not both."
        if cores is None:
            if nodes is None or cores_per_node is None:
                raise ValueError(msg)
        elif nodes is not None or cores_per_node is not None:
            raise ValueError(msg)

        if extra_scheduler is None:
            extra_scheduler = []

        # If any is a list, then all should be a list
        n = _tuple_lengths(cores, nodes, cores_per_node, partition, extra_scheduler)
        single_job_script = n is None
        cores = _maybe_as_tuple(cores, n, check_type=int)
        self.nodes = nodes = _maybe_as_tuple(nodes, n, check_type=int)
        self.cores_per_node = cores_per_node = _maybe_as_tuple(
            cores_per_node,
            n,
            check_type=int,
        )
        self.partition = partition = _maybe_as_tuple(partition, n, check_type=str)
        extra_scheduler = _maybe_as_tuple(extra_scheduler, n, check_type=list)
        if cores_per_node is not None:
            if single_job_script:
                assert isinstance(cores_per_node, int)
                assert isinstance(nodes, int)
                assert isinstance(extra_scheduler, list)
                extra_scheduler.append(f"--ntasks-per-node={cores_per_node}")
                cores = cores_per_node * nodes
            else:
                assert isinstance(cores_per_node, tuple)
                assert isinstance(nodes, tuple)
                assert isinstance(extra_scheduler, tuple)
                for lst, cpn in zip(extra_scheduler, cores_per_node):
                    assert isinstance(lst, list)
                    lst.append(f"--ntasks-per-node={cpn}")
                cores = tuple(cpn * n for cpn, n in zip(cores_per_node, nodes))

        if partition is not None:
            if single_job_script:
                assert isinstance(partition, str)
                assert isinstance(extra_scheduler, list)
                if partition not in self.partitions:
                    msg = f"Invalid partition: {partition}, only {self.partitions} are available."
                    raise ValueError(msg)
                extra_scheduler.append(f"--partition={partition}")
            else:
                if any(p not in self.partitions for p in partition):
                    msg = f"Invalid partition: {partition}, only {self.partitions} are available."
                    raise ValueError(msg)
                assert isinstance(extra_scheduler, tuple)
                for lst, p in zip(extra_scheduler, partition):
                    assert isinstance(lst, list)
                    lst.append(f"--partition={p}")

        if exclusive:
            if single_job_script:
                assert isinstance(extra_scheduler, list)
                extra_scheduler.append("--exclusive")
            else:
                assert isinstance(extra_scheduler, tuple)
                for lst in extra_scheduler:
                    assert isinstance(lst, list)
                    lst.append("--exclusive")

        assert cores is not None
        super().__init__(
            cores,
            python_executable=python_executable,
            log_folder=log_folder,
            mpiexec_executable=mpiexec_executable,
            executor_type=executor_type,
            num_threads=num_threads,
            extra_scheduler=extra_scheduler,  # type: ignore[arg-type]
            extra_env_vars=extra_env_vars,
            extra_script=extra_script,
            batch_folder=batch_folder,
        )
        # SLURM specific
        self.mpiexec_executable = mpiexec_executable or "srun --mpi=pmi2"

    def __getstate__(self) -> dict[str, Any]:
        """Get the state of the SLURM scheduler."""
        state = super().__getstate__()
        state["cores"] = self._cores
        state["nodes"] = self._nodes
        state["cores_per_node"] = self._cores_per_node
        state["partition"] = self._partition
        state["exclusive"] = self.exclusive
        state["extra_scheduler"] = self.__extra_scheduler
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state of the SLURM scheduler."""
        self.__init__(**state)  # type: ignore[misc]

    def _ipyparallel(self, *, index: int | None = None) -> tuple[str, tuple[str, ...]]:
        cores = self._get_cores(index=index)
        job_id = self._JOB_ID_VARIABLE
        profile = "${profile}"
        # We need to reserve one core for the controller
        if self.nodes is not None and self.partition is not None and self.exclusive:
            if self.single_job_script:
                partition = self.partition
                nodes = self.nodes
            else:
                assert isinstance(self.partition, list)
                assert isinstance(self.nodes, list)
                assert index is not None
                partition = self.partition[index]
                nodes = self.nodes[index]
            assert isinstance(partition, str)
            assert isinstance(nodes, int)
            # Limit the number of cores to the maximum number of cores per node
            max_cores_per_node = self.partitions[partition]
            tot_cores = nodes * max_cores_per_node
            cores = min(cores, tot_cores - 1)
        else:  # noqa: PLR5501
            if self.single_job_script:
                assert isinstance(self.cores, int)
                cores = self.cores - 1
            else:
                assert isinstance(self.cores, tuple)
                assert index is not None
                cores = self.cores[index] - 1

        start = textwrap.dedent(
            f"""\
            profile=adaptive_scheduler_{job_id}

            echo "Creating profile {profile}"
            ipython profile create {profile}

            echo "Launching controller"
            ipcontroller --ip="*" --profile={profile} --log-to-file &
            sleep 10

            echo "Launching engines"
            srun --ntasks {cores} ipengine \\
                --profile={profile} \\
                --cluster-id='' \\
                --log-to-file &

            echo "Starting the Python script"
            srun --ntasks 1 {self.python_executable} {self.launcher} \\
            """,
        )
        custom = (f"    --profile {profile}",)
        return start, custom

    def job_script(self, options: dict[str, Any], *, index: int | None = None) -> str:
        """Get a jobscript in string form.

        Returns
        -------
        job_script
            A job script that can be submitted to SLURM.
        index
            The index of the job that is being run. This is used when
            specifying different resources for different jobs.
        """
        cores = self._get_cores(index=index)
        job_script = textwrap.dedent(
            f"""\
            #!/bin/bash
            #SBATCH --ntasks {cores}
            #SBATCH --no-requeue
            {{extra_scheduler}}

            {{extra_env_vars}}

            {{extra_script}}

            {{executor_specific}}
            """,
        )

        return job_script.format(
            extra_scheduler=self.extra_scheduler(index=index),
            extra_env_vars=self.extra_env_vars,
            extra_script=self.extra_script,
            executor_specific=self._executor_specific("${NAME}", options, index=index),
        )

    def start_job(self, name: str, *, index: int | None = None) -> None:
        """Writes a job script and submits it to the scheduler."""
        if self.single_job_script:
            name_prefix = name.rsplit("-", 1)[0]
        else:
            name_prefix = name
            with self.batch_fname(name_prefix).open("w", encoding="utf-8") as f:
                assert self._command_line_options is not None
                options = dict(self._command_line_options)  # copy
                options["--n"] = self._get_cores(index=index)
                if self.executor_type == "ipyparallel":
                    options["--n"] -= 1
                job_script = self.job_script(options, index=index)
                f.write(job_script)

        (output_fname,) = self.output_fnames(name)
        output_str = str(output_fname).replace(self._JOB_ID_VARIABLE, "%A")
        output_opt = f"--output {output_str}"
        name_opt = f"--job-name {name}"
        submit_cmd = (
            f"{self.submit_cmd} {name_opt} {output_opt} {self.batch_fname(name_prefix)}"
        )
        run_submit(submit_cmd, name)

    def queue(self, *, me_only: bool = True) -> dict[str, dict[str, str]]:
        """Get the queue of jobs."""
        python_format = {
            "JobID": 100,
            "Name": 100,
            "state": 100,
            "NumNodes": 100,
            "NumTasks": 100,
            "ReasonList": 4000,
            "SubmitTime": 100,
            "StartTime": 100,
            "UserName": 100,
            "Partition": 100,
        }  # (key -> length) mapping

        slurm_format = ",".join(f"{k}:{v}" for k, v in python_format.items())
        squeue_executable = find_executable("squeue")
        assert isinstance(squeue_executable, str)
        cmd = [
            squeue_executable,
            rf'--Format=",{slurm_format},"',
            "--noheader",
            "--array",
        ]
        if me_only:
            username = getpass.getuser()
            cmd.append(f"--user={username}")
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
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
        running = {info.pop("JobID"): info for info in squeue}
        for info in running.values():
            info["job_name"] = info.pop("Name")
        return running

    @cached_property
    def partitions(self) -> dict[str, int]:
        """Get the partitions of the SLURM scheduler."""
        return slurm_partitions()  # type: ignore[return-value]


def _get_ncores(partition: str) -> int | None:
    numbers = re.findall(r"\d+", partition)
    if not numbers:
        return None
    return int(numbers[0])


@lru_cache(maxsize=1)
def slurm_partitions(
    *,
    timeout: int = 5,
    with_ncores: bool = True,
) -> list[str] | dict[str, int | None]:
    """Get the available slurm partitions, raises subprocess.TimeoutExpired after timeout."""
    output = subprocess.run(
        ["sinfo", "-ahO", "partition"],
        capture_output=True,
        timeout=timeout,
        check=False,
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
