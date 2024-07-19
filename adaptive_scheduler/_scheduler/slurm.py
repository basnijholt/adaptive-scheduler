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

from adaptive_scheduler._scheduler.base_scheduler import BaseScheduler, _maybe_call
from adaptive_scheduler._scheduler.common import run_submit

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    from adaptive_scheduler.utils import EXECUTOR_TYPES


T = TypeVar("T")


def _maybe_as_tuple(
    x: T | tuple[T | Callable[[], T], ...] | None,
    n: int | None,
    *,
    check_type: type | None = None,
) -> tuple[T | Callable[[], T], ...] | T | None:
    if x is None:
        return None
    if check_type is not None and not isinstance(x, check_type | tuple):
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

    ``cores``, ``nodes``, ``cores_per_node``, ``extra_scheduler``,
    ``executor_type``, ``extra_script``, ``exclusive``, ``extra_env_vars``,
    ``num_threads`` and ``partition`` can be either a single value or a tuple of
    values. If a tuple is given, then the length of the tuple should be the same
    as the number of learners (jobs) that are run. This allows for different
    resources for different jobs. The tuple elements are also allowed to be
    callables without arguments, which will be called when the job is submitted.
    These callables should return the value that is needed. See the type hints
    for the allowed types.

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
        The executor that is used, by default `concurrent.futures.ProcessPoolExecutor` is used.
        One can use ``"ipyparallel"``, ``"dask-mpi"``, ``"mpi4py"``,
        ``"loky"``, ``"sequential"``, or ``"process-pool"``.
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

    def __init__(  # noqa: PLR0915
        self,
        *,
        cores: int | tuple[int | None | Callable[[], int | None], ...] | None = None,
        nodes: int | tuple[int | None | Callable[[], int | None], ...] | None = None,
        cores_per_node: int | tuple[int | None | Callable[[], int | None], ...] | None = None,
        partition: str | tuple[str | None | Callable[[], str | None], ...] | None = None,
        exclusive: bool | tuple[bool | Callable[[], bool], ...] = False,
        python_executable: str | None = None,
        log_folder: str | Path = "",
        mpiexec_executable: str | None = None,
        executor_type: EXECUTOR_TYPES
        | tuple[EXECUTOR_TYPES | Callable[[], EXECUTOR_TYPES], ...] = "process-pool",
        num_threads: int | tuple[int | Callable[[], int], ...] = 1,
        extra_scheduler: list[str] | tuple[list[str] | Callable[[], list[str]], ...] | None = None,
        extra_env_vars: list[str] | tuple[list[str] | Callable[[], list[str]], ...] | None = None,
        extra_script: str | tuple[str | Callable[[], str], ...] | None = None,
        batch_folder: str | Path = "",
    ) -> None:
        """Initialize the scheduler."""
        # Store the original values
        self._cores = cores
        self._nodes = nodes
        self._cores_per_node = cores_per_node
        self._partition = partition
        self._executor_type = executor_type
        self._num_threads = num_threads
        self._exclusive = exclusive
        self.__extra_scheduler = extra_scheduler
        self.__extra_env_vars = extra_env_vars
        self.__extra_script = extra_script

        msg = "Specify either `nodes` and `cores_per_node`, or only `cores`, not both."
        if cores is None:
            if nodes is None or cores_per_node is None:
                raise ValueError(msg)
        elif nodes is not None or cores_per_node is not None:
            raise ValueError(msg)

        if extra_scheduler is None:
            extra_scheduler = []
        if extra_env_vars is None:
            extra_env_vars = []
        if extra_script is None:
            extra_script = ""

        # If any is a tuple, then all should be a tuple
        n = _tuple_lengths(
            cores,
            nodes,
            cores_per_node,
            partition,
            executor_type,
            num_threads,
            exclusive,
            extra_scheduler,
            extra_env_vars,
            extra_script,
        )
        single_job_script = n is None
        cores = _maybe_as_tuple(cores, n, check_type=int)  # type: ignore[arg-type]
        self.nodes = _maybe_as_tuple(nodes, n, check_type=int)  # type: ignore[arg-type]
        self.cores_per_node = _maybe_as_tuple(cores_per_node, n, check_type=int)  # type: ignore[arg-type]
        self.partition = _maybe_as_tuple(partition, n, check_type=str)  # type: ignore[arg-type]
        executor_type = _maybe_as_tuple(executor_type, n, check_type=str)  # type: ignore[assignment]
        num_threads = _maybe_as_tuple(num_threads, n, check_type=int)  # type: ignore[assignment]
        self.exclusive = _maybe_as_tuple(exclusive, n, check_type=bool)
        extra_scheduler = _maybe_as_tuple(extra_scheduler, n, check_type=list)
        extra_env_vars = _maybe_as_tuple(extra_env_vars, n, check_type=list)
        extra_script = _maybe_as_tuple(extra_script, n, check_type=str)

        _validate_partition(self.partition, self.partitions)
        if cores is None and single_job_script:
            assert isinstance(self.cores_per_node, int)
            assert isinstance(self.nodes, int)
            cores = self.cores_per_node * self.nodes
        elif not single_job_script:
            # When cores is a tuple with callables, they might return None, in which
            # case we calculate the cores from the nodes and cores_per_node.
            if cores is None:
                assert isinstance(self.cores_per_node, tuple)
                assert isinstance(self.nodes, tuple)
                cores = tuple(
                    _cores(cores=None, cores_per_node=cpn, nodes=n)
                    for cpn, n in zip(self.cores_per_node, self.nodes, strict=True)
                )
            elif self.cores_per_node is not None:
                assert isinstance(cores, tuple)
                assert isinstance(self.cores_per_node, tuple)
                assert isinstance(self.nodes, tuple)
                cores = tuple(
                    _cores(cores=c, cores_per_node=cpn, nodes=n)
                    for c, cpn, n in zip(cores, self.cores_per_node, self.nodes, strict=True)
                )
        assert cores is not None
        super().__init__(
            cores,
            python_executable=python_executable,
            log_folder=log_folder,
            mpiexec_executable=mpiexec_executable,
            executor_type=executor_type,
            num_threads=num_threads,
            extra_scheduler=extra_scheduler,
            extra_env_vars=extra_env_vars,
            extra_script=extra_script,
            batch_folder=batch_folder,
        )
        # SLURM specific
        self.mpiexec_executable = mpiexec_executable or "srun --mpi=pmi2"

    def _extra_scheduler_list(self, *, index: int | None = None) -> list[str]:  # noqa: PLR0912
        slurm_args = []
        if self.cores_per_node is not None:
            if self.single_job_script:
                assert isinstance(self.cores_per_node, int)
                assert isinstance(self.nodes, int)
                cores_per_node = self.cores_per_node
            else:
                assert isinstance(self.cores_per_node, tuple)
                assert isinstance(self.nodes, tuple)
                assert index is not None
                cores_per_node = _maybe_call(self.cores_per_node[index])
            if cores_per_node is not None:
                slurm_args.append(f"--ntasks-per-node={cores_per_node}")

        if self.partition is not None:
            if self.single_job_script:
                assert isinstance(self.partition, str)
                partition = self.partition
            else:
                assert isinstance(self.partition, tuple)
                assert index is not None
                partition = _maybe_call(self.partition[index])
                _validate_partition(partition, self.partitions)
            if partition is not None:
                slurm_args.append(f"--partition={partition}")

        if self.single_job_script:
            assert isinstance(self.exclusive, bool)
            exclusive = self.exclusive
        else:
            assert isinstance(self.exclusive, tuple)
            assert index is not None
            exclusive = _maybe_call(self.exclusive[index])
        if exclusive:
            slurm_args.append("--exclusive")

        if self.single_job_script:
            assert isinstance(self._extra_scheduler, list)
            slurm_args.extend(self._extra_scheduler)
        else:
            assert index is not None
            assert isinstance(self._extra_scheduler, tuple)
            slurm_args.extend(_maybe_call(self._extra_scheduler[index]))
        return slurm_args

    def __getstate__(self) -> dict[str, Any]:
        """Get the state of the SLURM scheduler."""
        state = super().__getstate__()
        state["cores"] = self._cores
        state["nodes"] = self._nodes
        state["cores_per_node"] = self._cores_per_node
        state["partition"] = self._partition
        state["executor_type"] = self._executor_type
        state["num_threads"] = self._num_threads
        state["exclusive"] = self._exclusive
        state["extra_scheduler"] = self.__extra_scheduler
        state["extra_env_vars"] = self.__extra_env_vars
        state["extra_script"] = self.__extra_script
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
                assert isinstance(self.partition, tuple)
                assert isinstance(self.nodes, tuple)
                assert index is not None
                partition = _maybe_call(self.partition[index])
                nodes = _maybe_call(self.nodes[index])
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
                cores = _maybe_call(self.cores[index]) - 1

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
            extra_env_vars=self.extra_env_vars(index=index),
            extra_script=self.extra_script(index=index),
            executor_specific=self._executor_specific("${NAME}", options, index=index),
        )

    def start_job(self, name: str, *, index: int | None = None) -> None:
        """Writes a job script and submits it to the scheduler."""
        if self.single_job_script:
            name_prefix = name.rsplit("-", 1)[0]
        else:
            name_prefix = name
            assert index is not None
            options = self._multi_job_script_options(index)
            self.write_job_script(name_prefix, options, index=index)

        (output_fname,) = self.output_fnames(name)
        output_str = str(output_fname).replace(self._JOB_ID_VARIABLE, "%A")
        output_opt = f"--output {output_str}"
        name_opt = f"--job-name {name}"
        submit_cmd = f"{self.submit_cmd} {name_opt} {output_opt} {self.batch_fname(name_prefix)}"
        run_submit(submit_cmd, name)

    @staticmethod
    def queue(*, me_only: bool = True) -> dict[str, dict[str, str]]:
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

        if "squeue: error" in output or "slurm_load_jobs error" in output or proc.returncode != 0:
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

    @staticmethod
    def cancel_jobs(name: str, *, dry: bool = False) -> None:
        """Cancel jobs with names matching the pattern '{name}-{i}' where i is an integer.

        Parameters
        ----------
        name
            The base name of the jobs to cancel. Jobs with names that start with '{name}-'
            followed by an integer will be canceled.
        dry
            If True, perform a dry run and print the job IDs that would be canceled without
            actually canceling them. Default is False.

        Raises
        ------
        RuntimeError
            If there is an error while canceling the jobs.

        Examples
        --------
        >>> SLURM.cancel_jobs("my_job")
        # Cancels all running jobs with names like "my_job-1", "my_job-2", etc.

        >>> SLURM.cancel_jobs("my_job", dry=True)
        # Prints the job IDs that would be canceled without actually canceling them.

        """
        running_jobs = SLURM.queue()
        job_ids_to_cancel = []

        for job_id, job_info in running_jobs.items():
            job_name = job_info["job_name"]
            if job_name.startswith(f"{name}-"):
                suffix = job_name[len(name) + 1 :]
                if suffix.isdigit():
                    job_ids_to_cancel.append(job_id)

        if job_ids_to_cancel:
            job_ids_str = ",".join(job_ids_to_cancel)
            cmd = f"{SLURM._cancel_cmd} {job_ids_str}"
            if dry:
                print(f"Dry run: would cancel jobs with IDs: {job_ids_str}")
            else:
                try:
                    subprocess.run(cmd.split(), check=True)
                except subprocess.CalledProcessError as e:
                    msg = f"Failed to cancel jobs with name {name}. Error: {e}"
                    raise RuntimeError(
                        msg,
                    ) from e
        else:
            print(f"No running jobs found with name pattern '{name}-<integer>'")


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
    try:
        output = subprocess.run(
            ["sinfo", "-ahO", "partition"],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return {} if with_ncores else []
    lines = output.stdout.decode("utf-8").split("\n")
    partitions = sorted(partition for line in lines if (partition := line.strip()))
    # Sort partitions alphabetically, but put the default partition first
    partitions = sorted(partitions, key=lambda s: ("*" not in s, s))
    # Remove asterisk, which is used for default partition
    partitions = [partition.replace("*", "") for partition in partitions]
    if not with_ncores:
        return partitions

    return {partition: _get_ncores(partition) for partition in partitions}


def _cores(
    cores: int | None | Callable[[], int | None],
    cores_per_node: int | None | Callable[[], int | None],
    nodes: int | None | Callable[[], int | None],
) -> int | Callable[[], int]:
    if isinstance(cores, int):
        return cores
    if callable(cores) or callable(cores_per_node) or callable(nodes):
        return lambda: _maybe_call(cores) or _maybe_call(cores_per_node) * (_maybe_call(nodes) or 1)
    return cores or cores_per_node * (nodes or 1)  # type: ignore[operator]


def _at_least_tuple(x: Any) -> tuple[Any, ...]:
    """Convert x to a tuple if it is not already a tuple."""
    return x if isinstance(x, tuple) else (x,)


def _validate_partition(
    partition: str | tuple[str | None | Callable[[], str | None], ...] | None,
    partitions: dict[str, int],
) -> None:
    if partition is None:
        return
    for p in _at_least_tuple(partition):
        if callable(p) or p is None:
            continue
        if p not in partitions:
            msg = f"Invalid partition: {p}, only {partitions} are available."
            raise ValueError(msg)
