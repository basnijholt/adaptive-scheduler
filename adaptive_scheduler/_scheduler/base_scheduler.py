"""BaseScheduler for Adaptive Scheduler."""

from __future__ import annotations

import abc
import subprocess
import sys
import textwrap
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from adaptive_scheduler._scheduler.common import run_submit
from adaptive_scheduler.utils import EXECUTOR_TYPES, _progress

if TYPE_CHECKING:
    from typing import Any, ClassVar


_MULTI_LINE_BREAK = " \\\n    "


class BaseScheduler(abc.ABC):
    """Base object for a Scheduler.

    Parameters
    ----------
    cores
        Number of cores per job (so per learner.)
    python_executable
        The Python executable that should run adaptive-scheduler. By default
        it uses the same Python as where this function is called.
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
        arguments, e.g. ``["--exclusive=user", "--time=1"]``.
    extra_env_vars
        Extra environment variables that are exported in the job
        script. e.g. ``["TMPDIR='/scratch'", "PYTHONPATH='my_dir:$PYTHONPATH'"]``.
    extra_script
        Extra script that will be executed after any environment variables are set,
        but before the main scheduler is run.
    batch_folder
        The folder in which to put the batch files.

    Returns
    -------
    `BaseScheduler` object.

    """

    _ext: ClassVar[str]
    _submit_cmd: ClassVar[str]
    _options_flag: ClassVar[str]
    _cancel_cmd: ClassVar[str]
    _JOB_ID_VARIABLE: ClassVar[str] = "${JOB_ID}"

    def __init__(
        self,
        cores: int | tuple[int, ...],
        *,
        python_executable: str | None = None,
        log_folder: str | Path = "",
        mpiexec_executable: str | None = None,
        executor_type: EXECUTOR_TYPES | tuple[EXECUTOR_TYPES, ...] = "process-pool",
        num_threads: int | tuple[int, ...] = 1,
        extra_scheduler: list[str] | tuple[list[str], ...] | None = None,
        extra_env_vars: list[str] | tuple[list[str], ...] | None = None,
        extra_script: str | tuple[str, ...] | None = None,
        batch_folder: str | Path = "",
    ) -> None:
        """Initialize the scheduler."""
        self.cores = cores
        self.python_executable = python_executable or sys.executable
        self.log_folder = log_folder
        self.batch_folder = batch_folder
        self.mpiexec_executable = mpiexec_executable or "mpiexec"
        self.executor_type = executor_type
        self.num_threads = num_threads
        self._extra_scheduler = extra_scheduler
        self._extra_env_vars = extra_env_vars
        self._extra_script = extra_script if extra_script is not None else ""
        # This attribute is set in JobManager._setup ATM (hacky)
        self._command_line_options: dict[str, Any] | None = None

    @abc.abstractmethod
    def queue(self, *, me_only: bool = True) -> dict[str, dict]:
        """Get the current running and pending jobs.

        Parameters
        ----------
        me_only
            Only see your jobs.

        Returns
        -------
        queue
            Mapping of ``job_id`` -> `dict` with ``name`` and ``state``, for
            example ``{job_id: {"job_name": "TEST_JOB-1", "state": "R" or "Q"}}``.

        Notes
        -----
        This function might return extra information about the job, however
        this is not used elsewhere in this package.

        """

    def queue_df(self) -> pd.DataFrame:
        """Get the current running and pending jobs as a `pandas.DataFrame`."""
        queue = self.queue()
        return pd.DataFrame(queue).transpose()

    @property
    def ext(self) -> str:
        """The extension of the job script."""
        return self._ext

    @property
    def submit_cmd(self) -> str:
        """Command to start a job, e.g. ``qsub fname.batch`` or ``sbatch fname.sbatch``."""
        return self._submit_cmd

    @abc.abstractmethod
    def job_script(self, options: dict[str, Any], *, index: int | None = None) -> str:
        """Get a jobscript in string form.

        Returns
        -------
        job_script
            A job script that can be submitted to the scheduler.
        index
            The index of the job that is being run. This is used when
            specifying different resources for different jobs.

        """

    @property
    def single_job_script(self) -> bool:
        return isinstance(self.cores, int)

    def _get_executor_type(self, *, index: int | None = None) -> str:
        if self.single_job_script:
            assert isinstance(self.executor_type, str)
            return self.executor_type
        assert index is not None
        return self.executor_type[index]

    def batch_fname(self, name: str) -> Path:
        """The filename of the job script."""
        if self.batch_folder:
            batch_folder = Path(self.batch_folder)
            batch_folder.mkdir(exist_ok=True, parents=True)
        else:
            batch_folder = Path.cwd()
        return batch_folder / f"{name}{self.ext}"

    @staticmethod
    def sanatize_job_id(job_id: str) -> str:
        """Sanatize the job_id."""
        return job_id

    def job_names_to_job_ids(self, *job_names: str) -> list[str]:
        """Get the job_ids from the job_names in the queue."""
        queue = self.queue()
        job_name_to_id = {info["job_name"]: job_id for job_id, info in queue.items()}
        return [
            job_name_to_id[job_name]
            for job_name in job_names
            if job_name in job_name_to_id
        ]

    def cancel(
        self,
        job_names: list[str],
        *,
        with_progress_bar: bool = True,
        max_tries: int = 5,
    ) -> None:
        """Cancel all jobs in `job_names`.

        Parameters
        ----------
        job_names
            List of job names.
        with_progress_bar
            Display a progress bar using `tqdm`.
        max_tries
            Maximum number of attempts to cancel a job.

        """

        def cancel_jobs(job_ids: list[str]) -> None:
            for job_id in _progress(job_ids, with_progress_bar, "Canceling jobs"):
                cmd = f"{self._cancel_cmd} {job_id}".split()
                returncode = subprocess.run(
                    cmd,
                    stderr=subprocess.PIPE,
                    check=False,
                ).returncode
                if returncode != 0:
                    warnings.warn(
                        f"Couldn't cancel '{job_id}'.",
                        UserWarning,
                        stacklevel=2,
                    )

        job_names_set = set(job_names)
        for _ in range(max_tries):
            job_ids = self.job_names_to_job_ids(*job_names_set)
            if not job_ids:
                # no more running jobs
                break
            cancel_jobs(job_ids)
            time.sleep(0.5)

    def _expand_options(
        self,
        custom: tuple[str, ...],
        name: str,
        options: dict[str, Any],
    ) -> str:
        """Expand the options.

        This is used to expand the options that are passed to the job script.
        """
        log_fname = self.log_fname(name)
        return _MULTI_LINE_BREAK.join(
            (
                *custom,
                f"--log-fname {log_fname}",
                f"--job-id {self._JOB_ID_VARIABLE}",
                f"--name {name}",
                *(f"{k} {v}" if v is not None else k for k, v in options.items()),
            ),
        )

    def _get_cores(self, index: int | None = None) -> int:
        if self.single_job_script:
            cores = self.cores
        else:
            assert index is not None
            assert isinstance(self.cores, tuple)
            cores = self.cores[index]
        assert isinstance(cores, int)
        return cores

    def _mpi4py(self, *, index: int | None = None) -> tuple[str, ...]:
        cores = self._get_cores(index=index)
        return (
            f"{self.mpiexec_executable}",
            f"-n {cores} {self.python_executable}",
            f"-m mpi4py.futures {self.launcher}",
        )

    def _dask_mpi(self, *, index: int | None = None) -> tuple[str, ...]:
        cores = self._get_cores(index=index)
        return (
            f"{self.mpiexec_executable}",
            f"-n {cores} {self.python_executable} {self.launcher}",
        )

    def _ipyparallel(self, *, index: int | None = None) -> tuple[str, tuple[str, ...]]:
        cores = self._get_cores(index=index)
        job_id = self._JOB_ID_VARIABLE
        profile = "${profile}"
        start = textwrap.dedent(
            f"""\
            profile=adaptive_scheduler_{job_id}

            echo "Creating profile {profile}"
            ipython profile create {profile}

            echo "Launching controller"
            ipcontroller --ip="*" --profile={profile} --log-to-file &
            sleep 10

            echo "Launching engines"
            {self.mpiexec_executable} \\
                -n {cores-1} \\
                ipengine \\
                --profile={profile} \\
                --mpi \\
                --cluster-id='' \\
                --log-to-file &

            echo "Starting the Python script"
            {self.python_executable} {self.launcher} \\
            """,
        )
        custom = (f"    --profile {profile}", f"--n {cores-1}")
        return start, custom

    def _process_pool(self) -> tuple[str, ...]:
        return (f"{self.python_executable} {self.launcher}",)

    def _sequential_executor(self) -> tuple[str, ...]:
        return (f"{self.python_executable} {self.launcher}",)

    def _executor_specific(
        self,
        name: str,
        options: dict[str, Any],
        *,
        index: int | None = None,
    ) -> str:
        start = ""
        executor_type = self._get_executor_type(index=index)
        if executor_type == "mpi4py":
            opts = self._mpi4py(index=index)
        elif executor_type == "dask-mpi":
            opts = self._dask_mpi(index=index)
        elif executor_type == "ipyparallel":
            cores = self._get_cores(index=index)
            if cores <= 1:
                msg = (
                    "`ipyparalllel` uses 1 cores of the `adaptive.Runner` and"
                    " the rest of the cores for the engines, so use more than 1 core."
                )
                raise ValueError(msg)
            start, opts = self._ipyparallel(index=index)
        elif executor_type in ("process-pool", "loky"):
            opts = self._process_pool()
        elif executor_type == "sequential":
            opts = self._sequential_executor()
        else:
            msg = "Use 'ipyparallel', 'dask-mpi', 'mpi4py', 'loky', 'sequential', or 'process-pool'."
            raise NotImplementedError(msg)
        return start + self._expand_options(opts, name, options)

    def log_fname(self, name: str) -> Path:
        """The filename of the log (with JOB_ID_VARIABLE)."""
        if self.log_folder:
            log_folder = Path(self.log_folder)
            log_folder.mkdir(exist_ok=True)
        else:
            log_folder = Path.cwd()
        return log_folder / f"{name}-{self._JOB_ID_VARIABLE}.log"

    def output_fnames(self, name: str) -> list[Path]:
        """Scheduler output file names (with JOB_ID_VARIABLE)."""
        log_fname = self.log_fname(name)
        return [log_fname.with_suffix(".out")]

    @property
    def launcher(self) -> Path:
        from adaptive_scheduler import _server_support

        return Path(_server_support.__file__).parent / "launcher.py"

    def extra_scheduler(self, *, index: int | None = None) -> str:
        """Scheduler options that go in the job script."""
        if self._extra_scheduler is None:
            return ""
        if self.single_job_script:
            extra_scheduler = self._extra_scheduler
        else:
            assert index is not None
            extra_scheduler = self._extra_scheduler[index]  # type: ignore[assignment]
        assert isinstance(extra_scheduler, list)
        return "\n".join(f"#{self._options_flag} {arg}" for arg in extra_scheduler)

    def _get_num_threads(self, *, index: int | None = None) -> int:
        if self.single_job_script:
            assert isinstance(self.num_threads, int)
            return self.num_threads
        assert index is not None
        return self.num_threads[index]  # type: ignore[index]

    def extra_env_vars(self, *, index: int | None = None) -> str:
        """Environment variables that need to exist in the job script."""
        extra_env_vars: list[str]
        if self._extra_env_vars is None:
            extra_env_vars = []
        elif self.single_job_script:
            assert isinstance(self._extra_env_vars, list)
            extra_env_vars = self._extra_env_vars.copy()
        else:
            assert index is not None
            extra_env_vars = self._extra_env_vars[index].copy()  # type: ignore[union-attr]
        num_threads = self._get_num_threads(index=index)
        extra_env_vars.extend(
            [
                f"EXECUTOR_TYPE={self._get_executor_type(index=index)}",
                f"MKL_NUM_THREADS={num_threads}",
                f"OPENBLAS_NUM_THREADS={num_threads}",
                f"OMP_NUM_THREADS={num_threads}",
                f"NUMEXPR_NUM_THREADS={num_threads}",
            ],
        )
        return "\n".join(f"export {arg}" for arg in extra_env_vars)

    def extra_script(self, *, index: int | None = None) -> str:
        """Script that will be run before the main scheduler."""
        assert self._extra_script is not None
        if self.single_job_script:
            assert isinstance(self._extra_script, str)
            return self._extra_script
        assert index is not None
        assert isinstance(self._extra_script, tuple), self._extra_script
        return self._extra_script[index]

    def write_job_script(
        self,
        name: str,
        options: dict[str, Any],
        index: int | None = None,
    ) -> None:
        """Writes a job script."""
        with self.batch_fname(name).open("w", encoding="utf-8") as f:
            job_script = self.job_script(options, index=index)
            f.write(job_script)

    def _multi_job_script_options(self, index: int) -> dict[str, Any]:
        assert self._command_line_options is not None
        assert isinstance(self.cores, tuple)
        options = dict(self._command_line_options)  # copy
        executor_type = self._get_executor_type(index=index)
        options["--executor-type"] = executor_type
        options["--n"] = self._get_cores(index=index)
        if executor_type == "ipyparallel":
            options["--n"] -= 1
        return options

    def start_job(self, name: str, *, index: int | None = None) -> None:
        """Writes a job script and submits it to the scheduler."""
        if not self.single_job_script:
            assert index is not None
            options = self._multi_job_script_options(index)
            self.write_job_script(name, options, index=index)

        submit_cmd = f"{self.submit_cmd} {name} {self.batch_fname(name)}"
        run_submit(submit_cmd)

    def __getstate__(self) -> dict[str, Any]:
        """Return the state of the scheduler."""
        return {
            "cores": self.cores,
            "python_executable": self.python_executable,
            "log_folder": self.log_folder,
            "mpiexec_executable": self.mpiexec_executable,
            "executor_type": self.executor_type,
            "num_threads": self.num_threads,
            "extra_scheduler": self._extra_scheduler,
            "extra_env_vars": self._extra_env_vars,
            "extra_script": self._extra_script,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state of the scheduler."""
        self.__init__(**state)  # type: ignore[misc]
