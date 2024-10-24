from __future__ import annotations

import abc
import asyncio
import os
import time
import uuid
from concurrent.futures import Executor, Future
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from adaptive import SequenceLearner

import adaptive_scheduler

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from adaptive_scheduler.utils import (
        _DATAFRAME_FORMATS,
        EXECUTOR_TYPES,
        LOKY_START_METHODS,
        GoalTypes,
    )


class AdaptiveSchedulerExecutorBase(Executor):
    _run_manager: adaptive_scheduler.RunManager | None

    @abc.abstractmethod
    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:
        pass

    @abc.abstractmethod
    def finalize(self, *, start: bool = True) -> adaptive_scheduler.RunManager:
        """Finalize the executor and return the RunManager."""

    def map(  # type: ignore[override]
        self,
        fn: Callable[..., Any],
        /,
        *iterables: Iterable[Any],
        timeout: float | None = None,
        chunksize: int = 1,
    ) -> list[Future]:
        tasks = []
        if timeout is not None:
            msg = "Timeout not implemented"
            raise NotImplementedError(msg)
        if chunksize != 1:
            msg = "Chunksize not implemented"
            raise NotImplementedError(msg)
        for args in zip(*iterables, strict=True):
            task = self.submit(fn, *args)
            tasks.append(task)
        return tasks

    def shutdown(
        self,
        wait: bool = True,  # noqa: FBT001, FBT002
        *,
        cancel_futures: bool = False,
    ) -> None:
        if not wait:
            msg = "Non-waiting shutdown not implemented"
            raise NotImplementedError(msg)
        if cancel_futures:
            msg = "Cancelling futures not implemented"
            raise NotImplementedError(msg)
        if self._run_manager is not None:
            self._run_manager.cancel()


class TaskID(NamedTuple):
    learner_inded: int
    sequence_index: int


class SlurmTask(Future):
    """A `Future` that loads the result from a `SequenceLearner`."""

    __slots__ = ("executor", "task_id", "_state", "_last_mtime", "min_load_interval")

    def __init__(
        self,
        executor: SlurmExecutor,
        task_id: TaskID,
        min_load_interval: float = 1.0,
    ) -> None:
        super().__init__()
        self.executor = executor
        self.task_id = task_id
        self.min_load_interval: float = min_load_interval
        self._state: Literal["PENDING", "RUNNING", "FINISHED", "CANCELLED"] = "PENDING"
        self._last_mtime: float = 0

    def _get(self) -> Any | None:
        """Updates the state of the task and returns the result if the task is finished."""
        i_learner, index = self.task_id
        learner, fname = self._learner_and_fname(load=False)
        if self._state == "FINISHED":
            return learner.data[index]

        assert self.executor._run_manager is not None
        last_load_time = self.executor._run_manager._last_load_time.get(i_learner, 0)
        now = time.monotonic()
        time_since_last_load = now - last_load_time
        if time_since_last_load < self.min_load_interval:
            return None

        try:
            mtime = os.path.getmtime(fname)  # noqa: PTH204
        except FileNotFoundError:
            return None

        if self._last_mtime == mtime:
            return None
        self._last_mtime = mtime

        learner.load(fname)
        self.executor._run_manager._last_load_time[i_learner] = now

        if index in learner.data:
            result = learner.data[index]
            self.set_result(result)
            return result
        return None

    def __repr__(self) -> str:
        if self._state == "PENDING":
            self._get()
        return f"SLURMTask(task_id={self.task_id}, state={self._state})"

    def __str__(self) -> str:
        return self.__repr__()

    def _learner_and_fname(self, *, load: bool = True) -> tuple[SequenceLearner, str | Path]:
        i_learner, _ = self.task_id
        run_manager = self.executor._run_manager
        assert run_manager is not None
        learner: SequenceLearner = run_manager.learners[i_learner]  # type: ignore[index]
        fname = run_manager.fnames[i_learner]
        if load and not learner.done():
            learner.load(fname)
        return learner, fname

    def result(self, timeout: float | None = None) -> Any:
        if timeout is not None:
            msg = "Timeout not implemented for SLURMTask"
            raise NotImplementedError(msg)
        if self.executor._run_manager is None:
            msg = "RunManager not initialized. Call finalize() first."
            raise RuntimeError(msg)
        result = self._get()
        if self._state == "FINISHED":
            return result
        msg = "Task not finished"
        raise RuntimeError(msg)

    def __await__(self) -> Any:
        def wakeup() -> None:
            if not self.done():
                self._get()
                loop.call_later(1, wakeup)  # Schedule next check after 1 second
            else:
                fut.set_result(self.result())

        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        loop.call_soon(wakeup)
        yield from fut
        return self.result()

    async def __aiter__(self) -> Any:
        await self
        return self.result()


@dataclass
class SlurmExecutor(AdaptiveSchedulerExecutorBase):
    """An executor that runs jobs on SLURM.

    Similar to `concurrent.futures.Executor`, but for SLURM.
    A key difference is that ``submit()`` returns a `SLURMTask` instead of a `Future`
    and that ``finalize()`` must be called in order to start the jobs.

    Parameters
    ----------
    name
        The name of the job.
    folder
        The folder to save the adaptive_scheduler files such as logs, database,
        ``.sbatch``, pickled tasks, and results files in. If the folder exists and has
        results, the results will be loaded!
    partition
        The partition to use. If None, then the default partition will be used.
        (The one marked with a * in `sinfo`). Use
        `adaptive_scheduler.scheduler.slurm_partitions` to see the
        available partitions.
    nodes
        The number of nodes to use.
    cores_per_node
        The number of cores per node to use. If None, then all cores on the partition
        will be used.
    num_threads
        The number of threads to use.
    exclusive
        Whether to use exclusive nodes, adds ``"--exclusive"`` if True.
    executor_type
        The executor that is used, by default `concurrent.futures.ProcessPoolExecutor` is used.
        One can use ``"ipyparallel"``, ``"dask-mpi"``, ``"mpi4py"``,
        ``"loky"``, ``"sequential"``, or ``"process-pool"``.
    extra_scheduler
        Extra ``#SLURM`` (depending on scheduler type)
        arguments, e.g. ``["--exclusive=user", "--time=1"]`` or a tuple of lists,
        e.g. ``(["--time=10"], ["--time=20"]])`` for two jobs.
    goal
        The goal passed to the `adaptive.Runner`. Note that this function will
        be serialized and pasted in the ``job_script``. Can be a smart-goal
        that accepts
        ``Callable[[adaptive.BaseLearner], bool] | float | datetime | timedelta | None``.
        See `adaptive_scheduler.utils.smart_goal` for more information.
    check_goal_on_start
        Checks whether a learner is already done. Only works if the learner is loaded.
    runner_kwargs
        Extra keyword argument to pass to the `adaptive.Runner`. Note that this dict
        will be serialized and pasted in the ``job_script``.
    url
        The url of the database manager, with the format
        ``tcp://ip_of_this_machine:allowed_port.``. If None, a correct url will be chosen.
    save_interval
        Time in seconds between saving of the learners.
    log_interval
        Time in seconds between log entries.
    job_manager_interval
        Time in seconds between checking and starting jobs.
    kill_interval
        Check for `kill_on_error` string inside the log-files every `kill_interval` seconds.
    kill_on_error
        If ``error`` is a string and is found in the log files, the job will
        be cancelled and restarted. If it is a callable, it is applied
        to the log text. Must take a single argument, a list of
        strings, and return True if the job has to be killed, or
        False if not. Set to None if no `KillManager` is needed.
    overwrite_db
        Overwrite the existing database.
    job_manager_kwargs
        Keyword arguments for the `JobManager` function that aren't set in ``__init__`` here.
    kill_manager_kwargs
        Keyword arguments for the `KillManager` function that aren't set in ``__init__`` here.
    loky_start_method
        Loky start method, by default "loky".
    cleanup_first
        Cancel all previous jobs generated by the same RunManager and clean logfiles.
    save_dataframe
        Whether to periodically save the learner's data as a `pandas.DataFame`.
    dataframe_format
        The format in which to save the `pandas.DataFame`. See the type hint for the options.
    max_log_lines
        The maximum number of lines to display in the log viewer widget.
    max_fails_per_job
        Maximum number of times that a job can fail. This is here as a fail switch
        because a job might fail instantly because of a bug inside your code.
        The job manager will stop when
        ``n_jobs * total_number_of_jobs_failed > max_fails_per_job`` is true.
    max_simultaneous_jobs
        Maximum number of simultaneously running jobs. By default no more than 500
        jobs will be running. Keep in mind that if you do not specify a ``runner.goal``,
        jobs will run forever, resulting in the jobs that were not initially started
        (because of this `max_simultaneous_jobs` condition) to not ever start.
    quiet
        Whether to show a progress bar when creating learner files.
    extra_run_manager_kwargs
        Extra keyword arguments to pass to the `RunManager`.
    extra_scheduler_kwargs
        Extra keyword arguments to pass to the `adaptive_scheduler.scheduler.SLURM`.

    """

    # Same as slurm_run, except it has no learners, fnames, dependencies and initializers.
    # Additionally, the type hints for scheduler arguments are singular instead of tuples.

    # slurm_run: Specific to slurm_run
    name: str = "adaptive-scheduler"
    folder: str | Path | None = None  # slurm_run has no None default
    # slurm_run: SLURM scheduler arguments
    partition: str | None = None
    nodes: int | None = 1
    cores_per_node: int | None = None
    num_threads: int = 1
    exclusive: bool = False
    executor_type: EXECUTOR_TYPES = "process-pool"
    extra_scheduler: list[str] | None = None
    # slurm_run: Same as RunManager below (except dependencies and initializers)
    goal: GoalTypes | None = None
    check_goal_on_start: bool = True
    runner_kwargs: dict | None = None
    url: str | None = None
    save_interval: float = 300
    log_interval: float = 300
    job_manager_interval: float = 60
    kill_interval: float = 60
    kill_on_error: str | Callable[[list[str]], bool] | None = "srun: error:"
    overwrite_db: bool = True
    job_manager_kwargs: dict[str, Any] | None = None
    kill_manager_kwargs: dict[str, Any] | None = None
    loky_start_method: LOKY_START_METHODS = "loky"
    cleanup_first: bool = True
    save_dataframe: bool = True
    dataframe_format: _DATAFRAME_FORMATS = "pickle"
    max_log_lines: int = 500
    max_fails_per_job: int = 50
    max_simultaneous_jobs: int = 100
    quiet: bool = True  # `slurm_run` defaults to `False`
    # slurm_run: RunManager arguments
    extra_run_manager_kwargs: dict[str, Any] | None = None
    extra_scheduler_kwargs: dict[str, Any] | None = None
    # Internal
    _sequences: dict[Callable[..., Any], list[Any]] = field(default_factory=dict)
    _sequence_mapping: dict[Callable[..., Any], int] = field(default_factory=dict)
    _run_manager: adaptive_scheduler.RunManager | None = None

    def __post_init__(self) -> None:
        if self.folder is None:
            self.folder = Path.cwd() / ".adaptive_scheduler" / uuid.uuid4().hex  # type: ignore[operator]
        else:
            self.folder = Path(self.folder)

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> SlurmTask:
        if kwargs:
            msg = "Keyword arguments are not supported"
            raise ValueError(msg)
        if fn not in self._sequence_mapping:
            self._sequence_mapping[fn] = len(self._sequence_mapping)
        if len(args) != 1:
            msg = "Exactly one argument is required"
            raise ValueError(msg)
        sequence = self._sequences.setdefault(fn, [])
        i = len(sequence)
        sequence.append(args[0])
        task_id = TaskID(self._sequence_mapping[fn], i)
        return SlurmTask(self, task_id)

    def _to_learners(self) -> tuple[list[SequenceLearner], list[Path]]:
        learners = []
        fnames = []
        for func, args_kwargs_list in self._sequences.items():
            learner = SequenceLearner(func, args_kwargs_list)
            learners.append(learner)
            assert isinstance(self.folder, Path)
            name = func.__name__ if hasattr(func, "__name__") else ""
            fnames.append(self.folder / f"{name}-{uuid.uuid4().hex}.pickle")
        return learners, fnames

    def finalize(self, *, start: bool = True) -> adaptive_scheduler.RunManager:
        if self._run_manager is not None:
            msg = "RunManager already initialized. Create a new SlurmExecutor instance."
            raise RuntimeError(msg)
        learners, fnames = self._to_learners()
        assert self.folder is not None
        self._run_manager = adaptive_scheduler.slurm_run(
            learners=learners,
            fnames=fnames,
            # Specific to slurm_run
            name=self.name,
            folder=self.folder,
            # SLURM scheduler arguments
            partition=self.partition,
            nodes=self.nodes,
            cores_per_node=self.cores_per_node,
            num_threads=self.num_threads,
            exclusive=self.exclusive,
            executor_type=self.executor_type,
            extra_scheduler=self.extra_scheduler,
            # Same as RunManager below (except job_name, move_old_logs_to, and db_fname)
            goal=self.goal,
            check_goal_on_start=self.check_goal_on_start,
            runner_kwargs=self.runner_kwargs,
            url=self.url,
            save_interval=self.save_interval,
            log_interval=self.log_interval,
            job_manager_interval=self.job_manager_interval,
            kill_interval=self.kill_interval,
            kill_on_error=self.kill_on_error,
            overwrite_db=self.overwrite_db,
            job_manager_kwargs=self.job_manager_kwargs,
            kill_manager_kwargs=self.kill_manager_kwargs,
            loky_start_method=self.loky_start_method,
            cleanup_first=self.cleanup_first,
            save_dataframe=self.save_dataframe,
            dataframe_format=self.dataframe_format,
            max_log_lines=self.max_log_lines,
            max_fails_per_job=self.max_fails_per_job,
            max_simultaneous_jobs=self.max_simultaneous_jobs,
            quiet=self.quiet,
            # RunManager arguments
            extra_run_manager_kwargs=self.extra_run_manager_kwargs,
            extra_scheduler_kwargs=self.extra_scheduler_kwargs,
        )
        if start:
            self._run_manager.start()
        return self._run_manager

    def cleanup(self) -> None:
        assert self._run_manager is not None
        self._run_manager.cleanup(remove_old_logs_folder=True)

    def new(self) -> SlurmExecutor:
        """Create a new SlurmExecutor with the same parameters."""
        data = asdict(self)
        data["_run_manager"] = None
        data["_sequences"] = {}
        data["_sequence_mapping"] = {}
        return SlurmExecutor(**data)
