from __future__ import annotations

import abc
import asyncio
import datetime
import functools
import os
import time
import uuid
from concurrent.futures import Executor, Future
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import cloudpickle
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
        """Submit a task to the executor."""

    @abc.abstractmethod
    def finalize(self, *, start: bool = True) -> adaptive_scheduler.RunManager | None:
        """Finalize the executor and return the RunManager.

        Returns None if no learners were submitted.
        """

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
    learner_index: int
    sequence_index: int


class SlurmTask(Future):
    """A `Future` that loads the result from a `SequenceLearner`."""

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
        self._last_size: float = 0
        self._load_time: float = 0

        loop = asyncio.get_event_loop()
        self._background_task = loop.create_task(self._background_check())

    async def _background_check(self) -> None:
        """Periodically check if the task is done."""
        while not self.done():
            if self.executor._run_manager is not None:
                self._get()
            await asyncio.sleep(1)

    def _get(self) -> Any | None:  # noqa: PLR0911
        """Updates the state of the task and returns the result if the task is finished."""
        if self.done():
            return super().result(timeout=0)

        func_id, global_index = self.task_id
        try:
            learner_idx, local_index = self.executor._task_mapping[(func_id, global_index)]
        except KeyError as e:
            msg = "Task mapping not found; finalize() must be called first."
            raise RuntimeError(msg) from e
        # Now retrieve the correct learner and filename:
        run_manager = self.executor._run_manager
        assert run_manager is not None, "RunManager not initialized"
        learner = run_manager.learners[learner_idx]
        fname = run_manager.fnames[learner_idx]

        if learner.done():
            result = learner.data[local_index]
            self.set_result(result)
            return result

        assert self.executor._run_manager is not None
        last_load_time = self.executor._run_manager._last_load_time.get(learner_idx, 0)
        now = time.monotonic()
        time_since_last_load = now - last_load_time
        if time_since_last_load < self.min_load_interval:
            return None

        try:
            size = os.path.getsize(fname)  # noqa: PTH202
        except FileNotFoundError:
            return None

        if self._last_size == size:
            return None
        self._last_size = size

        learner.load(fname)
        self._load_time = time.monotonic() - now
        self.min_load_interval = max(1.0, 20.0 * self._load_time)
        self.executor._run_manager._last_load_time[learner_idx] = now

        if local_index in learner.data:
            result = learner.data[local_index]
            self.set_result(result)
            return result
        return None

    @functools.cached_property
    def _learner_and_fname(self) -> tuple[SequenceLearner, str | Path]:
        idx_learner, _ = self.task_id
        run_manager = self.executor._run_manager
        assert run_manager is not None, "RunManager not initialized"
        learner: SequenceLearner = run_manager.learners[idx_learner]  # type: ignore[index]
        fname = run_manager.fnames[idx_learner]
        return learner, fname

    def result(self, timeout: float | None = None) -> Any:
        """Return the result of the future if available.

        Since this is an async task, this method will only return if the result
        is immediately available. Use `await task` to wait for the result.
        """
        if timeout is not None:
            msg = "Timeout not implemented for SLURMTask"
            raise NotImplementedError(msg)

        if self.executor._run_manager is None:
            msg = "RunManager not initialized. Call finalize() first."
            raise RuntimeError(msg)

        # Do one check
        self._get()

        if not self.done():
            msg = (
                "Result not immediately available. "
                "Use 'await task' to wait for the result asynchronously."
            )
            raise RuntimeError(msg)

        return super().result(timeout=0)  # timeout=0 makes it non-blocking

    def cancel(self) -> bool:
        """Cancel the future and its background task."""
        self._background_task.cancel()
        return super().cancel()

    def __repr__(self) -> str:
        if not self.done():
            self._get()
        return f"SLURMTask(task_id={self.task_id}, state={self._state})"

    def __str__(self) -> str:
        return self.__repr__()

    def __await__(self) -> Any:
        """Allow using 'await task' to wait for the result."""
        return asyncio.wrap_future(self).__await__()


def _uuid_with_datetime() -> str:
    """Return a UUID with the current datetime."""
    # YYYYMMDD-HHMMSS-UUID
    return f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex}"  # noqa: DTZ005


class _SerializableFunctionSplatter:
    def __init__(self, func: Callable[..., Any]) -> None:
        self.func = func

    def __call__(self, args: Any) -> Any:
        return self.func(*args)

    def __getstate__(self) -> bytes:
        return cloudpickle.dumps(self.func)

    def __setstate__(self, state: bytes) -> None:
        self.func = cloudpickle.loads(state)


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
    size_per_learner
        The size of each learner. If None, the whole sequence is passed to the learner.

    """

    # Same as slurm_run, except it has no learners, fnames, dependencies and initializers.

    # slurm_run: Specific to slurm_run
    name: str = "adaptive-scheduler"
    folder: str | Path | None = None  # `slurm_run` defaults to None
    # slurm_run: SLURM scheduler arguments
    partition: str | tuple[str | Callable[[], str], ...] | None = None
    nodes: int | tuple[int | None | Callable[[], int | None], ...] | None = 1
    cores_per_node: int | tuple[int | None | Callable[[], int | None], ...] | None = (
        1  # `slurm_run` defaults to `None`
    )
    num_threads: int | tuple[int | Callable[[], int], ...] = 1
    exclusive: bool | tuple[bool | Callable[[], bool], ...] = False
    executor_type: EXECUTOR_TYPES | tuple[EXECUTOR_TYPES | Callable[[], EXECUTOR_TYPES], ...] = (
        "process-pool"
    )
    extra_scheduler: list[str] | tuple[list[str] | Callable[[], list[str]], ...] | None = None
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
    save_dataframe: bool = False  # `slurm_run` defaults to `True`
    dataframe_format: _DATAFRAME_FORMATS = "pickle"
    max_log_lines: int = 500
    max_fails_per_job: int = 50
    max_simultaneous_jobs: int = 100
    quiet: bool = True  # `slurm_run` defaults to `False`
    # slurm_run: RunManager arguments
    extra_run_manager_kwargs: dict[str, Any] | None = None
    extra_scheduler_kwargs: dict[str, Any] | None = None
    # Internal
    size_per_learner: int | None = None
    _sequences: dict[Callable[..., Any], list[Any]] = field(default_factory=dict)
    _sequence_mapping: dict[Callable[..., Any], int] = field(default_factory=dict)
    _disk_func_mapping: dict[Callable[..., Any], _DiskFunction] = field(default_factory=dict)
    _run_manager: adaptive_scheduler.RunManager | None = None
    _task_mapping: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.folder is None:
            self.folder = Path.cwd() / ".adaptive_scheduler" / _uuid_with_datetime()  # type: ignore[operator]
        else:
            self.folder = Path(self.folder)

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> SlurmTask:
        if kwargs:
            msg = "Keyword arguments are not supported"
            raise ValueError(msg)
        if fn not in self._sequence_mapping:
            self._sequence_mapping[fn] = len(self._sequence_mapping)
            assert fn not in self._disk_func_mapping
            assert isinstance(self.folder, Path)
            self._disk_func_mapping[fn] = _DiskFunction(
                fn,
                self.folder / f"{_name(fn)}-{uuid.uuid4().hex}.pickle",
            )
        sequence = self._sequences.setdefault(fn, [])
        i = len(sequence)
        sequence.append(args)
        task_id = TaskID(self._sequence_mapping[fn], i)
        return SlurmTask(self, task_id)

    def _to_learners(
        self,
    ) -> tuple[
        list[SequenceLearner],
        list[Path],
        dict[tuple[int, int], tuple[int, int]],
    ]:
        learners = []
        fnames = []
        task_mapping = {}
        learner_idx = 0
        for func, args_list in self._sequences.items():
            func_id = self._sequence_mapping[func]
            # Chunk the sequence if size_per_learner is set; otherwise one chunk.
            if self.size_per_learner is not None:
                chunked_args = [
                    args_list[i : i + self.size_per_learner]
                    for i in range(0, len(args_list), self.size_per_learner)
                ]
            else:
                chunked_args = [args_list]

            global_index = 0  # global index for tasks of this function
            for chunk in chunked_args:
                # Map each task in the chunk: global index -> (current learner, local index)
                for local_index in range(len(chunk)):
                    task_mapping[(func_id, global_index)] = (learner_idx, local_index)
                    global_index += 1

                disk_func = self._disk_func_mapping[func]
                ser_func = _SerializableFunctionSplatter(disk_func)
                learner = SequenceLearner(ser_func, chunk)
                learners.append(learner)
                name = _name(func)
                assert isinstance(self.folder, Path)
                fnames.append(self.folder / f"{name}-{learner_idx}-{uuid.uuid4().hex}.pickle")
                learner_idx += 1
        return learners, fnames, task_mapping

    def finalize(self, *, start: bool = True) -> adaptive_scheduler.RunManager | None:
        if self._run_manager is not None:
            msg = "RunManager already initialized. Create a new SlurmExecutor instance."
            raise RuntimeError(msg)
        learners, fnames, self._task_mapping = self._to_learners()
        if not learners:
            return None
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

    def new(self, update: dict[str, Any] | None = None) -> SlurmExecutor:
        """Create a new SlurmExecutor with the same parameters."""
        data = asdict(self)
        data["_run_manager"] = None
        data["_sequences"] = {}
        data["_sequence_mapping"] = {}
        if update is not None:
            data.update(update)
        return SlurmExecutor(**data)


def _name(func: Callable[..., Any]) -> str:
    return func.__name__ if hasattr(func, "__name__") else "func"


class _DiskFunction:
    def __init__(self, func: Callable[..., Any], fname: str | Path) -> None:
        self.fname = Path(fname)
        with self.fname.open("wb") as f:
            cloudpickle.dump(func, f)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    @functools.cached_property
    def func(self) -> Callable[..., Any]:
        return cloudpickle.loads(self.fname.read_bytes())
