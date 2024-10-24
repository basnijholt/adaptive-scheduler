from __future__ import annotations

import abc
import asyncio
import os
import time
import uuid
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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


class SLURMTask(Future):
    """A `Future` that loads the result from a `SequenceLearner`."""

    __slots__ = ("executor", "id_", "_state", "_last_mtime", "min_load_interval")

    def __init__(
        self,
        executor: SLURMExecutor,
        id_: tuple[int, int],
        min_load_interval: float = 1.0,
    ) -> None:
        super().__init__()
        self.executor = executor
        self.id_ = id_
        self._state: Literal["PENDING", "RUNNING", "FINISHED", "CANCELLED"] = "PENDING"
        self._last_mtime: float = 0
        self.min_load_interval: float = min_load_interval

    def _get(self) -> Any | None:
        """Updates the state of the task and returns the result if the task is finished."""
        i_learner, index = self.id_
        learner, fname = self._learner_and_fname(load=False)
        assert self.executor._run_manager is not None
        last_load_time = self.executor._run_manager._last_load_time.get(i_learner, 0)
        now = time.monotonic()
        time_since_last_load = now - last_load_time
        if time_since_last_load < self.min_load_interval:
            return None
        if self._state == "FINISHED":
            return learner.data[index]

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
        return f"SLURMTask(id_={self.id_}, state={self._state})"

    def _learner_and_fname(self, *, load: bool = True) -> tuple[SequenceLearner, str | Path]:
        i_learner, _ = self.id_
        run_manager = self.executor._run_manager
        assert run_manager is not None
        learner = run_manager.learners[i_learner]
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
class SLURMExecutor(AdaptiveSchedulerExecutorBase):
    # Same as slurm_run, except it has no dependencies and initializers.
    # Additionally, the type hints for scheduler arguments are singular instead of tuples.

    # Specific to slurm_run
    name: str = "adaptive-scheduler"
    folder: str | Path = ""
    # SLURM scheduler arguments
    partition: str | None = None
    nodes: int | None = 1
    cores_per_node: int | None = None
    num_threads: int = 1
    exclusive: bool = False
    executor_type: EXECUTOR_TYPES = "process-pool"
    extra_scheduler: list[str] | None = None
    # Same as RunManager below (except dependencies and initializers)
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
    # RunManager arguments
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

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> SLURMTask:
        if kwargs:
            msg = "Keyword arguments are not supported"
            raise ValueError(msg)
        if fn not in self._sequence_mapping:
            self._sequence_mapping[fn] = len(self._sequence_mapping)
        sequence = self._sequences.setdefault(fn, [])
        i = len(sequence)
        sequence.append(args)
        id_ = (self._sequence_mapping[fn], i)
        return SLURMTask(self, id_)

    def _to_learners(self) -> tuple[list[SequenceLearner], list[Path]]:
        learners = []
        fnames = []
        for func, args_kwargs_list in self._sequences.items():
            learner = SequenceLearner(func, args_kwargs_list)
            learners.append(learner)
            assert isinstance(self.folder, Path)
            fnames.append(self.folder / f"{func.__name__}.pickle")
        return learners, fnames

    def finalize(self, *, start: bool = True) -> adaptive_scheduler.RunManager:
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

    def cleanup(self) -> None: ...
