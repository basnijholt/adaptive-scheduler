from __future__ import annotations

import abc
import asyncio
import os
import uuid
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from adaptive import SequenceLearner

import adaptive_scheduler

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from adaptive_scheduler.utils import _DATAFRAME_FORMATS, EXECUTOR_TYPES, GoalTypes


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
    def __init__(self, executor: SLURMExecutor, id_: tuple[int, int]) -> None:
        super().__init__()
        self.executor = executor
        self.id_ = id_
        self._state: Literal["PENDING", "RUNNING", "FINISHED", "CANCELLED"] = "PENDING"
        self._last_mtime: float = 0

    def _get(self) -> Any | None:
        """Updates the state of the task and returns the result if the task is finished."""
        index = self.id_[1]
        learner, fname = self._learner_and_fname(load=False)

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
    partition: str | None = None
    nodes: int | None = 1
    cores_per_node: int | None = 1
    goal: GoalTypes | None = None
    folder: Path | None = None
    name: str = "adaptive"
    num_threads: int = 1
    save_interval: float = 300
    log_interval: float = 300
    job_manager_interval: float = 60
    cleanup_first: bool = True
    save_dataframe: bool = True
    dataframe_format: _DATAFRAME_FORMATS = "pickle"
    max_fails_per_job: int = 50
    max_simultaneous_jobs: int = 100
    exclusive: bool = False
    executor_type: EXECUTOR_TYPES = "process-pool"
    extra_scheduler: list[str] | None = None
    extra_run_manager_kwargs: dict[str, Any] | None = None
    extra_scheduler_kwargs: dict[str, Any] | None = None
    _sequences: dict[Callable[..., Any], list[Any]] = field(default_factory=dict)
    _sequence_mapping: dict[Callable[..., Any], int] = field(default_factory=dict)
    _quiet: bool = True
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
            assert self.folder is not None
            fnames.append(self.folder / f"{func.__name__}.pickle")
        return learners, fnames

    def finalize(self, *, start: bool = True) -> adaptive_scheduler.RunManager:
        learners, fnames = self._to_learners()
        assert self.folder is not None
        self._run_manager = adaptive_scheduler.slurm_run(
            learners=learners,
            fnames=fnames,
            partition=self.partition,
            nodes=self.nodes,
            cores_per_node=self.cores_per_node,
            goal=self.goal,
            folder=self.folder,
            name=self.name,
            num_threads=self.num_threads,
            save_interval=self.save_interval,
            log_interval=self.log_interval,
            job_manager_interval=self.job_manager_interval,
            cleanup_first=self.cleanup_first,
            save_dataframe=self.save_dataframe,
            dataframe_format=self.dataframe_format,
            max_fails_per_job=self.max_fails_per_job,
            max_simultaneous_jobs=self.max_simultaneous_jobs,
            exclusive=self.exclusive,
            executor_type=self.executor_type,
            extra_scheduler=self.extra_scheduler,
            extra_run_manager_kwargs=self.extra_run_manager_kwargs,
            extra_scheduler_kwargs=self.extra_scheduler_kwargs,
            quiet=self._quiet,
        )
        if start:
            self._run_manager.start()
        return self._run_manager
