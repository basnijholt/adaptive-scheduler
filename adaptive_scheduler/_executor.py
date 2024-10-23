import abc
from collections.abc import Callable, Iterable
from concurrent.futures import Executor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adaptive import SequenceLearner

import adaptive_scheduler
from adaptive_scheduler.utils import _DATAFRAME_FORMATS, EXECUTOR_TYPES, GoalTypes


class AdaptiveSchedulerExecutorBase(Executor):
    _run_manager: adaptive_scheduler.RunManager | None

    @abc.abstractmethod
    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
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
    ) -> list[Any]:
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


@dataclass
class SLURMExecutor(AdaptiveSchedulerExecutorBase):
    partition: str | None = None
    nodes: int | None = 1
    cores_per_node: int | None = 1
    goal: GoalTypes | None = None
    folder: str | Path = ""
    name: str = "adaptive"
    num_threads: int | None = 1
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
    _tasks: list[tuple[Callable[..., Any], tuple, dict]] = field(default_factory=list)
    _quiet: bool = True
    _run_manager: adaptive_scheduler.RunManager | None = None

    def __post_init__(self) -> None:
        self.folder = Path(self.folder)

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> None:
        self._tasks.append((fn, args, kwargs))

    def _to_learners(self) -> tuple[list[SequenceLearner], list[Path]]:
        sequences = {}
        for func, args, kwargs in self._tasks:
            sequences.setdefault(func, []).append((args, kwargs))
        learners = []
        fnames = []
        for func, args_kwargs_list in sequences.items():
            learner = SequenceLearner(func, args_kwargs_list)
            learners.append(learner)
            fnames.append(self.folder / f"{func.__name__}.pickle")
        return learners, fnames

    def finalize(self, *, start: bool = True) -> adaptive_scheduler.RunManager:
        learners, fnames = self._to_learners()
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
