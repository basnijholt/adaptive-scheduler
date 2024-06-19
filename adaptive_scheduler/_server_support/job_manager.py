from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from adaptive_scheduler.utils import _now, _serialize_to_b64, sleep_unless_task_is_done

from .base_manager import BaseManager
from .common import MaxRestartsReachedError, log

if TYPE_CHECKING:
    from adaptive_scheduler.scheduler import BaseScheduler
    from adaptive_scheduler.utils import (
        _DATAFRAME_FORMATS,
        LOKY_START_METHODS,
        GoalTypes,
    )

    from .database_manager import DatabaseManager


def command_line_options(
    *,
    scheduler: BaseScheduler,
    database_manager: DatabaseManager,
    runner_kwargs: dict[str, Any] | None = None,
    goal: GoalTypes,
    log_interval: float = 60,
    save_interval: float = 300,
    save_dataframe: bool = True,
    dataframe_format: _DATAFRAME_FORMATS = "pickle",
    loky_start_method: LOKY_START_METHODS = "loky",
) -> dict[str, Any]:
    """Return the command line options for the job_script.

    Parameters
    ----------
    scheduler
        A scheduler instance from `adaptive_scheduler.scheduler`.
    database_manager
        A database manager instance.
    runner_kwargs
        Extra keyword argument to pass to the `adaptive.Runner`. Note that this dict
        will be serialized and pasted in the ``job_script``.
    goal
        The goal passed to the `adaptive.Runner`. Note that this function will
        be serialized and pasted in the ``job_script``. Can be a smart-goal
        that accepts
        ``Callable[[adaptive.BaseLearner], bool] | float | datetime | timedelta | None``.
        See `adaptive_scheduler.utils.smart_goal` for more information.
    log_interval
        Time in seconds between log entries.
    save_interval
        Time in seconds between saving of the learners.
    save_dataframe
        Whether to periodically save the learner's data as a `pandas.DataFame`.
    dataframe_format
        The format in which to save the `pandas.DataFame`. See the type hint for the options.
    loky_start_method
        Loky start method, by default "loky".

    Returns
    -------
    dict
        The command line options for the job_script.

    """
    if runner_kwargs is None:
        runner_kwargs = {}
    runner_kwargs["goal"] = goal
    base64_runner_kwargs = _serialize_to_b64(runner_kwargs)

    opts = {
        "--url": database_manager.url,
        "--log-interval": log_interval,
        "--save-interval": save_interval,
        "--serialized-runner-kwargs": base64_runner_kwargs,
    }
    if scheduler.single_job_script:
        # if `cores` or `executor_type` is a tuple then we set it
        # in `Scheduler.start_job`
        assert isinstance(scheduler.cores, int)
        n = scheduler.cores
        if scheduler.executor_type == "ipyparallel":
            n -= 1
        opts["--n"] = n
        opts["--executor-type"] = scheduler.executor_type

    opts["--loky-start-method"] = loky_start_method

    if save_dataframe:
        opts["--dataframe-format"] = dataframe_format
        opts["--save-dataframe"] = None  # type: ignore[assignment]
    return opts


class JobManager(BaseManager):
    """Job manager.

    Parameters
    ----------
    job_names
        List of unique names used for the jobs with the same length as
        `learners`. Note that a job name does not correspond to a certain
        specific learner.
    database_manager
        A `DatabaseManager` instance.
    scheduler
        A scheduler instance from `adaptive_scheduler.scheduler`.
    interval
        Time in seconds between checking and starting jobs.
    max_simultaneous_jobs
        Maximum number of simultaneously running jobs. By default no more than 500
        jobs will be running. Keep in mind that if you do not specify a ``runner.goal``,
        jobs will run forever, resulting in the jobs that were not initially started
        (because of this `max_simultaneous_jobs` condition) to not ever start.
    max_fails_per_job
        Maximum number of times that a job can fail. This is here as a fail switch
        because a job might fail instantly because of a bug inside your code.
        The job manager will stop when
        ``n_jobs * total_number_of_jobs_failed > max_fails_per_job`` is true.
    save_dataframe
        Whether to periodically save the learner's data as a `pandas.DataFame`.
    dataframe_format
        The format in which to save the `pandas.DataFame`. See the type hint for the options.
    loky_start_method
        Loky start method, by default "loky".
    log_interval
        Time in seconds between log entries.
    save_interval
        Time in seconds between saving of the learners.
    runner_kwargs
        Extra keyword argument to pass to the `adaptive.Runner`. Note that this dict
        will be serialized and pasted in the ``job_script``.
    goal
        The goal passed to the `adaptive.Runner`. Note that this function will
        be serialized and pasted in the ``job_script``. Can be a smart-goal
        that accepts
        ``Callable[[adaptive.BaseLearner], bool] | float | datetime | timedelta | None``.
        See `adaptive_scheduler.utils.smart_goal` for more information.

    Attributes
    ----------
    n_started : int
        Total number of jobs started by the `JobManager`.

    """

    def __init__(
        self,
        job_names: list[str],
        database_manager: DatabaseManager,
        scheduler: BaseScheduler,
        interval: float = 30,
        *,
        max_simultaneous_jobs: int = 100,
        max_fails_per_job: int = 50,
        # Command line launcher options
        save_dataframe: bool = True,
        dataframe_format: _DATAFRAME_FORMATS = "pickle",
        loky_start_method: LOKY_START_METHODS = "loky",
        log_interval: float = 60,
        save_interval: float = 300,
        runner_kwargs: dict[str, Any] | None = None,
        goal: GoalTypes | None = None,
    ) -> None:
        super().__init__()
        self.job_names = job_names
        self.database_manager = database_manager
        self.scheduler = scheduler
        self.interval = interval
        self.max_simultaneous_jobs = max_simultaneous_jobs
        self.max_fails_per_job = max_fails_per_job
        # Other attributes
        self.n_started = 0
        self._request_times: dict[str, str] = {}

        # Command line launcher options
        self.save_dataframe = save_dataframe
        self.dataframe_format = dataframe_format
        self.loky_start_method = loky_start_method
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.runner_kwargs = runner_kwargs
        self.goal = goal

    @property
    def max_job_starts(self) -> int:
        """Equivalent to ``self.max_fails_per_job * len(self.job_names)``."""
        return self.max_fails_per_job * len(self.job_names)

    def _queued(self, queue: dict[str, dict[str, Any]]) -> set[str]:
        return {job["job_name"] for job in queue.values() if job["job_name"] in self.job_names}

    def _setup(self) -> None:
        name_prefix = self.job_names[0].rsplit("-", 1)[0]
        options = command_line_options(
            scheduler=self.scheduler,
            database_manager=self.database_manager,
            runner_kwargs=self.runner_kwargs,
            log_interval=self.log_interval,
            save_interval=self.save_interval,
            save_dataframe=self.save_dataframe,
            dataframe_format=self.dataframe_format,
            goal=self.goal,
            loky_start_method=self.loky_start_method,
        )
        # hack to get the options in the job_script  # noqa: FIX004
        self.scheduler._command_line_options = options
        if self.scheduler.single_job_script:
            self.scheduler.write_job_script(name_prefix, options, index=None)

    async def _update_database_and_get_not_queued(
        self,
    ) -> tuple[set[str], set[str]] | None:
        running = self.scheduler.queue(me_only=True)
        self.database_manager.update(running)  # in case some jobs died
        queued = self._queued(running)  # running `job_name`s
        not_queued = set(self.job_names) - queued
        n_done = self.database_manager.n_done()
        if n_done == len(self.job_names):
            return None  # we are finished!
        n_to_schedule = max(0, len(not_queued) - n_done)
        return queued, set(list(not_queued)[:n_to_schedule])

    async def _start_new_jobs(
        self,
        not_queued: set[str],
        queued: set[str],
    ) -> None:
        num_jobs_to_start = min(
            len(not_queued),
            self.max_simultaneous_jobs - len(queued),
        )
        for _ in range(num_jobs_to_start):
            index, fname = self.database_manager._choose_fname()
            if index == -1:
                log.debug("No jobs can be scheduled currently because of dependencies.")
                return
            job_name = not_queued.pop()
            queued.add(job_name)
            log.debug(
                f"Starting `job_name={job_name}` with `index={index}` and `fname={fname}`",
            )
            await asyncio.to_thread(self.scheduler.start_job, job_name, index=index)
            self.database_manager._confirm_submitted(index, job_name)

            self.n_started += 1
            self._request_times[job_name] = _now()

    async def _manage(self) -> None:
        while True:
            try:
                update = await self._update_database_and_get_not_queued()
                if update is None:  # we are finished!
                    return
                queued, not_queued = update
                await self._start_new_jobs(not_queued, queued)
                if self.n_started > self.max_job_starts:
                    msg = "Too many jobs failed, your Python code probably has a bug."
                    raise MaxRestartsReachedError(msg)  # noqa: TRY301
                if await sleep_unless_task_is_done(
                    self.database_manager.task,  # type: ignore[arg-type]
                    self.interval,
                ):  # if true, we are done
                    return
            except asyncio.CancelledError:  # noqa: PERF203
                log.info("task was cancelled because of a CancelledError")
                raise
            except MaxRestartsReachedError as e:
                log.exception(
                    "too many jobs have failed, cancelling the job manager",
                    n_started=self.n_started,
                    max_fails_per_job=self.max_fails_per_job,
                    max_job_starts=self.max_job_starts,
                    exception=str(e),
                )
                raise
            except Exception as e:  # noqa: BLE001
                log.exception("got exception when starting a job", exception=str(e))
                if await sleep_unless_task_is_done(
                    self.database_manager.task,  # type: ignore[arg-type]
                    5,
                ):  # if true, we are done
                    return
