from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from .base_manager import BaseManager
from .common import MaxRestartsReachedError, log

if TYPE_CHECKING:
    from adaptive_scheduler.scheduler import BaseScheduler

    from .database_manager import DatabaseManager


class JobManager(BaseManager):
    """Job manager.

    Parameters
    ----------
    job_names : list
        List of unique names used for the jobs with the same length as
        `learners`. Note that a job name does not correspond to a certain
        specific learner.
    database_manager : `DatabaseManager`
        A `DatabaseManager` instance.
    scheduler : `~adaptive_scheduler.scheduler.BaseScheduler`
        A scheduler instance from `adaptive_scheduler.scheduler`.
    interval : int, default: 30
        Time in seconds between checking and starting jobs.
    max_simultaneous_jobs : int, default: 500
        Maximum number of simultaneously running jobs. By default no more than 500
        jobs will be running. Keep in mind that if you do not specify a ``runner.goal``,
        jobs will run forever, resulting in the jobs that were not initially started
        (because of this `max_simultaneous_jobs` condition) to not ever start.
    max_fails_per_job : int, default: 40
        Maximum number of times that a job can fail. This is here as a fail switch
        because a job might fail instantly because of a bug inside `run_script`.
        The job manager will stop when
        ``n_jobs * total_number_of_jobs_failed > max_fails_per_job`` is true.

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
        interval: int = 30,
        *,
        max_simultaneous_jobs: int = 100,
        max_fails_per_job: int = 50,
    ) -> None:
        super().__init__()
        self.job_names = job_names
        self.database_manager = database_manager
        self.scheduler = scheduler
        self.interval = interval
        self.max_simultaneous_jobs = max_simultaneous_jobs
        self.max_fails_per_job = max_fails_per_job

        self.n_started = 0

    @property
    def max_job_starts(self) -> int:
        """Equivalent to ``self.max_fails_per_job * len(self.job_names)``."""
        return self.max_fails_per_job * len(self.job_names)

    def _queued(self, queue: dict[str, dict[str, Any]]) -> set[str]:
        return {
            job["job_name"]
            for job in queue.values()
            if job["job_name"] in self.job_names
        }

    async def _manage(self) -> None:
        while True:
            try:
                running = self.scheduler.queue(me_only=True)
                self.database_manager.update(running)  # in case some jobs died

                queued = self._queued(running)  # running `job_name`s
                not_queued = set(self.job_names) - queued

                n_done = self.database_manager.n_done()

                if n_done == len(self.job_names):
                    # we are finished!
                    self.database_manager.task.cancel()  # type: ignore[union-attr]
                    return

                n_to_schedule = max(0, len(not_queued) - n_done)
                not_queued = set(list(not_queued)[:n_to_schedule])
                while not_queued:
                    # start new jobs
                    if len(queued) < self.max_simultaneous_jobs:
                        job_name = not_queued.pop()
                        queued.add(job_name)
                        await asyncio.to_thread(self.scheduler.start_job, job_name)
                        self.n_started += 1
                    else:
                        break
                if self.n_started > self.max_job_starts:
                    msg = "Too many jobs failed, your Python code probably has a bug."
                    raise MaxRestartsReachedError(msg)  # noqa: TRY301
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
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
                await asyncio.sleep(5)
