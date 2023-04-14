from __future__ import annotations

import asyncio
import os
from typing import Callable

from adaptive_scheduler.scheduler import BaseScheduler
from adaptive_scheduler.utils import (
    _remove_or_move_files,
)

from .base_manager import BaseManager
from .common import log
from .database_manager import DatabaseManager


def logs_with_string_or_condition(
    error: str | Callable[[list[str]], bool],
    database_manager: DatabaseManager,
) -> list[tuple[str, list[str]]]:
    """Get jobs that have `string` (or apply a callable) inside their log-file.

    Either use `string` or `error`.

    Parameters
    ----------
    error : str or callable
        String that is searched for or callable that is applied
        to the log text. Must take a single argument, a list of
        strings, and return True if the job has to be killed, or
        False if not.
    database_manager : `DatabaseManager`
        A `DatabaseManager` instance.

    Returns
    -------
    has_string : dict
        A list ``(job_name, fnames)``, which have the string inside their log-file.
    """
    if isinstance(error, str):
        has_error = lambda lines: error in "".join(lines)  # noqa: E731
    elif callable(error):
        has_error = error
    else:
        raise ValueError("`error` can only be a `str` or `callable`.")

    def file_has_error(fname):
        if not os.path.exists(fname):
            return False
        with open(fname, encoding="utf-8") as f:
            lines = f.readlines()
        return has_error(lines)

    have_error = []
    for entry in database_manager.as_dicts():
        fnames = entry["output_logs"]
        if entry["job_id"] is not None and any(file_has_error(f) for f in fnames):
            all_fnames = [*fnames, entry["log_fname"]]
            have_error.append((entry["job_name"], all_fnames))
    return have_error


class KillManager(BaseManager):
    """Kill manager.

    Automatically cancel jobs that contain an error (or other condition)
    in the log files.

    Parameters
    ----------
    scheduler : `~adaptive_scheduler.scheduler.BaseScheduler`
        A scheduler instance from `adaptive_scheduler.scheduler`.
    database_manager : `DatabaseManager`
        A `DatabaseManager` instance.
    error : str or callable, default: "srun: error:"
        If ``error`` is a string and is found in the log files, the job will
        be cancelled and restarted. If it is a callable, it is applied
        to the log text. Must take a single argument, a list of
        strings, and return True if the job has to be killed, or
        False if not.
    interval : int, default: 600
        Time in seconds between checking for the condition.
    max_cancel_tries : int, default: 5
        Try maximum `max_cancel_tries` times to cancel a job.
    move_to : str, optional
        If a job is cancelled the log is either removed (if ``move_to=None``)
        or moved to a folder (e.g. if ``move_to='old_logs'``).
    """

    def __init__(
        self,
        scheduler: BaseScheduler,
        database_manager: DatabaseManager,
        error: str | Callable[[list[str]], bool] = "srun: error:",
        interval: int = 600,
        max_cancel_tries: int = 5,
        move_to: str | None = None,
    ) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.database_manager = database_manager
        self.error = error
        self.interval = interval
        self.max_cancel_tries = max_cancel_tries
        self.move_to = move_to

        self.cancelled: list[str] = []
        self.deleted: list[str] = []

    async def _manage(self) -> None:
        while True:
            try:
                self.database_manager.update()
                failed_jobs = logs_with_string_or_condition(
                    self.error,
                    self.database_manager,
                )

                to_cancel: list[str] = []
                to_delete: list[str] = []
                for job_name, fnames in failed_jobs:
                    to_cancel.append(job_name)
                    to_delete.extend(fnames)

                self.scheduler.cancel(
                    to_cancel,
                    with_progress_bar=False,
                    max_tries=self.max_cancel_tries,
                )
                _remove_or_move_files(
                    to_delete,
                    with_progress_bar=False,
                    move_to=self.move_to,
                )
                self.cancelled.extend(to_cancel)
                self.deleted.extend(to_delete)
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                log.info("task was cancelled because of a CancelledError")
                raise
            except Exception as e:
                log.exception("got exception in kill manager", exception=str(e))
                await asyncio.sleep(self.interval)
