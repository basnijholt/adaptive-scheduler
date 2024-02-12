from __future__ import annotations

import datetime
import json
import os
from contextlib import suppress
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from typing import Any

    from adaptive_scheduler.scheduler import BaseScheduler

    from .database_manager import DatabaseManager


def _get_infos(fname: str, *, only_last: bool = True) -> list[dict[str, Any]]:
    status_lines: list[dict[str, Any]] = []
    with open(fname, encoding="utf-8") as f:  # noqa: PTH123
        lines = f.readlines()
        for line in reversed(lines):
            with suppress(Exception):
                info = json.loads(line)
                if info["event"] == "current status":
                    status_lines.append(info)
                    if only_last:
                        return status_lines
        return status_lines


def parse_log_files(
    database_manager: DatabaseManager,
    scheduler: BaseScheduler,
    *,
    only_last: bool = True,
) -> pd.DataFrame:
    """Parse the log-files and convert it to a `~pandas.core.frame.DataFrame`.

    Parameters
    ----------
    job_names
        List of job names.
    database_manager
        A `DatabaseManager` instance.
    scheduler
        A scheduler instance from `adaptive_scheduler.scheduler`.
    only_last
        Only look use the last printed status message.

    Returns
    -------
    `~pandas.core.frame.DataFrame`

    """
    _queue = scheduler.queue()
    database_manager.update(_queue)

    infos = []
    for entry in database_manager.as_dicts():
        log_fname = entry["log_fname"]
        if log_fname is None or not os.path.exists(log_fname):  # noqa: PTH110
            continue
        for info_dict in _get_infos(log_fname, only_last=only_last):
            info_dict.pop("event")  # this is always "current status"
            info_dict["timestamp"] = datetime.datetime.strptime(  # noqa: DTZ007
                info_dict["timestamp"],
                "%Y-%m-%d %H:%M.%S",
            )
            info_dict["elapsed_time"] = pd.to_timedelta(info_dict["elapsed_time"])
            info_dict.update(entry)
            infos.append(info_dict)

    for info_dict in infos:
        info_from_queue = _queue.get(info_dict["job_id"])
        if info_from_queue is None:
            continue
        info_dict["state"] = info_from_queue["state"]
        info_dict["job_name"] = info_from_queue["job_name"]

    return pd.DataFrame(infos)
