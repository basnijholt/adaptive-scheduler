from __future__ import annotations

import asyncio
import itertools
import logging
import shutil
import socket
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
import zmq
import zmq.asyncio
import zmq.ssh
from rich.console import Console

from adaptive_scheduler.utils import (
    _progress,
    _remove_or_move_files,
)

if TYPE_CHECKING:
    from adaptive_scheduler.scheduler import BaseScheduler

console = Console()

logger = logging.getLogger("adaptive_scheduler.server")
logger.setLevel(logging.INFO)
log = structlog.wrap_logger(logger)


class MaxRestartsReachedError(Exception):
    """Max restarts reached.

    Jobs can fail instantly because of an error in
    your Python code which results jobs being started indefinitely.
    """


def get_allowed_url() -> str:
    """Get an allowed url for the database manager.

    Returns
    -------
    url
        An url that can be used for the database manager, with the format
        ``tcp://ip_of_this_machine:allowed_port.``.

    """
    ip = socket.gethostbyname(socket.gethostname())
    port = zmq.ssh.tunnel.select_random_ports(1)[0]
    return f"tcp://{ip}:{port}"


def _get_matching_files(f: Path, variable: str) -> list[Path]:
    modified_name = f.name.replace(variable, "*")
    return list(f.parent.glob(modified_name))


def _get_all_files(job_names: list[str], scheduler: BaseScheduler) -> set[Path]:
    log_fnames = [scheduler.log_fname(name) for name in job_names]
    _output_fnames = [scheduler.output_fnames(name) for name in job_names]
    output_fnames: list[Path] = list(itertools.chain(*_output_fnames))
    batch_fnames = [scheduler.batch_fname(name) for name in job_names]
    fnames = log_fnames + output_fnames + batch_fnames
    all_files = []
    for f in fnames:
        matching_files = _get_matching_files(f, scheduler._JOB_ID_VARIABLE)
        all_files.extend(matching_files)
    # For schedulers that use a single batch file
    name_prefix = job_names[0].rsplit("-", 1)[0]
    batch_file = scheduler.batch_fname(name_prefix)
    if batch_file.exists():
        all_files.append(batch_file)
    return set(all_files)


def cleanup_scheduler_files(
    job_names: list[str],
    scheduler: BaseScheduler,
    *,
    with_progress_bar: bool = True,
    move_to: str | Path | None = None,
) -> None:
    """Cleanup the scheduler log-files files.

    Parameters
    ----------
    job_names
        List of job names.
    scheduler
        A scheduler instance from `adaptive_scheduler.scheduler`.
    with_progress_bar
        Display a progress bar using `tqdm`.
    move_to
        Move the file to a different directory.
        If None the file is removed.
    log_file_folder
        The folder in which to delete the log-files.

    """
    to_rm = _get_all_files(job_names, scheduler)

    _remove_or_move_files(
        to_rm,
        with_progress_bar=with_progress_bar,
        move_to=move_to,
        desc="Removing logs and batch files",
    )


IPYTHON_PROFILE_PATTERN = "profile_adaptive_scheduler_"


def _ipython_profiles() -> list[Path]:
    return list(Path("~/.ipython/").expanduser().glob(f"{IPYTHON_PROFILE_PATTERN}*"))


def _delete_old_ipython_profiles(
    scheduler: BaseScheduler,
    *,
    with_progress_bar: bool = True,
) -> None:
    # We need the job_ids because only job_names wouldn't be
    # enough information. There might be other job_managers
    # running.

    running_job_ids = set(scheduler.queue().keys())
    to_delete = [
        folder
        for folder in _ipython_profiles()
        if str(folder).split(IPYTHON_PROFILE_PATTERN)[1] not in running_job_ids
    ]

    with ThreadPoolExecutor(256) as ex:
        desc = "Submitting deleting old IPython profiles tasks"
        pbar = _progress(to_delete, desc=desc)
        futs = [ex.submit(shutil.rmtree, folder) for folder in pbar]
        desc = "Finishing deleting old IPython profiles"
        for fut in _progress(futs, with_progress_bar, desc=desc):
            fut.result()


def periodically_clean_ipython_profiles(
    scheduler: BaseScheduler,
    interval: float = 600,
) -> asyncio.Task:
    """Periodically remove old IPython profiles.

    In the `RunManager.cleanup` method the profiles will be removed. However,
    one might want to remove them earlier.

    Parameters
    ----------
    scheduler
        A scheduler instance from `adaptive_scheduler.scheduler`.
    interval
        The interval at which to remove old profiles.

    Returns
    -------
    asyncio.Task

    """
    if isinstance(scheduler.executor_type, tuple):
        msg = (
            "This function is not implemented for multiple executors."
            " Please open an issue on GitHub if you need this feature."
        )
        raise NotImplementedError(msg)

    async def clean(interval: float) -> None:
        while True:
            with suppress(Exception):
                _delete_old_ipython_profiles(scheduler, with_progress_bar=False)
            await asyncio.sleep(interval)

    ioloop = asyncio.get_event_loop()
    coro = clean(interval)
    return ioloop.create_task(coro)


def _maybe_path(fname: str | Path | None) -> Path | None:  # pragma: no cover
    """Convert a string to a Path or return None."""
    if fname is None:
        return None
    if isinstance(fname, str):
        return Path(fname)
    return fname
