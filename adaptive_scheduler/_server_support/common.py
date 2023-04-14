from __future__ import annotations

import asyncio
import glob
import logging
import os
import shutil
import socket
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

import structlog
import zmq
import zmq.asyncio
import zmq.ssh
from rich.console import Console

from adaptive_scheduler.scheduler import BaseScheduler
from adaptive_scheduler.utils import (
    _progress,
    _remove_or_move_files,
)

console = Console()


logger = logging.getLogger("adaptive_scheduler.server")
logger.setLevel(logging.INFO)
log = structlog.wrap_logger(logger)


class MaxRestartsReached(Exception):
    """Jobs can fail instantly because of a error in
    your Python code which results jobs being started indefinitely.
    """


def get_allowed_url() -> str:
    """Get an allowed url for the database manager.

    Returns
    -------
    url : str
        An url that can be used for the database manager, with the format
        ``tcp://ip_of_this_machine:allowed_port.``.
    """
    ip = socket.gethostbyname(socket.gethostname())
    port = zmq.ssh.tunnel.select_random_ports(1)[0]
    return f"tcp://{ip}:{port}"


def _get_all_files(job_names: list[str], scheduler: BaseScheduler) -> list[str]:
    log_fnames = [scheduler.log_fname(name) for name in job_names]
    output_fnames = [scheduler.output_fnames(name) for name in job_names]
    output_fnames = sum(output_fnames, [])
    batch_fnames = [scheduler.batch_fname(name) for name in job_names]
    fnames = log_fnames + output_fnames + batch_fnames
    all_files = [glob.glob(f.replace(scheduler._JOB_ID_VARIABLE, "*")) for f in fnames]
    all_files = sum(all_files, [])

    # For schedulers that use a single batch file
    name_prefix = job_names[0].rsplit("-", 1)[0]
    batch_file = scheduler.batch_fname(name_prefix)
    if os.path.exists(batch_file):
        all_files.append(batch_file)
    return all_files


def cleanup_scheduler_files(
    job_names: list[str],
    scheduler: BaseScheduler,
    with_progress_bar: bool = True,
    move_to: str | None = None,
) -> None:
    """Cleanup the scheduler log-files files.

    Parameters
    ----------
    job_names : list
        List of job names.
    scheduler : `~adaptive_scheduler.scheduler.BaseScheduler`
        A scheduler instance from `adaptive_scheduler.scheduler`.
    with_progress_bar : bool, default: True
        Display a progress bar using `tqdm`.
    move_to : str, default: None
        Move the file to a different directory.
        If None the file is removed.
    log_file_folder : str, default: ''
        The folder in which to delete the log-files.
    """
    to_rm = _get_all_files(job_names, scheduler)

    _remove_or_move_files(
        to_rm,
        with_progress_bar,
        move_to,
        "Removing logs and batch files",
    )


def _delete_old_ipython_profiles(
    scheduler: BaseScheduler,
    with_progress_bar: bool = True,
) -> None:
    if scheduler.executor_type != "ipyparallel":
        return
    # We need the job_ids because only job_names wouldn't be
    # enough information. There might be other job_managers
    # running.
    pattern = "profile_adaptive_scheduler_"
    profile_folders = glob.glob(os.path.expanduser(f"~/.ipython/{pattern}*"))

    running_job_ids = set(scheduler.queue().keys())
    to_delete = [
        folder
        for folder in profile_folders
        if folder.split(pattern)[1] not in running_job_ids
    ]

    with ThreadPoolExecutor(256) as ex:
        desc = "Submitting deleting old IPython profiles tasks"
        pbar = _progress(to_delete, desc=desc)
        futs = [ex.submit(shutil.rmtree, folder) for folder in pbar]
        desc = "Finishing deleting old IPython profiles"
        for fut in _progress(futs, with_progress_bar, desc=desc):
            fut.result()


def periodically_clean_ipython_profiles(scheduler, interval: int = 600):
    """Periodically remove old IPython profiles.

    In the `RunManager.cleanup` method the profiles will be removed. However,
    one might want to remove them earlier.

    Parameters
    ----------
    scheduler : `~adaptive_scheduler.scheduler.BaseScheduler`
        A scheduler instance from `adaptive_scheduler.scheduler`.
    interval : int, default: 600
        The interval at which to remove old profiles.

    Returns
    -------
    asyncio.Task
    """

    async def clean(interval: int) -> None:
        while True:
            with suppress(Exception):
                _delete_old_ipython_profiles(scheduler, with_progress_bar=False)
            await asyncio.sleep(interval)

    ioloop = asyncio.get_event_loop()
    coro = clean(interval)
    return ioloop.create_task(coro)
