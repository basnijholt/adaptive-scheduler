"""Client support for Adaptive Scheduler."""

from __future__ import annotations

import datetime
import json
import logging
import os
import socket
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import psutil
import structlog
import zmq

from adaptive_scheduler.utils import (
    _deserialize,
    _get_npoints,
    _serialize,
    fname_to_learner,
    log_exception,
    sleep_unless_task_is_done,
)

if TYPE_CHECKING:
    import argparse
    import asyncio
    from collections.abc import Callable
    from pathlib import Path

    from adaptive import AsyncRunner, BaseLearner


def _dumps(event_dict: dict[str, Any], **kwargs: Any) -> str:
    """Custom json.dumps to ensure 'event' key is always first in the JSON output."""
    event = event_dict.pop("event", None)
    return json.dumps({"event": event, **event_dict}, **kwargs)


ctx = zmq.Context()
logger = logging.getLogger("adaptive_scheduler.client")
logger.setLevel(logging.INFO)
log = structlog.wrap_logger(
    logger,
    processors=[
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M.%S", utc=False),
        structlog.processors.JSONRenderer(serializer=_dumps),
    ],
)


def add_log_file_handler(log_fname: str | Path) -> None:  # pragma: no cover
    """Add a file handler to the logger."""
    fh = logging.FileHandler(log_fname)
    logger.addHandler(fh)


def get_learner(
    url: str,
    log_fname: str,
    job_id: str,
    job_name: str,
) -> tuple[BaseLearner, str | list[str], Callable[[], None] | None]:
    """Get a learner from the database (running at `url`).

    This learner's process will be logged in `log_fname`
    and running under `job_id`.

    Parameters
    ----------
    url
        The url of the database manager running via
        (`adaptive_scheduler.server_support.manage_database`).
    log_fname
        The filename of the log-file. Should be passed in the job-script.
    job_id
        The job_id of the process the job. Should be passed in the job-script.
    job_name
        The name of the job. Should be passed in the job-script.

    Returns
    -------
    learner
        Learner that is chosen.
    fname
        The filename of the learner that was chosen.
    initializer
        A function that runs before the process is forked.

    """
    log.info(
        "trying to get learner",
        job_id=job_id,
        log_fname=log_fname,
        job_name=job_name,
    )
    with ctx.socket(zmq.REQ) as socket:
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, 300_000)  # timeout after 300s
        socket.connect(url)
        socket.send_serialized(("start", job_id, log_fname, job_name), _serialize)
        log.info("sent start signal, going to wait 300s for a reply.")
        socket.setsockopt(zmq.RCVTIMEO, 300_000)  # timeout after 300s
        reply = socket.recv_serialized(_deserialize)
        log.info("got reply", reply=str(reply))
        if reply is None:
            msg = "No learners to be run."
            exception = RuntimeError(msg)
            log_exception(log, msg, exception)
            raise exception
        if isinstance(reply, Exception):
            log_exception(log, "got an exception", exception=reply)
            raise reply
        fname = reply
        learner, initializer = fname_to_learner(fname, return_initializer=True)
        log.info("got fname and loaded learner")

    log.info("picked a learner")
    return learner, fname, initializer


def tell_done(url: str, fname: str | list[str]) -> None:
    """Tell the database that the learner has reached it's goal.

    Parameters
    ----------
    url
        The url of the database manager running via
        (`adaptive_scheduler.server_support.manage_database`).
    fname
        The filename of the learner that is done.

    """
    log.info("goal reached! 🎉🎊🥳")
    with ctx.socket(zmq.REQ) as socket:
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(url)
        socket.setsockopt(zmq.SNDTIMEO, 300_000)  # timeout after 300s
        socket.send_serialized(("stop", fname), _serialize)
        socket.setsockopt(zmq.RCVTIMEO, 300_000)  # timeout after 300s
        log.info("sent stop signal, going to wait 300s for a reply", fname=fname)
        socket.recv_serialized(_deserialize)  # Needed because of socket type


def _get_log_entry(runner: AsyncRunner, npoints_start: int) -> dict[str, Any]:
    learner = runner.learner
    info: dict[str, float | str] = {}
    Δt = datetime.timedelta(seconds=runner.elapsed_time())  # noqa: N806, PLC2401
    info["elapsed_time"] = str(Δt)
    info["overhead"] = runner.overhead()
    npoints = _get_npoints(learner)
    if npoints is not None:
        info["npoints"] = npoints
        Δnpoints = npoints - npoints_start  # noqa: N806, PLC2401
        with suppress(ZeroDivisionError):
            # Δt.seconds could be zero if the job is done when starting
            info["npoints/s"] = Δnpoints / Δt.seconds
    with suppress(Exception):
        info["latest_loss"] = learner._cache["loss"]
    with suppress(AttributeError):
        info["nlearners"] = len(learner.learners)
        if "npoints" in info:
            info["npoints/learner"] = info["npoints"] / info["nlearners"]  # type: ignore[operator]
    info["cpu_usage"] = psutil.cpu_percent()
    info["mem_usage"] = psutil.virtual_memory().percent
    for k, v in psutil.cpu_times()._asdict().items():
        info[f"cputimes.{k}"] = v
    return info


def log_now(runner: AsyncRunner, npoints_start: int) -> None:
    """Create a log message now."""
    info = _get_log_entry(runner, npoints_start)
    log.info("current status", **info)


def log_info(runner: AsyncRunner, interval: float = 300) -> asyncio.Task:
    """Log info in the job's logfile, similar to `runner.live_info`.

    Parameters
    ----------
    runner
        Adaptive Runner instance.
    interval
        Time in seconds between log entries.

    """

    async def coro(runner: AsyncRunner, interval: float) -> None:
        log.info(f"started logger on hostname {socket.gethostname()}")  # noqa: G004
        learner = runner.learner
        npoints_start = _get_npoints(learner)
        assert npoints_start is not None
        log.info("npoints at start", npoints=npoints_start)
        while runner.status() == "running":
            if await sleep_unless_task_is_done(runner.task, interval):
                break
            log_now(runner, npoints_start)
        log.info("runner status changed", status=runner.status())
        log.info("current status", **_get_log_entry(runner, npoints_start))

    return runner.ioloop.create_task(coro(runner, interval))


def args_to_env(args: argparse.Namespace, prefix: str = "ADAPTIVE_SCHEDULER_") -> None:
    """Convert parsed arguments to environment variables."""
    env_vars = {}
    for arg, value in vars(args).items():
        if value is not None:
            env_vars[f"{prefix}{arg.upper()}"] = str(value)
    os.environ.update(env_vars)
    log.info("set environment variables", **env_vars)
