import asyncio
import datetime
import logging
import socket
from contextlib import suppress
from typing import Any, Dict, List, Tuple, Union

import psutil
import structlog
import zmq
from adaptive import AsyncRunner, BaseLearner

from adaptive_scheduler.utils import log_exception

ctx = zmq.Context()
logger = logging.getLogger("adaptive_scheduler.client")
logger.setLevel(logging.INFO)
log = structlog.wrap_logger(
    logger,
    processors=[
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M.%S", utc=False),
        structlog.processors.JSONRenderer(),
    ],
)


def _add_log_file_handler(log_file):
    fh = logging.FileHandler(log_file)
    logger.addHandler(fh)


def get_learner(
    learners: List[BaseLearner], fnames: List[str], url: str, log_file: str, job_id: str
) -> None:
    """Get a learner from the database running at `url` and this learner's
    process will be logged in `log_file` and running under `job_id`.

    Parameters
    ----------
    learners : list of `adaptive.BaseLearner` isinstances
        List of `learners` corresponding to `fnames`.
    fnames : list
        List of `fnames` corresponding to `learners`.
    url : str
        The url of the database manager running via
        (`adaptive_scheduler.server_support.manage_database`).
    log_file : str
        The filename of the log-file. Should be passed in the job-script.
    job_id : str
        The job_id of the process the job.

    Returns
    -------
    fname : str
        The filename of the learner that was chosen.
    """
    _add_log_file_handler(log_file)
    log.info("trying to get learner", job_id=job_id, log_file=log_file)
    with ctx.socket(zmq.REQ) as socket:
        socket.connect(url)
        socket.send_pyobj(("start", job_id, log_file))
        log.info(f"sent start signal")
        reply = socket.recv_pyobj()
        log.info("got reply", reply=str(reply))
        if reply is None:
            msg = f"No learners to be run."
            exception = RuntimeError(msg)
            log_exception(log, msg, exception)
            raise exception
        elif isinstance(reply, Exception):
            log_exception(log, "got an exception", exception=reply)
            raise reply
        else:
            fname = reply
            log.info(f"got fname")

    def maybe_lst(fname: Union[Tuple[str], str]):
        if isinstance(fname, tuple):
            # TinyDB converts tuples to lists
            fname = list(fname)
        return fname

    try:
        learner = next(l for l, f in zip(learners, fnames) if maybe_lst(f) == fname)
    except StopIteration:
        msg = "Learner with this fname doesn't exist in the database."
        exception = UserWarning(msg)
        log_exception(log, msg, exception)
        raise exception

    log.info("picked a learner")
    return learner, fname


def tell_done(url: str, fname: str) -> None:
    """Tell the database that the learner has reached it's goal.

    Parameters
    ----------
    url : str
        The url of the database manager running via
        (`adaptive_scheduler.server_support.manage_database`).
    fname : str
        The filename of the learner that is done.
    """
    log.info("goal reached! 🎉🎊🥳")
    with ctx.socket(zmq.REQ) as socket:
        socket.connect(url)
        socket.send_pyobj(("stop", fname))
        log.info("sent stop signal", fname=fname)
        socket.recv_pyobj()  # Needed because of socket type


def _get_npoints(learner: BaseLearner) -> int:
    with suppress(AttributeError):
        return learner.npoints
    with suppress(AttributeError):
        # If the Learner is a BalancingLearner
        return sum(l.npoints for l in learner.learners)


def _get_log_entry(runner: AsyncRunner, npoints_start: int) -> Dict[str, Any]:
    learner = runner.learner
    info = {}
    Δt = datetime.timedelta(seconds=runner.elapsed_time())
    info["elapsed_time"] = str(Δt)
    info["overhead"] = runner.overhead()
    npoints = _get_npoints(learner)
    if npoints is not None:
        info["npoints"] = _get_npoints(learner)
        Δnpoints = npoints - npoints_start
        with suppress(ZeroDivisionError):
            # Δt.seconds could be zero if the job is done when starting
            info["npoints/s"] = Δnpoints / Δt.seconds
    with suppress(Exception):
        info["latest_loss"] = learner._cache["loss"]
    with suppress(AttributeError):
        info["nlearners"] = len(learner.learners)
        if "npoints" in info:
            info["npoints/learner"] = info["npoints"] / info["nlearners"]
    info["cpu_usage"] = psutil.cpu_percent()
    info["mem_usage"] = psutil.virtual_memory().percent
    return info


def log_info(runner: AsyncRunner, interval=300) -> asyncio.Task:
    """Log info in the job's logfile, similar to `runner.live_info`.

    Parameters
    ----------
    runner : `adaptive.Runner` instance
    interval : int, default: 300
        Time in seconds between log entries.

    Returns
    -------
    asyncio.Task
    """

    async def coro(runner, interval):
        log.info(f"started logger on hostname {socket.gethostname()}")
        learner = runner.learner
        npoints_start = _get_npoints(learner)
        log.info("npoints at start", npoints=npoints_start)
        while runner.status() == "running":
            await asyncio.sleep(interval)
            info = _get_log_entry(runner, npoints_start)
            log.info("current status", **info)
        log.info("runner status changed", status=runner.status())
        log.info("current status", **_get_log_entry(runner, npoints_start))

    return runner.ioloop.create_task(coro(runner, interval))
