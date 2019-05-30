import asyncio
import datetime
import socket
import structlog
import zmq
from contextlib import suppress

from adaptive_scheduler._scheduler import get_job_id

ctx = zmq.Context()
log = structlog.get_logger("adaptive_scheduler.client")


def get_learner(url, learners, fnames):
    """Get a learner from the database running at `url`.

    Parameters
    ----------
    url : str
        The url of the database manager running via
        (`adaptive_scheduler.server_support.manage_database`).
    learners : list of `adaptive.BaseLearner` isinstances
        List of `learners` corresponding to `fnames`.
    fnames : list
        List of `fnames` corresponding to `learners`.

    Returns
    -------
    fname : str
        The filename of the learner that was chosen.
    """
    job_id = get_job_id()
    log.info(f"trying to get learner", job_id=job_id)
    with ctx.socket(zmq.REQ) as socket:
        socket.connect(url)
        socket.send_pyobj(("start", job_id))
        log.info(f"sent start signal")
        reply = socket.recv_pyobj()
        log.info("got reply", reply=str(reply))
        if reply is None:
            msg = f"No learners to be run for {job_id}."
            log.exception(msg)
            raise RuntimeError(msg)
        elif isinstance(reply, Exception):
            log.exception("got an exception")
            raise reply
        else:
            fname = reply
            log.info(f"got fname")

    def maybe_lst(fname):
        if isinstance(fname, tuple):
            # TinyDB converts tuples to lists
            fname = list(fname)
        return fname

    try:
        learner = next(l for l, f in zip(learners, fnames) if maybe_lst(f) == fname)
    except StopIteration:
        msg = "Learner with this fname doesn't exist in the database."
        log.exception(msg)
        raise UserWarning(msg)

    log.info("picked a learner")
    return learner, fname


def tell_done(url, fname):
    """Tell the database that the learner has reached it's goal.

    Parameters
    ----------
    url : str
        The url of the database manager running via
        (`adaptive_scheduler.server_support.manage_database`).
    fname : str
        The filename of the learner that is done.
    """
    with ctx.socket(zmq.REQ) as socket:
        socket.connect(url)
        socket.send_pyobj(("stop", fname))
        log.info("sent stop signal", fname=fname)
        socket.recv_pyobj()  # Needed because of socket type


def log_info(runner, interval=300):
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

    def get_npoints(learner):
        with suppress(AttributeError):
            return learner.npoints
        with suppress(AttributeError):
            # If the Learner is a BalancingLearner
            return sum(l.npoints for l in learner.learners)

    async def coro(runner, interval):
        log.info(f"started logger on hostname {socket.gethostname()}")
        learner = runner.learner
        npoints_start = get_npoints(learner)
        log.info(f"npoints at start {npoints_start}")
        while runner.status() == "running":
            await asyncio.sleep(interval)
            info = {}
            Δt = datetime.timedelta(seconds=runner.elapsed_time())
            info["elapsed_time"] = str(Δt)
            info["overhead"] = f"{runner.overhead():.2f}%"
            npoints = get_npoints(learner)
            if npoints is not None:
                info["npoints"] = get_npoints(learner)
                Δnpoints = npoints - npoints_start
                info["npoints/s"] = f"{Δnpoints / Δt.seconds:.3f}"
            with suppress(Exception):
                info["latest loss"] = f'{learner._cache["loss"]:.3f}'
            with suppress(AttributeError):
                info["nlearners"] = len(learner.learners)
                if "npoints" in info:
                    info["npoints/learner"] = info["npoints"] / info["nlearners"]
            log.info(f"current status", **info)
        log.info(f"runner statues changed to {runner.status()}")

    return runner.ioloop.create_task(coro(runner, interval))
