import asyncio
import datetime
import structlog
import zmq
from contextlib import suppress

from adaptive_scheduler._scheduler import get_job_id

ctx = zmq.Context()
log = structlog.get_logger("adaptive_scheduler.client")


def get_learner(url, learners, fnames):
    job_id = get_job_id()
    log.info(f"trying to get learner", job_id=job_id)
    with ctx.socket(zmq.REQ) as socket:
        socket.connect(url)
        socket.send_pyobj(("start", job_id))
        log.info(f"sent start signal")
        reply = socket.recv_pyobj()
        log.info("got reply", reply=reply)
        if reply is None:
            msg = f"No learners to be run for {job_id}."
            log.exception(msg)
            raise RuntimeError(msg)
        elif isinstance(reply, Exception):
            log.exception("got an exception", reply=str(reply))
            raise reply
        else:
            fname = reply
            log.info(f"got fname", fname=fname)

    def maybe_lst(fname):
        if isinstance(fname, tuple):
            # TinyDB converts tuples to lists
            fname = list(fname)
        return fname

    learner = next(lrn for lrn, fn in zip(learners, fnames) if maybe_lst(fn) == fname)
    log.info("picked a learner")
    return learner, fname


def tell_done(url, fname):
    with ctx.socket(zmq.REQ) as socket:
        socket.connect(url)
        socket.send_pyobj(("stop", fname))
        log.info("sent stop signal", fname=fname)
        socket.recv_pyobj()  # Needed because of socket type


def log_info(runner, interval=300):
    """Log info in the terminal, similar to ``runner.live_info()``."""

    async def coro(runner, interval):
        while runner.status() == "running":
            await asyncio.sleep(interval)
            info = {}
            dt = datetime.timedelta(seconds=runner.elapsed_time())
            info["elapsed_time"] = str(dt)
            info["overhead"] = f"{runner.overhead():.2f}%"
            with suppress(AttributeError):
                info["npoints"] = runner.learner.npoints
            with suppress(AttributeError):
                # If the Learner is a BalancingLearner
                info["npoints"] = sum(l.npoints for l in runner.learner.learners)
            if "npoints" in info:
                info["npoint_per_second"] = f'{info["npoints"] / dt.seconds:.3f}'
            with suppress(Exception):
                info["latest loss"] = f'{runner.learner._cache["loss"]:.3f}'
            log.info(f"current status", **info)
        log.info(f"runner statues changed to {runner.status()}")

    return runner.ioloop.create_task(coro(runner, interval))
