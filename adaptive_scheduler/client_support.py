import structlog
import zmq

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
    learner = next(lrn for lrn, fn in zip(learners, fnames) if fn == fname)
    log.info("picked a learner")
    return learner, fname


def tell_done(url, fname):
    with ctx.socket(zmq.REQ) as socket:
        socket.connect(url)
        socket.send_pyobj(("stop", fname))
        log.info("sent stop signal", fname=fname)
        socket.recv_pyobj()  # Needed because of socket type
