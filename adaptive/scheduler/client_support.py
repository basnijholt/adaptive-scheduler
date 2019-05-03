import os
import zmq

ctx = zmq.Context()


def get_learner(url, learners, combos):
    with ctx.socket(zmq.REQ) as socket:
        socket.connect(url)
        job_id = os.environ.get("SLURM_JOB_ID", "UNKNOWN")
        socket.send_pyobj(("start", job_id))
        fname, combo = socket.recv_pyobj()
    learner = next(lrn for lrn, c in zip(learners, combos) if c == combo)
    return learner, fname


def tell_done(url, fname):
    with ctx.socket(zmq.REQ) as socket:
        socket.connect(url)
        socket.send_pyobj(("stop", fname))
        socket.recv_pyobj()  # Needed because of socket type
