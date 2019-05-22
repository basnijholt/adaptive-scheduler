import asyncio
import concurrent.futures
import logging
import os
import socket
import subprocess
import time

import structlog
import zmq
import zmq.asyncio
import zmq.ssh
from tinydb import Query, TinyDB

from adaptive_scheduler._scheduler import ext, make_job_script, queue, submit_cmd

ctx = zmq.asyncio.Context()

logger = logging.getLogger("adaptive_scheduler.server")
logger.setLevel(logging.INFO)
log = structlog.wrap_logger(logger)


class MaxRestartsReached(Exception):
    """Jobs can fail instantly because of a error in
    your Python code which results jobs being started indefinitely."""


def _dispatch(request, db_fname):
    request_type, request_arg = request
    log.debug("got a request", request=request)
    try:
        if request_type == "start":
            job_id = request_arg  # workers send us their slurm ID for us to fill in
            # give the worker a job and send back the fname to the worker
            fname = _choose_fname(db_fname, job_id)
            log.debug("choose a fname", fname=fname, job_id=job_id)
            return fname
        elif request_type == "stop":
            fname = request_arg  # workers send us the fname they were given
            log.debug("got a stop request", fname=fname)
            return _done_with_learner(db_fname, fname)  # reset the job_id to None
    except Exception as e:
        return e


async def manage_database(url, db_fname):
    """Database manager co-routine.

    Parameters
    ----------
    url : str
        The url of the database manager, with the format
        ``tcp://ip_of_this_machine:allowed_port.``. Use `get_allowed_url`
        to get a `url` that will work.
    db_name : str
        Filename of the database, e.g. 'running.json'

    Returns
    -------
    coroutine
    """
    log.debug("started database")
    socket = ctx.socket(zmq.REP)
    socket.bind(url)
    try:
        while True:
            request = await socket.recv_pyobj()
            reply = _dispatch(request, db_fname)
            await socket.send_pyobj(reply)
    finally:
        socket.close()


async def manage_jobs(
    job_names,
    db_fname,
    ioloop,
    cores=8,
    job_script_function=make_job_script,
    run_script="run_learner.py",
    python_executable=None,
    interval=30,
    *,
    max_fails_per_job=100,
):
    """Job manager co-routine.

    Parameters
    ----------
    job_names : list
        List of unique names used for the jobs with the same length as
        `learners`. Note that a job name does not correspond to a certain
        specific learner.
    db_fname : str
        Filename of the database, e.g. 'running.json'
    ioloop : asyncio.AbstractEventLoop instance
        A running eventloop.
    cores : int
        Number of cores per job (so per learner.)
    job_script_function : callable, default: adaptive_scheduler.slurm/pbs.make_job_script
        A function with the following signature:
        ``job_script(name, cores, run_script, python_executable)`` that returns
        a job script in string form. See ``adaptive_scheduler/slurm.py`` or
        ``adaptive_scheduler/pbs.py`` for an example.
    run_script : str
        Filename of the script that is run on the nodes. Inside this script we
        query the database and run the learner.
    python_executable : str, default: sys.executable
        The Python executable that should run the `run_script`. By default
        it uses the same Python as where this function is called.
    interval : int, default: 30
        Time in seconds between checking and starting jobs.
    max_fails_per_job : int, default: 100
        Maximum number of times that a job can fail. This is here as a fail switch
        because a job might fail instantly because of a bug inside `run_script`.
        The job manager will stop when
        ``n_jobs * total_number_of_jobs_failed > max_fails_per_job`` is true.

    Returns
    -------
    coroutine
    """
    n_started = 0
    max_job_starts = max_fails_per_job * len(job_names)
    with concurrent.futures.ProcessPoolExecutor() as ex:
        while True:
            try:
                running = queue()
                _update_db(db_fname, running)  # in case some jobs died
                running_job_names = {
                    job["name"] for job in running.values() if job["name"] in job_names
                }
                n_jobs_done = _get_n_jobs_done(db_fname)
                to_start = len(job_names) - len(running_job_names) - n_jobs_done
                for job_name in job_names:
                    if job_name not in running_job_names and to_start > 0:
                        await ioloop.run_in_executor(
                            ex,
                            _start_job,
                            job_name,
                            cores,
                            job_script_function,
                            run_script,
                            python_executable,
                        )
                        to_start -= 1
                        n_started += 1
                if n_started > max_job_starts:
                    raise MaxRestartsReached(
                        "Too many jobs failed, your Python code probably has a bug."
                    )
                await asyncio.sleep(interval)
            except concurrent.futures.CancelledError:
                log.exception("task was cancelled because of a CancelledError")
                raise
            except MaxRestartsReached as e:
                log.exception(
                    "too many jobs have failed, cancelling the job manager",
                    n_started=n_started,
                    max_fails_per_job=max_fails_per_job,
                    max_job_starts=max_job_starts,
                    exception=str(e),
                )
                raise
            except Exception as e:
                log.exception("got exception when starting a job", exception=str(e))
                await asyncio.sleep(5)


def _start_job(name, cores, job_script_function, run_script, python_executable):
    with open(name + ext, "w") as f:
        job_script = job_script_function(name, cores, run_script, python_executable)
        f.write(job_script)

    returncode = None
    while returncode != 0:
        returncode = subprocess.run(
            f"{submit_cmd} {name}{ext}".split(), stderr=subprocess.PIPE
        ).returncode
        time.sleep(0.5)


def get_allowed_url():
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


def create_empty_db(db_fname, fnames):
    """Create an empty database that keeps track of fname -> (job_id, is_done).

    Parameters
    ----------
    db_fname : str
        Filename of the database, e.g. 'running.json'
    fnames : list
        List of `fnames` corresponding to `learners`.
    """
    entries = [dict(fname=fname, job_id=None, is_done=False) for fname in fnames]
    if os.path.exists(db_fname):
        os.remove(db_fname)
    with TinyDB(db_fname) as db:
        db.insert_multiple(entries)


def get_database(db_fname):
    """Get the database as a list of dicts."""
    with TinyDB(db_fname) as db:
        return db.all()


def _update_db(db_fname, running):
    """If the job_id isn't running anymore, replace it with None."""
    with TinyDB(db_fname) as db:
        doc_ids = [entry.doc_id for entry in db.all() if entry["job_id"] not in running]
        db.update({"job_id": None}, doc_ids=doc_ids)


def _choose_fname(db_fname, job_id):
    Entry = Query()
    with TinyDB(db_fname) as db:
        assert not db.contains(Entry.job_id == job_id)
        entry = db.get((Entry.job_id == None) & (Entry.is_done == False))  # noqa: E711
        log.debug("chose fname", entry=entry)
        if entry is None:
            return
        db.update({"job_id": job_id}, doc_ids=[entry.doc_id])
    return entry["fname"]


def _done_with_learner(db_fname, fname):
    Entry = Query()
    with TinyDB(db_fname) as db:
        db.update({"job_id": None, "is_done": True}, Entry.fname == fname)


def _get_n_jobs_done(db_fname):
    Entry = Query()
    with TinyDB(db_fname) as db:
        return db.count(Entry.is_done == True)  # noqa: E711
