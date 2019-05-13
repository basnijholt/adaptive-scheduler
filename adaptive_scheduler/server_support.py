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


def dispatch(request, db_fname):
    request_type, request_arg = request
    log.debug("got a request", request=request)
    try:
        if request_type == "start":
            job_id = request_arg  # workers send us their slurm ID for us to fill in
            # give the worker a job and send back the fname to the worker
            fname = choose_fname(db_fname, job_id)
            log.debug("choose a fname", fname=fname, job_id=job_id)
            return fname
        elif request_type == "stop":
            fname = request_arg  # workers send us the fname they were given
            log.debug("got a stop request", fname=fname)
            return done_with_learner(db_fname, fname)  # reset the job_id to None
    except Exception as e:
        return e


async def manage_database(address, db_fname):
    log.debug("started database")
    socket = ctx.socket(zmq.REP)
    socket.bind(address)
    try:
        while True:
            request = await socket.recv_pyobj()
            reply = dispatch(request, db_fname)
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
):
    with concurrent.futures.ProcessPoolExecutor() as ex:
        while True:
            try:
                running = queue()
                update_db(db_fname, running)  # in case some jobs died
                running_job_names = {
                    job["name"] for job in running.values() if job["name"] in job_names
                }
                n_jobs_done = get_n_jobs_done(db_fname)
                to_start = len(job_names) - len(running_job_names) - n_jobs_done
                for job_name in job_names:
                    if job_name not in running_job_names and to_start > 0:
                        await ioloop.run_in_executor(
                            ex,
                            start_job,
                            job_name,
                            cores,
                            job_script_function,
                            run_script,
                            python_executable,
                        )
                        to_start -= 1
                await asyncio.sleep(interval)
            except concurrent.futures.CancelledError:
                log.exception("task was cancelled because of a CancelledError")
                raise
            except Exception as e:
                log.exception("got exception when starting a job", exception=str(e))
                await asyncio.sleep(5)


def create_empty_db(db_fname, fnames):
    entries = [dict(fname=fname, job_id=None, is_done=False) for fname in fnames]
    if os.path.exists(db_fname):
        os.remove(db_fname)
    with TinyDB(db_fname) as db:
        db.insert_multiple(entries)


def update_db(db_fname, running):
    """If the job_id isn't running anymore, replace it with None."""
    with TinyDB(db_fname) as db:
        doc_ids = [entry.doc_id for entry in db.all() if entry["job_id"] not in running]
        db.update({"job_id": None}, doc_ids=doc_ids)


def choose_fname(db_fname, job_id):
    Entry = Query()
    with TinyDB(db_fname) as db:
        assert not db.contains(Entry.job_id == job_id)
        entry = db.get((Entry.job_id == None) & (Entry.is_done == False))  # noqa: E711
        log.debug("chose fname", entry=entry)
        if entry is None:
            return
        db.update({"job_id": job_id}, doc_ids=[entry.doc_id])
    return entry["fname"]


def done_with_learner(db_fname, fname):
    Entry = Query()
    with TinyDB(db_fname) as db:
        db.update({"job_id": None, "is_done": True}, Entry.fname == fname)


def start_job(name, cores, job_script_function, run_script, python_executable):
    with open(name + ext, "w") as f:
        job_script = job_script_function(name, cores, run_script, python_executable)
        f.write(job_script)

    returncode = None
    while returncode != 0:
        returncode = subprocess.run(
            f"{submit_cmd} {name}{ext}".split(), stderr=subprocess.PIPE
        ).returncode
        time.sleep(0.5)


def get_n_jobs_done(db_fname):
    Entry = Query()
    with TinyDB(db_fname) as db:
        return db.count(Entry.is_done == True)  # noqa: E711


def get_allowed_url():
    ip = socket.gethostbyname(socket.gethostname())
    port = zmq.ssh.tunnel.select_random_ports(1)[0]
    return f"tcp://{ip}:{port}"


def get_database(db_fname):
    with TinyDB(db_fname) as db:
        return db.all()
