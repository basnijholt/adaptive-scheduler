import asyncio
from concurrent.futures import ProcessPoolExecutor
import os
import socket
import subprocess
import time

from tinydb import TinyDB, Query
import zmq
import zmq.asyncio
import zmq.ssh

from adaptive_scheduler.slurm import make_sbatch, check_running

ctx = zmq.asyncio.Context()


def dispatch(request, db_fname):
    request_type, request_arg = request

    if request_type == "start":
        job_id = request_arg  # workers send us their slurm ID for us to fill in
        # give the worker a job and send back the fname and combo to the worker
        return choose_combo(db_fname, job_id)

    elif request_type == "stop":
        fname = request_arg  # workers send us the fname they were given
        return done_with_learner(db_fname, fname)  # reset the job_id to None

    else:
        print(f"unknown request type: {request_type}")


async def manage_database(address, db_fname):
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
    job_script_function=make_sbatch,
    run_script="run_learner.py",
    python_executable=None,
    interval=30,
):
    with ProcessPoolExecutor() as ex:
        while True:
            running = check_running()
            update_db(db_fname, running)  # in case some jobs died
            running_job_names = {job["job_name"] for job in running.values()}
            for job_name in job_names:
                if job_name not in running_job_names:
                    await ioloop.run_in_executor(
                        ex,
                        start_job,
                        job_name,
                        cores,
                        job_script_function,
                        run_script,
                        python_executable,
                    )
            await asyncio.sleep(interval)


def create_empty_db(db_fname, fnames, combos):
    entries = [
        dict(fname=fname, combo=combo, job_id=None, is_done=False)
        for fname, combo in zip(fnames, combos)
    ]
    if os.path.exists(db_fname):
        os.remove(db_fname)
    with TinyDB(db_fname) as db:
        db.insert_multiple(entries)


def update_db(db_fname, running):
    """If the job_id isn't running anymore, replace it with None."""
    with TinyDB(db_fname) as db:
        doc_ids = [entry.doc_id for entry in db.all() if entry["job_id"] not in running]
        db.update({"job_id": None}, doc_ids=doc_ids)


def choose_combo(db_fname, job_id):
    Entry = Query()
    with TinyDB(db_fname) as db:
        entry = db.get(Entry.job_id == None)
        db.update({"job_id": job_id}, doc_ids=[entry.doc_id])
    return entry["fname"], entry["combo"]


def done_with_learner(db_fname, fname):
    Entry = Query()
    with TinyDB(db_fname) as db:
        db.update({"job_id": None, "is_done": True}, Entry.fname == fname)


def start_job(name, cores, job_script_function, run_script, python_executable):
    with open(name + ".sbatch", "w") as f:
        job_script = job_script_function(name, cores, run_script, python_executable)
        f.write(job_script)

    returncode = None
    while returncode != 0:
        returncode = subprocess.run(
            f"sbatch {name}.sbatch".split(), stderr=subprocess.PIPE
        ).returncode
        time.sleep(0.5)


def get_allowed_url():
    ip = socket.gethostbyname(socket.gethostname())
    port = zmq.ssh.tunnel.select_random_ports(1)[0]
    return f"tcp://{ip}:{port}"
