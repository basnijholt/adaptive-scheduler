import asyncio
import concurrent.futures
import datetime
import glob
import json
import logging
import os
import socket
import textwrap
import time
from contextlib import suppress
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union

import adaptive
import dill
import pandas as pd
import structlog
import zmq
import zmq.asyncio
import zmq.ssh
from tinydb import Query, TinyDB

from adaptive_scheduler.scheduler import BaseScheduler


ctx = zmq.asyncio.Context()

logger = logging.getLogger("adaptive_scheduler.server")
logger.setLevel(logging.INFO)
log = structlog.wrap_logger(logger)


class MaxRestartsReached(Exception):
    """Jobs can fail instantly because of a error in
    your Python code which results jobs being started indefinitely."""


def _dispatch(request: Tuple[str, ...], db_fname: str):
    request_type, *request_arg = request
    log.debug("got a request", request=request)
    try:
        if request_type == "start":
            # workers send us their slurm ID for us to fill in
            job_id, log_fname, job_name = request_arg
            kwargs = dict(job_id=job_id, log_fname=log_fname, job_name=job_name)
            # give the worker a job and send back the fname to the worker
            fname = _choose_fname(db_fname, **kwargs)
            log.debug("choose a fname", fname=fname, **kwargs)
            return fname
        elif request_type == "stop":
            fname = request_arg[0]  # workers send us the fname they were given
            log.debug("got a stop request", fname=fname)
            _done_with_learner(db_fname, fname)  # reset the job_id to None
    except Exception as e:
        return e


_DATABASE_MANAGER_DOC = """\
{first_line}

Parameters
----------
url : str
    The url of the database manager, with the format
    ``tcp://ip_of_this_machine:allowed_port.``. Use `get_allowed_url`
    to get a `url` that will work.
db_fname : str
    Filename of the database, e.g. 'running.json'

Returns
-------
{returns}
"""


async def manage_database(url: str, db_fname: str) -> Coroutine:
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


manage_database.__doc__ = _DATABASE_MANAGER_DOC.format(
    first_line="Database manager co-routine.", returns="coroutine"
)


def start_database_manager(url: str, db_fname: str) -> asyncio.Task:
    ioloop = asyncio.get_event_loop()
    coro = manage_database(url, db_fname)
    return ioloop.create_task(coro)


start_database_manager.__doc__ = _DATABASE_MANAGER_DOC.format(
    first_line="Start database manager task.", returns="asyncio.Task"
)

_JOB_MANAGER_DOC = """{first_line}

Parameters
----------
job_names : list
    List of unique names used for the jobs with the same length as
    `learners`. Note that a job name does not correspond to a certain
    specific learner.
db_fname : str
    Filename of the database, e.g. 'running.json'
{extra_args}
scheduler : `~adaptive_scheduler.scheduler.BaseScheduler`
    A scheduler instance from `adaptive_scheduler.scheduler`.
interval : int, default: 30
    Time in seconds between checking and starting jobs.
max_simultaneous_jobs : int, default: 5000
    Maximum number of simultaneously running jobs. By default no more than 5000
    jobs will be running. Keep in mind that if you do not specify a ``runner.goal``,
    jobs will run forever, resulting in the jobs that were not initially started
    (because of this `max_simultaneous_jobs` condition) to not ever start.
max_fails_per_job : int, default: 40
    Maximum number of times that a job can fail. This is here as a fail switch
    because a job might fail instantly because of a bug inside `run_script`.
    The job manager will stop when
    ``n_jobs * total_number_of_jobs_failed > max_fails_per_job`` is true.

Returns
-------
{returns}
"""


async def manage_jobs(
    job_names: List[str],
    db_fname: str,
    ioloop,
    scheduler: BaseScheduler,
    interval: int = 30,
    *,
    max_simultaneous_jobs: int = 5000,
    max_fails_per_job: int = 100,
) -> Coroutine:
    n_started = 0
    max_job_starts = max_fails_per_job * len(job_names)
    with concurrent.futures.ProcessPoolExecutor() as ex:
        while True:
            try:
                running = scheduler.queue()
                _update_db(db_fname, running)  # in case some jobs died
                queued = {
                    j["job_name"]
                    for j in running.values()
                    if j["job_name"] in job_names
                }
                not_queued = set(job_names) - queued

                n_done = _get_n_jobs_done(db_fname)

                for _ in range(n_done):
                    # remove jobs that are finished
                    if not_queued:
                        # A job might still be running but can at the same
                        # time be marked as finished in the db. Therefore
                        # we added the `if not_queued` clause.
                        not_queued.pop()

                if n_done == len(job_names):
                    # we are finished!
                    return

                while not_queued:
                    if len(queued) < max_simultaneous_jobs:
                        job_name = not_queued.pop()
                        queued.add(job_name)
                        await ioloop.run_in_executor(ex, scheduler.start_job, job_name)
                        n_started += 1
                    else:
                        break
                if n_started > max_job_starts:
                    raise MaxRestartsReached(
                        "Too many jobs failed, your Python code probably has a bug."
                    )
                await asyncio.sleep(interval)
            except concurrent.futures.CancelledError:
                log.info("task was cancelled because of a CancelledError")
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


manage_jobs.__doc__ = _JOB_MANAGER_DOC.format(
    first_line="Job manager co-routine.",
    returns="coroutine",
    extra_args="ioloop : `asyncio.AbstractEventLoop` instance\n    A running eventloop.",
)


def start_job_manager(
    job_names: List[str],
    db_fname: str,
    scheduler: BaseScheduler,
    interval: int = 30,
    *,
    max_simultaneous_jobs: int = 5000,
    max_fails_per_job: int = 40,
) -> asyncio.Task:
    ioloop = asyncio.get_event_loop()
    coro = manage_jobs(
        job_names,
        db_fname,
        ioloop,
        scheduler,
        interval,
        max_simultaneous_jobs=max_simultaneous_jobs,
        max_fails_per_job=max_fails_per_job,
    )
    return ioloop.create_task(coro)


start_job_manager.__doc__ = _JOB_MANAGER_DOC.format(
    first_line="Start the job manager task.", returns="asyncio.Task", extra_args=""
)


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


def create_empty_db(db_fname: str, fnames: List[str]) -> None:
    """Create an empty database that keeps track of
    fname -> (job_id, is_done, log_fname, job_name).

    Parameters
    ----------
    db_fname : str
        Filename of the database, e.g. 'running.json'
    fnames : list
        List of `fnames` corresponding to `learners`.
    """
    defaults = dict(job_id=None, is_done=False, log_fname=None, job_name=None)
    entries = [dict(fname=fname, **defaults) for fname in fnames]
    if os.path.exists(db_fname):
        os.remove(db_fname)
    with TinyDB(db_fname) as db:
        db.insert_multiple(entries)


def get_database(db_fname: str) -> List[Dict[str, Any]]:
    """Get the database as a list of dicts."""
    with TinyDB(db_fname) as db:
        return db.all()


def _update_db(db_fname: str, running: Dict[str, dict]) -> None:
    """If the job_id isn't running anymore, replace it with None."""
    with TinyDB(db_fname) as db:
        doc_ids = [entry.doc_id for entry in db.all() if entry["job_id"] not in running]
        db.update({"job_id": None, "job_name": None}, doc_ids=doc_ids)


def _choose_fname(db_fname: str, job_id: str, log_fname: str, job_name: str) -> str:
    Entry = Query()
    with TinyDB(db_fname) as db:
        if db.contains(Entry.job_id == job_id):
            entry = db.get(Entry.job_id == job_id)
            fname = entry["fname"]  # already running
            raise Exception(
                f"The job_id {job_id} already exists in the database and "
                f"runs {fname}. You might have forgotten to use the "
                "`if __name__ == '__main__': ...` idom in your code. Read the "
                "warning in the [mpi4py](https://bit.ly/2HAk0GG) documentation."
            )
        entry = db.get((Entry.job_id == None) & (Entry.is_done == False))  # noqa: E711
        log.debug("choose fname", entry=entry)
        if entry is None:
            return
        db.update(
            {"job_id": job_id, "log_fname": log_fname, "job_name": job_name},
            doc_ids=[entry.doc_id],
        )
    return entry["fname"]


def _done_with_learner(db_fname: str, fname: str) -> None:
    Entry = Query()
    with TinyDB(db_fname) as db:
        reset = dict(job_id=None, is_done=True, job_name=None)
        db.update(reset, Entry.fname == fname)


def _get_n_jobs_done(db_fname: str) -> int:
    Entry = Query()
    with TinyDB(db_fname) as db:
        return db.count(Entry.is_done == True)  # noqa: E711


def _get_entry(job_name, db_fname):
    Entry = Query()
    with TinyDB(db_fname) as db:
        return db.get(Entry.job_name == job_name)


def _get_output_fnames(job_name, db_fname, scheduler):
    entry = _get_entry(job_name, db_fname)
    if entry is None or entry["job_id"] is None:
        return
    job_id = entry["job_id"].split(".")[0]  # for PBS
    output_fnames = [
        f.replace(scheduler._JOB_ID_VARIABLE, job_id)
        for f in scheduler.output_fnames(job_name)
    ]
    return output_fnames


def logs_with_string_or_condition(
    error: Union[str, Callable[[List[str]], bool]],
    db_fname: List[str],
    scheduler: BaseScheduler,
) -> Dict[str, list]:
    """Get jobs that have `string` (or apply a callable) inside their log-file.

    Either use `string` or `error`.

    Parameters
    ----------
    error : str or callable
        String that is searched for or callable that is applied
        to the log text. Must take a single argument, a list of
        strings, and return True if the job has to be killed, or
        False if not.
    db_fname : str, default: "running.json"
        Filename of the database, e.g. 'running.json'.
    scheduler : `~adaptive_scheduler.scheduler.BaseScheduler`
        A scheduler instance from `adaptive_scheduler.scheduler`.

    Returns
    -------
    has_string : dict
        A dictionary of ``job_id -> (job_name, fnames)``,
        which have the string inside their log-file.
    """

    if isinstance(error, str):
        has_error = lambda lines: error in "".join(lines)  # noqa: E731
    elif callable(error):
        has_error = error
    else:
        raise ValueError("`error` can only be a `str` or `callable`.")

    def file_has_error(fname):
        if not os.path.exists(fname):
            return False
        with open(fname) as f:
            lines = f.readlines()
        return has_error(lines)

    have_error = {}
    for entry in get_database(db_fname):
        if entry["job_id"] is None:
            continue
        output_fnames = _get_output_fnames(entry["job_name"], db_fname, scheduler)
        if any(file_has_error(f) for f in output_fnames):
            have_error[entry["job_id"]] = entry["job_name"], output_fnames
    return have_error


async def manage_killer(
    job_names: List[str],
    scheduler: BaseScheduler,
    error: Union[str, Callable[[List[str]], bool]] = "srun: error:",
    interval: int = 600,
    max_cancel_tries: int = 5,
    move_to: Optional[str] = None,
    db_fname: str = "running.json",
) -> Coroutine:
    # It seems like tasks that print the error message do not always stop working
    # I think it only stops working when the error happens on a node where the logger runs.
    from adaptive_scheduler.utils import _remove_or_move_files

    while True:
        try:
            failed_jobs = logs_with_string_or_condition(error, db_fname, scheduler)
            to_cancel = []
            to_delete = []

            # get cancel/delete only the processes/logs that are running now
            for job_id in scheduler.queue().keys():
                if job_id in failed_jobs:
                    job_name, fnames = failed_jobs[job_id]
                    to_cancel.append(job_name)
                    to_delete += fnames

            scheduler.cancel(
                to_cancel, with_progress_bar=False, max_tries=max_cancel_tries
            )
            _remove_or_move_files(to_delete, with_progress_bar=False, move_to=move_to)
            await asyncio.sleep(interval)
        except concurrent.futures.CancelledError:
            log.info("task was cancelled because of a CancelledError")
            raise
        except Exception as e:
            log.exception("got exception in kill manager", exception=str(e))


_KILL_MANAGER_DOC = """{first_line}

Automatically cancel jobs that contain an error (or other condition)
in the log files.

Parameters
----------
job_names : list
    List of unique names used for the jobs with the same length as
    `learners`.
scheduler : `~adaptive_scheduler.scheduler.BaseScheduler`
    A scheduler instance from `adaptive_scheduler.scheduler`.
error : str or callable, default: "srun: error:"
    If ``error`` is a string and is found in the log files, the job will
    be cancelled and restarted. If it is a callable, it is applied
    to the log text. Must take a single argument, a list of
    strings, and return True if the job has to be killed, or
    False if not.
interval : int, default: 600
    Time in seconds between checking for the condition.
max_cancel_tries : int, default: 5
    Try maximum `max_cancel_tries` times to cancel a job.
move_to : str, optional
    If a job is cancelled the log is either removed (if ``move_to=None``)
    or moved to a folder (e.g. if ``move_to='old_logs'``).
db_fname : str, default: "running.json"
    Filename of the database, e.g. 'running.json'.

Returns
-------
{returns}
"""

manage_killer.__doc__ = _KILL_MANAGER_DOC.format(
    first_line="Kill manager co-routine.", returns="coroutine"
)


def start_kill_manager(
    job_names: List[str],
    scheduler: BaseScheduler,
    error: Union[str, Callable[[List[str]], bool]] = "srun: error:",
    interval: int = 600,
    max_cancel_tries: int = 5,
    move_to: Optional[str] = None,
    db_fname: str = "running.json",
) -> asyncio.Task:
    ioloop = asyncio.get_event_loop()
    coro = manage_killer(
        job_names, scheduler, error, interval, max_cancel_tries, move_to, db_fname
    )
    return ioloop.create_task(coro)


start_kill_manager.__doc__ = _KILL_MANAGER_DOC.format(
    first_line="Start the kill manager task.", returns="asyncio.Task"
)


def _make_default_run_script(
    url: str,
    learners_file: str,
    save_interval: int,
    log_interval: int,
    goal: Optional[Callable[[adaptive.BaseLearner], bool]] = None,
    runner_kwargs: Optional[Dict[str, Any]] = None,
    run_script_fname: str = "run_learner.py",
    executor_type: str = "mpi4py",
):
    default_runner_kwargs = dict(shutdown_executor=True)
    runner_kwargs = dict(default_runner_kwargs, goal=goal, **(runner_kwargs or {}))
    serialized_runner_kwargs = dill.dumps(runner_kwargs)

    if executor_type == "mpi4py":
        import_line = "from mpi4py.futures import MPIPoolExecutor"
        executor_line = "MPIPoolExecutor()"
    elif executor_type == "ipyparallel":
        import_line = "from adaptive_scheduler.utils import connect_to_ipyparallel"
        executor_line = "connect_to_ipyparallel(profile=args.profile, n=args.n)"
    elif executor_type == "dask-mpi":
        try:
            import dask_mpi  # noqa: F401
        except ModuleNotFoundError as e:
            msg = "You need to have 'dask-mpi' installed to use `executor_type='dask-mpi'`."
            raise Exception(msg) from e
        import_line = "from distributed import Client"
        executor_line = "Client()"
    else:
        raise NotImplementedError("Use 'ipyparallel', 'dask-mpi' or 'mpi4py'.")

    if os.path.dirname(learners_file):
        raise RuntimeError(
            f"The {learners_file} needs to be in the same"
            " directory as where this is run from."
        )

    learners_module = os.path.splitext(learners_file)[0]

    template = textwrap.dedent(
        f"""\
    # {run_script_fname}, automatically generated
    # by `adaptive_scheduler.server_support._make_default_run_script()`.
    import argparse
    from contextlib import suppress

    import adaptive
    import dill
    from adaptive_scheduler import client_support
    {import_line}


    # the file that defines the learners we created above
    from {learners_module} import learners, fnames

    if __name__ == "__main__":  # ← use this, see warning @ https://bit.ly/2HAk0GG

        # parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--profile', action="store", dest="profile", type=str)
        parser.add_argument('--n', action="store", dest="n", type=int)
        parser.add_argument('--log-fname', action="store", dest="log_fname", type=str)
        parser.add_argument('--job-id', action="store", dest="job_id", type=str)
        parser.add_argument('--name', action="store", dest="name", type=str)
        args = parser.parse_args()

        # the address of the "database manager"
        url = "{url}"

        # ask the database for a learner that we can run which we log in `args.log_fname`
        learner, fname = client_support.get_learner(learners, fnames, url, args.log_fname, args.job_id, args.name)

        # load the data
        with suppress(Exception):
            learner.load(fname)

        # connect to the executor
        executor = {executor_line}

        # this is serialized by dill.dumps
        runner_kwargs = dill.loads({serialized_runner_kwargs})

        # run until `some_goal` is reached with an `MPIPoolExecutor`
        runner = adaptive.Runner(learner, executor=executor, **runner_kwargs)

        # periodically save the data (in case the job dies)
        runner.start_periodic_saving(dict(fname=fname), interval={save_interval})

        # log progress info in the job output script, optional
        client_support.log_info(runner, interval={log_interval})

        # block until runner goal reached
        runner.ioloop.run_until_complete(runner.task)

        # save once more after the runner is done
        learner.save(fname)

        # tell the database that this learner has reached its goal
        client_support.tell_done(url, fname)
    """
    )
    if executor_type == "dask-mpi":
        template = "from dask_mpi import initialize; initialize()\n" + template

    with open(run_script_fname, "w") as f:
        f.write(template)
    return run_script_fname


def _get_infos(fname: str, only_last: bool = True):
    status_lines: List[str] = []
    with open(fname) as f:
        lines = f.readlines()
        for line in reversed(lines):
            with suppress(Exception):
                info = json.loads(line)
                if info["event"] == "current status":
                    status_lines.append(info)
                    if only_last:
                        return status_lines
        return status_lines


def parse_log_files(  # noqa: C901
    job_names: List[str], db_fname: str, scheduler, only_last: bool = True
):
    """Parse the log-files and convert it to a `~pandas.core.frame.DataFrame`.

    This only works if you use `adaptive_scheduler.client_support.log_info`
    inside your ``run_script``.

    Parameters
    ----------
    job_names : list
        List of job names.
    db_fname : str, optional
        The database filename. If passed, ``fname`` will be populated.
    scheduler : `~adaptive_scheduler.scheduler.BaseScheduler`
        A scheduler instance from `adaptive_scheduler.scheduler`.
    only_last : bool, default: True
        Only look use the last printed status message.

    Returns
    -------
    `~pandas.core.frame.DataFrame`
    """

    infos = []
    for entry in get_database(db_fname):
        log_fname = entry["log_fname"]
        if log_fname is None:
            continue
        for info in _get_infos(log_fname, only_last):
            info.pop("event")  # this is always "current status"
            info["timestamp"] = datetime.datetime.strptime(
                info["timestamp"], "%Y-%m-%d %H:%M.%S"
            )
            info["elapsed_time"] = pd.to_timedelta(info["elapsed_time"])
            info.update(entry)
            infos.append(info)

    # Polulate state and job_name from the queue
    _queue = scheduler.queue()

    for info in infos:
        info_from_queue = _queue.get(info["job_id"])
        if info_from_queue is None:
            continue
        info["state"] = info_from_queue["state"]
        info["job_name"] = info_from_queue["job_name"]

    return pd.DataFrame(infos)


class RunManager:
    """A convenience tool that starts the job, database, and kill manager.

    Parameters
    ----------
    scheduler : `~adaptive_scheduler.scheduler.BaseScheduler`
        A scheduler instance from `adaptive_scheduler.scheduler`.
    goal : callable, default: None
        The goal passed to the `adaptive.Runner`. Note that this function will
        be serialized and pasted in the ``run_script``.
    runner_kwargs : dict, default: None
        Extra keyword argument to pass to the `adaptive.Runner`. Note that this dict
        will be serialized and pasted in the ``run_script``.
    url : str, default: None
        The url of the database manager, with the format
        ``tcp://ip_of_this_machine:allowed_port.``. If None, a correct url will be chosen.
    learners_file : str, default: "learners_file.py"
        The module that defined the variables ``learners`` and ``fnames``.
    save_interval : int, default: 300
        Time in seconds between saving of the learners.
    log_interval : int, default: 300
        Time in seconds between log entries.
    job_name : str, default: "adaptive-scheduler"
        From this string the job names will be created, e.g.
        ``["adaptive-scheduler-1", "adaptive-scheduler-2", ...]``.
    job_manager_interval : int, default: 60
        Time in seconds between checking and starting jobs.
    kill_interval : int, default: 60
        Check for `kill_on_error` string inside the log-files every `kill_interval` seconds.
    kill_on_error : str or callable, default: "srun: error:"
        If ``error`` is a string and is found in the log files, the job will
        be cancelled and restarted. If it is a callable, it is applied
        to the log text. Must take a single argument, a list of
        strings, and return True if the job has to be killed, or
        False if not.
    move_old_logs_to : str, default: "old_logs"
        Move logs of killed jobs to this directory. If None the logs will be deleted.
    db_fname : str, default: "running.json"
        Filename of the database, e.g. 'running.json'.
    overwrite_db : bool, default: True
        Overwrite the existing database.
    start_job_manager_kwargs : dict, default: None
        Keyword arguments for the `start_job_manager` function that aren't set in ``__init__`` here.
    start_kill_manager_kwargs : dict, default: None
        Keyword arguments for the `start_kill_manager` function that aren't set in ``__init__`` here.

    Examples
    --------
    Here is an example of using the `RunManager` with a modified ``job_script_function``.

    >>> import adaptive_scheduler
    >>> scheduler = adaptive_scheduler.scheduler.DefaultScheduler(cores=10)
    >>> run_manager = adaptive_scheduler.server_support.RunManager(
    ...     scheduler=scheduler
    ... ).start()

    Or an example using `ipyparallel.Client`.

    >>> from functools import partial
    >>> import adaptive_scheduler
    >>> scheduler = adaptive_scheduler.scheduler.DefaultScheduler(
    ...     cores=10, executor_type="ipyparallel",
    ... )
    >>> def goal(learner):
    ...     return learner.npoints > 2000
    >>> run_manager = adaptive_scheduler.server_support.RunManager(
    ...     scheduler=scheduler,
    ...     learners_file="learners_file.py",
    ...     goal=goal,
    ...     log_interval=30,
    ...     save_interval=30,
    ... )
    >>> run_manager.start()

    """

    def __init__(
        self,
        scheduler: BaseScheduler,
        goal: Optional[Callable[[adaptive.BaseLearner], bool]] = None,
        runner_kwargs: Optional[dict] = None,
        url: Optional[str] = None,
        learners_file: str = "learners_file.py",
        save_interval: int = 300,
        log_interval: int = 300,
        job_name: str = "adaptive-scheduler",
        job_manager_interval: int = 60,
        kill_interval: int = 60,
        kill_on_error: Union[str, Callable[[List[str]], bool], None] = "srun: error:",
        move_old_logs_to: Optional[str] = "old_logs",
        db_fname: str = "running.json",
        overwrite_db: bool = True,
        start_job_manager_kwargs: Optional[Dict[str, Any]] = None,
        start_kill_manager_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Set from arguments
        self.scheduler = scheduler
        self.goal = goal
        self.runner_kwargs = runner_kwargs
        self.learners_file = learners_file
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.job_name = job_name
        self.job_manager_interval = job_manager_interval
        self.kill_interval = kill_interval
        self.kill_on_error = kill_on_error
        self.move_old_logs_to = move_old_logs_to
        self.db_fname = db_fname
        self.overwrite_db = overwrite_db
        self.start_job_manager_kwargs = start_job_manager_kwargs or {}
        self.start_kill_manager_kwargs = start_kill_manager_kwargs or {}

        # Set in methods
        self.job_task = None
        self.database_task = None
        self.kill_task = None
        self.start_time = None
        self.end_time = None

        # Set on init
        self.url = url or get_allowed_url()
        self.learners_module = self._get_learners_file()
        self._set_job_names()
        self.is_started = False
        self.ioloop = asyncio.get_event_loop()

    def start(self):
        """Start running the `RunManager`."""

        async def _start():
            await self.job_task
            self.end_time = time.time()

        self._write_db()
        self._start_database_manager()
        self._start_job_manager()
        if self.kill_on_error is not None:
            self._start_kill_manager()
        self.is_started = True
        self.start_time = time.time()
        self.ioloop.create_task(_start())
        return self

    def _get_learners_file(self):
        from importlib.util import module_from_spec, spec_from_file_location

        spec = spec_from_file_location("learners_file", self.learners_file)
        learners_file = module_from_spec(spec)
        spec.loader.exec_module(learners_file)
        return learners_file

    def _write_db(self) -> None:
        if os.path.exists(self.db_fname) and not self.overwrite_db:
            return
        create_empty_db(self.db_fname, self.learners_module.fnames)

    def _set_job_names(self) -> None:
        learners = self.learners_module.learners
        self.job_names = [f"{self.job_name}-{i}" for i in range(len(learners))]

    def _start_job_manager(self) -> None:

        self.run_script = _make_default_run_script(
            self.url,
            self.learners_file,
            self.save_interval,
            self.log_interval,
            self.goal,
            self.runner_kwargs,
            self.scheduler.run_script,
            self.scheduler.executor_type,
        )

        self.job_task = start_job_manager(
            self.job_names,
            self.db_fname,
            scheduler=self.scheduler,
            interval=self.job_manager_interval,
            **self.start_job_manager_kwargs,
        )

    def _start_database_manager(self) -> None:
        self.database_task = start_database_manager(self.url, self.db_fname)

    def _start_kill_manager(self) -> None:
        if self.kill_on_error is None:
            return
        self.kill_task = start_kill_manager(
            self.job_names,
            self.scheduler,
            error=self.kill_on_error,
            interval=self.kill_interval,
            move_to=self.move_old_logs_to,
            db_fname=self.db_fname,
            **self.start_kill_manager_kwargs,
        )

    def cancel(self) -> None:
        """Cancel the manager tasks and the jobs in the queue."""
        if self.job_task is not None:
            self.job_task.cancel()
            self.database_task.cancel()
        if self.kill_task is not None:
            self.kill_task.cancel()
        self.scheduler.cancel(self.job_names)

    def cleanup(self) -> None:
        """Cleanup the log and batch files.

        If the `RunManager` is not running, the ``run_script.py`` file
        will also be removed.
        """
        from adaptive_scheduler.utils import (
            _delete_old_ipython_profiles,
            _remove_or_move_files,
        )

        scheduler = self.scheduler
        with suppress(FileNotFoundError):
            if self.status() != "running":
                os.remove(scheduler.run_script)

        running_job_ids = set(scheduler.queue().keys())
        if scheduler.executor_type == "ipyparallel":
            _delete_old_ipython_profiles(running_job_ids)

        log_fnames = [scheduler.log_fname(name) for name in self.job_names]
        output_fnames = [scheduler.output_fnames(name) for name in self.job_names]
        output_fnames = sum(output_fnames, [])
        batch_fnames = [scheduler.batch_fname(name) for name in self.job_names]
        fnames = log_fnames + output_fnames + batch_fnames
        to_rm = [glob.glob(f.replace(scheduler._JOB_ID_VARIABLE, "*")) for f in fnames]
        to_rm = sum(to_rm, [])
        _remove_or_move_files(
            to_rm, True, self.move_old_logs_to, "Removing logs and batch files"
        )

    def parse_log_files(self, only_last: bool = True):
        """Parse the log-files and convert it to a `~pandas.core.frame.DataFrame`.

        Parameters
        ----------
        only_last : bool, default: True
            Only look use the last printed status message.

        Returns
        -------
        df : `~pandas.core.frame.DataFrame`

        """
        return parse_log_files(self.job_names, self.db_fname, self.scheduler, only_last)

    def task_status(self) -> None:
        r"""Print the stack of the `asyncio.Task`\s."""
        if self.job_task is not None:
            self.job_task.print_stack()
        if self.database_task is not None:
            self.database_task.print_stack()
        if self.kill_task is not None:
            self.kill_task.print_stack()

    def get_database(self) -> List[Dict[str, Any]]:
        """Get the database as a list of dicts."""
        return get_database(self.db_fname)

    def load_learners(self) -> None:
        """Load the learners in parallel using `adaptive_scheduler.utils.load_parallel`."""
        from adaptive_scheduler.utils import load_parallel

        load_parallel(self.learners_module.learners, self.learners_module.fnames)

    def elapsed_time(self) -> float:
        """Total time elapsed since the RunManager was started."""
        if not self.is_started:
            return 0

        if self.job_task.done():
            end_time = self.end_time
            if end_time is None:
                # task was cancelled before it began
                assert self.job_task.cancelled()
                return 0
        else:
            end_time = time.time()
        return end_time - self.start_time

    def status(self) -> str:
        """Return the current status of the RunManager."""
        if not self.is_started:
            return "not yet started"
        try:
            self.job_task.result()
        except asyncio.InvalidStateError:
            return "running"
        except asyncio.CancelledError:
            status = "cancelled"
        except Exception:
            status = "failed"
        else:
            status = "finished"

        if self.end_time is None:
            self.end_time = time.time()
        return status

    def info(self) -> None:
        """Display information about the `RunManager`.

        Returns an interactive ipywidget that can be
        visualized in a Jupyter notebook.
        """
        from ipywidgets import Layout, Button, VBox, HBox, HTML
        from IPython.display import display

        status = HTML(value=self._info_html())

        layout = Layout(width="200px")
        buttons = [
            Button(description="update info", layout=layout, button_color="lightgreen"),
            Button(description="cancel jobs", layout=layout, button_style="danger"),
            Button(
                description="cleanup log and batch files",
                layout=layout,
                button_style="danger",
            ),
        ]
        buttons = {b.description: b for b in buttons}

        def update(_):
            status.value = self._info_html()

        def cancel(_):
            self.cancel()
            update(_)

        def cleanup(_):
            self.cleanup()
            update(_)

        buttons["cancel jobs"].on_click(cancel)
        buttons["cleanup log and batch files"].on_click(cleanup)
        buttons["update info"].on_click(update)

        buttons = VBox(list(buttons.values()))
        display(
            HBox(
                (status, buttons),
                layout=Layout(border="solid 1px", width="400px", align_items="center"),
            )
        )

    def _info_html(self) -> str:
        jobs = [
            job
            for job in self.scheduler.queue().values()
            if job["job_name"] in self.job_names
        ]
        n_running = sum(job["state"] in ("RUNNING", "R") for job in jobs)
        n_pending = sum(job["state"] in ("PENDING", "Q") for job in jobs)
        n_done = sum(job["is_done"] for job in self.get_database())

        status = self.status()
        color = {
            "cancelled": "orange",
            "not yet started": "orange",
            "running": "blue",
            "failed": "red",
            "finished": "green",
        }[status]

        info = [
            ("status", f'<font color="{color}">{status}</font>'),
            ("# running jobs", f'<font color="blue">{n_running}</font>'),
            ("# pending jobs", f'<font color="orange">{n_pending}</font>'),
            ("# finished jobs", f'<font color="green">{n_done}</font>'),
            ("elapsed time", datetime.timedelta(seconds=self.elapsed_time())),
        ]

        with suppress(Exception):
            df = self.parse_log_files()
            abbr = '<abbr title="{}">{}</abbr>'  # creates a tooltip
            from_logs = [
                ("# of points", df.npoints.sum()),
                ("mean CPU usage", f"{df.cpu_usage.mean().round(1)} %"),
                ("mean memory usage", f"{df.mem_usage.mean().round(1)} %"),
                ("mean overhead", f"{df.overhead.mean().round(1)} %"),
            ]
            for key in ["npoints/s", "latest_loss", "nlearners"]:
                with suppress(Exception):
                    from_logs.append((f"mean {key}", f"{df[key].mean().round(1)}"))
            msg = "this is extracted from the log files, so it might not be up-to-date"
            info.extend([(abbr.format(msg, k), v) for k, v in from_logs])

        template = '<dt class="ignore-css">{}</dt><dd>{}</dd>'
        table = "\n".join(template.format(k, v) for k, v in info)

        return f"""
            <dl>
            {table}
            </dl>
        """

    def _repr_html_(self) -> None:
        return self.info()


def periodically_clean_ipython_profiles(scheduler, interval: int = 600):
    """Periodically remove old IPython profiles.

    In the `RunManager.cleanup` method the profiles will be removed. However,
    one might want to remove them earlier.

    Parameters
    ----------
    interval : int, default: 600
        The interval at which to remove old profiles.

    Returns
    -------
    asyncio.Task
    """

    async def clean(interval):
        from adaptive_scheduler.utils import _delete_old_ipython_profiles

        while True:
            with suppress(Exception):
                running_job_ids = set(scheduler.queue().keys())
                _delete_old_ipython_profiles(running_job_ids, with_progress_bar=False)
            await asyncio.sleep(interval)

    ioloop = asyncio.get_event_loop()
    coro = clean(interval)
    return ioloop.create_task(coro)
