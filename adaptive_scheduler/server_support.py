import abc
import asyncio
import concurrent.futures
import datetime
import glob
import json
import logging
import os
import pickle
import shutil
import socket
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Union

import adaptive
import cloudpickle
import jinja2
import pandas as pd
import structlog
import zmq
import zmq.asyncio
import zmq.ssh
from tinydb import Query, TinyDB

from adaptive_scheduler.scheduler import BaseScheduler
from adaptive_scheduler.utils import (
    _deserialize,
    _progress,
    _remove_or_move_files,
    _serialize,
    load_parallel,
    maybe_lst,
)
from adaptive_scheduler.widgets import info

ctx = zmq.asyncio.Context()

logger = logging.getLogger("adaptive_scheduler.server")
logger.setLevel(logging.INFO)
log = structlog.wrap_logger(logger)


class MaxRestartsReached(Exception):
    """Jobs can fail instantly because of a error in
    your Python code which results jobs being started indefinitely."""


class _BaseManager(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self.ioloop: Optional[asyncio.events.AbstractEventLoop] = None
        self._coro: Optional[Coroutine] = None
        self.task: Optional[asyncio.Task] = None

    def start(self):
        if self.is_started:
            raise Exception(f"{self.__class__} is already started!")
        self._setup()
        self.ioloop = asyncio.get_event_loop()
        self._coro = self._manage()
        self.task = self.ioloop.create_task(self._coro)
        return self

    @property
    def is_started(self) -> bool:
        return self.task is not None

    def cancel(self) -> Optional[bool]:
        if self.is_started:
            return self.task.cancel()

    def _setup(self):
        """Is run in the beginning of `self.start`."""
        pass

    @abc.abstractmethod
    async def _manage(self) -> None:
        pass


class DatabaseManager(_BaseManager):
    """Database manager.

    Parameters
    ----------
    url : str
        The url of the database manager, with the format
        ``tcp://ip_of_this_machine:allowed_port.``. Use `get_allowed_url`
        to get a `url` that will work.
    scheduler : `~adaptive_scheduler.scheduler.BaseScheduler`
        A scheduler instance from `adaptive_scheduler.scheduler`.
    db_fname : str
        Filename of the database, e.g. 'running.json'.
    learners : list of `adaptive.BaseLearner` isinstances
        List of `learners` corresponding to `fnames`.
    fnames : list
        List of `fnames` corresponding to `learners`.
    overwrite_db : bool, default: True
        Overwrite the existing database upon starting.

    Attributes
    ----------
    failed : list
        A list of entries that have failed and have been removed from the database.
    """

    def __init__(
        self,
        url: str,
        scheduler: BaseScheduler,
        db_fname: str,
        learners: List[adaptive.BaseLearner],
        fnames: Union[List[str], List[List[str]]],
        overwrite_db: bool = True,
    ) -> None:
        super().__init__()
        self.url = url
        self.scheduler = scheduler
        self.db_fname = db_fname
        self.learners = learners
        self.fnames = fnames
        self.overwrite_db = overwrite_db

        self.defaults = dict(
            job_id=None, is_done=False, log_fname=None, job_name=None, output_logs=[]
        )

        self._last_reply: Union[str, Exception, None] = None
        self._last_request: Optional[Tuple[str, ...]] = None
        self.failed: List[Dict[str, Any]] = []

    def _setup(self) -> None:
        if os.path.exists(self.db_fname) and not self.overwrite_db:
            return
        self.create_empty_db()

    def update(self, queue: Optional[Dict[str, Dict[str, str]]] = None) -> None:
        """If the ``job_id`` isn't running anymore, replace it with None."""
        if queue is None:
            queue = self.scheduler.queue(me_only=True)

        with TinyDB(self.db_fname) as db:
            failed = [
                entry
                for entry in db.all()
                if (entry["job_id"] is not None) and (entry["job_id"] not in queue)
            ]
            self.failed.extend(failed)
            doc_ids = [e.doc_id for e in failed]
            db.update({"job_id": None, "job_name": None}, doc_ids=doc_ids)

    def n_done(self) -> int:
        Entry = Query()
        with TinyDB(self.db_fname) as db:
            return db.count(Entry.is_done == True)  # noqa: E712

    def create_empty_db(self) -> None:
        """Create an empty database that keeps track of
        ``fname -> (job_id, is_done, log_fname, job_name)``.
        """
        entries = [dict(fname=fname, **self.defaults) for fname in self.fnames]
        if os.path.exists(self.db_fname):
            os.remove(self.db_fname)
        with TinyDB(self.db_fname) as db:
            db.insert_multiple(entries)

    def as_dicts(self) -> List[Dict[str, str]]:
        with TinyDB(self.db_fname) as db:
            return db.all()

    def _output_logs(self, job_id: str, job_name: str):
        job_id = self.scheduler.sanatize_job_id(job_id)
        output_fnames = self.scheduler.output_fnames(job_name)
        return [
            f.replace(self.scheduler._JOB_ID_VARIABLE, job_id) for f in output_fnames
        ]

    def _start_request(
        self, job_id: str, log_fname: str, job_name: str
    ) -> Optional[str]:
        Entry = Query()
        with TinyDB(self.db_fname) as db:
            if db.contains(Entry.job_id == job_id):
                entry = db.get(Entry.job_id == job_id)
                fname = entry["fname"]  # already running
                raise Exception(
                    f"The job_id {job_id} already exists in the database and "
                    f"runs {fname}. You might have forgotten to use the "
                    "`if __name__ == '__main__': ...` idom in your code. Read the "
                    "warning in the [mpi4py](https://bit.ly/2HAk0GG) documentation."
                )
            entry = db.get(
                (Entry.job_id == None) & (Entry.is_done == False)  # noqa: E711
            )
            log.debug("choose fname", entry=entry)
            if entry is None:
                return None
            db.update(
                {
                    "job_id": job_id,
                    "log_fname": log_fname,
                    "job_name": job_name,
                    "output_logs": self._output_logs(job_id, job_name),
                },
                doc_ids=[entry.doc_id],
            )
        return entry["fname"]

    def _stop_request(self, fname: Union[str, List[str]]) -> None:
        fname = maybe_lst(fname)  # if a BalancingLearner
        Entry = Query()
        with TinyDB(self.db_fname) as db:
            reset = dict(job_id=None, is_done=True, job_name=None)
            assert (
                db.get(Entry.fname == fname) is not None
            )  # make sure the entry exists
            db.update(reset, Entry.fname == fname)

    def _stop_requests(self, fnames: List[Union[str, List[str]]]) -> None:
        # Same as `_stop_request` but optimized for processing many `fnames` at once
        fnames = {str(maybe_lst(fname)) for fname in fnames}
        with TinyDB(self.db_fname) as db:
            reset = dict(job_id=None, is_done=True, job_name=None)
            doc_ids = [e.doc_id for e in db.all() if str(e["fname"]) in fnames]
            db.update(reset, doc_ids=doc_ids)

    def _dispatch(self, request: Tuple[str, ...]) -> Union[str, Exception, None]:
        request_type, *request_arg = request
        log.debug("got a request", request=request)
        try:
            if request_type == "start":
                # workers send us their slurm ID for us to fill in
                job_id, log_fname, job_name = request_arg
                kwargs = dict(job_id=job_id, log_fname=log_fname, job_name=job_name)
                # give the worker a job and send back the fname to the worker
                fname = self._start_request(**kwargs)
                if fname is None:
                    raise RuntimeError("No more learners to run in the database.")
                learner = next(
                    learner
                    for learner, f in zip(self.learners, self.fnames)
                    if maybe_lst(f) == fname
                )
                log.debug("choose a fname", fname=fname, **kwargs)
                return learner, fname
            elif request_type == "stop":
                fname = request_arg[0]  # workers send us the fname they were given
                log.debug("got a stop request", fname=fname)
                self._stop_request(fname)  # reset the job_id to None
                return None
        except Exception as e:
            return e

    async def _manage(self) -> None:
        """Database manager co-routine.

        Returns
        -------
        coroutine
        """
        log.debug("started database")
        socket = ctx.socket(zmq.REP)
        socket.bind(self.url)
        try:
            while True:
                try:
                    self._last_request = await socket.recv_serialized(_deserialize)
                except zmq.error.Again:
                    log.exception(
                        "socket.recv_serialized failed in the DatabaseManager"
                        " with `zmq.error.Again`."
                    )
                except pickle.UnpicklingError as e:
                    if r"\x03" in str(e):
                        # Empty frame received.
                        # TODO: not sure why this happens
                        pass
                    else:
                        log.exception(
                            "socket.recv_serialized failed in the DatabaseManager"
                            " with `pickle.UnpicklingError` in _deserialize."
                        )
                else:
                    self._last_reply = self._dispatch(self._last_request)
                    await socket.send_serialized(self._last_reply, _serialize)
        finally:
            socket.close()


class JobManager(_BaseManager):
    """Job manager.

    Parameters
    ----------
    job_names : list
        List of unique names used for the jobs with the same length as
        `learners`. Note that a job name does not correspond to a certain
        specific learner.
    database_manager : `DatabaseManager`
        A `DatabaseManager` instance.
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

    Attributes
    ----------
    n_started : int
        Total number of jobs started by the `JobManager`.
    """

    def __init__(
        self,
        job_names: List[str],
        database_manager: DatabaseManager,
        scheduler: BaseScheduler,
        interval: int = 30,
        *,
        max_simultaneous_jobs: int = 5000,
        max_fails_per_job: int = 100,
    ):
        super().__init__()
        self.job_names = job_names
        self.database_manager = database_manager
        self.scheduler = scheduler
        self.interval = interval
        self.max_simultaneous_jobs = max_simultaneous_jobs
        self.max_fails_per_job = max_fails_per_job

        self.n_started = 0

    @property
    def max_job_starts(self) -> int:
        """Equivalent to ``self.max_fails_per_job * len(self.job_names)``"""
        return self.max_fails_per_job * len(self.job_names)

    def _queued(self, queue) -> Set[str]:
        return {
            job["job_name"]
            for job in queue.values()
            if job["job_name"] in self.job_names
        }

    async def _manage(self) -> None:
        with concurrent.futures.ProcessPoolExecutor() as ex:
            while True:
                try:
                    running = self.scheduler.queue(me_only=True)
                    self.database_manager.update(running)  # in case some jobs died

                    queued = self._queued(running)  # running `job_name`s
                    not_queued = set(self.job_names) - queued

                    n_done = self.database_manager.n_done()
                    if n_done == len(self.job_names):
                        # we are finished!
                        self.database_manager.task.cancel()
                        return
                    else:
                        n_to_schedule = max(0, len(not_queued) - n_done)
                        not_queued = set(list(not_queued)[:n_to_schedule])

                    while not_queued:
                        # start new jobs
                        if len(queued) <= self.max_simultaneous_jobs:
                            job_name = not_queued.pop()
                            queued.add(job_name)
                            await self.ioloop.run_in_executor(
                                ex, self.scheduler.start_job, job_name
                            )
                            self.n_started += 1
                        else:
                            break
                    if self.n_started > self.max_job_starts:
                        raise MaxRestartsReached(
                            "Too many jobs failed, your Python code probably has a bug."
                        )
                    await asyncio.sleep(self.interval)
                except concurrent.futures.CancelledError:
                    log.info("task was cancelled because of a CancelledError")
                    raise
                except MaxRestartsReached as e:
                    log.exception(
                        "too many jobs have failed, cancelling the job manager",
                        n_started=self.n_started,
                        max_fails_per_job=self.max_fails_per_job,
                        max_job_starts=self.max_job_starts,
                        exception=str(e),
                    )
                    raise
                except Exception as e:
                    log.exception("got exception when starting a job", exception=str(e))
                    await asyncio.sleep(5)


def logs_with_string_or_condition(
    error: Union[str, Callable[[List[str]], bool]], database_manager: DatabaseManager
) -> List[Tuple[str, List[str]]]:
    """Get jobs that have `string` (or apply a callable) inside their log-file.

    Either use `string` or `error`.

    Parameters
    ----------
    error : str or callable
        String that is searched for or callable that is applied
        to the log text. Must take a single argument, a list of
        strings, and return True if the job has to be killed, or
        False if not.
    database_manager : `DatabaseManager`
        A `DatabaseManager` instance.

    Returns
    -------
    has_string : dict
        A list ``(job_name, fnames)``, which have the string inside their log-file.
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

    have_error = []
    for entry in database_manager.as_dicts():
        fnames = entry["output_logs"]
        if entry["job_id"] is not None and any(file_has_error(f) for f in fnames):
            all_fnames = fnames + [entry["log_fname"]]
            have_error.append((entry["job_name"], all_fnames))
    return have_error


class KillManager(_BaseManager):
    """Kill manager.

    Automatically cancel jobs that contain an error (or other condition)
    in the log files.

    Parameters
    ----------
    scheduler : `~adaptive_scheduler.scheduler.BaseScheduler`
        A scheduler instance from `adaptive_scheduler.scheduler`.
    database_manager : `DatabaseManager`
        A `DatabaseManager` instance.
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
    """

    def __init__(
        self,
        scheduler: BaseScheduler,
        database_manager: DatabaseManager,
        error: Union[str, Callable[[List[str]], bool]] = "srun: error:",
        interval: int = 600,
        max_cancel_tries: int = 5,
        move_to: Optional[str] = None,
    ):
        super().__init__()
        self.scheduler = scheduler
        self.database_manager = database_manager
        self.error = error
        self.interval = interval
        self.max_cancel_tries = max_cancel_tries
        self.move_to = move_to

        self.cancelled: List[str] = []
        self.deleted: List[str] = []

    async def _manage(self) -> None:
        while True:
            try:
                self.database_manager.update()

                failed_jobs = logs_with_string_or_condition(
                    self.error, self.database_manager
                )

                to_cancel: List[str] = []
                to_delete: List[str] = []
                for job_name, fnames in failed_jobs:
                    to_cancel.append(job_name)
                    to_delete.extend(fnames)

                self.scheduler.cancel(
                    to_cancel, with_progress_bar=False, max_tries=self.max_cancel_tries
                )
                _remove_or_move_files(
                    to_delete, with_progress_bar=False, move_to=self.move_to
                )
                self.cancelled.extend(to_cancel)
                self.deleted.extend(to_delete)
                await asyncio.sleep(self.interval)
            except concurrent.futures.CancelledError:
                log.info("task was cancelled because of a CancelledError")
                raise
            except Exception as e:
                log.exception("got exception in kill manager", exception=str(e))


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


def _make_default_run_script(
    url: str,
    save_interval: int,
    log_interval: int,
    goal: Union[Callable[[adaptive.BaseLearner], bool], None] = None,
    runner_kwargs: Optional[Dict[str, Any]] = None,
    run_script_fname: str = "run_learner.py",
    executor_type: str = "mpi4py",
) -> None:
    default_runner_kwargs = dict(shutdown_executor=True)
    runner_kwargs = dict(default_runner_kwargs, goal=goal, **(runner_kwargs or {}))
    serialized_runner_kwargs = cloudpickle.dumps(runner_kwargs)

    if executor_type not in ("mpi4py", "ipyparallel", "dask-mpi", "process-pool"):
        raise NotImplementedError(
            "Use 'ipyparallel', 'dask-mpi', 'mpi4py' or 'process-pool'."
        )

    if executor_type == "dask-mpi":
        try:
            import dask_mpi  # noqa: F401
        except ModuleNotFoundError as e:
            msg = "You need to have 'dask-mpi' installed to use `executor_type='dask-mpi'`."
            raise Exception(msg) from e

    with open(Path(__file__).parent / "run_script.py.j2", encoding="utf-8") as f:
        empty = "".join(f.readlines())

    template = jinja2.Template(empty).render(
        run_script_fname=run_script_fname,
        executor_type=executor_type,
        url=url,
        serialized_runner_kwargs=serialized_runner_kwargs,
        save_interval=save_interval,
        log_interval=log_interval,
    )

    with open(run_script_fname, "w", encoding="utf-8") as f:
        f.write(template)


def _get_infos(fname: str, only_last: bool = True) -> List[str]:
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


def parse_log_files(
    job_names: List[str],
    database_manager: DatabaseManager,
    scheduler,
    only_last: bool = True,
) -> pd.DataFrame:
    """Parse the log-files and convert it to a `~pandas.core.frame.DataFrame`.

    This only works if you use `adaptive_scheduler.client_support.log_info`
    inside your ``run_script``.

    Parameters
    ----------
    job_names : list
        List of job names.
    database_manager : `DatabaseManager`
        A `DatabaseManager` instance.
    scheduler : `~adaptive_scheduler.scheduler.BaseScheduler`
        A scheduler instance from `adaptive_scheduler.scheduler`.
    only_last : bool, default: True
        Only look use the last printed status message.

    Returns
    -------
    `~pandas.core.frame.DataFrame`
    """

    _queue = scheduler.queue()
    database_manager.update(_queue)

    infos = []
    for entry in database_manager.as_dicts():
        log_fname = entry["log_fname"]
        if log_fname is None or not os.path.exists(log_fname):
            continue
        for info_dict in _get_infos(log_fname, only_last):
            info_dict.pop("event")  # this is always "current status"
            info_dict["timestamp"] = datetime.datetime.strptime(
                info_dict["timestamp"], "%Y-%m-%d %H:%M.%S"
            )
            info_dict["elapsed_time"] = pd.to_timedelta(info_dict["elapsed_time"])
            info_dict.update(entry)
            infos.append(info_dict)

    for info_dict in infos:
        info_from_queue = _queue.get(info_dict["job_id"])
        if info_from_queue is None:
            continue
        info_dict["state"] = info_from_queue["state"]
        info_dict["job_name"] = info_from_queue["job_name"]

    return pd.DataFrame(infos)


def _get_all_files(job_names: List[str], scheduler: BaseScheduler) -> List[str]:
    log_fnames = [scheduler.log_fname(name) for name in job_names]
    output_fnames = [scheduler.output_fnames(name) for name in job_names]
    output_fnames = sum(output_fnames, [])
    batch_fnames = [scheduler.batch_fname(name) for name in job_names]
    fnames = log_fnames + output_fnames + batch_fnames
    all_files = [glob.glob(f.replace(scheduler._JOB_ID_VARIABLE, "*")) for f in fnames]
    all_files = sum(all_files, [])

    # For schedulers that use a single batch file
    name_prefix = job_names[0].rsplit("-", 1)[0]
    batch_file = scheduler.batch_fname(name_prefix)
    if os.path.exists(batch_file):
        all_files.append(batch_file)
    return all_files


def cleanup(
    job_names: List[str],
    scheduler: BaseScheduler,
    with_progress_bar: bool = True,
    move_to: Optional[str] = None,
) -> None:
    """Cleanup the scheduler log-files files.

    Parameters
    ----------
    job_names : list
        List of job names.
    scheduler : `~adaptive_scheduler.scheduler.BaseScheduler`
        A scheduler instance from `adaptive_scheduler.scheduler`.
    with_progress_bar : bool, default: True
        Display a progress bar using `tqdm`.
    move_to : str, default: None
        Move the file to a different directory.
        If None the file is removed.
    log_file_folder : str, default: ''
        The folder in which to delete the log-files.
    """

    to_rm = _get_all_files(job_names, scheduler)

    _remove_or_move_files(
        to_rm, with_progress_bar, move_to, "Removing logs and batch files"
    )


def _delete_old_ipython_profiles(
    scheduler: BaseScheduler, with_progress_bar: bool = True
) -> None:

    if scheduler.executor_type != "ipyparallel":
        return
    # We need the job_ids because only job_names wouldn't be
    # enough information. There might be other job_managers
    # running.
    pattern = "profile_adaptive_scheduler_"
    profile_folders = glob.glob(os.path.expanduser(f"~/.ipython/{pattern}*"))

    running_job_ids = set(scheduler.queue().keys())
    to_delete = [
        folder
        for folder in profile_folders
        if not folder.split(pattern)[1] in running_job_ids
    ]

    with ThreadPoolExecutor(256) as ex:
        desc = "Submitting deleting old IPython profiles tasks"
        pbar = _progress(to_delete, desc=desc)
        futs = [ex.submit(shutil.rmtree, folder) for folder in pbar]
        desc = "Finishing deleting old IPython profiles"
        for fut in _progress(futs, with_progress_bar, desc=desc):
            fut.result()


class RunManager(_BaseManager):
    """A convenience tool that starts the job, database, and kill manager.

    Parameters
    ----------
    scheduler : `~adaptive_scheduler.scheduler.BaseScheduler`
        A scheduler instance from `adaptive_scheduler.scheduler`.
    learners : list of `adaptive.BaseLearner` isinstances
        List of `learners` corresponding to `fnames`.
    fnames : list
        List of `fnames` corresponding to `learners`.
    goal : callable, default: None
        The goal passed to the `adaptive.Runner`. Note that this function will
        be serialized and pasted in the ``run_script``.
    check_goal_on_start : bool, default: True
        Checks whether a learner is already done. Only works if the learner is loaded.
    runner_kwargs : dict, default: None
        Extra keyword argument to pass to the `adaptive.Runner`. Note that this dict
        will be serialized and pasted in the ``run_script``.
    url : str, default: None
        The url of the database manager, with the format
        ``tcp://ip_of_this_machine:allowed_port.``. If None, a correct url will be chosen.
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
        False if not. Set to None if no `KillManager` is needed.
    move_old_logs_to : str, default: "old_logs"
        Move logs of killed jobs to this directory. If None the logs will be deleted.
    db_fname : str, default: "running.json"
        Filename of the database, e.g. 'running.json'.
    overwrite_db : bool, default: True
        Overwrite the existing database.
    job_manager_kwargs : dict, default: None
        Keyword arguments for the `JobManager` function that aren't set in ``__init__`` here.
    kill_manager_kwargs : dict, default: None
        Keyword arguments for the `KillManager` function that aren't set in ``__init__`` here.

    Attributes
    ----------
    job_names : list
        List of job_names. Generated with ``self.job_name``.
    database_manager : `DatabaseManager`
        The database manager.
    job_manager : `JobManager`
        The job manager.
    kill_manager : `KillManager` or None
        The kill manager.
    start_time : float or None
        Time at which ``self.start()`` is called.
    end_time : float or None
        Time at which the jobs are all done or at which ``self.cancel()`` is called.

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
    ...     goal=goal,
    ...     log_interval=30,
    ...     save_interval=30,
    ... )
    >>> run_manager.start()

    """

    def __init__(
        self,
        scheduler: BaseScheduler,
        learners: List[adaptive.BaseLearner],
        fnames: List[str],
        goal: Union[Callable[[adaptive.BaseLearner], bool], None] = None,
        check_goal_on_start: bool = True,
        runner_kwargs: Optional[dict] = None,
        url: Optional[str] = None,
        save_interval: int = 300,
        log_interval: int = 300,
        job_name: str = "adaptive-scheduler",
        job_manager_interval: int = 60,
        kill_interval: int = 60,
        kill_on_error: Union[str, Callable[[List[str]], bool], None] = "srun: error:",
        move_old_logs_to: Optional[str] = "old_logs",
        db_fname: Optional[str] = None,
        overwrite_db: bool = True,
        job_manager_kwargs: Optional[Dict[str, Any]] = None,
        kill_manager_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        # Set from arguments
        self.scheduler = scheduler
        self.goal = goal
        self.check_goal_on_start = check_goal_on_start
        self.runner_kwargs = runner_kwargs
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.job_name = job_name
        self.job_manager_interval = job_manager_interval
        self.kill_interval = kill_interval
        self.kill_on_error = kill_on_error
        self.move_old_logs_to = move_old_logs_to
        self.db_fname = db_fname or f"{job_name}-database.json"
        self.overwrite_db = overwrite_db
        self.job_manager_kwargs = job_manager_kwargs or {}
        self.kill_manager_kwargs = kill_manager_kwargs or {}

        # Set in methods
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # Set on init
        self.learners = learners
        self.fnames = fnames

        if isinstance(self.fnames[0], (list, tuple)):
            # For a BalancingLearner
            assert isinstance(self.fnames[0][0], str)
        else:
            assert isinstance(self.fnames[0], str)

        self.job_names = [f"{self.job_name}-{i}" for i in range(len(self.learners))]
        self.url = url or get_allowed_url()
        self.database_manager = DatabaseManager(
            self.url,
            self.scheduler,
            self.db_fname,
            self.learners,
            self.fnames,
            self.overwrite_db,
        )
        self.job_manager = JobManager(
            self.job_names,
            self.database_manager,
            scheduler=self.scheduler,
            interval=self.job_manager_interval,
            **self.job_manager_kwargs,
        )

        if self.kill_on_error is not None:
            self.kill_manager = KillManager(
                scheduler=self.scheduler,
                database_manager=self.database_manager,
                error=self.kill_on_error,
                interval=self.kill_interval,
                move_to=self.move_old_logs_to,
                **self.kill_manager_kwargs,
            )
        else:
            self.kill_manager = None

    def _setup(self) -> None:
        _make_default_run_script(
            self.url,
            self.save_interval,
            self.log_interval,
            self.goal,
            self.runner_kwargs,
            self.scheduler.run_script,
            self.scheduler.executor_type,
        )
        self.database_manager.start()
        if self.check_goal_on_start:
            # Check if goal already reached
            # Only works after the `database_manager` has started.
            done_fnames = [
                fname
                for fname, learner in zip(self.fnames, self.learners)
                if self.goal(learner)
            ]
            self.database_manager._stop_requests(done_fnames)
        self.job_manager.start()
        if self.kill_manager:
            self.kill_manager.start()
        self.start_time = time.time()

    async def _manage(self) -> None:
        await self.job_manager.task
        self.end_time = time.time()

    def cancel(self) -> None:
        """Cancel the manager tasks and the jobs in the queue."""
        self.database_manager.cancel()
        self.job_manager.cancel()
        self.kill_manager.cancel()
        self.scheduler.cancel(self.job_names)
        if self.task is not None:
            self.task.cancel()
        self.end_time = time.time()

    def cleanup(self, remove_old_logs_folder=False) -> None:
        """Cleanup the log and batch files.

        If the `RunManager` is not running, the ``run_script.py`` file
        will also be removed.
        """
        with suppress(FileNotFoundError):
            if self.status() != "running":
                os.remove(self.scheduler.run_script)

        _delete_old_ipython_profiles(self.scheduler)

        cleanup(
            job_names=self.job_names,
            scheduler=self.scheduler,
            with_progress_bar=True,
            move_to=self.move_old_logs_to,
        )
        if remove_old_logs_folder:
            with suppress(FileNotFoundError):
                shutil.rmtree(self.move_old_logs_to)

    def parse_log_files(self, only_last: bool = True) -> pd.DataFrame:
        """Parse the log-files and convert it to a `~pandas.core.frame.DataFrame`.

        Parameters
        ----------
        only_last : bool, default: True
            Only look use the last printed status message.

        Returns
        -------
        df : `~pandas.core.frame.DataFrame`

        """
        return parse_log_files(
            self.job_names, self.database_manager, self.scheduler, only_last
        )

    def task_status(self) -> None:
        r"""Print the stack of the `asyncio.Task`\s."""
        if self.job_manager.task is not None:
            self.job_manager.task.print_stack()
        if self.database_manager.task is not None:
            self.database_manager.task.print_stack()
        if self.kill_manager.task is not None:
            self.kill_manager.task.print_stack()
        if self.task is not None:
            self.task.print_stack()

    def get_database(self) -> List[Dict[str, Any]]:
        """Get the database as a `pandas.DataFrame`."""
        return pd.DataFrame(self.database_manager.as_dicts())

    def load_learners(self) -> None:
        """Load the learners in parallel using `adaptive_scheduler.utils.load_parallel`."""
        load_parallel(self.learners, self.fnames)

    def elapsed_time(self) -> float:
        """Total time elapsed since the `RunManager` was started."""
        if not self.is_started:
            return 0

        if self.job_manager.task.done():
            end_time = self.end_time
            if end_time is None:
                # task was cancelled before it began
                assert self.job_manager.task.cancelled()
                return 0
        else:
            end_time = time.time()
        return end_time - self.start_time

    def status(self) -> str:
        """Return the current status of the `RunManager`."""
        if not self.is_started:
            return "not yet started"

        try:
            self.job_manager.task.result()
        except asyncio.InvalidStateError:
            status = "running"
        except asyncio.CancelledError:
            status = "cancelled"
        except Exception:
            status = "failed"
            print("JobManager failed because of the following")
            traceback.print_exc()
        else:
            status = "finished"

        try:
            self.database_manager.task.result()
        except (asyncio.InvalidStateError, asyncio.CancelledError):
            pass
        except Exception:
            status = "failed"
            print("DatabaseManager failed because of the following")
            traceback.print_exc()

        if status == "running":
            return "running"

        if self.end_time is None:
            self.end_time = time.time()
        return status

    def _repr_html_(self) -> None:
        return info(self)

    def info(self):
        return info(self)


def periodically_clean_ipython_profiles(scheduler, interval: int = 600):
    """Periodically remove old IPython profiles.

    In the `RunManager.cleanup` method the profiles will be removed. However,
    one might want to remove them earlier.

    Parameters
    ----------
    scheduler : `~adaptive_scheduler.scheduler.BaseScheduler`
        A scheduler instance from `adaptive_scheduler.scheduler`.
    interval : int, default: 600
        The interval at which to remove old profiles.

    Returns
    -------
    asyncio.Task
    """

    async def clean(interval):
        while True:
            with suppress(Exception):
                _delete_old_ipython_profiles(scheduler, with_progress_bar=False)
            await asyncio.sleep(interval)

    ioloop = asyncio.get_event_loop()
    coro = clean(interval)
    return ioloop.create_task(coro)
