import asyncio
import concurrent.futures
import datetime
import logging
import os
import socket
import subprocess
import textwrap
import time
from contextlib import suppress
from typing import Any, Coroutine, Dict, List, Optional

import dill
import structlog
import zmq
import zmq.asyncio
import zmq.ssh
from tinydb import Query, TinyDB

from adaptive_scheduler._scheduler import (
    ext,
    make_job_script,
    queue,
    submit_cmd,
    cancel,
)

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


def start_database_manager(url: str, db_fname: str):
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
cores : int
    Number of cores per job (so per learner.)
job_script_function : callable, default: `adaptive_scheduler.slurm.make_job_script` or `adaptive_scheduler.pbs.make_job_script` depending on the scheduler.
    A function with the following signature:
    ``job_script(name, cores, run_script, python_executable)`` that returns
    a job script in string form. See ``adaptive_scheduler/slurm.py`` or
    ``adaptive_scheduler/pbs.py`` for an example.
run_script : str
    Filename of the script that is run on the nodes. Inside this script we
    query the database and run the learner.
python_executable : str, default: `sys.executable`
    The Python executable that should run the `run_script`. By default
    it uses the same Python as where this function is called.
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
    job_names,
    db_fname,
    ioloop,
    cores=8,
    job_script_function=make_job_script,
    run_script="run_learner.py",
    python_executable=None,
    interval=30,
    *,
    max_simultaneous_jobs=5000,
    max_fails_per_job=100,
) -> Coroutine:
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
                if n_jobs_done == len(job_names):
                    return
                to_start = len(job_names) - len(running_job_names) - n_jobs_done
                to_start = min(max_simultaneous_jobs, to_start)
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
    job_names,
    db_fname,
    cores=8,
    job_script_function=make_job_script,
    run_script="run_learner.py",
    python_executable=None,
    interval=30,
    *,
    max_simultaneous_jobs=5000,
    max_fails_per_job=40,
) -> asyncio.Task:
    ioloop = asyncio.get_event_loop()
    coro = manage_jobs(
        job_names,
        db_fname,
        ioloop,
        cores,
        job_script_function,
        run_script,
        python_executable,
        interval,
        max_simultaneous_jobs=max_simultaneous_jobs,
        max_fails_per_job=max_fails_per_job,
    )
    return ioloop.create_task(coro)


start_job_manager.__doc__ = _JOB_MANAGER_DOC.format(
    first_line="Start the job manager task.", returns="asyncio.Task", extra_args=""
)


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


def create_empty_db(db_fname: str, fnames: List[str]):
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


def get_database(db_fname: str) -> List[Dict[str, Any]]:
    """Get the database as a list of dicts."""
    with TinyDB(db_fname) as db:
        return db.all()


def _update_db(db_fname: str, running: Dict[str, dict]):
    """If the job_id isn't running anymore, replace it with None."""
    with TinyDB(db_fname) as db:
        doc_ids = [entry.doc_id for entry in db.all() if entry["job_id"] not in running]
        db.update({"job_id": None}, doc_ids=doc_ids)


def _choose_fname(db_fname: str, job_id: str):
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
        log.debug("chose fname", entry=entry)
        if entry is None:
            return
        db.update({"job_id": job_id}, doc_ids=[entry.doc_id])
    return entry["fname"]


def _done_with_learner(db_fname: str, fname: str):
    Entry = Query()
    with TinyDB(db_fname) as db:
        db.update({"job_id": None, "is_done": True}, Entry.fname == fname)


def _get_n_jobs_done(db_fname: str):
    Entry = Query()
    with TinyDB(db_fname) as db:
        return db.count(Entry.is_done == True)  # noqa: E711


async def kill_failed(
    job_names: List[str],
    error: str = "srun: error:",
    interval: int = 600,
    max_cancel_tries: int = 5,
    move_to: Optional[str] = None,
) -> Coroutine:
    """XXX: IS NOT FULLY WORKING/TESTED YET"""
    # It seems like tasks that print the error message do not always stop working
    # I think it only stops working when the error happens on a node where the logger runs.
    from adaptive_scheduler.utils import _remove_or_move_files, logs_with_string

    while True:
        try:
            failed_jobs = logs_with_string(job_names, error)
            to_cancel = []
            to_delete = []

            # get cancel/delete only the processes/logs that are running nowg
            for job_id, info in queue().items():
                job_name = info["name"]
                if job_id in failed_jobs.get(job_name, []):
                    to_cancel.append(job_name)
                    to_delete.append(f"{job_name}-{job_id}.out")

            cancel(to_cancel, with_progress_bar=False, max_tries=max_cancel_tries)
            _remove_or_move_files(to_delete, with_progress_bar=False, move_to=move_to)
            await asyncio.sleep(interval)
        except concurrent.futures.CancelledError:
            log.info("task was cancelled because of a CancelledError")
            raise
        except Exception as e:
            log.exception("got exception in kill manager", exception=str(e))


def start_kill_manager(
    job_names: List[str],
    error: str = "srun: error:",
    interval: int = 600,
    max_cancel_tries: int = 5,
    move_to: Optional[str] = None,
) -> asyncio.Task:
    """XXX: IS NOT FULLY WORKING/TESTED YET"""
    ioloop = asyncio.get_event_loop()
    coro = kill_failed(job_names, error, interval, max_cancel_tries, move_to)
    return ioloop.create_task(coro)


def _make_default_run_script(
    url,
    learners_file,
    save_interval,
    log_interval,
    goal=None,
    run_script_fname="run_learner.py",
):
    serialized_goal = dill.dumps(goal)

    template = textwrap.dedent(
        f"""\
    # {run_script_fname}, automatically generated
    # by `adaptive_scheduler.server_support._make_default_run_script()`.
    import adaptive
    import dill
    from adaptive_scheduler import client_support
    from mpi4py.futures import MPIPoolExecutor

    # the file that defines the learners we created above
    from {learners_file.rstrip(".py")} import learners, fnames

    if __name__ == "__main__":  # ← use this, see warning @ https://bit.ly/2HAk0GG
        # the address of the "database manager"
        url = "{url}"

        # ask the database for a learner that we can run
        learner, fname = client_support.get_learner(url, learners, fnames)

        # load the data
        learner.load(fname)

        # this is serialized by dill.dumps
        goal = dill.loads({serialized_goal})

        # run until `some_goal` is reached with an `MPIPoolExecutor`
        runner = adaptive.Runner(
            learner, executor=MPIPoolExecutor(), shutdown_executor=True, goal=goal
        )

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
    with open(run_script_fname, "w") as f:
        f.write(template)
    return run_script_fname


class RunManager:
    """A convenience tool that starts the job, database, and kill manager.

    Parameters
    ----------
    run_script : str, default: None
        Filename of the script that is run on the nodes. Inside this script we
        query the database and run the learner. If None, a standard script
        will be created.
    goal : callable, default: None
        The goal passed to the `adaptive.Runner`. Note that this function will
        be serialized and pasted in the job script. If using a custom ``run_script``.
        the goal is ignored.
    url : str, default: None
        The url of the database manager, with the format
        ``tcp://ip_of_this_machine:allowed_port.``. If None, a correct url will be chosen.
        You only **need** to specify this when using a custom ``run_script``.
    learners_file : str, default: "learners_file.py"
        The module that defined the variables ``learners`` and ``fnames``.
    save_interval : int, default: 300
        Time in seconds between saving of the learners.
    log_interval : int, default: 300
        Time in seconds between log entries.
    job_name : str, default: "adaptive-scheduler"
        From this string the job names will be created, e.g.
        ``["adaptive-scheduler-1", "adaptive-scheduler-2", ...]``.
    job_script_function : callable, default: `adaptive_scheduler.slurm.make_job_script` or `adaptive_scheduler.pbs.make_job_script` depending on the scheduler.
        A function with the following signature:
        ``job_script(name, cores, run_script, python_executable)`` that returns
        a job script in string form. See ``adaptive_scheduler/slurm.py`` or
        ``adaptive_scheduler/pbs.py`` for an example.
    cores_per_job : int, default: 1
        Number of cores per job (so per learner.)
    job_manager_interval : int, default: 60
        Time in seconds between checking and starting jobs.
    kill_interval : int, default: 60
        Check for `kill_on_error` string inside the log-files every `kill_interval` seconds.
    kill_on_error : str, default: "srun: error:"
        If this error is encountered in the log-files the job is killed.
    move_logs_to : str, default: "old_logs"
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
    >>> from functools import partial
    >>> from adaptive_scheduler.slurm import make_job_script
    >>> job_script_function = partial(
    ...     make_job_script,
    ...     extra_sbatch=[
    ...         "--ntasks-per-node=12",
    ...         "--cpus-per-task=2",
    ...         "--time=5:00:35",
    ...         "--exclusive",
    ...     ],
    ...     mpiexec_executable="srun --mpi=pmi2",
    ... )
    >>> run_manager = adaptive_scheduler.server_support.RunManager(
    ...     job_script_function=job_script_function, cores_per_job=12, overwrite_db=True
    ... ).start()
    """

    def __init__(
        self,
        run_script: Optional[str] = None,
        goal: Optional[callable] = None,
        url: Optional[str] = None,
        learners_file: str = "learners_file.py",
        save_interval: int = 300,
        log_interval: int = 300,
        job_name: str = "adaptive-scheduler",
        job_script_function: callable = make_job_script,
        cores_per_job: int = 1,
        job_manager_interval: int = 60,
        kill_interval: int = 60,
        kill_on_error: str = "srun: error:",
        move_logs_to: Optional[str] = "old_logs",
        db_fname: str = "running.json",
        overwrite_db: bool = True,
        start_job_manager_kwargs: Optional[dict] = None,
        start_kill_manager_kwargs: Optional[dict] = None,
    ):
        # Set from arguments
        self.run_script = run_script
        self.goal = goal
        self.learners_file = learners_file
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.job_name = job_name
        self.job_script_function = job_script_function
        self.cores_per_job = cores_per_job
        self.job_manager_interval = job_manager_interval
        self.kill_interval = kill_interval
        self.kill_on_error = kill_on_error
        self.move_logs_to = move_logs_to
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

        # Check incompatible arguments
        if goal is not None and run_script is not None:
            raise ValueError("Do not pass a goal if you are using a custom run_script.")

    def start(self):
        async def _start():
            await self.job_task
            self.end_time = time.time()

        self.write_db()
        self._start_database_manager()
        self._start_job_manager(**self.start_job_manager_kwargs)
        self._start_kill_manager(**self.start_kill_manager_kwargs)
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

    def write_db(self) -> None:
        if os.path.exists(self.db_fname) and not self.overwrite_db:
            return
        create_empty_db(self.db_fname, self.learners_module.fnames)

    def _set_job_names(self) -> None:
        learners = self.learners_module.learners
        self.job_names = [f"{self.job_name}-{i}" for i in range(len(learners))]

    def _start_job_manager(self) -> None:

        if self.run_script is None:
            self.run_script = _make_default_run_script(
                self.url,
                self.learners_file,
                self.save_interval,
                self.log_interval,
                self.goal,
            )

        self.job_task = start_job_manager(
            self.job_names,
            self.db_fname,
            cores=self.cores_per_job,
            interval=self.job_manager_interval,
            run_script=self.run_script,
            job_script_function=self.job_script_function,
        )

    def _start_database_manager(self) -> None:
        self.database_task = start_database_manager(self.url, self.db_fname)

    def _start_kill_manager(self) -> None:
        self.kill_task = start_kill_manager(
            self.job_names,
            error=self.kill_on_error,
            interval=self.kill_interval,
            move_to=self.move_logs_to,
            **self.start_kill_manager_kwargs,
        )

    def cancel(self):
        if self.job_task is not None:
            self.job_task.cancel()
            self.database_task.cancel()
            self.kill_task.cancel()
        return cancel(self.job_names)

    def cleanup(self):
        from adaptive_scheduler.utils import cleanup_files

        return cleanup_files(self.job_names)

    def parse_log_files(self, only_last=True):
        """Parse the log-files and convert it to a `~pandas.core.frame.DataFrame`.

        Parameters
        ----------
        only_last : bool, default: True
            Only look use the last printed status message.

        Returns
        -------
        `~pandas.core.frame.DataFrame`
        """
        from adaptive_scheduler.utils import parse_log_files

        return parse_log_files(self.job_names, only_last, self.db_fname)

    def task_status(self):
        return (
            self.job_task.print_stack(),
            self.database_task.print_stack(),
            self.kill_task.print_stack(),
        )

    def get_database(self) -> List[Dict[str, Any]]:
        return get_database(self.db_fname)

    def elapsed_time(self):
        """Return the total time elapsed since the RunManager
        was started."""
        if self.job_task.done():
            end_time = self.end_time
            if end_time is None:
                # task was cancelled before it began
                assert self.job_task.cancelled()
                return 0
        else:
            end_time = time.time()
        return end_time - self.start_time

    def _task_status(self, task):
        try:
            task.result()
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

    def status(self):
        return self._task_status(self.job_task)

    def info(self, *, update_interval=0.5):
        """Display information about the run_manager.
        Returns an interactive ipywidget that can be
        visualized in a Jupyter notebook.
        """
        from ipywidgets import Layout, Button, VBox, HBox, HTML
        from IPython.display import display

        status = HTML(value=self._info_html(update_interval))

        layout = Layout(width="150px")
        buttons = [
            Button(description="update info", layout=layout, button_color="lightgreen"),
            Button(description="cancel jobs", layout=layout, button_style="danger"),
            Button(
                description="cleanup log files", layout=layout, button_style="danger"
            ),
        ]
        buttons = {b.description: b for b in buttons}

        def update(_):
            status.value = self._info_html(update_interval)

        buttons["cancel jobs"].on_click(lambda _: self.cancel())
        buttons["cleanup log files"].on_click(lambda _: self.cleanup())
        buttons["update info"].on_click(update)

        buttons = VBox(list(buttons.values()))
        display(
            HBox(
                (status, buttons),
                layout=Layout(border="solid 1px", width="400px", align_items="center"),
            )
        )

    def _info_html(self, update_interval):
        jobs = [job for job in queue().values() if job["name"] in self.job_names]
        n_running = sum(job["state"] in ("RUNNING", "R") for job in jobs)
        n_pending = sum(job["state"] in ("PENDING", "P") for job in jobs)
        n_done = sum(job["is_done"] for job in self.get_database())

        status = self.status()
        color = {
            "cancelled": "orange",
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
            msg = "this is extracted from the log files, so it might not be up-to-date"
            info.extend([(abbr.format(msg, k), v) for k, v in from_logs])

        template = '<dt class="ignore-css">{}</dt><dd>{}</dd>'
        table = "\n".join(template.format(k, v) for k, v in info)

        return f"""
            <dl>
            {table}
            </dl>
        """

    def _repr_html_(self):
        return self.info()
