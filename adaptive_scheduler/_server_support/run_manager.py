from __future__ import annotations

import asyncio
import shutil
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from adaptive_scheduler.utils import (
    LOKY_START_METHODS,
    GoalTypes,
    _at_least_adaptive_version,
    _time_between,
    fname_to_learner_fname,
    load_dataframes,
    load_parallel,
    sleep_unless_task_is_done,
    smart_goal,
)
from adaptive_scheduler.widgets import info

from .base_manager import BaseManager
from .common import (
    _delete_old_ipython_profiles,
    _maybe_path,
    cleanup_scheduler_files,
    console,
    get_allowed_url,
)
from .database_manager import DatabaseManager
from .job_manager import JobManager
from .kill_manager import KillManager
from .parse_logs import parse_log_files

if TYPE_CHECKING:
    import adaptive

    from adaptive_scheduler.scheduler import BaseScheduler
    from adaptive_scheduler.utils import _DATAFRAME_FORMATS


class RunManager(BaseManager):
    """A convenience tool that starts the job, database, and kill manager.

    Parameters
    ----------
    scheduler : `~adaptive_scheduler.scheduler.BaseScheduler`
        A scheduler instance from `adaptive_scheduler.scheduler`.
    learners : list of `adaptive.BaseLearner` isinstances
        List of `learners` corresponding to `fnames`.
    fnames : list
        List of `fnames` corresponding to `learners`.
    goal : callable
        The goal passed to the `adaptive.Runner`. Note that this function will
        be serialized and pasted in the ``job_script``. Can be a smart-goal
        that accepts
        ``Callable[[adaptive.BaseLearner], bool] | int | float | datetime | timedelta | None``.
        See `adaptive_scheduler.utils.smart_goal` for more information.
    initializers : list of callables
        List of functions that are called before the job starts, can populate
        a cache.
    check_goal_on_start : bool
        Checks whether a learner is already done. Only works if the learner is loaded.
    runner_kwargs : dict
        Extra keyword argument to pass to the `adaptive.Runner`. Note that this dict
        will be serialized and pasted in the ``job_script``.
    url : str
        The url of the database manager, with the format
        ``tcp://ip_of_this_machine:allowed_port.``. If None, a correct url will be chosen.
    save_interval : int
        Time in seconds between saving of the learners.
    log_interval : int
        Time in seconds between log entries.
    job_name : str
        From this string the job names will be created, e.g.
        ``["adaptive-scheduler-1", "adaptive-scheduler-2", ...]``.
    job_manager_interval : int
        Time in seconds between checking and starting jobs.
    kill_interval : int
        Check for `kill_on_error` string inside the log-files every `kill_interval` seconds.
    kill_on_error : str or callable
        If ``error`` is a string and is found in the log files, the job will
        be cancelled and restarted. If it is a callable, it is applied
        to the log text. Must take a single argument, a list of
        strings, and return True if the job has to be killed, or
        False if not. Set to None if no `KillManager` is needed.
    move_old_logs_to : str
        Move logs of killed jobs to this directory. If None the logs will be deleted.
    db_fname : str
        Filename of the database, e.g. 'running.json'.
    overwrite_db : bool
        Overwrite the existing database.
    job_manager_kwargs : dict
        Keyword arguments for the `JobManager` function that aren't set in ``__init__`` here.
    kill_manager_kwargs : dict
        Keyword arguments for the `KillManager` function that aren't set in ``__init__`` here.
    loky_start_method : str
        Loky start method, by default "loky".
    cleanup_first : bool
        Cancel all previous jobs generated by the same RunManager and clean logfiles.
    save_dataframe : bool
        Whether to periodically save the learner's data as a `pandas.DataFame`.
    dataframe_format : str
        The format in which to save the `pandas.DataFame`. See the type hint for the options.
    max_log_lines : int
        The maximum number of lines to display in the log viewer widget.

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
        learners: list[adaptive.BaseLearner],
        fnames: list[str] | list[Path],
        *,
        goal: GoalTypes | None = None,
        check_goal_on_start: bool = True,
        runner_kwargs: dict | None = None,
        url: str | None = None,
        save_interval: int | float = 300,
        log_interval: int | float = 300,
        job_name: str = "adaptive-scheduler",
        job_manager_interval: int | float = 60,
        kill_interval: int | float = 60,
        kill_on_error: str | Callable[[list[str]], bool] | None = "srun: error:",
        move_old_logs_to: str | Path | None = "old_logs",
        db_fname: str | Path | None = None,
        overwrite_db: bool = True,
        job_manager_kwargs: dict[str, Any] | None = None,
        kill_manager_kwargs: dict[str, Any] | None = None,
        loky_start_method: LOKY_START_METHODS = "loky",
        cleanup_first: bool = False,
        save_dataframe: bool = False,
        dataframe_format: _DATAFRAME_FORMATS = "pickle",
        max_log_lines: int = 500,
        max_fails_per_job: int = 50,
        max_simultaneous_jobs: int = 100,
        initializers: list[Callable[[], None]] | None = None,
    ) -> None:
        super().__init__()

        # Set from arguments
        self.scheduler = scheduler
        self.goal = smart_goal(goal, learners)
        self.check_goal_on_start = check_goal_on_start
        self.runner_kwargs = runner_kwargs
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.job_name = job_name
        self.job_manager_interval = job_manager_interval
        self.kill_interval = kill_interval
        self.kill_on_error = kill_on_error
        self.move_old_logs_to = _maybe_path(move_old_logs_to)
        self.db_fname = Path(db_fname or f"{job_name}-database.json")
        self.overwrite_db = overwrite_db
        self.job_manager_kwargs = job_manager_kwargs or {}
        self.kill_manager_kwargs = kill_manager_kwargs or {}
        self.loky_start_method = loky_start_method
        self.save_dataframe = save_dataframe
        self.dataframe_format = dataframe_format
        self.max_log_lines = max_log_lines
        self.max_fails_per_job = max_fails_per_job
        self.max_simultaneous_jobs = max_simultaneous_jobs
        self.initializers = initializers
        # Track job start times, (job_name, start_time) -> request_time
        self._job_start_time_dict: dict[tuple[str, str], str] = {}

        for key in ["max_fails_per_job", "max_simultaneous_jobs"]:
            if key in self.job_manager_kwargs:
                msg = (
                    f"The `{key}` argument is not allowed in `job_manager_kwargs`."
                    " Please specify it in `RunManager.__init__` instead.",
                )
                raise ValueError(msg)

        if self.save_dataframe:
            _at_least_adaptive_version("0.14.0", "save_dataframe")

        # Set in methods
        self.start_time: float | None = None
        self.end_time: float | None = None
        self._start_one_by_one_task: tuple[
            asyncio.Future,
            list[asyncio.Task],
        ] | None = None

        # Set on init
        self.learners = learners
        self.fnames = fnames

        if isinstance(self.fnames[0], (list, tuple)):
            # For a BalancingLearner
            assert isinstance(self.fnames[0][0], (str, Path))
        else:
            assert isinstance(self.fnames[0], (str, Path))

        self.job_names = [f"{self.job_name}-{i}" for i in range(len(self.learners))]

        if cleanup_first:
            self.scheduler.cancel(self.job_names)
            self.cleanup(remove_old_logs_folder=True)

        self.url = url or get_allowed_url()
        self.database_manager = DatabaseManager(
            url=self.url,
            scheduler=self.scheduler,
            db_fname=self.db_fname,
            learners=self.learners,
            fnames=self.fnames,
            overwrite_db=self.overwrite_db,
            initializers=self.initializers,
        )
        self.job_manager = JobManager(
            self.job_names,
            self.database_manager,
            scheduler=self.scheduler,
            interval=self.job_manager_interval,
            max_fails_per_job=self.max_fails_per_job,
            max_simultaneous_jobs=self.max_simultaneous_jobs,
            # Laucher command line options
            save_dataframe=self.save_dataframe,
            dataframe_format=self.dataframe_format,
            loky_start_method=self.loky_start_method,
            log_interval=self.log_interval,
            save_interval=self.save_interval,
            runner_kwargs=self.runner_kwargs,
            goal=self.goal,
            **self.job_manager_kwargs,
        )
        self.kill_manager: KillManager | None
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
        self.database_manager.start()
        if self.check_goal_on_start:
            # Check if goal already reached
            # Only works after the `database_manager` has started.
            done_fnames = [
                fname
                for fname, learner in zip(self.fnames, self.learners)
                if self.goal(learner)
            ]
            self.database_manager._stop_requests(done_fnames)  # type: ignore[arg-type]
        self.job_manager.start()
        if self.kill_manager:
            self.kill_manager.start()
        self.start_time = time.time()

    def start(self, wait_for: RunManager | None = None) -> RunManager:  # type: ignore[override]
        """Start the RunManager and optionally wait for another RunManager to finish."""
        if wait_for is not None:
            self._start_one_by_one_task = start_one_by_one(wait_for, self)
        else:
            super().start()
        return self

    async def _manage(self) -> None:
        assert self.job_manager.task is not None
        while not self.job_manager.task.done():
            if self.job_manager._request_times:
                for job in self.database_manager.as_dicts():
                    start_time = job["start_time"]
                    job_name = job["job_name"]
                    # Check if the job actually started (not cancelled)
                    if (
                        start_time is not None
                        and job_name in self.job_manager._request_times
                        and (job_name, start_time) not in self._job_start_time_dict
                    ):
                        request_time = self.job_manager._request_times.pop(job_name)
                        self._job_start_time_dict[job_name, start_time] = request_time

            if await sleep_unless_task_is_done(
                self.database_manager.task,  # type: ignore[arg-type]
                5,
            ):  # if true, we are done
                break

        self.end_time = time.time()

    def job_starting_times(self) -> list[float]:
        """Return the starting times of the jobs."""
        return [
            _time_between(end, start)
            for (_, start), end in self._job_start_time_dict.items()
        ]

    def cancel(self) -> None:
        """Cancel the manager tasks and the jobs in the queue."""
        self.database_manager.cancel()
        self.job_manager.cancel()
        if self.kill_manager is not None:
            self.kill_manager.cancel()
        self.scheduler.cancel(self.job_names)
        if self.task is not None:
            self.task.cancel()
        self.end_time = time.time()
        if self._start_one_by_one_task is not None:
            self._start_one_by_one_task[0].cancel()

    def cleanup(self, *, remove_old_logs_folder: bool = False) -> None:
        """Cleanup the log and batch files."""
        for fname in self.fnames:
            fname_cloudpickle = fname_to_learner_fname(fname)
            with suppress(FileNotFoundError):
                fname_cloudpickle.unlink()

        _delete_old_ipython_profiles(self.scheduler)

        cleanup_scheduler_files(
            job_names=self.job_names,
            scheduler=self.scheduler,
            with_progress_bar=True,
            move_to=self.move_old_logs_to,
        )
        if remove_old_logs_folder and self.move_old_logs_to is not None:
            with suppress(FileNotFoundError):
                shutil.rmtree(self.move_old_logs_to)

    def parse_log_files(self, *, only_last: bool = True) -> pd.DataFrame:
        """Parse the log-files and convert it to a `~pandas.core.frame.DataFrame`.

        Parameters
        ----------
        only_last : bool
            Only look use the last printed status message.

        Returns
        -------
        df : `~pandas.core.frame.DataFrame`

        """
        return parse_log_files(
            self.database_manager,
            self.scheduler,
            only_last=only_last,
        )

    def task_status(self) -> None:
        r"""Print the stack of the `asyncio.Task`\s."""
        if self.job_manager.task is not None:
            self.job_manager.task.print_stack()
        if self.database_manager.task is not None:
            self.database_manager.task.print_stack()
        if self.kill_manager is not None and self.kill_manager.task is not None:
            self.kill_manager.task.print_stack()
        if self.task is not None:
            self.task.print_stack()

    def get_database(self) -> pd.DataFrame:
        """Get the database as a `pandas.DataFrame`."""
        return pd.DataFrame(self.database_manager.as_dicts())

    def load_learners(self) -> None:
        """Load the learners in parallel using `adaptive_scheduler.utils.load_parallel`."""
        load_parallel(self.learners, self.fnames)

    def elapsed_time(self) -> float:
        """Total time elapsed since the `RunManager` was started."""
        if not self.is_started:
            return 0
        assert self.job_manager.task is not None  # for mypy
        if self.job_manager.task.done():
            end_time = self.end_time
            if end_time is None:
                # task was cancelled before it began
                assert self.job_manager.task.cancelled()
                return 0
        else:
            end_time = time.time()
        return end_time - self.start_time  # type: ignore[operator]

    def status(self) -> str:
        """Return the current status of the `RunManager`."""
        if not self.is_started:
            return "not yet started"

        try:
            assert self.job_manager.task is not None
            self.job_manager.task.result()
        except asyncio.InvalidStateError:
            status = "running"
        except asyncio.CancelledError:
            status = "cancelled"
        except Exception:  # noqa: BLE001
            status = "failed"
            console.log("`JobManager` failed because of the following")
            console.print_exception(show_locals=True)
        else:
            status = "finished"

        try:
            assert self.database_manager.task is not None  # for mypy
            self.database_manager.task.result()
        except (asyncio.InvalidStateError, asyncio.CancelledError):
            pass
        except Exception:  # noqa: BLE001
            status = "failed"
            console.log("`DatabaseManager` failed because of the following")
            console.print_exception(show_locals=True)
        if status == "running":
            return "running"

        if self.end_time is None:
            self.end_time = time.time()
        return status

    def _repr_html_(self) -> None:
        return info(self)

    def info(self) -> None:
        return info(self)

    def load_dataframes(self) -> pd.DataFrame:
        """Load the `pandas.DataFrame`s with the most recently saved learners data."""
        if not self.save_dataframe:
            msg = "The `save_dataframe` option was not set to True."
            raise ValueError(msg)
        return load_dataframes(self.fnames, format=self.dataframe_format)  # type: ignore[return-value]


async def _wait_for_finished(
    manager_first: RunManager,
    manager_second: RunManager,
    goal: Callable[[RunManager], bool] | None = None,
    interval: int | float = 120,
) -> None:
    if goal is None:
        assert manager_first.task is not None  # for mpypy
        await manager_first.task
    else:
        while not goal(manager_first):
            await asyncio.sleep(interval)
    manager_second.start()


def _start_after(
    manager_first: RunManager,
    manager_second: RunManager,
    goal: Callable[[RunManager], bool] | None = None,
    interval: int | float = 120,
) -> asyncio.Task:
    if manager_second.is_started:
        msg = "The second manager must not be started yet."
        raise ValueError(msg)
    coro = _wait_for_finished(manager_first, manager_second, goal, interval)
    return asyncio.create_task(coro)


def start_one_by_one(
    *run_managers: RunManager,
    goal: Callable[[RunManager], bool] | None = None,
    interval: int | float = 120,
) -> tuple[asyncio.Future, list[asyncio.Task]]:
    """Start a list of RunManagers after each other.

    Parameters
    ----------
    run_managers : list[RunManager]
        A list of RunManagers.
    goal : callable
        A callable that takes a RunManager as argument and returns a boolean.
        If `goal` is not None, the RunManagers will be started after `goal`
        returns True for the previous RunManager. If `goal` is None, the
        RunManagers will be started after the previous RunManager has finished.
    interval : int
        The interval at which to check if `goal` is True. Only used if `goal`
        is not None.

    Returns
    -------
    tuple[asyncio.Future, list[asyncio.Future]]
        The first element is the grouped task that starts all RunManagers.
        The second element is a list of tasks that start each RunManager.

    Examples
    --------
    >>> manager_1 = adaptive_scheduler.slurm_run(
    ...     learners[:5],
    ...     fnames[:5],
    ...     partition="hb120rsv2-low",
    ...     goal=0.01,
    ...     name="first",
    ... )
    >>> manager_1.start()
    >>> manager_2 = adaptive_scheduler.slurm_run(
    ...     learners[5:],
    ...     fnames[5:],
    ...     partition="hb120rsv2-low",
    ...     goal=0.01,
    ...     name="second",
    ... )
    >>> # Start second when the first RunManager has more than 1000 points.
    >>> def start_goal(run_manager):
    ...     df = run_manager.parse_log_files()
    ...     npoints = df.get("npoints")
    ...     if npoints is None:
    ...         return False
    ...     return npoints.sum() > 1000
    >>> tasks = adaptive_scheduler.start_one_by_one(
    ...     manager_1,
    ...     manager_2,
    ...     goal=start_goal,
    ... )

    """
    uniques = ["job_name", "db_fname"]
    for u in uniques:
        if len({getattr(r, u) for r in run_managers}) != len(run_managers):
            msg = (
                f"All `RunManager`s must have a unique {u}."
                " If using `slurm_run` these are controlled through the `name` argument.",
            )
            raise ValueError(msg)

    tasks = [
        _start_after(run_managers[i], run_managers[i + 1], goal, interval)
        for i in range(len(run_managers) - 1)
    ]
    return asyncio.gather(*tasks), tasks
