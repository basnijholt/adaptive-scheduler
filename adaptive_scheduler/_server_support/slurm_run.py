from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from adaptive_scheduler.scheduler import SLURM, slurm_partitions
from adaptive_scheduler.utils import _get_default_args

from .common import console
from .run_manager import RunManager

if TYPE_CHECKING:
    from collections.abc import Callable

    import adaptive

    from adaptive_scheduler.utils import (
        _DATAFRAME_FORMATS,
        EXECUTOR_TYPES,
        LOKY_START_METHODS,
        GoalTypes,
    )


def slurm_run(
    learners: list[adaptive.BaseLearner],
    fnames: list[str] | list[Path],
    *,
    # Specific to slurm_run
    name: str = "adaptive-scheduler",
    folder: str | Path = "",
    # SLURM scheduler arguments
    partition: str | tuple[str | Callable[[], str], ...] | None = None,
    nodes: int | tuple[int | None | Callable[[], int | None], ...] | None = 1,
    cores_per_node: int | tuple[int | None | Callable[[], int | None], ...] | None = None,
    num_threads: int | tuple[int | Callable[[], int], ...] = 1,
    exclusive: bool | tuple[bool | Callable[[], bool], ...] = False,
    executor_type: EXECUTOR_TYPES
    | tuple[EXECUTOR_TYPES | Callable[[], EXECUTOR_TYPES], ...] = "process-pool",
    extra_scheduler: list[str] | tuple[list[str] | Callable[[], list[str]], ...] | None = None,
    # Same as RunManager below (except job_name, move_old_logs_to, and db_fname)
    goal: GoalTypes | None = None,
    check_goal_on_start: bool = True,
    dependencies: dict[int, list[int]] | None = None,
    runner_kwargs: dict | None = None,
    url: str | None = None,
    save_interval: float = 300,
    log_interval: float = 300,
    job_manager_interval: float = 60,
    kill_interval: float = 60,
    kill_on_error: str | Callable[[list[str]], bool] | None = "srun: error:",
    overwrite_db: bool = True,
    job_manager_kwargs: dict[str, Any] | None = None,
    kill_manager_kwargs: dict[str, Any] | None = None,
    loky_start_method: LOKY_START_METHODS = "loky",
    cleanup_first: bool = True,
    save_dataframe: bool = True,
    dataframe_format: _DATAFRAME_FORMATS = "pickle",
    max_log_lines: int = 500,
    max_fails_per_job: int = 50,
    max_simultaneous_jobs: int = 100,
    initializers: list[Callable[[], None]] | None = None,
    quiet: bool = False,
    # RunManager arguments
    extra_run_manager_kwargs: dict[str, Any] | None = None,
    extra_scheduler_kwargs: dict[str, Any] | None = None,
) -> RunManager:
    """Run adaptive on a SLURM cluster.

    ``cores_per_node``, ``nodes``, ``extra_scheduler``,
    ``executor_type``, ``exclusive``,
    ``num_threads`` and ``partition`` can be either a single value or a tuple of
    values. If a tuple is given, then the length of the tuple should be the same
    as the number of learners (jobs) that are run. This allows for different
    resources for different jobs. The tuple elements are also allowed to be
    callables without arguments, which will be called when the job is submitted.
    These callables should return the value that is needed. See the type hints
    for the allowed types.

    Parameters
    ----------
    learners
        A list of learners.
    fnames
        A list of filenames to save the learners.
    name
        The name of the job.
    folder
        The folder to save the adaptive_scheduler files such as logs, database,
        and ``.sbatch`` files in.
    partition
        The partition to use. If None, then the default partition will be used.
        (The one marked with a * in `sinfo`). Use
        `adaptive_scheduler.scheduler.slurm_partitions` to see the
        available partitions.
    nodes
        The number of nodes to use.
    cores_per_node
        The number of cores per node to use. If None, then all cores on the partition
        will be used.
    num_threads
        The number of threads to use.
    exclusive
        Whether to use exclusive nodes, adds ``"--exclusive"`` if True.
    executor_type
        The executor that is used, by default `concurrent.futures.ProcessPoolExecutor` is used.
        One can use ``"ipyparallel"``, ``"dask-mpi"``, ``"mpi4py"``,
        ``"loky"``, ``"sequential"``, or ``"process-pool"``.
    extra_scheduler
        Extra ``#SLURM`` (depending on scheduler type)
        arguments, e.g. ``["--exclusive=user", "--time=1"]`` or a tuple of lists,
        e.g. ``(["--time=10"], ["--time=20"]])`` for two jobs.
    goal
        The goal passed to the `adaptive.Runner`. Note that this function will
        be serialized and pasted in the ``job_script``. Can be a smart-goal
        that accepts
        ``Callable[[adaptive.BaseLearner], bool] | float | datetime | timedelta | None``.
        See `adaptive_scheduler.utils.smart_goal` for more information.
    check_goal_on_start
        Checks whether a learner is already done. Only works if the learner is loaded.
    dependencies
        Dictionary of dependencies, e.g., ``{1: [0]}`` means that the ``learners[1]``
        depends on the ``learners[0]``. This means that the ``learners[1]`` will only
        start when the ``learners[0]`` is done.
    runner_kwargs
        Extra keyword argument to pass to the `adaptive.Runner`. Note that this dict
        will be serialized and pasted in the ``job_script``.
    url
        The url of the database manager, with the format
        ``tcp://ip_of_this_machine:allowed_port.``. If None, a correct url will be chosen.
    save_interval
        Time in seconds between saving of the learners.
    log_interval
        Time in seconds between log entries.
    job_manager_interval
        Time in seconds between checking and starting jobs.
    kill_interval
        Check for `kill_on_error` string inside the log-files every `kill_interval` seconds.
    kill_on_error
        If ``error`` is a string and is found in the log files, the job will
        be cancelled and restarted. If it is a callable, it is applied
        to the log text. Must take a single argument, a list of
        strings, and return True if the job has to be killed, or
        False if not. Set to None if no `KillManager` is needed.
    overwrite_db
        Overwrite the existing database.
    job_manager_kwargs
        Keyword arguments for the `JobManager` function that aren't set in ``__init__`` here.
    kill_manager_kwargs
        Keyword arguments for the `KillManager` function that aren't set in ``__init__`` here.
    loky_start_method
        Loky start method, by default "loky".
    cleanup_first
        Cancel all previous jobs generated by the same RunManager and clean logfiles.
    save_dataframe
        Whether to periodically save the learner's data as a `pandas.DataFame`.
    dataframe_format
        The format in which to save the `pandas.DataFame`. See the type hint for the options.
    max_log_lines
        The maximum number of lines to display in the log viewer widget.
    max_fails_per_job
        Maximum number of times that a job can fail. This is here as a fail switch
        because a job might fail instantly because of a bug inside your code.
        The job manager will stop when
        ``n_jobs * total_number_of_jobs_failed > max_fails_per_job`` is true.
    max_simultaneous_jobs
        Maximum number of simultaneously running jobs. By default no more than 500
        jobs will be running. Keep in mind that if you do not specify a ``runner.goal``,
        jobs will run forever, resulting in the jobs that were not initially started
        (because of this `max_simultaneous_jobs` condition) to not ever start.
    initializers
        List of functions that are called before the job starts, can populate
        a cache.
    quiet
        Whether to show a progress bar when creating learner files.
    extra_run_manager_kwargs
        Extra keyword arguments to pass to the `RunManager`.
    extra_scheduler_kwargs
        Extra keyword arguments to pass to the `adaptive_scheduler.scheduler.SLURM`.

    Returns
    -------
    RunManager

    """
    if " " in name:
        msg = "The name should not contain spaces."
        raise ValueError(msg)
    if partition is None:
        partitions = slurm_partitions()
        assert isinstance(partitions, dict)
        partition, ncores = next(iter(partitions.items()))
        if not quiet:
            console.log(
                f"Using default partition {partition} (The one marked"
                f" with a '*' in `sinfo`) with {ncores} cores."
                " Use `adaptive_scheduler.scheduler.slurm_partitions`"
                " to see the available partitions.",
            )
    if (
        executor_type == "process-pool"
        and nodes is not None
        and (
            nodes > 1 if isinstance(nodes, int) else any(n > 1 for n in nodes if isinstance(n, int))
        )
    ):
        msg = "process-pool can maximally use a single node, use e.g., ipyparallel for multi node."
        raise ValueError(msg)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if cores_per_node is None:
        if isinstance(partition, tuple) and any(callable(p) for p in partition):
            msg = "cores_per_node must be given if partition is a callable."
            raise ValueError(msg)
        partitions = slurm_partitions()
        assert isinstance(partitions, dict)
        cores_per_node = (
            tuple(partitions[p] for p in partition)  # type: ignore[misc,index]
            if isinstance(partition, tuple)
            else partitions[partition]
        )

    if extra_scheduler_kwargs is None:
        extra_scheduler_kwargs = {}
    if extra_scheduler is not None:
        # "extra_scheduler" used to be passed via the extra_scheduler_kwargs
        # this ensures backwards compatibility
        assert "extra_scheduler" not in extra_scheduler_kwargs
        extra_scheduler_kwargs["extra_scheduler"] = extra_scheduler

    slurm_kwargs = dict(
        _get_default_args(SLURM),
        nodes=nodes,
        cores_per_node=cores_per_node,
        partition=partition,
        log_folder=folder / "logs",
        batch_folder=folder / "batch_scripts",
        executor_type=executor_type,
        num_threads=num_threads,
        exclusive=exclusive,
        **extra_scheduler_kwargs,
    )
    scheduler = SLURM(**slurm_kwargs)
    # Below are the defaults for the RunManager
    kw = dict(
        _get_default_args(RunManager),
        goal=goal,
        check_goal_on_start=check_goal_on_start,
        dependencies=dependencies,
        runner_kwargs=runner_kwargs,
        url=url,
        save_interval=save_interval,
        log_interval=log_interval,
        job_name=name,
        job_manager_interval=job_manager_interval,
        kill_interval=kill_interval,
        kill_on_error=kill_on_error,
        move_old_logs_to=folder / "old_logs",
        db_fname=folder / f"{name}.db.json",
        overwrite_db=overwrite_db,
        job_manager_kwargs=job_manager_kwargs,
        kill_manager_kwargs=kill_manager_kwargs,
        loky_start_method=loky_start_method,
        cleanup_first=cleanup_first,
        save_dataframe=save_dataframe,
        dataframe_format=dataframe_format,
        max_log_lines=max_log_lines,
        max_fails_per_job=max_fails_per_job,
        max_simultaneous_jobs=max_simultaneous_jobs,
        initializers=initializers,
        quiet=quiet,
        scheduler=scheduler,
        learners=learners,
        fnames=fnames,
    )
    if extra_run_manager_kwargs is None:
        extra_run_manager_kwargs = {}
    return RunManager(**dict(kw, **extra_run_manager_kwargs))
