from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from adaptive_scheduler.scheduler import SLURM, slurm_partitions
from adaptive_scheduler.utils import _get_default_args

from .common import console
from .run_manager import RunManager

if TYPE_CHECKING:
    import adaptive

    from adaptive_scheduler.utils import _DATAFRAME_FORMATS, EXECUTOR_TYPES, GoalTypes


def slurm_run(
    learners: list[adaptive.BaseLearner],
    fnames: list[str] | list[Path],
    *,
    partition: str | None = None,
    nodes: int = 1,
    cores_per_node: int | None = None,
    goal: GoalTypes | None = None,
    folder: str | Path = "",
    name: str = "adaptive",
    num_threads: int = 1,
    save_interval: int | float = 300,
    log_interval: int | float = 300,
    cleanup_first: bool = True,
    save_dataframe: bool = True,
    dataframe_format: _DATAFRAME_FORMATS = "pickle",
    max_fails_per_job: int = 50,
    max_simultaneous_jobs: int = 100,
    exclusive: bool = True,
    executor_type: EXECUTOR_TYPES = "process-pool",
    extra_run_manager_kwargs: dict[str, Any] | None = None,
    extra_scheduler_kwargs: dict[str, Any] | None = None,
    initializers: list[Callable[[], None]] | None = None,
) -> RunManager:
    """Run adaptive on a SLURM cluster.

    Parameters
    ----------
    learners : list[adaptive.BaseLearner]
        A list of learners.
    fnames : list[str]
        A list of filenames to save the learners.
    partition : str
        The partition to use. If None, then the default partition will be used.
        (The one marked with a * in `sinfo`). Use
        `adaptive_scheduler.scheduler.slurm_partitions` to see the
        available partitions.
    nodes : int
        The number of nodes to use.
    cores_per_node : int
        The number of cores per node to use. If None, then all cores on the partition
        will be used.
    goal : callable, int, float, datetime.timedelta, datetime.datetime
        The goal of the adaptive run. If None, then the run will continue
        indefinitely.
    folder : str or pathlib.Path
        The folder to save the learners in.
    name : str
        The name of the job.
    num_threads : int
        The number of threads to use.
    save_interval : int
        The interval at which to save the learners.
    log_interval : int
        The interval at which to log the status of the run.
    cleanup_first : bool
        Whether to clean up the folder before starting the run.
    save_dataframe : bool
        Whether to save the `pandas.DataFrame`s with the learners data.
    dataframe_format : str
        The format to save the `pandas.DataFrame`s in. See
        `adaptive_scheduler.utils.save_dataframes` for more information.
    max_fails_per_job : int
        The maximum number of times a job can fail before it is cancelled.
    max_simultaneous_jobs : int
        The maximum number of simultaneous jobs.
    executor_type : str
        The type of executor to use. One of "ipyparallel", "dask-mpi", "mpi4py",
        "loky", or "process-pool".
    exclusive : bool
        Whether to use exclusive nodes, adds ``"--exclusive"`` if True.
    extra_run_manager_kwargs : dict
        Extra keyword arguments to pass to the `RunManager`.
    extra_scheduler_kwargs : dict
        Extra keyword arguments to pass to the `SLURMScheduler`.
    initializers : list of callables
        List of functions that are called before the job starts, can populate
        a cache.

    Returns
    -------
    RunManager
    """
    if partition is None:
        partitions = slurm_partitions()
        assert isinstance(partitions, dict)
        partition, ncores = next(iter(partitions.items()))
        console.log(
            f"Using default partition {partition} (The one marked"
            f" with a '*' in `sinfo`) with {ncores} cores."
            " Use `adaptive_scheduler.scheduler.slurm_partitions`"
            " to see the available partitions.",
        )
    if executor_type == "process-pool" and nodes > 1:
        msg = (
            "process-pool can maximally use a single node,"
            " use e.g., ipyparallel for multi node.",
        )
        raise ValueError(msg)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if cores_per_node is None:
        cores_per_node = slurm_partitions()[partition]  # type: ignore[call-overload]
    kw = dict(
        _get_default_args(SLURM),
        nodes=nodes,
        cores_per_node=cores_per_node,
        partition=partition,
        log_folder=folder / "logs",
        batch_folder=folder / "batch_scripts",
        executor_type=executor_type,
        num_threads=num_threads,
    )
    if extra_scheduler_kwargs is None:
        extra_scheduler_kwargs = {}
    scheduler = SLURM(**dict(kw, exclusive=exclusive, **extra_scheduler_kwargs))
    # Below are the defaults for the RunManager
    kw = dict(
        _get_default_args(RunManager),
        scheduler=scheduler,
        learners=learners,
        fnames=fnames,
        goal=goal,
        save_interval=save_interval,
        log_interval=log_interval,
        move_old_logs_to=folder / "old_logs",
        db_fname=folder / f"{name}.db.json",
        job_name=name,
        cleanup_first=cleanup_first,
        save_dataframe=save_dataframe,
        dataframe_format=dataframe_format,
        max_fails_per_job=max_fails_per_job,
        max_simultaneous_jobs=max_simultaneous_jobs,
        initializers=initializers,
    )
    if extra_run_manager_kwargs is None:
        extra_run_manager_kwargs = {}
    return RunManager(**dict(kw, **extra_run_manager_kwargs))
