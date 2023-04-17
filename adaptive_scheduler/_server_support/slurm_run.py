from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

from adaptive_scheduler.scheduler import SLURM, slurm_partitions
from adaptive_scheduler.utils import _get_default_args

from .common import console
from .run_manager import RunManager

if TYPE_CHECKING:
    import datetime

    import adaptive

    from adaptive_scheduler.utils import EXECUTOR_TYPES


def slurm_run(
    learners: list[adaptive.BaseLearner],
    fnames: list[str] | list[Path],
    *,
    partition: str | None = None,
    nodes: int = 1,
    cores_per_node: int | None = None,
    goal: Callable[[adaptive.BaseLearner], bool]
    | int
    | float
    | datetime.timedelta
    | datetime.datetime
    | None = None,
    folder: str | Path = "",
    name: str = "adaptive",
    num_threads: int = 1,
    save_interval: int | float = 300,
    log_interval: int | float = 300,
    cleanup_first: bool = True,
    save_dataframe: bool = True,
    dataframe_format: Literal[
        "parquet",
        "csv",
        "hdf",
        "pickle",
        "feather",
        "excel",
        "json",
    ] = "parquet",
    max_fails_per_job: int = 50,
    max_simultaneous_jobs: int = 100,
    exclusive: bool = True,
    executor_type: EXECUTOR_TYPES = "process-pool",
    extra_run_manager_kwargs: dict[str, Any] | None = None,
    extra_scheduler_kwargs: dict[str, Any] | None = None,
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
    nodes : int, default: 1
        The number of nodes to use.
    cores_per_node : int, default: None
        The number of cores per node to use. If None, then all cores on the partition
        will be used.
    goal : callable, int, float, datetime.timedelta, datetime.datetime, default: None
        The goal of the adaptive run. If None, then the run will continue
        indefinitely.
    folder : str or pathlib.Path, default: ""
        The folder to save the learners in.
    name : str, default: "adaptive"
        The name of the job.
    num_threads : int, default: 1
        The number of threads to use.
    save_interval : int, default: 300
        The interval at which to save the learners.
    log_interval : int, default: 300
        The interval at which to log the status of the run.
    cleanup_first : bool, default: True
        Whether to clean up the folder before starting the run.
    save_dataframe : bool, default: True
        Whether to save the `pandas.DataFrame`s with the learners data.
    dataframe_format : str, default: "parquet"
        The format to save the `pandas.DataFrame`s in. See
        `adaptive_scheduler.utils.save_dataframes` for more information.
    max_fails_per_job : int, default: 50
        The maximum number of times a job can fail before it is cancelled.
    max_simultaneous_jobs : int, default: 500
        The maximum number of simultaneous jobs.
    executor_type : str, default: "process-pool"
        The type of executor to use. One of "ipyparallel", "dask-mpi", "mpi4py",
        "loky", or "process-pool".
    exclusive : bool, default: True
        Whether to use exclusive nodes, adds ``"--exclusive"`` if True.
    extra_run_manager_kwargs : dict, default: None
        Extra keyword arguments to pass to the `RunManager`.
    extra_scheduler_kwargs : dict, default: None
        Extra keyword arguments to pass to the `SLURMScheduler`.

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
        db_fname=f"{name}.db.json",
        job_name=name,
        cleanup_first=cleanup_first,
        save_dataframe=save_dataframe,
        dataframe_format=dataframe_format,
        max_fails_per_job=max_fails_per_job,
        max_simultaneous_jobs=max_simultaneous_jobs,
    )
    if extra_run_manager_kwargs is None:
        extra_run_manager_kwargs = {}
    return RunManager(**dict(kw, **extra_run_manager_kwargs))
