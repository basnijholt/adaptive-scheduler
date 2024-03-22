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
    partition: str | tuple[str, ...] | None = None,
    nodes: int | tuple[int, ...] = 1,
    cores_per_node: int | tuple[int, ...] | None = None,
    goal: GoalTypes | None = None,
    folder: str | Path = "",
    name: str = "adaptive",
    num_threads: int | tuple[int, ...] = 1,
    save_interval: float = 300,
    log_interval: float = 300,
    cleanup_first: bool = True,
    save_dataframe: bool = True,
    dataframe_format: _DATAFRAME_FORMATS = "pickle",
    max_fails_per_job: int = 50,
    max_simultaneous_jobs: int = 100,
    exclusive: bool | tuple[bool, ...] = True,
    executor_type: EXECUTOR_TYPES | tuple[EXECUTOR_TYPES, ...] = "process-pool",
    extra_scheduler: list[str] | tuple[list[str], ...] | None = None,
    extra_run_manager_kwargs: dict[str, Any] | None = None,
    extra_scheduler_kwargs: dict[str, Any] | None = None,
    initializers: list[Callable[[], None]] | None = None,
) -> RunManager:
    """Run adaptive on a SLURM cluster.

    ``cores``, ``nodes``, ``cores_per_node``, ``extra_scheduler``,
    ``executor_type``, ``extra_script``, ``exclusive``, ``extra_env_vars``,
    ``num_threads`` and ``partition`` can be either a single value or a tuple of
    values. If a tuple is given, then the length of the tuple should be the same
    as the number of learners (jobs) that are run. This allows for different
    resources for different jobs.

    Parameters
    ----------
    learners
        A list of learners.
    fnames
        A list of filenames to save the learners.
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
    goal
        The goal of the adaptive run. If None, then the run will continue
        indefinitely.
    folder
        The folder to save the adaptive_scheduler files such as logs, database,
        and ``.sbatch`` files in.
    name
        The name of the job.
    num_threads
        The number of threads to use.
    save_interval
        The interval at which to save the learners.
    log_interval
        The interval at which to log the status of the run.
    cleanup_first
        Whether to clean up the folder before starting the run.
    save_dataframe
        Whether to save the `pandas.DataFrame`s with the learners data.
    dataframe_format
        The format to save the `pandas.DataFrame`s in. See
        `adaptive_scheduler.utils.save_dataframes` for more information.
    max_fails_per_job
        The maximum number of times a job can fail before it is cancelled.
    max_simultaneous_jobs
        The maximum number of simultaneous jobs.
    executor_type
        The executor that is used, by default `concurrent.futures.ProcessPoolExecutor` is used.
        One can use ``"ipyparallel"``, ``"dask-mpi"``, ``"mpi4py"``,
        ``"loky"``, ``"sequential"``, or ``"process-pool"``.
    exclusive
        Whether to use exclusive nodes, adds ``"--exclusive"`` if True.
    extra_scheduler
        Extra ``#SLURM`` (depending on scheduler type)
        arguments, e.g. ``["--exclusive=user", "--time=1"]`` or a tuple of lists,
        e.g. ``(["--time=10"], ["--time=20"]])`` for two jobs.
    extra_run_manager_kwargs
        Extra keyword arguments to pass to the `RunManager`.
    extra_scheduler_kwargs
        Extra keyword arguments to pass to the `adaptive_scheduler.scheduler.SLURM`.
    initializers
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
    if executor_type == "process-pool" and (
        nodes > 1 if isinstance(nodes, int) else any(n > 1 for n in nodes)
    ):
        msg = (
            "process-pool can maximally use a single node,"
            " use e.g., ipyparallel for multi node.",
        )
        raise ValueError(msg)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if cores_per_node is None:
        partitions = slurm_partitions()
        assert isinstance(partitions, dict)
        cores_per_node = (
            tuple(partitions[p] for p in partition)  # type: ignore[misc]
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
