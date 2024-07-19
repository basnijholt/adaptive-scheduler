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

    from adaptive_scheduler.utils import _DATAFRAME_FORMATS, EXECUTOR_TYPES, GoalTypes


def slurm_run(
    learners: list[adaptive.BaseLearner],
    fnames: list[str] | list[Path],
    *,
    partition: str | tuple[str | Callable[[], str], ...] | None = None,
    nodes: int | tuple[int | None | Callable[[], int | None], ...] | None = 1,
    cores_per_node: int | tuple[int | None | Callable[[], int | None], ...] | None = None,
    goal: GoalTypes | None = None,
    folder: str | Path = "",
    name: str = "adaptive",
    dependencies: dict[int, list[int]] | None = None,
    num_threads: int | tuple[int | Callable[[], int], ...] = 1,
    save_interval: float = 300,
    log_interval: float = 300,
    job_manager_interval: float = 60,
    cleanup_first: bool = True,
    save_dataframe: bool = True,
    dataframe_format: _DATAFRAME_FORMATS = "pickle",
    max_fails_per_job: int = 50,
    max_simultaneous_jobs: int = 100,
    exclusive: bool | tuple[bool | Callable[[], bool], ...] = False,
    executor_type: EXECUTOR_TYPES
    | tuple[EXECUTOR_TYPES | Callable[[], EXECUTOR_TYPES], ...] = "process-pool",
    extra_scheduler: list[str] | tuple[list[str] | Callable[[], list[str]], ...] | None = None,
    extra_run_manager_kwargs: dict[str, Any] | None = None,
    extra_scheduler_kwargs: dict[str, Any] | None = None,
    initializers: list[Callable[[], None]] | None = None,
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
    dependencies
        Dictionary of dependencies, e.g., ``{1: [0]}`` means that the ``learners[1]``
        depends on the ``learners[0]``. This means that the ``learners[1]`` will only
        start when the ``learners[0]`` is done.
    num_threads
        The number of threads to use.
    save_interval
        The interval at which to save the learners.
    log_interval
        The interval at which to log the status of the run.
    job_manager_interval
        The interval at which the job manager checks the status of the jobs and
        submits new jobs.
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
        scheduler=scheduler,
        learners=learners,
        fnames=fnames,
        goal=goal,
        save_interval=save_interval,
        log_interval=log_interval,
        move_old_logs_to=folder / "old_logs",
        db_fname=folder / f"{name}.db.json",
        job_name=name,
        dependencies=dependencies,
        cleanup_first=cleanup_first,
        save_dataframe=save_dataframe,
        dataframe_format=dataframe_format,
        max_fails_per_job=max_fails_per_job,
        max_simultaneous_jobs=max_simultaneous_jobs,
        initializers=initializers,
        job_manager_interval=job_manager_interval,
    )
    if extra_run_manager_kwargs is None:
        extra_run_manager_kwargs = {}
    return RunManager(**dict(kw, **extra_run_manager_kwargs))
