from __future__ import annotations

import argparse
import base64
from contextlib import suppress
from typing import TYPE_CHECKING, Any, get_args

import adaptive
import cloudpickle

from adaptive_scheduler import client_support
from adaptive_scheduler.utils import (
    _DATAFRAME_FORMATS,
    LOKY_START_METHODS,
)

if TYPE_CHECKING:
    from adaptive_scheduler.scheduler import BaseScheduler
    from adaptive_scheduler.utils import EXECUTOR_TYPES, GoalTypes

    from .database_manager import DatabaseManager


def _serialize_to_b64(x: Any) -> str:
    serialized_x = cloudpickle.dumps(x)
    return base64.b64encode(serialized_x).decode("utf-8")


def _deserialize_from_b64(x: str) -> Any:
    bytes_ = base64.b64decode(x)
    return cloudpickle.loads(bytes_)


def command_line_options(
    *,
    scheduler: BaseScheduler,
    database_manager: DatabaseManager,
    runner_kwargs: dict[str, Any] | None = None,
    goal: GoalTypes,
    log_interval: int | float = 60,
    save_interval: int | float = 300,
    save_dataframe: bool = True,
    dataframe_format: _DATAFRAME_FORMATS = "parquet",
) -> dict[str, Any]:
    """Return the command line options for the job_script."""
    if runner_kwargs is None:
        runner_kwargs = {}
    runner_kwargs["goal"] = goal
    base64_runner_kwargs = _serialize_to_b64(runner_kwargs)
    n = scheduler.cores
    if scheduler.executor_type == "ipyparallel":
        n -= 1

    opts = {
        "--n": n,
        "--url": database_manager.url,
        "--executor-type": scheduler.executor_type,
        "--log-interval": log_interval,
        "--save-interval": save_interval,
        "--serialized-runner-kwargs": base64_runner_kwargs,
    }
    if scheduler.executor_type == "loky":
        opts["--loky-start-method"] = scheduler.loky_start_method
    if save_dataframe:
        opts["--dataframe-format"] = dataframe_format
        opts["--save-dataframe"] = None
    return opts


def _get_executor(
    executor_type: EXECUTOR_TYPES,
    profile: str | None,
    n: int,
    loky_start_method: LOKY_START_METHODS,
) -> Any:
    if executor_type == "mpi4py":
        from mpi4py import MPI
        from mpi4py.futures import MPIPoolExecutor

        MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads)
        return MPIPoolExecutor()
    if executor_type == "ipyparallel":
        from adaptive_scheduler.utils import connect_to_ipyparallel

        assert profile is not None
        return connect_to_ipyparallel(profile=profile, n=n)
    if executor_type == "dask-mpi":
        from dask_mpi import initialize

        initialize()
        from distributed import Client

        return Client()
    if executor_type == "process-pool":
        import loky

        loky.backend.context.set_start_method(loky_start_method)
        return loky.get_reusable_executor(max_workers=n)
    msg = f"Unknown executor_type: {executor_type}"
    raise ValueError(msg)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store", type=str, default=None)
    parser.add_argument("--n", action="store", dest="n", type=int)
    parser.add_argument("--log-fname", action="store", type=str)
    parser.add_argument("--job-id", action="store", type=str)
    parser.add_argument("--name", action="store", dest="name", type=str, required=True)
    parser.add_argument("--url", action="store", type=str, required=True)

    parser.add_argument("--save-dataframe", action="store_true", default=False)
    parser.add_argument(
        "--dataframe-format",
        action="store",
        type=str,
        choices=get_args(_DATAFRAME_FORMATS),
    )
    parser.add_argument(
        "--executor-type",
        action="store",
        type=str,
        default="process-pool",
    )
    parser.add_argument(
        "--loky-start-method",
        action="store",
        type=str,
        default="loky",
        choices=get_args(LOKY_START_METHODS),
    )
    parser.add_argument("--log-interval", action="store", type=float, default=60)
    parser.add_argument(
        "--save-interval",
        action="store",
        type=float,
        default=120,
    )
    parser.add_argument("--serialized-runner-kwargs", action="store", type=str)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    client_support.log.info("parsed args", **vars(args))

    # ask the database for a learner that we can run which we log in `args.log_fname`
    learner, fname = client_support.get_learner(
        args.url,
        args.log_fname,
        args.job_id,
        args.name,
    )

    # load the data
    with suppress(Exception):
        learner.load(fname)
    npoints_start = learner.npoints

    executor = _get_executor(
        args.executor_type,
        args.profile,
        args.n,
        args.loky_start_method,
    )

    runner_kwargs = _deserialize_from_b64(args.serialized_runner_kwargs)

    runner_kwargs.setdefault("shutdown_executor", True)
    runner = adaptive.Runner(learner, executor=executor, **runner_kwargs)

    # periodically save the data (in case the job dies)
    runner.start_periodic_saving({"fname": fname}, interval=args.save_interval)

    if args.save_dataframe:
        from adaptive_scheduler.utils import save_dataframe

        save_method = save_dataframe(fname, format=args.dataframe_format)
        runner.start_periodic_saving(interval=args.save_interval, method=save_method)

    # log progress info in the job output script, optional
    _log_task = client_support.log_info(runner, interval=args.log_interval)

    # block until runner goal reached
    runner.ioloop.run_until_complete(runner.task)

    # save once more after the runner is done
    learner.save(fname)

    if args.save_dataframe:
        save_method(learner)

    # log once more after the runner is done
    client_support.log_now(runner, npoints_start)

    # tell the database that this learner has reached its goal
    client_support.tell_done(args.url, fname)


if __name__ == "__main__":
    main()
