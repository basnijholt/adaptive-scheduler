from __future__ import annotations

import os

if os.environ.get("EXECUTOR_TYPE") == "dask-mpi":
    from dask_mpi import initialize

    initialize()

import argparse
import os
from contextlib import suppress
from typing import TYPE_CHECKING, Any, get_args

import adaptive
import cloudpickle

from adaptive_scheduler import client_support
from adaptive_scheduler.utils import (
    _DATAFRAME_FORMATS,
    LOKY_START_METHODS,
    WrappedFunction,
    _deserialize_from_b64,
)

if TYPE_CHECKING:
    from adaptive_scheduler.utils import EXECUTOR_TYPES


if os.environ.get("EXECUTOR_TYPE") == "mpi4py":
    from mpi4py import MPI

    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads)


def _get_executor(
    executor_type: EXECUTOR_TYPES,
    profile: str | None,
    n: int,
    loky_start_method: LOKY_START_METHODS | None,
) -> Any:
    if executor_type == "mpi4py":
        from mpi4py.futures import MPIPoolExecutor

        return MPIPoolExecutor()
    if executor_type == "ipyparallel":
        from adaptive_scheduler.utils import connect_to_ipyparallel

        assert profile is not None
        return connect_to_ipyparallel(profile=profile, n=n)
    if executor_type == "dask-mpi":
        from distributed import Client

        return Client()
    if executor_type == "loky":
        import loky

        assert loky_start_method is not None
        loky.backend.context.set_start_method(loky_start_method)
        return loky.get_reusable_executor(max_workers=n)
    if executor_type == "process-pool":
        from concurrent.futures import ProcessPoolExecutor

        return ProcessPoolExecutor(max_workers=n)
    if executor_type == "sequential":
        from adaptive.runner import SequentialExecutor

        return SequentialExecutor()
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
    """The main function that is called by the launcher script."""
    args = _parse_args()
    client_support.add_log_file_handler(args.log_fname)
    client_support.log.info("parsed args", **vars(args))
    client_support.args_to_env(args)

    # ask the server for a fname and learner
    learner, fname, initializer = client_support.get_learner(
        args.url,
        args.log_fname,
        args.job_id,
        args.name,
    )

    if initializer is not None:
        initializer()

    if args.executor_type == "process-pool":
        learner.function = WrappedFunction(learner.function)

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
