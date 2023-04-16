from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import cloudpickle
import jinja2

if TYPE_CHECKING:
    import adaptive

    from adaptive_scheduler.utils import _DATAFRAME_FORMATS


def _is_dask_mpi_installed() -> bool:  # pragma: no cover
    return find_spec("dask_mpi") is not None


def _make_default_run_script(
    url: str,
    save_interval: int | float,
    log_interval: int | float,
    *,
    goal: Callable[[adaptive.BaseLearner], bool] | None = None,
    runner_kwargs: dict[str, Any] | None = None,
    run_script_fname: str | Path = "run_learner.py",
    executor_type: str = "mpi4py",
    loky_start_method: Literal[
        "loky",
        "loky_int_main",
        "spawn",
        "fork",
        "forkserver",
    ] = "loky",
    save_dataframe: bool = False,
    dataframe_format: _DATAFRAME_FORMATS = "parquet",
) -> None:
    default_runner_kwargs = {"shutdown_executor": True}
    runner_kwargs = dict(default_runner_kwargs, goal=goal, **(runner_kwargs or {}))
    serialized_runner_kwargs = cloudpickle.dumps(runner_kwargs)

    if executor_type not in ("mpi4py", "ipyparallel", "dask-mpi", "process-pool"):
        msg = "Use 'ipyparallel', 'dask-mpi', 'mpi4py' or 'process-pool'."
        raise NotImplementedError(msg)

    if executor_type == "dask-mpi" and not _is_dask_mpi_installed():
        msg = "You need to have 'dask-mpi' installed to use `executor_type='dask-mpi'`."
        raise ModuleNotFoundError(msg)
    run_script_template = Path(__file__).parent / "run_script.py.j2"
    with run_script_template.open(encoding="utf-8") as f:
        empty = "".join(f.readlines())
    run_script_fname = Path(run_script_fname)
    template = jinja2.Template(empty).render(
        run_script_fname=run_script_fname,
        executor_type=executor_type,
        url=url,
        serialized_runner_kwargs=serialized_runner_kwargs,
        save_interval=save_interval,
        log_interval=log_interval,
        loky_start_method=loky_start_method,
        save_dataframe=save_dataframe,
        dataframe_format=dataframe_format,
    )

    with run_script_fname.open("w", encoding="utf-8") as f:
        f.write(template)
