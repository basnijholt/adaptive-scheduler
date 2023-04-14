"""Tests for the run script functionality."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from adaptive_scheduler.server_support import _make_default_run_script

RUN_SCRIPT_FNAME = "run_learner.py"
MIN_POINTS = 10


@pytest.fixture()
def _remove_run_script() -> None:
    """Remove the run script if it exists."""
    yield
    if Path(RUN_SCRIPT_FNAME).exists():
        Path(RUN_SCRIPT_FNAME).unlink()


@pytest.mark.parametrize(
    ("executor_type", "expected_string"),
    [
        ("mpi4py", "mpi4py"),
        ("ipyparallel", "ipyparallel"),
        ("dask-mpi", "from distributed import Client"),
        ("process-pool", "loky"),
    ],
)
@pytest.mark.usefixtures("_remove_run_script")
def test_make_default_run_script(
    executor_type: str,
    expected_string: str,
) -> None:
    """Test that the run script is created correctly."""
    url = "http://localhost:1234"
    save_interval = 10
    log_interval = 5

    def goal(learner: Any) -> bool:
        return learner.npoints >= MIN_POINTS

    runner_kwargs = {"max_npoints": 100}
    with patch(
        "adaptive_scheduler._server_support.run_script._is_dask_mpi_installed",
        return_value=True,
    ):
        _make_default_run_script(
            url,
            save_interval,
            log_interval,
            goal=goal,
            runner_kwargs=runner_kwargs,
            executor_type=executor_type,
        )

    if executor_type == "dask-mpi":
        with patch(
            "adaptive_scheduler._server_support.run_script._is_dask_mpi_installed",
            return_value=False,
        ), pytest.raises(
            ModuleNotFoundError,
            match="You need to have 'dask-mpi' installed",
        ):
            _make_default_run_script(
                url,
                save_interval,
                log_interval,
                goal=goal,
                runner_kwargs=runner_kwargs,
                executor_type=executor_type,
            )

    assert Path(RUN_SCRIPT_FNAME).exists()

    with Path(RUN_SCRIPT_FNAME).open(encoding="utf-8") as f:
        content = f.read()

    assert expected_string in content
    assert url in content


@pytest.mark.usefixtures("_remove_run_script")
def test_make_default_run_script_invalid_executor_type() -> None:
    """Test that an error is raised when an invalid executor type is given."""
    url = "http://localhost:1234"
    save_interval = 10
    log_interval = 5

    def goal(learner: Any) -> bool:
        return learner.npoints >= MIN_POINTS

    runner_kwargs = {"max_npoints": 100}

    with pytest.raises(NotImplementedError):
        _make_default_run_script(
            url,
            save_interval,
            log_interval,
            goal=goal,
            runner_kwargs=runner_kwargs,
            executor_type="invalid_executor",
        )
