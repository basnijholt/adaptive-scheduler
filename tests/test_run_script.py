import os
from unittest.mock import patch

import pytest

from adaptive_scheduler.server_support import _make_default_run_script


@pytest.fixture()
def remove_run_script() -> None:
    yield
    run_script_fname = "run_learner.py"
    if os.path.exists(run_script_fname):
        os.remove(run_script_fname)


@pytest.mark.parametrize(
    ("executor_type", "expected_string"),
    [
        ("mpi4py", "mpi4py"),
        ("ipyparallel", "ipyparallel"),
        ("dask-mpi", "from distributed import Client"),
        ("process-pool", "loky"),
    ],
)
def test_make_default_run_script(executor_type, expected_string, remove_run_script):
    url = "http://localhost:1234"
    save_interval = 10
    log_interval = 5

    def goal(learner):
        return learner.npoints >= 10

    runner_kwargs = {"max_npoints": 100}
    with patch(
        "adaptive_scheduler.server_support._is_dask_mpi_installed",
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
            "adaptive_scheduler.server_support._is_dask_mpi_installed",
            return_value=False,
        ), pytest.raises(
            Exception,
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

    run_script_fname = "run_learner.py"
    assert os.path.exists(run_script_fname)

    with open(run_script_fname, encoding="utf-8") as f:
        content = f.read()

    assert expected_string in content
    assert url in content


def test_make_default_run_script_invalid_executor_type(remove_run_script):
    url = "http://localhost:1234"
    save_interval = 10
    log_interval = 5

    def goal(learner):
        return learner.npoints >= 10

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
