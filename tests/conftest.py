"""Tests for conftest module."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import adaptive
import pytest

from adaptive_scheduler.server_support import (
    DatabaseManager,
    JobManager,
    get_allowed_url,
)

from .helpers import PARTITIONS, MockScheduler, get_socket

if TYPE_CHECKING:
    from collections.abc import Generator

    import zmq.asyncio


@pytest.fixture()
def mock_scheduler(tmp_path: Path) -> MockScheduler:
    """Fixture for creating a MockScheduler instance."""
    return MockScheduler(log_folder=str(tmp_path), cores=8)


@pytest.fixture()
def db_manager(
    mock_scheduler: MockScheduler,
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
    tmp_path: Path,
) -> DatabaseManager:
    """Fixture for creating a DatabaseManager instance."""
    url = get_allowed_url()
    db_fname = str(tmp_path / "test_db.json")
    return DatabaseManager(url, mock_scheduler, db_fname, learners, fnames)


def func(x: float) -> float:
    """A simple quadratic function."""
    return x**2


@pytest.fixture(
    params=[adaptive.Learner1D, adaptive.BalancingLearner, adaptive.SequenceLearner],
)
def learners(
    request: pytest.FixtureRequest,
) -> (
    list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner]
):
    """Fixture for creating a list of adaptive.Learner1D instances."""
    learner_class = request.param
    if learner_class is adaptive.Learner1D:
        learner1 = adaptive.Learner1D(func, bounds=(-1, 1))
        learner2 = adaptive.Learner1D(func, bounds=(-1, 1))
        return [learner1, learner2]
    if learner_class is adaptive.BalancingLearner:
        learner1 = adaptive.Learner1D(func, bounds=(-1, 1))
        learner2 = adaptive.Learner1D(func, bounds=(-1, 1))
        learner3 = adaptive.Learner1D(func, bounds=(-1, 1))
        learner4 = adaptive.Learner1D(func, bounds=(-1, 1))
        return [
            adaptive.BalancingLearner([learner1, learner2]),
            adaptive.BalancingLearner([learner3, learner4]),
        ]
    if learner_class is adaptive.SequenceLearner:
        learner1 = adaptive.SequenceLearner(func, sequence=list(range(200)))
        learner2 = adaptive.SequenceLearner(func, sequence=list(range(200)))
        return [learner1, learner2]
    msg = f"Learner type '{type(learner_class)}' not implemented"
    raise NotImplementedError(msg)


@pytest.fixture(params=[Path, str])
def fnames(
    request: pytest.FixtureRequest,
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    tmp_path: Path,
) -> list[Path] | list[str] | list[list[Path]] | list[list[str]]:
    """Fixture for creating a list of filenames for learners."""
    type_ = request.param
    if isinstance(learners[0], (adaptive.Learner1D, adaptive.SequenceLearner)):
        return [type_(tmp_path / f"learner{i}.pkl") for i, _ in enumerate(learners)]
    if isinstance(learners[0], adaptive.BalancingLearner):
        return [
            [
                type_(tmp_path / f"bal_learner{j}_{i}.json")
                for j, _ in enumerate(learner.learners)
            ]
            for i, learner in enumerate(learners)
        ]
    msg = f"Learner type '{type(learners[0])}' not implemented"
    raise NotImplementedError(msg)


@pytest.fixture()
def socket(db_manager: DatabaseManager) -> zmq.asyncio.Socket:
    """Fixture for creating a ZMQ socket."""
    with get_socket(db_manager) as socket:
        yield socket


@pytest.fixture()
def job_manager(
    db_manager: DatabaseManager,
    mock_scheduler: MockScheduler,
) -> JobManager:
    """Fixture for creating a JobManager instance."""
    job_names = ["job1", "job2"]
    return JobManager(job_names, db_manager, mock_scheduler, interval=0.05)


@pytest.fixture()
def _mock_slurm_partitions_output() -> Generator[None, None, None]:
    """Mock `slurm_partitions` function."""
    mock_output = "hb120v2-low\nhb60-high\nnc24-low*\nnd40v2-mpi\n"
    with patch("adaptive_scheduler._scheduler.slurm.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=mock_output.encode("utf-8"))
        yield


@pytest.fixture()
def _mock_slurm_partitions() -> Generator[None, None, None]:
    """Mock `slurm_partitions` function."""
    with patch(
        "adaptive_scheduler._scheduler.slurm.slurm_partitions",
    ) as slurm_partitions, patch(
        "adaptive_scheduler._server_support.slurm_run.slurm_partitions",
    ) as slurm_partitions_imported:
        slurm_partitions.return_value = PARTITIONS
        slurm_partitions_imported.return_value = PARTITIONS
        yield


@pytest.fixture()
def _mock_slurm_queue() -> Generator[None, None, None]:
    """Mock `SLURM.queue` function."""
    with patch(
        "adaptive_scheduler._scheduler.slurm.SLURM.queue",
    ) as queue:
        queue.return_value = {}
        yield
