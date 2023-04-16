"""Tests for conftest module."""
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import zmq.asyncio
from adaptive import Learner1D

from adaptive_scheduler.server_support import (
    DatabaseManager,
    JobManager,
    get_allowed_url,
)

from .helpers import PARTITIONS, MockScheduler

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture()
def mock_scheduler(tmp_path: Path) -> MockScheduler:
    """Fixture for creating a MockScheduler instance."""
    return MockScheduler(log_folder=str(tmp_path), cores=8)


@pytest.fixture()
def db_manager(
    mock_scheduler: MockScheduler,
    learners: list[Learner1D],
    fnames: list[str],
    tmp_path: Path,
) -> DatabaseManager:
    """Fixture for creating a DatabaseManager instance."""
    url = get_allowed_url()
    db_fname = str(tmp_path / "test_db.json")
    return DatabaseManager(url, mock_scheduler, db_fname, learners, fnames)


def func(x: float) -> float:
    """A simple quadratic function."""
    return x**2


@pytest.fixture()
def learners() -> list[Learner1D]:
    """Fixture for creating a list of Learner1D instances."""
    learner1 = Learner1D(func, bounds=(-1, 1))
    learner2 = Learner1D(func, bounds=(-1, 1))
    return [learner1, learner2]


@pytest.fixture()
def fnames(learners: list[Learner1D], tmp_path: Path) -> list[str]:
    """Fixture for creating a list of filenames for learners."""
    return [str(tmp_path / f"learner{i}.pkl") for i, _ in enumerate(learners)]


@pytest.fixture()
def socket(db_manager: DatabaseManager) -> zmq.asyncio.Socket:
    """Fixture for creating a ZMQ socket."""
    ctx = zmq.asyncio.Context.instance()
    socket = ctx.socket(zmq.REQ)
    socket.connect(db_manager.url)
    yield socket
    socket.close()


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
