"""Tests for conftest module."""

from pathlib import Path

import pytest
import zmq.asyncio
from adaptive import Learner1D

from adaptive_scheduler.server_support import (
    DatabaseManager,
    JobManager,
    get_allowed_url,
)

from .helpers import MockScheduler


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
