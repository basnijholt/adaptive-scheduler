import asyncio
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from adaptive_scheduler._server_support.common import (
    MaxRestartsReachedError,
    _get_all_files,
    _ipython_profiles,
    cleanup_scheduler_files,
    get_allowed_url,
    periodically_clean_ipython_profiles,
)

from .helpers import MockScheduler


@contextmanager
def temporary_working_directory(path: Path) -> None:
    """Context manager for temporarily changing the working directory."""
    original_cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_cwd)


@pytest.mark.asyncio()
async def test_periodically_clean_ipython_profiles(
    mock_scheduler: MockScheduler,
) -> None:
    """Test that the periodic cleaning of IPython profiles works."""
    task = periodically_clean_ipython_profiles(mock_scheduler, interval=0.1)
    await asyncio.sleep(0.2)
    task.cancel()


def test_get_allowed_url() -> None:
    """Test that the allowed URL is a TCP URL."""
    url = get_allowed_url()
    assert url.startswith("tcp://")
    url2 = get_allowed_url()
    assert url != url2


def test_cleanup_scheduler_files(mock_scheduler: MockScheduler, tmp_path: Path) -> None:
    """Test that the scheduler files are cleaned up correctly."""
    job_names = ["job1", "job2"]
    for name in job_names:
        (tmp_path / f"{name}.mock").touch()
        (tmp_path / f"{name}-JOBID.out").touch()
        (tmp_path / f"{name}-JOBID.log").touch()

    assert len(list(tmp_path.glob("*"))) == 6
    move_to = tmp_path / "moved"
    with temporary_working_directory(tmp_path):
        cleanup_scheduler_files(job_names, mock_scheduler, move_to=move_to)
    assert len(list(tmp_path.glob("*"))) == 1
    assert len(list(move_to.glob("*"))) == 6


def test__get_all_files(mock_scheduler: MockScheduler, tmp_path: Path) -> None:
    """Test that all files are returned correctly."""
    job_names = ["job1", "job2"]
    for name in job_names:
        (tmp_path / f"{name}.mock").touch()
        (tmp_path / f"{name}-JOBID.out").touch()
        (tmp_path / f"{name}-JOBID.log").touch()
    with temporary_working_directory(tmp_path):
        all_files = _get_all_files(job_names, mock_scheduler)
        assert len(all_files) == 6


def test__ipython_profiles() -> None:
    """Test that the IPython profiles are returned correctly."""
    profiles = _ipython_profiles()
    assert isinstance(profiles, list)
    assert all(isinstance(p, Path) for p in profiles)


def test_MaxRestartsReachedError() -> None:
    """Test that the MaxRestartsReachedError is raised correctly."""
    with pytest.raises(MaxRestartsReachedError) as excinfo:
        raise MaxRestartsReachedError("Max restarts reached.")
    assert str(excinfo.value) == "Max restarts reached."
