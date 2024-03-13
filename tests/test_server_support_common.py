"""Tests for the common module of the server_support module."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from adaptive_scheduler._server_support.common import (
    _get_all_files,
    _ipython_profiles,
    cleanup_scheduler_files,
    get_allowed_url,
    periodically_clean_ipython_profiles,
)

from .helpers import MockScheduler, temporary_working_directory


@pytest.mark.asyncio()
async def test_periodically_clean_ipython_profiles(
    mock_scheduler: MockScheduler,
) -> None:
    """Test that the periodic cleaning of IPython profiles works."""
    task = periodically_clean_ipython_profiles(
        mock_scheduler,
        interval=0.1,  # type: ignore[arg-type]
    )
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
