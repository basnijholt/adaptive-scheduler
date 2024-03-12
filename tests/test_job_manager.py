"""Test module for JobManager."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from adaptive_scheduler.server_support import JobManager, MaxRestartsReachedError


@pytest.mark.asyncio()
async def test_job_manager_init(job_manager: JobManager) -> None:
    """Test the initialization of JobManager."""
    job_manager.database_manager.start()
    job_manager.start()
    assert job_manager.task is not None


@pytest.mark.asyncio()
async def test_job_manager_queued(job_manager: JobManager) -> None:
    """Test the _queued method of JobManager."""
    job_manager.scheduler.start_job("job1")
    job_manager.scheduler.start_job("job2")
    job_manager.scheduler.update_queue("job1", "running")  # type: ignore[attr-defined]
    job_manager.scheduler.update_queue("job2", "running")  # type: ignore[attr-defined]
    assert job_manager._queued(job_manager.scheduler.queue()) == {"job1", "job2"}


@pytest.mark.asyncio()
async def test_job_manager_manage_max_restarts_reached(job_manager: JobManager) -> None:
    """Test the JobManager when the maximum restarts are reached."""
    job_manager.n_started = 105
    # Should fail after n_started > n_learners * max_fails_per_job
    # Which is `105 > 2 * 50 = True`
    job_manager.scheduler._queue_info = {}  # type: ignore[attr-defined]
    job_manager.database_manager.start()
    job_manager.start()
    assert job_manager.task is not None
    await asyncio.sleep(0.1)
    with pytest.raises(
        MaxRestartsReachedError,
        match="Too many jobs failed, your Python code probably has a bug",
    ):
        job_manager.task.result()


@pytest.mark.asyncio()
async def test_job_manager_manage_start_jobs(job_manager: JobManager) -> None:
    """Test the JobManager when managing the start of jobs."""
    job_manager.database_manager.n_done = MagicMock(return_value=0)  # type: ignore[method-assign]
    job_manager.scheduler._queue_info = {}  # type: ignore[attr-defined]
    job_manager.max_simultaneous_jobs = 2
    job_manager.database_manager.start()
    job_manager.start()
    await asyncio.sleep(0.1)
    assert set(job_manager.scheduler._started_jobs) == {"job1", "job2"}  # type: ignore[attr-defined]


@pytest.mark.asyncio()
async def test_job_manager_manage_start_max_simultaneous_jobs(
    job_manager: JobManager,
) -> None:
    """Test the JobManager when managing the maximum simultaneous jobs."""
    job_manager.database_manager.n_done = MagicMock(return_value=0)  # type: ignore[method-assign]
    job_manager.scheduler._queue_info = {}  # type: ignore[attr-defined]
    job_manager.interval = 0.1  # type: ignore[assignment]
    job_manager.max_simultaneous_jobs = 1
    job_manager.database_manager.start()
    job_manager.start()
    assert job_manager.task is not None
    await asyncio.sleep(0.15)
    assert len(job_manager.scheduler._started_jobs) == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio()
async def test_job_manager_manage_cancelled_error(
    job_manager: JobManager,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the JobManager when a CancelledError occurs."""
    caplog.set_level(logging.INFO)
    job_manager.database_manager.start()
    job_manager.start()
    assert job_manager.task is not None

    timeout = 0.1
    await asyncio.sleep(timeout)
    assert job_manager.database_manager.task is not None
    assert not job_manager.database_manager.task.done()
    assert not job_manager.task.done()
    job_manager.task.cancel()
    await asyncio.sleep(timeout)
    with pytest.raises(asyncio.CancelledError):
        await job_manager.task

    assert "task was cancelled because of a CancelledError" in caplog.text


@pytest.mark.asyncio()
async def test_job_manager_manage_n_done_equal_job_names(
    job_manager: JobManager,
) -> None:
    """Test the JobManager when n_done equals the number of job names."""
    job_manager.database_manager.n_done = MagicMock(  # type: ignore[method-assign]
        return_value=len(job_manager.job_names),
    )
    job_manager.database_manager.start()
    job_manager.start()
    assert job_manager.task is not None
    await asyncio.sleep(0.1)
    assert job_manager.task.done()
    assert job_manager.task.result() is None


@pytest.mark.asyncio()
async def test_job_manager_manage_generic_exception(
    job_manager: JobManager,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the JobManager when a generic exception occurs."""

    def raise_exception(*_args: Any, **_kwargs: Any) -> None:
        msg = "Test exception"
        raise ValueError(msg)

    caplog.set_level(logging.ERROR)
    job_manager.scheduler.start_job = MagicMock(side_effect=raise_exception)  # type: ignore[method-assign]
    job_manager.database_manager.start()
    job_manager.start()
    assert job_manager.task is not None
    await asyncio.sleep(0.15)
    assert "got exception when starting a job" in caplog.text
    assert "Test exception" in caplog.text
