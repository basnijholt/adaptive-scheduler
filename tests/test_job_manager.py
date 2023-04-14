import asyncio
import logging
from unittest.mock import MagicMock

import pytest

from adaptive_scheduler.server_support import JobManager, MaxRestartsReached


@pytest.mark.asyncio
async def test_job_manager_init(job_manager: JobManager) -> None:
    job_manager.database_manager.start()
    job_manager.start()
    assert job_manager.task is not None


@pytest.mark.asyncio
async def test_job_manager_queued(job_manager: JobManager):
    job_manager.scheduler.update_queue("job1", "running")
    job_manager.scheduler.update_queue("job2", "running")
    assert job_manager._queued(job_manager.scheduler.queue()) == {"job1", "job2"}


@pytest.mark.asyncio
async def test_job_manager_manage_max_restarts_reached(job_manager: JobManager) -> None:
    job_manager.n_started = 105
    # Should fail after n_started > n_learners * max_fails_per_job
    # Which is `105 > 2 * 50 = True`
    # job_manager.database_manager.n_done = MagicMock(return_value=0)
    job_manager.scheduler.queue_info = {}
    job_manager.database_manager.start()
    job_manager.start()
    await asyncio.sleep(0.1)
    with pytest.raises(
        MaxRestartsReached,
        match="Too many jobs failed, your Python code probably has a bug",
    ):
        job_manager.task.result()


@pytest.mark.asyncio
async def test_job_manager_manage_start_jobs(job_manager: JobManager) -> None:
    job_manager.database_manager.n_done = MagicMock(return_value=0)
    job_manager.scheduler.queue_info = {}
    job_manager.max_simultaneous_jobs = 2
    job_manager.database_manager.start()
    job_manager.start()
    await asyncio.sleep(0.1)
    assert set(job_manager.scheduler._started_jobs) == {"job1", "job2"}


@pytest.mark.asyncio
async def test_job_manager_manage_start_max_simultaneous_jobs(
    job_manager: JobManager,
) -> None:
    job_manager.database_manager.n_done = MagicMock(return_value=0)
    job_manager.scheduler.queue_info = {}
    job_manager.interval = 0.1
    job_manager.max_simultaneous_jobs = 1
    job_manager.database_manager.start()
    job_manager.start()
    await asyncio.sleep(0.15)
    assert len(job_manager.scheduler._started_jobs) == 1


@pytest.mark.asyncio
async def test_job_manager_manage_cancelled_error(
    job_manager: JobManager, caplog
) -> None:
    caplog.set_level(logging.INFO)
    job_manager.database_manager.start()
    job_manager.start()

    timeout = 0.1
    try:
        await asyncio.wait_for(job_manager.task, timeout=timeout)
    except asyncio.TimeoutError:
        job_manager.task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await job_manager.task

    assert "task was cancelled because of a CancelledError" in caplog.text


@pytest.mark.asyncio
async def test_job_manager_manage_n_done_equal_job_names(
    job_manager: JobManager,
) -> None:
    job_manager.database_manager.n_done = MagicMock(
        return_value=len(job_manager.job_names)
    )
    job_manager.database_manager.start()
    job_manager.start()
    await asyncio.sleep(0.1)
    assert job_manager.task.done() and job_manager.task.result() is None


@pytest.mark.asyncio
async def test_job_manager_manage_generic_exception(
    job_manager: JobManager, caplog
) -> None:
    def raise_exception(*args, **kwargs):  # noqa: ANN003, ANN002
        raise ValueError("Test exception")

    caplog.set_level(logging.ERROR)
    job_manager.scheduler.start_job = MagicMock(side_effect=raise_exception)
    job_manager.database_manager.start()
    job_manager.start()
    await asyncio.sleep(0.15)
    assert "got exception when starting a job" in caplog.text
    assert "Test exception" in caplog.text
