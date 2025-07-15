from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adaptive_scheduler._server_support.job_manager import JobManager
from adaptive_scheduler._server_support.llm_manager import LLMManager
from adaptive_scheduler._server_support.run_manager import RunManager


@pytest.mark.asyncio
async def test_diagnose_failed_job() -> None:
    """Test that the diagnose_failed_job method returns a diagnosis."""
    llm_manager = LLMManager()
    job_id = "test_job"
    with patch(
        "adaptive_scheduler._server_support.llm_manager.aiofiles.open",
    ) as mock_open:
        mock_open.return_value.__aenter__.return_value.read.return_value = (
            "This is a log file with an error."
        )
        diagnosis = await llm_manager.diagnose_failed_job(job_id)
    assert isinstance(diagnosis, str)


@pytest.mark.asyncio
async def test_chat() -> None:
    """Test that the chat method returns a response."""
    llm_manager = LLMManager()
    message = "Hello, world!"
    response = await llm_manager.chat(message)
    assert isinstance(response, str)


@pytest.fixture
def run_manager() -> RunManager:
    """A RunManager instance with a mocked scheduler."""
    scheduler = MagicMock()
    learners = [MagicMock()]
    fnames = ["test_fname"]
    return RunManager(
        scheduler,
        learners,
        fnames,
        with_llm=True,
    )


def test_run_manager_with_llm(run_manager: RunManager) -> None:
    """Test that the RunManager initializes with an LLMManager."""
    assert isinstance(run_manager.llm_manager, LLMManager)


@pytest.mark.asyncio
@patch("asyncio.to_thread")
async def test_job_manager_diagnoses_failed_job(
    mock_to_thread: MagicMock,
    run_manager: RunManager,
) -> None:
    """Test that the JobManager diagnoses a failed job."""
    # Mock the queue to return an empty dict
    mock_to_thread.return_value = {}

    # Add a failed job to the database
    run_manager.database_manager.failed.append(
        {"job_id": "test_job", "is_done": False},
    )

    # Create a JobManager with the RunManager's components
    job_manager = JobManager(
        ["test_job_name"],
        run_manager.database_manager,
        run_manager.scheduler,
        llm_manager=run_manager.llm_manager,
    )

    # Run the _update_database_and_get_not_queued method
    await job_manager._update_database_and_get_not_queued()

    # Check that a diagnosis was cached
    assert "test_job" in run_manager.llm_manager._diagnoses_cache
