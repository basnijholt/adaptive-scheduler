"""Tests for the LLMManager."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import ipywidgets as ipyw
import pytest

from adaptive_scheduler._server_support.job_manager import JobManager
from adaptive_scheduler._server_support.llm_manager import LLMManager
from adaptive_scheduler._server_support.run_manager import RunManager
from adaptive_scheduler.widgets import info

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_diagnose_failed_job(llm_manager: LLMManager) -> None:
    """Test that the diagnose_failed_job method returns a diagnosis."""
    llm_manager.llm.agenerate = AsyncMock(
        return_value=MagicMock(
            generations=[[MagicMock(text="diagnosis")]],
        ),
    )
    job_id = "test_job"
    llm_manager.db_manager.failed = [{"job_id": job_id, "job_name": "test_job_name"}]
    llm_manager.db_manager.scheduler.job_script.return_value = "job script"
    llm_manager.db_manager.get.return_value = MagicMock(index=0)
    with (
        patch(
            "adaptive_scheduler._server_support.llm_manager.aiofiles.open",
        ) as mock_open,
        patch.object(
            llm_manager,
            "_get_log_file_paths",
            return_value=["some/path/to/log.txt"],
        ),
    ):
        mock_open.return_value.__aenter__.return_value.read.return_value = (
            "This is a log file with an error."
        )
        diagnosis = await llm_manager.diagnose_failed_job(job_id)
    assert diagnosis == "diagnosis"


@pytest.mark.asyncio
async def test_chat(llm_manager: LLMManager) -> None:
    """Test that the chat method returns a response."""
    llm_manager.llm.agenerate = AsyncMock(
        return_value=MagicMock(
            generations=[[MagicMock(text="response")]],
        ),
    )
    message = "Hello, world!"
    response = await llm_manager.chat(message)
    assert response == "response"


@pytest.fixture
@patch("adaptive_scheduler._server_support.llm_manager.ChatOpenAI")
def llm_manager(mock_chat_openai: MagicMock) -> LLMManager:  # noqa: ARG001
    """An LLMManager instance with a mocked ChatOpenAI."""
    return LLMManager(db_manager=MagicMock())


@pytest.fixture
@patch("adaptive_scheduler._server_support.llm_manager.ChatOpenAI")
def run_manager(mock_chat_openai: MagicMock) -> RunManager:  # noqa: ARG001
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


@patch("adaptive_scheduler._server_support.llm_manager.ChatGoogleGenerativeAI")
def test_run_manager_with_google_llm(mock_chat_google: MagicMock) -> None:
    """Test that the RunManager initializes with a Google LLM."""
    scheduler = MagicMock()
    learners = [MagicMock()]
    fnames = ["test_fname"]
    run_manager = RunManager(
        scheduler,
        learners,
        fnames,
        with_llm=True,
        llm_manager_kwargs={"model_provider": "google"},
    )
    assert isinstance(run_manager.llm_manager, LLMManager)
    mock_chat_google.assert_called_once()


@pytest.mark.asyncio
@patch("asyncio.to_thread")
async def test_job_manager_diagnoses_failed_job_async(
    mock_to_thread: MagicMock,
    run_manager: RunManager,
) -> None:
    """Test that the JobManager diagnoses a failed job asynchronously."""
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

    with patch.object(
        run_manager.llm_manager,
        "diagnose_failed_job",
        new_callable=AsyncMock,
    ) as mock_diagnose:
        mock_diagnose.return_value = "diagnosis"
        # Run the _update_database_and_get_not_queued method
        await job_manager._update_database_and_get_not_queued()

        # Check that diagnose_failed_job was called
        mock_diagnose.assert_awaited_once_with("test_job")


@pytest.mark.asyncio
async def test_llm_manager_cache(llm_manager: LLMManager) -> None:
    """Test that the LLMManager caches diagnoses."""
    job_id = "test_job"
    llm_manager.db_manager.failed = [{"job_id": job_id, "job_name": "test_job_name"}]
    llm_manager.db_manager.scheduler.job_script.return_value = "job script"
    llm_manager.db_manager.get.return_value = MagicMock(index=0)
    llm_manager.llm.agenerate = AsyncMock(
        return_value=MagicMock(
            generations=[[MagicMock(text="diagnosis")]],
        ),
    )
    with (
        patch.object(
            llm_manager,
            "_read_log_files",
            return_value="log content",
        ),
        patch.object(
            llm_manager,
            "_get_log_file_paths",
            return_value=["some/path/to/log.txt"],
        ),
    ):
        await llm_manager.diagnose_failed_job(job_id)
        await llm_manager.diagnose_failed_job(job_id)
        llm_manager.llm.agenerate.assert_called_once()


@pytest.mark.asyncio
async def test_diagnose_failed_job_file_not_found(llm_manager: LLMManager) -> None:
    """Test that the diagnose_failed_job method handles a missing log file."""
    job_id = "test_job"
    with patch.object(
        llm_manager,
        "_get_log_file_paths",
        return_value=[],
    ):
        diagnosis = await llm_manager.diagnose_failed_job(job_id)
    assert "Could not find log files" in diagnosis


@pytest.mark.asyncio
async def test_chat_history(llm_manager: LLMManager) -> None:
    """Test that the chat history is maintained."""
    llm_manager.llm.agenerate = AsyncMock(
        return_value=MagicMock(
            generations=[[MagicMock(text="response")]],
        ),
    )
    await llm_manager.chat("Hello")
    await llm_manager.chat("How are you?")
    assert len(llm_manager._chat_history) == 4


@pytest.mark.asyncio
async def test_read_log_files(llm_manager: LLMManager, tmp_path: Path) -> None:
    """Test that the _read_log_files method reads a file asynchronously."""
    log_content = "This is a test log file."
    log_file = tmp_path / "job_test.log"
    log_file.write_text(log_content)

    read_content = await llm_manager._read_log_files([str(log_file)])
    assert read_content == log_content


def test_chat_widget_refresh_button(run_manager: RunManager) -> None:
    """Test that the chat widget's refresh button updates the failed jobs list."""
    from adaptive_scheduler.widgets import chat_widget

    # Initially, there are no failed jobs
    run_manager.database_manager.failed = []
    widget = chat_widget(run_manager)
    refresh_button = widget.children[1]
    dropdown = widget.children[2]
    assert dropdown.options == ()

    # A job fails
    run_manager.database_manager.failed.append({"job_id": "failed_job_1"})

    # Click the refresh button
    refresh_button.click()

    # The dropdown should now contain the failed job
    assert dropdown.options == ("failed_job_1",)


def test_info_widget_without_llm() -> None:
    """Test that the info widget does not show the chat button if with_llm=False."""
    scheduler = MagicMock()
    learners = [MagicMock()]
    fnames = ["test_fname"]
    run_manager = RunManager(
        scheduler,
        learners,
        fnames,
        with_llm=False,
    )
    widget = info(run_manager, display_widget=False)
    buttons = widget.children[0].children[1].children
    assert not any("chat" in b.description.lower() for b in buttons if isinstance(b, ipyw.Button))
