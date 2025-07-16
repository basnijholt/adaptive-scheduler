"""Tests for the LLMManager."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adaptive_scheduler._server_support.job_manager import JobManager
from adaptive_scheduler._server_support.llm_manager import LLMManager
from adaptive_scheduler._server_support.run_manager import RunManager

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_diagnose_failed_job() -> None:
    """Test that the diagnose_failed_job method returns a diagnosis."""
    llm_manager = LLMManager()
    job_id = "test_job"
    with patch("adaptive_scheduler._server_support.llm_manager.aiofiles.open") as mock_open:
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
def llm_manager() -> LLMManager:
    """An LLMManager instance."""
    return LLMManager()


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
    with (
        patch.object(
            llm_manager,
            "_read_log_file",
            return_value="log content",
        ),
        patch.object(
            llm_manager,
            "_simulate_llm_call",
            return_value="diagnosis",
        ) as mock_llm_call,
    ):
        await llm_manager.diagnose_failed_job(job_id)
        await llm_manager.diagnose_failed_job(job_id)
        mock_llm_call.assert_called_once()


@pytest.mark.asyncio
async def test_diagnose_failed_job_file_not_found() -> None:
    """Test that the diagnose_failed_job method handles a missing log file."""
    llm_manager = LLMManager()
    job_id = "test_job"
    with patch(
        "adaptive_scheduler._server_support.llm_manager.aiofiles.open",
    ) as mock_open:
        mock_open.side_effect = FileNotFoundError
        diagnosis = await llm_manager.diagnose_failed_job(job_id)
    assert "log file not found" in diagnosis.lower()


@pytest.mark.asyncio
async def test_chat_history(llm_manager: LLMManager) -> None:
    """Test that the chat history is maintained."""
    await llm_manager.chat("Hello")
    await llm_manager.chat("How are you?")
    assert len(llm_manager._chat_history) == 4


@pytest.mark.asyncio
async def test_read_log_file(tmp_path: Path) -> None:
    """Test that the _read_log_file method reads a file asynchronously."""
    llm_manager = LLMManager()
    log_content = "This is a test log file."
    log_file = tmp_path / "job_test.log"
    log_file.write_text(log_content)

    read_content = await llm_manager._read_log_file(str(log_file))
    assert read_content == log_content


@pytest.mark.asyncio
async def test_chat_widget_callbacks(run_manager: RunManager) -> None:
    """Test the async callbacks of the chat_widget."""
    from adaptive_scheduler.widgets import chat_widget

    # Mock the chat_widget dependencies
    with (
        patch("ipywidgets.Text") as mock_text,
        patch("ipywidgets.Textarea") as mock_textarea,
        patch("ipywidgets.Dropdown") as mock_dropdown,
        patch("ipywidgets.VBox"),
        patch("adaptive_scheduler.widgets._add_title"),
    ):
        # Create instances of the mocked widgets
        text_input = mock_text.return_value
        chat_history = mock_textarea.return_value
        failed_job_dropdown = mock_dropdown.return_value

        # Call the widget function to set up the callbacks
        chat_widget(run_manager)

        # --- Test on_submit ---
        # Get the on_submit wrapper from the mock
        on_submit_wrapper = text_input.on_submit.call_args[0][0]

        # Get a reference to the value mock before it's changed
        chat_history_value_mock = chat_history.value
        chat_history_value_mock.__iadd__.return_value = chat_history_value_mock

        # Mock the chat method
        with patch.object(
            run_manager.llm_manager,
            "chat",
            return_value="Test response",
        ) as mock_chat:
            # Simulate the submission
            text_input.value = "Test message"
            on_submit_wrapper(text_input)
            await asyncio.sleep(0)  # allow the task to run

            # Assertions
            mock_chat.assert_called_once_with("Test message")
            iadd_calls = chat_history_value_mock.__iadd__.call_args_list
            assert "You: Test message" in iadd_calls[0].args[0]
            assert "LLM: Test response" in iadd_calls[1].args[0]

        # --- Test on_failed_job_change ---
        # Get the on_failed_job_change wrapper from the mock
        on_failed_job_change_wrapper = failed_job_dropdown.observe.call_args[0][0]

        # Mock the diagnose_failed_job method
        with patch.object(
            run_manager.llm_manager,
            "diagnose_failed_job",
            return_value="Test diagnosis",
        ) as mock_diagnose:
            # Simulate the dropdown change
            change = {"new": "test_job_id"}
            on_failed_job_change_wrapper(change)
            await asyncio.sleep(0)  # allow the task to run

            # Assertions
            mock_diagnose.assert_called_once_with("test_job_id")
            assert "Diagnosis for job test_job_id" in chat_history.value
            assert "Test diagnosis" in chat_history.value
