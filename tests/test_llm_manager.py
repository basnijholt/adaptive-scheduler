"""Tests for the LLMManager."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import aiofiles
import ipywidgets as ipyw
import pytest
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from adaptive_scheduler._server_support.llm_manager import LLMManager
from adaptive_scheduler._server_support.run_manager import RunManager
from adaptive_scheduler.widgets import chat_widget, info

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.mark.asyncio
async def test_diagnose_failed_job(llm_manager: LLMManager) -> None:
    """Test that the diagnose_failed_job method returns a diagnosis."""
    llm_manager.agent_executor.ainvoke = AsyncMock(
        return_value={"messages": [AIMessage(content="diagnosis")]},
    )
    job_id = "test_job"
    llm_manager.db_manager.failed = [
        {"job_id": job_id, "job_name": "test_job_name", "output_logs": []},
    ]
    with (
        patch.object(llm_manager.db_manager, "as_dicts", return_value=[]),
        patch("adaptive_scheduler._server_support.llm_manager.aiofiles.open") as mock_open,
        patch.object(llm_manager, "_get_log_file_paths", return_value=["some/path/to/log.txt"]),
    ):
        mock_open.return_value.__aenter__.return_value.read.return_value = (
            "This is a log file with an error."
        )
        diagnosis = await llm_manager.diagnose_failed_job(job_id)
    assert diagnosis == "diagnosis"


@pytest.mark.asyncio
async def test_chat(llm_manager: LLMManager) -> None:
    """Test that the chat method returns a response."""
    llm_manager.agent_executor.ainvoke = AsyncMock(
        return_value={"messages": [AIMessage(content="response")]},
    )
    message = "Hello, world!"
    response = await llm_manager.chat(message)
    assert response == "response"


@pytest.fixture
def llm_manager() -> LLMManager:
    """An LLMManager instance with a mocked ChatOpenAI."""
    with patch(
        "adaptive_scheduler._server_support.llm_manager.ChatOpenAI",
    ) as mock_chat_openai:
        mock_chat_openai.spec = ChatOpenAI
        llm_manager = LLMManager(db_manager=MagicMock())
        llm_manager.agent_executor = MagicMock()
        llm_manager.agent_executor.ainvoke = AsyncMock()
        return llm_manager


@pytest.fixture
def run_manager() -> Generator[RunManager]:
    """A RunManager instance with a mocked scheduler."""
    with patch(
        "adaptive_scheduler._server_support.run_manager.LLMManager",
    ) as mock_llm_manager:
        scheduler = MagicMock()
        learners = [MagicMock()]
        fnames = ["test_fname"]
        rm = RunManager(
            scheduler,
            learners,
            fnames,
            llm_manager_kwargs={},
        )
        rm.llm_manager = mock_llm_manager.return_value
        yield rm
        rm.cancel()


def test_run_manager_with_llm(run_manager: RunManager) -> None:
    """Test that the RunManager initializes with an LLMManager."""
    assert isinstance(run_manager.llm_manager, MagicMock)


def test_run_manager_with_google_llm() -> None:
    """Test that the RunManager initializes with a Google LLM."""
    with patch(
        "adaptive_scheduler._server_support.llm_manager.ChatGoogleGenerativeAI",
    ) as mock_chat_google:
        mock_chat_google.spec = ChatGoogleGenerativeAI
        scheduler = MagicMock()
        learners = [MagicMock()]
        fnames = ["test_fname"]
        run_manager = RunManager(
            scheduler,
            learners,
            fnames,
            llm_manager_kwargs={"model_provider": "google"},
        )
        assert isinstance(run_manager.llm_manager, LLMManager)
        mock_chat_google.assert_called_once()
        run_manager.cancel()


@pytest.mark.asyncio
async def test_llm_manager_cache(llm_manager: LLMManager) -> None:
    """Test that the LLMManager caches diagnoses."""
    job_id = "test_job"
    llm_manager.db_manager.failed = [
        {"job_id": job_id, "job_name": "test_job_name", "output_logs": []},
    ]
    with patch.object(llm_manager.db_manager, "as_dicts", return_value=[]):
        llm_manager.agent_executor.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="diagnosis")]},
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
        llm_manager.agent_executor.ainvoke.assert_called_once()


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
async def test_read_log_files(llm_manager: LLMManager, tmp_path: Path) -> None:
    """Test that the _read_log_files method reads a file asynchronously."""
    log_content = "This is a test log file."
    log_file = tmp_path / "job_test.log"
    log_file.write_text(log_content)

    read_content = await llm_manager._read_log_files([log_file])
    assert read_content == log_content


@pytest.mark.asyncio
async def test_read_log_files_async(llm_manager: LLMManager, tmp_path: Path) -> None:
    """Test that the _read_log_files method reads a file asynchronously."""
    log_content = "This is a test log file."
    log_file = tmp_path / "job_test.log"
    async with aiofiles.open(log_file, "w") as f:
        await f.write(log_content)

    read_content = await llm_manager._read_log_files([log_file])
    assert read_content == log_content


@pytest.mark.asyncio
async def test_read_log_files_not_found(llm_manager: LLMManager) -> None:
    """Test that the _read_log_files method handles a missing file."""
    log_file = "non_existent_file.log"
    read_content = await llm_manager._read_log_files([Path(log_file)])
    assert "Log file not found" in read_content


def test_chat_widget_refresh_button(run_manager: RunManager) -> None:
    """Test that the chat widget's refresh button updates the failed jobs list."""
    run_manager.database_manager.failed = []
    widget = chat_widget(run_manager)
    (
        _,  # title
        refresh_button,
        failed_job_dropdown,
        yolo_checkbox,
        chat_history,
        text_input,
    ) = widget.children
    assert failed_job_dropdown.options == ()

    run_manager.database_manager.failed.append({"job_id": "failed_job_1"})
    refresh_button.click()
    assert failed_job_dropdown.options == ("failed_job_1",)


def test_info_widget_without_llm() -> None:
    """Test that the info widget does not show the chat button if with_llm=False."""
    scheduler = MagicMock()
    learners = [MagicMock()]
    fnames = ["test_fname"]
    run_manager = RunManager(
        scheduler,
        learners,
        fnames,
    )
    widget = info(run_manager, display_widget=False)
    buttons = widget.children[0].children[1].children
    assert not any("chat" in b.description.lower() for b in buttons if isinstance(b, ipyw.Button))
    run_manager.cancel()


@pytest.mark.asyncio
async def test_llm_manager_list_response_handling(llm_manager: LLMManager) -> None:
    """Test that list responses are properly formatted."""
    # Test with list response
    list_response = [
        "The job log indicates a NameError in my_code.py on line 4.",
        "python\nimport numpy as np\ndef h(x, offset=0, width=0.1):\n return x + width**2",
        "Reason for the fix: The variable was misspelled.",
    ]

    llm_manager.agent_executor.ainvoke = AsyncMock(
        return_value={"messages": [AIMessage(content=list_response)]},
    )

    result = await llm_manager.chat("Test message")

    # Should be joined with newlines
    expected = "\n".join(list_response)
    assert result == expected
    assert isinstance(result, str)
    assert "The job log indicates" in result
    assert "python\nimport numpy" in result
    assert "Reason for the fix" in result


@pytest.mark.asyncio
async def test_llm_manager_string_response_handling(llm_manager: LLMManager) -> None:
    """Test that string responses are unchanged."""
    # Test with string response
    string_response = "This is a normal string response"

    llm_manager.agent_executor.ainvoke = AsyncMock(
        return_value={"messages": [AIMessage(content=string_response)]},
    )

    result = await llm_manager.chat("Test message")

    # Should be unchanged
    assert result == string_response
    assert isinstance(result, str)


def test_llm_manager_yolo_mode() -> None:
    """Test that YOLO mode skips approval checks."""
    from unittest.mock import patch

    from adaptive_scheduler._server_support.llm_manager import LLMManager

    # Create a real LLMManager instance in YOLO mode
    db_manager = MagicMock()

    with patch("adaptive_scheduler._server_support.llm_manager.ChatOpenAI"):
        yolo_manager = LLMManager(db_manager=db_manager, yolo=True)
        non_yolo_manager = LLMManager(db_manager=db_manager, yolo=False)

        # Both should have the same tools (no human_approval tool anymore)
        yolo_tools = [tool.name for tool in yolo_manager.toolkit.get_tools()]
        non_yolo_tools = [tool.name for tool in non_yolo_manager.toolkit.get_tools()]

        # Should only have file management tools, no human_approval
        expected_tools = ["read_file", "write_file", "list_directory", "move_file"]
        assert yolo_tools == expected_tools
        assert non_yolo_tools == expected_tools

        # The difference is in the yolo flag
        assert yolo_manager.yolo is True
        assert non_yolo_manager.yolo is False
