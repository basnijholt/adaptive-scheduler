"""Tests for the LLMManager."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import ipywidgets as ipyw
import pytest
from langchain.schema import AIMessage, HumanMessage
from langchain.tools import BaseTool
from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from adaptive_scheduler._server_support.job_manager import JobManager
from adaptive_scheduler._server_support.llm_manager import ApprovalTool, LLMManager
from adaptive_scheduler._server_support.run_manager import RunManager
from adaptive_scheduler.widgets import info

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.mark.asyncio
async def test_diagnose_failed_job(llm_manager: LLMManager) -> None:
    """Test that the diagnose_failed_job method returns a diagnosis."""
    llm_manager.agent_executor.ainvoke = AsyncMock(return_value={"output": "diagnosis"})
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
    llm_manager.agent_executor.ainvoke = AsyncMock(return_value={"output": "response"})
    message = "Hello, world!"
    response = await llm_manager.chat(message)
    assert response == "response"


@pytest.fixture
def llm_manager() -> LLMManager:
    """An LLMManager instance with a mocked ChatOpenAI."""
    with (
        patch("adaptive_scheduler._server_support.llm_manager.AgentExecutor"),
        patch(
            "adaptive_scheduler._server_support.llm_manager.create_openai_functions_agent",
        ),
        patch(
            "adaptive_scheduler._server_support.llm_manager.ChatOpenAI",
        ) as mock_chat_openai,
    ):
        mock_chat_openai.spec = ChatOpenAI
        llm_manager = LLMManager(db_manager=MagicMock())
        llm_manager.agent_executor = AsyncMock()
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
    with (
        patch(
            "adaptive_scheduler._server_support.llm_manager.ChatGoogleGenerativeAI",
        ) as mock_chat_google,
        patch("adaptive_scheduler._server_support.llm_manager.AgentExecutor"),
        patch(
            "adaptive_scheduler._server_support.llm_manager.create_openai_functions_agent",
        ),
    ):
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
async def test_job_manager_diagnoses_failed_job_async(run_manager: RunManager) -> None:
    """Test that the JobManager diagnoses a failed job asynchronously."""
    run_manager.scheduler.queue.return_value = {}  # type: ignore[attr-defined]
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
    llm_manager.db_manager.failed = [
        {"job_id": job_id, "job_name": "test_job_name", "output_logs": []},
    ]
    with patch.object(llm_manager.db_manager, "as_dicts", return_value=[]):
        llm_manager.agent_executor.ainvoke = AsyncMock(return_value={"output": "diagnosis"})
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
async def test_chat_history(llm_manager: LLMManager) -> None:
    """Test that the chat history is maintained."""

    async def mock_ainvoke(inputs: dict[str, str]) -> dict[str, str]:
        llm_manager.memory.save_context(inputs, {"output": "response"})
        return {"output": "response"}

    llm_manager.agent_executor.ainvoke = AsyncMock(side_effect=mock_ainvoke)

    await llm_manager.chat("Hello")
    await llm_manager.chat("How are you?")

    history = llm_manager.memory.chat_memory.messages
    assert len(history) == 4
    assert isinstance(history[0], HumanMessage)
    assert history[0].content == "Hello"
    assert isinstance(history[1], AIMessage)
    assert history[1].content == "response"
    assert isinstance(history[2], HumanMessage)
    assert history[2].content == "How are you?"
    assert isinstance(history[3], AIMessage)
    assert history[3].content == "response"


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
    import aiofiles

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
    from adaptive_scheduler.widgets import chat_widget

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
async def test_approval_tool() -> None:
    """Test the approval mechanism of the ApprovalTool."""
    mock_tool = AsyncMock(spec=BaseTool)
    mock_tool.name = "mock_tool"
    mock_tool.description = "mock_tool_description"
    mock_tool.args_schema = None
    mock_tool._arun = AsyncMock(return_value="tool result")

    mock_llm_manager = MagicMock(spec=LLMManager)
    mock_llm_manager.ask_approval = MagicMock()
    mock_llm_manager.approval_queue = asyncio.Queue()

    tool = ApprovalTool(tool=mock_tool, llm_manager=mock_llm_manager)

    # Test with approval
    mock_llm_manager.yolo = False
    await mock_llm_manager.approval_queue.put("approve")
    result = await tool._arun("test input")
    assert result == "tool result"
    mock_llm_manager.ask_approval.assert_called_once()
    mock_tool._arun.assert_awaited_once_with("test input")

    # Test with denial
    mock_llm_manager.ask_approval.reset_mock()
    mock_tool._arun.reset_mock()
    await mock_llm_manager.approval_queue.put("deny")
    result = await tool._arun("test input")
    assert result == "Action cancelled by user."
    mock_llm_manager.ask_approval.assert_called_once()
    mock_tool._arun.assert_not_awaited()

    # Test with YOLO mode
    mock_llm_manager.ask_approval.reset_mock()
    mock_tool._arun.reset_mock()
    mock_llm_manager.yolo = True
    result = await tool._arun("test input")
    assert result == "tool result"
    mock_llm_manager.ask_approval.assert_not_called()
    mock_tool._arun.assert_awaited_once_with("test input")
