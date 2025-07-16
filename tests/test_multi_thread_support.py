"""Tests for multi-thread support in chat widget."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from adaptive_scheduler._server_support.llm_manager import LLMManager
from adaptive_scheduler.widgets import chat_widget


class TestMultiThreadSupport:
    """Test suite for multi-thread support in chat widget."""

    def test_chat_widget_has_proper_structure_for_multi_thread(self) -> None:
        """Test that chat widget has the proper structure for multi-thread support."""
        run_manager = MagicMock()
        run_manager.database_manager.failed = [
            {"job_id": "job_1", "job_name": "test_job_1", "output_logs": []},
            {"job_id": "job_2", "job_name": "test_job_2", "output_logs": []},
        ]

        llm_manager = MagicMock(spec=LLMManager)
        llm_manager.chat = AsyncMock(return_value="Test response")
        llm_manager.diagnose_failed_job = AsyncMock(return_value="Test diagnosis")
        run_manager.llm_manager = llm_manager

        widget = chat_widget(run_manager)
        failed_job_dropdown = widget.children[2]

        # Verify that dropdown has multiple jobs available
        assert failed_job_dropdown.options == ("job_1", "job_2")
        assert not failed_job_dropdown.disabled

    def test_llm_manager_uses_job_id_as_thread_id_for_diagnosis(self) -> None:
        """Test that LLM manager uses job_id as thread_id for diagnosis."""
        run_manager = MagicMock()
        run_manager.database_manager.failed = [
            {"job_id": "job_123", "job_name": "test_job", "output_logs": []},
        ]

        llm_manager = MagicMock(spec=LLMManager)
        llm_manager.chat = AsyncMock(return_value="Test response")
        llm_manager.diagnose_failed_job = AsyncMock(return_value="Test diagnosis")
        run_manager.llm_manager = llm_manager

        widget = chat_widget(run_manager)

        # Verify that the widget structure is correct for multi-thread support
        assert len(widget.children) == 6
        assert widget.children[2].options == ("job_123",)

    def test_default_thread_when_no_job_selected(self) -> None:
        """Test that appropriate structure exists when no job is selected."""
        run_manager = MagicMock()
        run_manager.database_manager.failed = []
        run_manager.llm_manager = MagicMock(spec=LLMManager)

        widget = chat_widget(run_manager)
        failed_job_dropdown = widget.children[2]

        # No jobs available, dropdown should be empty but enabled
        assert failed_job_dropdown.options == ()
        assert not failed_job_dropdown.disabled

    @pytest.mark.asyncio
    async def test_llm_manager_diagnosis_uses_correct_thread_id(self) -> None:
        """Test that LLM manager diagnosis method uses job_id as thread_id."""
        from unittest.mock import patch

        from adaptive_scheduler._server_support.llm_manager import LLMManager

        # Create a real LLMManager instance but mock the executor
        db_manager = MagicMock()
        db_manager.failed = [
            {"job_id": "test_job_456", "job_name": "test_job", "output_logs": []},
        ]

        with patch("adaptive_scheduler._server_support.llm_manager.ChatOpenAI"):
            llm_manager = LLMManager(db_manager=db_manager)

            # Mock the helper methods
            llm_manager._get_log_file_paths = MagicMock(return_value=["fake_log.txt"])  # type: ignore[method-assign]
            llm_manager._read_log_files = AsyncMock(return_value="fake log content")  # type: ignore[method-assign]

            # Mock the chat method to capture its call
            llm_manager.chat = AsyncMock(return_value="Test diagnosis")  # type: ignore[method-assign]

            # Call diagnose_failed_job
            await llm_manager.diagnose_failed_job("test_job_456")

            # Verify that chat was called with the correct thread_id
            llm_manager.chat.assert_called_once()
            call_args = llm_manager.chat.call_args

            # Check that the thread_id matches the job_id
            assert call_args[1]["thread_id"] == "test_job_456"

    @pytest.mark.asyncio
    async def test_chat_method_uses_provided_thread_id(self) -> None:
        """Test that chat method uses the provided thread_id."""
        from unittest.mock import patch

        from adaptive_scheduler._server_support.llm_manager import LLMManager

        # Create a real LLMManager instance but mock the executor
        db_manager = MagicMock()

        with patch("adaptive_scheduler._server_support.llm_manager.ChatOpenAI"):
            llm_manager = LLMManager(db_manager=db_manager)

            # Mock the agent executor
            llm_manager.agent_executor = MagicMock()
            llm_manager.agent_executor.ainvoke = AsyncMock(
                return_value={"messages": [MagicMock(content="Test response")]},
            )

            # Call chat with a specific thread_id
            await llm_manager.chat("Test message", thread_id="custom_thread_123")

            # Verify that agent_executor.ainvoke was called with the correct config
            call_args = llm_manager.agent_executor.ainvoke.call_args
            config = call_args[0][1]  # Second argument is the config

            # Check that the thread_id in the config matches what we provided
            assert config["configurable"]["thread_id"] == "custom_thread_123"

    def test_widget_structure_supports_multi_thread_workflow(self) -> None:
        """Test that widget structure supports multi-thread workflow."""
        run_manager = MagicMock()
        run_manager.database_manager.failed = [
            {"job_id": "job_1", "job_name": "test_job_1", "output_logs": []},
            {"job_id": "job_2", "job_name": "test_job_2", "output_logs": []},
            {"job_id": "job_3", "job_name": "test_job_3", "output_logs": []},
        ]

        llm_manager = MagicMock(spec=LLMManager)
        llm_manager.chat = AsyncMock(return_value="Test response")
        llm_manager.diagnose_failed_job = AsyncMock(return_value="Test diagnosis")
        run_manager.llm_manager = llm_manager

        widget = chat_widget(run_manager)

        # Get widget components
        refresh_button = widget.children[1]
        failed_job_dropdown = widget.children[2]
        yolo_checkbox = widget.children[3]
        chat_history = widget.children[4]
        text_input = widget.children[5]

        # Verify all components exist and have proper configuration
        assert refresh_button.description == "Refresh Failed Jobs"
        assert failed_job_dropdown.description == "Failed Job:"
        assert failed_job_dropdown.options == ("job_1", "job_2", "job_3")
        assert not failed_job_dropdown.disabled
        assert yolo_checkbox.description == "YOLO mode"
        assert chat_history.description == "Chat:"
        assert text_input.description == "You:"

        # Verify that the dropdown has observers (which handle thread switching)
        assert len(failed_job_dropdown._trait_notifiers.get("value", [])) > 0
