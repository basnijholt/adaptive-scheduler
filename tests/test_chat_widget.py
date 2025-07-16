"""Tests for the chat widget functionality."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from adaptive_scheduler._server_support.llm_manager import LLMManager
from adaptive_scheduler.widgets import chat_widget


class TestChatWidget:
    """Test suite for the chat widget."""

    def test_chat_widget_structure(self) -> None:
        """Test that the chat widget has the correct structure."""
        # Create mock run manager with failed jobs
        run_manager = MagicMock()
        run_manager.database_manager.failed = [
            {"job_id": "job_1", "job_name": "test_job_1", "output_logs": []},
            {"job_id": "job_2", "job_name": "test_job_2", "output_logs": []},
        ]

        # Create mock LLM manager
        llm_manager = MagicMock(spec=LLMManager)
        llm_manager.chat = AsyncMock(return_value="Test response")
        llm_manager.diagnose_failed_job = AsyncMock(return_value="Test diagnosis")
        run_manager.llm_manager = llm_manager

        # Create the chat widget
        widget = chat_widget(run_manager)

        # Check widget structure
        assert len(widget.children) == 6

        # Get components
        title = widget.children[0]
        refresh_button = widget.children[1]
        failed_job_dropdown = widget.children[2]
        yolo_checkbox = widget.children[3]
        chat_history = widget.children[4]
        text_input = widget.children[5]

        # Check types
        assert title.__class__.__name__ == "HTML"
        assert refresh_button.__class__.__name__ == "Button"
        assert failed_job_dropdown.__class__.__name__ == "Dropdown"
        assert yolo_checkbox.__class__.__name__ == "Checkbox"
        assert chat_history.__class__.__name__ == "HTML"
        assert text_input.__class__.__name__ == "Text"

        # Check initial state
        assert failed_job_dropdown.options == ("job_1", "job_2")
        assert not failed_job_dropdown.disabled
        assert "Hello!" in chat_history.value
        assert not yolo_checkbox.value

    def test_chat_widget_empty_failed_jobs(self) -> None:
        """Test that the chat widget handles empty failed jobs correctly."""
        run_manager = MagicMock()
        run_manager.database_manager.failed = []
        run_manager.llm_manager = MagicMock(spec=LLMManager)

        widget = chat_widget(run_manager)
        failed_job_dropdown = widget.children[2]

        assert failed_job_dropdown.options == ()
        assert not failed_job_dropdown.disabled

    def test_chat_widget_refresh_button_functionality(self) -> None:
        """Test that the refresh button works correctly."""
        run_manager = MagicMock()
        run_manager.database_manager.failed = [
            {"job_id": "job_1", "job_name": "test_job_1", "output_logs": []},
        ]
        run_manager.llm_manager = MagicMock(spec=LLMManager)

        widget = chat_widget(run_manager)
        refresh_button = widget.children[1]
        failed_job_dropdown = widget.children[2]

        # Initially should have one job
        assert failed_job_dropdown.options == ("job_1",)
        assert not failed_job_dropdown.disabled

        # Add another job
        run_manager.database_manager.failed.append(
            {"job_id": "job_2", "job_name": "test_job_2", "output_logs": []},
        )

        # Refresh should update the dropdown
        refresh_button.click()
        assert failed_job_dropdown.options == ("job_1", "job_2")
        assert not failed_job_dropdown.disabled

        # Remove all jobs
        run_manager.database_manager.failed = []
        refresh_button.click()
        assert failed_job_dropdown.options == ()
        # The dropdown should still be enabled when there are no failed jobs
        assert not failed_job_dropdown.disabled

    def test_chat_widget_dropdown_disabled_state(self) -> None:
        """Test that the dropdown disabled state works correctly."""
        run_manager = MagicMock()
        run_manager.database_manager.failed = [
            {"job_id": "job_1", "job_name": "test_job_1", "output_logs": []},
        ]
        run_manager.llm_manager = MagicMock(spec=LLMManager)

        widget = chat_widget(run_manager)
        refresh_button = widget.children[1]
        failed_job_dropdown = widget.children[2]

        # Initially should be enabled with one job
        assert not failed_job_dropdown.disabled
        assert failed_job_dropdown.options == ("job_1",)

        # Clear failed jobs
        run_manager.database_manager.failed = []
        refresh_button.click()

        # Should have no options but still be enabled
        assert not failed_job_dropdown.disabled
        assert failed_job_dropdown.options == ()

        # Add job back
        run_manager.database_manager.failed = [
            {"job_id": "job_2", "job_name": "test_job_2", "output_logs": []},
        ]
        refresh_button.click()

        # Should be enabled again
        assert not failed_job_dropdown.disabled
        assert failed_job_dropdown.options == ("job_2",)

    def test_chat_widget_no_llm_manager(self) -> None:
        """Test that the chat widget handles missing LLM manager gracefully."""
        run_manager = MagicMock()
        run_manager.database_manager.failed = [
            {"job_id": "job_1", "job_name": "test_job_1", "output_logs": []},
        ]
        run_manager.llm_manager = None

        widget = chat_widget(run_manager)
        chat_history = widget.children[4]

        # Should still create the widget
        assert len(widget.children) == 6

        # Chat history should have initial message
        assert "Hello!" in chat_history.value

    def test_chat_widget_components_configuration(self) -> None:
        """Test that widget components are properly configured."""
        run_manager = MagicMock()
        run_manager.database_manager.failed = [
            {"job_id": "job_1", "job_name": "test_job_1", "output_logs": []},
        ]

        llm_manager = MagicMock(spec=LLMManager)
        llm_manager.chat = AsyncMock(return_value="Test response")
        llm_manager.diagnose_failed_job = AsyncMock(return_value="Test diagnosis")
        run_manager.llm_manager = llm_manager

        widget = chat_widget(run_manager)

        # Get components
        refresh_button = widget.children[1]
        failed_job_dropdown = widget.children[2]
        yolo_checkbox = widget.children[3]
        chat_history = widget.children[4]
        text_input = widget.children[5]

        # Check component properties
        assert refresh_button.description == "Refresh Failed Jobs"
        assert failed_job_dropdown.description == "Failed Job:"
        assert yolo_checkbox.description == "YOLO mode"
        assert chat_history.description == "Chat:"
        assert text_input.description == "You:"
        assert text_input.placeholder == "Ask a question..."

    @pytest.mark.asyncio
    async def test_chat_widget_basic_functionality(self) -> None:
        """Test basic chat widget functionality."""
        run_manager = MagicMock()
        run_manager.database_manager.failed = []

        llm_manager = MagicMock(spec=LLMManager)
        llm_manager.chat = AsyncMock(return_value="Test response")
        llm_manager.yolo = False
        run_manager.llm_manager = llm_manager

        widget = chat_widget(run_manager)
        text_input = widget.children[5]
        chat_history = widget.children[4]
        yolo_checkbox = widget.children[3]

        # Test YOLO mode setting
        yolo_checkbox.value = True
        assert yolo_checkbox.value

        # Test that the LLM manager exists
        assert run_manager.llm_manager is not None

        # Test that the chat history has the initial message
        assert "Hello!" in chat_history.value

        # Test that text input is configured correctly
        assert text_input.value == ""
        assert not text_input.disabled


class TestChatWidgetIntegration:
    """Integration tests for the chat widget."""

    def test_chat_widget_with_multiple_scenarios(self) -> None:
        """Test various scenarios with the chat widget."""
        run_manager = MagicMock()
        run_manager.llm_manager = MagicMock(spec=LLMManager)

        # Test with no failed jobs
        run_manager.database_manager.failed = []
        widget = chat_widget(run_manager)
        dropdown = widget.children[2]
        assert not dropdown.disabled

        # Test with one failed job
        run_manager.database_manager.failed = [
            {"job_id": "job_1", "job_name": "test_job_1", "output_logs": []},
        ]
        widget = chat_widget(run_manager)
        dropdown = widget.children[2]
        assert not dropdown.disabled
        assert dropdown.options == ("job_1",)

        # Test with multiple failed jobs
        run_manager.database_manager.failed = [
            {"job_id": "job_1", "job_name": "test_job_1", "output_logs": []},
            {"job_id": "job_2", "job_name": "test_job_2", "output_logs": []},
            {"job_id": "job_3", "job_name": "test_job_3", "output_logs": []},
        ]
        widget = chat_widget(run_manager)
        dropdown = widget.children[2]
        assert not dropdown.disabled
        assert dropdown.options == ("job_1", "job_2", "job_3")

    def test_chat_widget_observer_management(self) -> None:
        """Test that the observer management works correctly during refresh."""
        run_manager = MagicMock()
        run_manager.database_manager.failed = [
            {"job_id": "job_1", "job_name": "test_job_1", "output_logs": []},
        ]
        run_manager.llm_manager = MagicMock(spec=LLMManager)

        widget = chat_widget(run_manager)
        refresh_button = widget.children[1]
        failed_job_dropdown = widget.children[2]

        # The dropdown should have observers
        assert len(failed_job_dropdown._trait_notifiers.get("value", [])) > 0

        # After refresh, observers should still be there
        refresh_button.click()
        assert len(failed_job_dropdown._trait_notifiers.get("value", [])) > 0

        # Clear jobs and refresh
        run_manager.database_manager.failed = []
        refresh_button.click()

        # Observers should still be there
        assert len(failed_job_dropdown._trait_notifiers.get("value", [])) > 0
        assert not failed_job_dropdown.disabled
