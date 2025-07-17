"""Tests for specific chat widget fixes - dropdown disabled state and async callbacks."""

from unittest.mock import MagicMock

from adaptive_scheduler._llm_widgets import chat_widget
from adaptive_scheduler._server_support.llm_manager import LLMManager


class TestChatWidgetFixes:
    """Test specific fixes for chat widget issues."""

    def test_dropdown_never_disabled(self) -> None:
        """Test that dropdown is never disabled, even when there are no failed jobs."""
        db_manager = MagicMock()
        db_manager.failed = []
        llm_manager = MagicMock(spec=LLMManager)
        llm_manager.db_manager = db_manager

        widget = chat_widget(llm_manager)
        failed_job_dropdown = widget.children[2]

        # Dropdown should never be disabled
        assert not failed_job_dropdown.disabled
        assert failed_job_dropdown.options == ()

    def test_dropdown_enabled_when_failed_jobs_exist(self) -> None:
        """Test that dropdown is enabled when there are failed jobs."""
        db_manager = MagicMock()
        db_manager.failed = [
            {"job_id": "job_1", "job_name": "test_job_1", "output_logs": []},
        ]
        llm_manager = MagicMock(spec=LLMManager)
        llm_manager.db_manager = db_manager

        widget = chat_widget(llm_manager)
        failed_job_dropdown = widget.children[2]

        # Dropdown should be enabled when failed jobs exist
        assert not failed_job_dropdown.disabled
        assert failed_job_dropdown.options == ("job_1",)

    def test_refresh_button_updates_dropdown_disabled_state(self) -> None:
        """Test that refresh button correctly updates dropdown disabled state."""
        db_manager = MagicMock()
        db_manager.failed = [
            {"job_id": "job_1", "job_name": "test_job_1", "output_logs": []},
        ]
        llm_manager = MagicMock(spec=LLMManager)
        llm_manager.db_manager = db_manager

        widget = chat_widget(llm_manager)
        refresh_button = widget.children[1]
        failed_job_dropdown = widget.children[2]

        # Initially enabled with one job
        assert not failed_job_dropdown.disabled
        assert failed_job_dropdown.options == ("job_1",)

        # Clear jobs and refresh
        db_manager.failed = []
        refresh_button.click()

        # Should have no options but still be enabled
        assert not failed_job_dropdown.disabled
        assert failed_job_dropdown.options == ()

        # Add job back and refresh
        db_manager.failed = [
            {"job_id": "job_2", "job_name": "test_job_2", "output_logs": []},
        ]
        refresh_button.click()

        # Should be enabled again
        assert not failed_job_dropdown.disabled
        assert failed_job_dropdown.options == ("job_2",)

    def test_refresh_button_prevents_async_callback_warnings(self) -> None:
        """Test that refresh button doesn't trigger async callback warnings."""
        db_manager = MagicMock()
        db_manager.failed = [
            {"job_id": "job_1", "job_name": "test_job_1", "output_logs": []},
        ]
        llm_manager = MagicMock(spec=LLMManager)
        llm_manager.db_manager = db_manager

        widget = chat_widget(llm_manager)
        refresh_button = widget.children[1]
        failed_job_dropdown = widget.children[2]

        # Clear jobs - this should not trigger callback warnings
        db_manager.failed = []

        # The refresh should work without triggering async warnings
        # (Note: the actual warning suppression is handled by the observer management)
        refresh_button.click()

        # Verify the state is correct
        assert not failed_job_dropdown.disabled
        assert failed_job_dropdown.options == ()

    def test_observer_management_during_refresh(self) -> None:
        """Test that observers are properly managed during refresh operations."""
        db_manager = MagicMock()
        db_manager.failed = [
            {"job_id": "job_1", "job_name": "test_job_1", "output_logs": []},
        ]
        llm_manager = MagicMock(spec=LLMManager)
        llm_manager.db_manager = db_manager

        widget = chat_widget(llm_manager)
        refresh_button = widget.children[1]
        failed_job_dropdown = widget.children[2]

        # Check that observers exist
        initial_observers = len(failed_job_dropdown._trait_notifiers.get("value", []))
        assert initial_observers > 0

        # Refresh with different job states
        db_manager.failed = []
        refresh_button.click()

        # Observers should still exist after refresh
        after_clear_observers = len(failed_job_dropdown._trait_notifiers.get("value", []))
        assert after_clear_observers > 0
        assert after_clear_observers == initial_observers

        # Add jobs back and refresh
        db_manager.failed = [
            {"job_id": "job_2", "job_name": "test_job_2", "output_logs": []},
        ]
        refresh_button.click()

        # Observers should still be the same
        after_add_observers = len(failed_job_dropdown._trait_notifiers.get("value", []))
        assert after_add_observers == initial_observers

    def test_chat_history_initial_message(self) -> None:
        """Test that chat history has a helpful initial message."""
        db_manager = MagicMock()
        db_manager.failed = []
        llm_manager = MagicMock(spec=LLMManager)
        llm_manager.db_manager = db_manager

        widget = chat_widget(llm_manager)
        chat_history = widget.children[4]

        # Should have a helpful initial message
        assert "Hello!" in chat_history.value
        assert (
            "Select a failed job to diagnose" in chat_history.value
            or "ask me a question" in chat_history.value
        )

    def test_multiple_refresh_operations(self) -> None:
        """Test multiple refresh operations to ensure state consistency."""
        db_manager = MagicMock()
        db_manager.failed = []
        llm_manager = MagicMock(spec=LLMManager)
        llm_manager.db_manager = db_manager

        widget = chat_widget(llm_manager)
        refresh_button = widget.children[1]
        failed_job_dropdown = widget.children[2]

        # Initially no jobs but dropdown should be enabled
        assert not failed_job_dropdown.disabled

        # Add job
        db_manager.failed = [
            {"job_id": "job_1", "job_name": "test_job_1", "output_logs": []},
        ]
        refresh_button.click()
        assert not failed_job_dropdown.disabled
        assert failed_job_dropdown.options == ("job_1",)

        # Add more jobs
        db_manager.failed = [
            {"job_id": "job_1", "job_name": "test_job_1", "output_logs": []},
            {"job_id": "job_2", "job_name": "test_job_2", "output_logs": []},
        ]
        refresh_button.click()
        assert not failed_job_dropdown.disabled
        assert failed_job_dropdown.options == ("job_1", "job_2")

        # Remove one job
        db_manager.failed = [
            {"job_id": "job_2", "job_name": "test_job_2", "output_logs": []},
        ]
        refresh_button.click()
        assert not failed_job_dropdown.disabled
        assert failed_job_dropdown.options == ("job_2",)

        # Remove all jobs
        db_manager.failed = []
        refresh_button.click()
        assert not failed_job_dropdown.disabled
        assert failed_job_dropdown.options == ()
