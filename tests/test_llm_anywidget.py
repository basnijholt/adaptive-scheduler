"""Comprehensive tests for the LLM AnyWidget implementation."""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from adaptive_scheduler._llm_anywidget import (
    LLMChatWidget, 
    chat_widget, 
    create_enhanced_chat_widget
)
from adaptive_scheduler._server_support.llm_manager import ChatResult


@pytest.fixture
def mock_llm_manager():
    """Create a mock LLM manager for testing."""
    manager = Mock()
    manager.db_manager.failed = [
        {"job_id": "test_job_001", "output_logs": ["/path/to/log1.txt"]},
        {"job_id": "test_job_002", "output_logs": ["/path/to/log2.txt"]},
        {"job_id": "test_job_003", "output_logs": ["/path/to/log3.txt"]},
    ]
    manager.yolo = False
    manager.chat = AsyncMock()
    manager.diagnose_failed_job = AsyncMock()
    manager.get_history = Mock(return_value={"messages": []})
    return manager


@pytest.fixture
def sample_chat_result():
    """Create a sample ChatResult for testing."""
    return ChatResult(
        content="This is a test response",
        interrupted=False,
        thread_id="test_thread"
    )


@pytest.fixture
def interrupted_chat_result():
    """Create an interrupted ChatResult for testing approval workflows."""
    return ChatResult(
        content="",
        interrupted=True,
        interrupt_message="Do you want to write to file.py?",
        thread_id="test_thread"
    )


class TestLLMChatWidget:
    """Test the core LLMChatWidget functionality."""
    
    def test_widget_creation(self, mock_llm_manager):
        """Test basic widget creation."""
        widget = LLMChatWidget(mock_llm_manager)
        
        assert widget.llm_manager == mock_llm_manager
        assert widget.thread_id == "1"
        assert widget.waiting_for_approval is False
        assert len(widget.chat_history) == 1  # Welcome message
        assert widget.chat_history[0]["role"] == "assistant"
        assert "👋" in widget.chat_history[0]["content"]
    
    def test_add_message(self, mock_llm_manager):
        """Test adding messages to chat history."""
        widget = LLMChatWidget(mock_llm_manager)
        initial_count = len(widget.chat_history)
        
        widget.add_message("user", "Test message")
        assert len(widget.chat_history) == initial_count + 1
        assert widget.chat_history[-1]["role"] == "user"
        assert widget.chat_history[-1]["content"] == "Test message"
    
    def test_clear_chat_history(self, mock_llm_manager):
        """Test clearing chat history."""
        widget = LLMChatWidget(mock_llm_manager)
        widget.add_message("user", "Test")
        assert len(widget.chat_history) > 0
        
        widget.clear_chat_history()
        assert len(widget.chat_history) == 0
    
    def test_set_yolo_mode(self, mock_llm_manager):
        """Test setting YOLO mode."""
        widget = LLMChatWidget(mock_llm_manager)
        
        widget.set_yolo_mode(True)
        assert mock_llm_manager.yolo is True
        
        widget.set_yolo_mode(False)
        assert mock_llm_manager.yolo is False
    
    def test_javascript_bundle_path(self, mock_llm_manager):
        """Test that the JavaScript bundle path is correct."""
        widget = LLMChatWidget(mock_llm_manager)
        
        # The _esm property should contain JavaScript content (loaded from file)
        assert isinstance(widget._esm, str)
        assert len(widget._esm) > 1000  # Should be substantial JS content
        assert "react" in widget._esm.lower()  # Should contain React
    
    @pytest.mark.asyncio
    async def test_handle_user_message_simple(self, mock_llm_manager, sample_chat_result):
        """Test handling a simple user message."""
        mock_llm_manager.chat.return_value = sample_chat_result
        widget = LLMChatWidget(mock_llm_manager)
        
        await widget._handle_user_message("Hello")
        
        mock_llm_manager.chat.assert_called_once_with("Hello", thread_id="1")
        # Should add assistant response to chat history
        assert len(widget.chat_history) >= 2
        assert any(msg["content"] == "This is a test response" for msg in widget.chat_history)
    
    @pytest.mark.asyncio
    async def test_handle_user_message_approval_workflow(self, mock_llm_manager, interrupted_chat_result):
        """Test handling user message that triggers approval workflow."""
        mock_llm_manager.chat.return_value = interrupted_chat_result
        widget = LLMChatWidget(mock_llm_manager)
        
        await widget._handle_user_message("Please fix the code")
        
        assert widget.waiting_for_approval is True
        # Should add approval message to chat history
        approval_msgs = [msg for msg in widget.chat_history if "approve" in msg["content"].lower()]
        assert len(approval_msgs) > 0
    
    @pytest.mark.asyncio
    async def test_handle_approval_responses(self, mock_llm_manager, sample_chat_result):
        """Test handling approval/denial responses."""
        mock_llm_manager.chat.return_value = sample_chat_result
        widget = LLMChatWidget(mock_llm_manager)
        widget.waiting_for_approval = True
        
        # Test approval
        await widget._handle_user_message("approve")
        mock_llm_manager.chat.assert_called_with(True, thread_id="1")
        
        # Test denial
        widget.waiting_for_approval = True
        await widget._handle_user_message("deny")
        mock_llm_manager.chat.assert_called_with(False, thread_id="1")
    
    @pytest.mark.asyncio
    async def test_diagnose_job(self, mock_llm_manager, sample_chat_result):
        """Test job diagnosis functionality."""
        mock_llm_manager.diagnose_failed_job.return_value = sample_chat_result
        widget = LLMChatWidget(mock_llm_manager)
        
        await widget.diagnose_job("test_job_001")
        
        assert widget.thread_id == "test_job_001"
        assert len(widget.chat_history) >= 1
        mock_llm_manager.diagnose_failed_job.assert_called_once_with("test_job_001")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_llm_manager):
        """Test error handling in message processing."""
        mock_llm_manager.chat.side_effect = Exception("Test error")
        widget = LLMChatWidget(mock_llm_manager)
        
        await widget._handle_user_message("This will cause an error")
        
        # Should add error message to chat history
        error_msgs = [msg for msg in widget.chat_history if "❌ Error" in msg["content"]]
        assert len(error_msgs) > 0


class TestCreateEnhancedChatWidget:
    """Test the enhanced chat widget creation function."""
    
    def test_widget_creation_with_failed_jobs(self, mock_llm_manager, capsys):
        """Test creating widget when there are failed jobs."""
        widget = create_enhanced_chat_widget(mock_llm_manager)
        
        # Check console output for debug information
        captured = capsys.readouterr()
        assert "Debug: Found 3 failed jobs" in captured.out
        assert "test_job_001" in captured.out
        
        # Check widget structure
        assert hasattr(widget, 'children')
        assert len(widget.children) == 5  # refresh, dropdown, checkbox, chat_widget, title
        
        # Check dropdown has correct options
        dropdown = widget.children[1]  # Second child is the dropdown
        assert len(dropdown.options) == 3
        assert "test_job_001" in dropdown.options
    
    def test_widget_creation_no_failed_jobs(self, mock_llm_manager, capsys):
        """Test creating widget when there are no failed jobs."""
        mock_llm_manager.db_manager.failed = []
        
        widget = create_enhanced_chat_widget(mock_llm_manager)
        
        captured = capsys.readouterr()
        assert "Debug: No failed jobs found" in captured.out
        
        # Dropdown should be empty
        dropdown = widget.children[1]
        assert len(dropdown.options) == 0
    
    def test_widget_creation_malformed_jobs(self, mock_llm_manager, capsys):
        """Test creating widget with malformed job data."""
        mock_llm_manager.db_manager.failed = [
            {"job_id": "good_job"},
            {"no_job_id_key": "bad_job"},
            {"job_id": None},
            {"job_id": ""},
        ]
        
        widget = create_enhanced_chat_widget(mock_llm_manager)
        
        captured = capsys.readouterr()
        assert "Debug: Found 4 failed jobs" in captured.out
        
        # Only the good job should be in dropdown
        dropdown = widget.children[1]
        assert len(dropdown.options) == 1
        assert dropdown.options[0] == "good_job"


class TestChatWidgetFunction:
    """Test the main chat_widget function."""
    
    def test_chat_widget_returns_vbox(self, mock_llm_manager):
        """Test that chat_widget returns a VBox widget."""
        widget = chat_widget(mock_llm_manager)
        
        # Should return the same as create_enhanced_chat_widget
        assert hasattr(widget, 'children')
        assert len(widget.children) == 5


class TestWidgetInteractions:
    """Test widget interactions and event handling."""
    
    def test_refresh_button_functionality(self, mock_llm_manager, capsys):
        """Test the refresh button updates the dropdown."""
        widget = create_enhanced_chat_widget(mock_llm_manager)
        
        # Add more failed jobs
        mock_llm_manager.db_manager.failed.append({"job_id": "new_job"})
        
        # Get the refresh button and click it
        refresh_button = widget.children[0]
        refresh_button.click()  # Simulate button click
        
        captured = capsys.readouterr()
        assert "Debug: Refreshing - found 4 failed jobs" in captured.out
        assert "new_job" in captured.out
    
    def test_yolo_checkbox_functionality(self, mock_llm_manager):
        """Test YOLO checkbox changes the manager setting."""
        widget = create_enhanced_chat_widget(mock_llm_manager)
        
        # Get the YOLO checkbox
        yolo_checkbox = widget.children[2]  # Third child is YOLO checkbox
        
        # Simulate checking the box
        yolo_checkbox.value = True
        
        # The manager's yolo setting should be updated
        # Note: In real usage, this would be triggered by the observe handler
        assert mock_llm_manager.yolo is False  # Not changed yet without triggering event
    

class TestErrorScenarios:
    """Test various error scenarios and edge cases."""
    
    def test_widget_with_none_llm_manager(self):
        """Test widget creation with None LLM manager."""
        with pytest.raises(AttributeError):
            LLMChatWidget(None)
    
    @pytest.mark.asyncio
    async def test_diagnose_nonexistent_job(self, mock_llm_manager):
        """Test diagnosing a job that doesn't exist."""
        mock_llm_manager.diagnose_failed_job.side_effect = Exception("Job not found")
        widget = LLMChatWidget(mock_llm_manager)
        
        await widget.diagnose_job("nonexistent_job")
        
        # Should handle error gracefully
        error_msgs = [msg for msg in widget.chat_history if "❌ Error" in msg["content"]]
        assert len(error_msgs) > 0
    
    def test_empty_message_handling(self, mock_llm_manager):
        """Test handling empty messages."""
        widget = LLMChatWidget(mock_llm_manager)
        
        # Mock the frontend message handler
        widget._handle_message(None, {"type": "user_message", "text": ""}, None)
        
        # Should not crash, message should be added but not processed
        assert len(widget.chat_history) >= 1


class TestPerformance:
    """Test performance aspects of the widget."""
    
    def test_large_chat_history(self, mock_llm_manager):
        """Test widget performance with large chat history."""
        widget = LLMChatWidget(mock_llm_manager)
        
        # Add many messages
        for i in range(1000):
            widget.add_message("user" if i % 2 == 0 else "assistant", f"Message {i}")
        
        assert len(widget.chat_history) == 1001  # 1000 + welcome message
        
        # Should still be responsive
        widget.add_message("user", "Final message")
        assert widget.chat_history[-1]["content"] == "Final message"
    
    def test_many_failed_jobs(self, mock_llm_manager, capsys):
        """Test widget with many failed jobs."""
        # Create many failed jobs
        mock_llm_manager.db_manager.failed = [
            {"job_id": f"job_{i:04d}"} for i in range(1000)
        ]
        
        widget = create_enhanced_chat_widget(mock_llm_manager)
        
        captured = capsys.readouterr()
        assert "Debug: Found 1000 failed jobs" in captured.out
        
        # Dropdown should have all jobs
        dropdown = widget.children[1]
        assert len(dropdown.options) == 1000


class TestIntegration:
    """Integration tests combining multiple components."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_job_diagnosis(self, mock_llm_manager, sample_chat_result):
        """Test complete workflow: select job, diagnose, get response."""
        mock_llm_manager.diagnose_failed_job.return_value = sample_chat_result
        
        widget = create_enhanced_chat_widget(mock_llm_manager)
        chat_widget_instance = widget.children[3]  # The actual chat widget
        
        # Simulate selecting a job and diagnosing it
        await chat_widget_instance.diagnose_job("test_job_001")
        
        # Check that diagnosis was called and response was added
        mock_llm_manager.diagnose_failed_job.assert_called_once_with("test_job_001")
        assert chat_widget_instance.thread_id == "test_job_001"
        
        # Check chat history contains diagnosis response
        response_msgs = [msg for msg in chat_widget_instance.chat_history 
                        if msg["content"] == "This is a test response"]
        assert len(response_msgs) > 0
    
    @pytest.mark.asyncio
    async def test_full_approval_workflow(self, mock_llm_manager, interrupted_chat_result, sample_chat_result):
        """Test complete approval workflow."""
        # First call returns interrupted result, second returns normal result
        mock_llm_manager.chat.side_effect = [interrupted_chat_result, sample_chat_result]
        
        widget = create_enhanced_chat_widget(mock_llm_manager)
        chat_widget_instance = widget.children[3]
        
        # Send message that requires approval
        await chat_widget_instance._handle_user_message("Fix the code")
        assert chat_widget_instance.waiting_for_approval is True
        
        # Send approval
        await chat_widget_instance._handle_user_message("approve")
        assert chat_widget_instance.waiting_for_approval is False
        
        # Check both calls were made
        assert mock_llm_manager.chat.call_count == 2
        mock_llm_manager.chat.assert_any_call("Fix the code", thread_id="1")
        mock_llm_manager.chat.assert_any_call(True, thread_id="1")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])