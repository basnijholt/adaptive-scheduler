from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import ipywidgets as ipyw
import sys
from pathlib import Path

# Add the assistant-ui-anywidget python directory to the path
_widget_path = Path(__file__).parent.parent / "assistant-ui-anywidget" / "python"
if str(_widget_path) not in sys.path:
    sys.path.insert(0, str(_widget_path))

from agent_widget import AgentWidget

from adaptive_scheduler._server_support.llm_manager import ChatResult

# Debug toggle - set to False to disable debug output
DEBUG_LLM_WIDGET = False

def _debug_print(*args, **kwargs):
    """Print debug message only if debugging is enabled."""
    if DEBUG_LLM_WIDGET:
        print(*args, **kwargs)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Coroutine

    from adaptive_scheduler.server_support import LLMManager


class LLMChatWidget(AgentWidget):
    """An AnyWidget-based chat interface for the LLMManager."""
    
    # Override the _esm path to point to the correct location
    _esm = str(Path(__file__).parent.parent / "assistant-ui-anywidget" / "frontend" / "dist" / "index.js")

    def __init__(self, llm_manager: LLMManager, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.llm_manager = llm_manager
        self.thread_id = "1"  # Default thread
        self.waiting_for_approval = False
        self._tasks: set[asyncio.Task] = set()
        
        # Add welcome message
        self.add_message("assistant", "👋 Hello! Select a failed job to diagnose or ask me a question.")

    def _handle_message(self, _, content, buffers=None):
        """Handle incoming messages from the frontend."""
        if content.get("type") == "user_message":
            user_text = content.get("text", "")
            
            # Add user message to chat history
            new_history = list(self.chat_history)
            new_history.append({"role": "user", "content": user_text})
            self.chat_history = new_history
            
            # Handle the message asynchronously
            self._create_task(self._handle_user_message(user_text))

    def _create_task(self, awaitable: Coroutine) -> None:
        """Run an async function as a background task."""
        try:
            task: asyncio.Task = asyncio.create_task(awaitable)
            self._tasks.add(task)
            
            def task_done_callback(t):
                self._tasks.discard(t)  # Use discard instead of remove to avoid KeyError
                if t.exception():
                    print(f"Debug: Task failed with exception: {t.exception()}")
                    import traceback
                    traceback.print_exception(type(t.exception()), t.exception(), t.exception().__traceback__)
            
            task.add_done_callback(task_done_callback)
            
        except Exception as e:
            print(f"Debug: Error creating task: {e}")
            import traceback
            traceback.print_exc()

    async def _handle_user_message(self, message: str) -> None:
        """Handle logic for when the user submits text."""
        if not message:
            return

        try:
            if self.waiting_for_approval and message.lower() in [
                "approve",
                "approved", 
                "deny",
                "denied",
            ]:
                is_approved = message.lower() in ["approve", "approved"]
                result = await self.llm_manager.chat(is_approved, thread_id=self.thread_id)
                self.waiting_for_approval = False
            else:
                result = await self.llm_manager.chat(message, thread_id=self.thread_id)
            
            await self._handle_llm_response(result)
            
        except Exception as e:
            self.add_message("assistant", f"❌ Error: {e}")

    async def _handle_llm_response(self, result: ChatResult) -> None:
        """Handle the LLM response."""
        if result.content:
            self.add_message("assistant", result.content)
        
        if result.interrupted:
            self.waiting_for_approval = True
            interrupt_msg = f"🤖 {result.interrupt_message}\n\n*Reply with 'approve' or 'deny' to continue.*"
            self.add_message("assistant", interrupt_msg)

    async def diagnose_job(self, job_id: str) -> None:
        """Diagnose a failed job."""
        self.thread_id = job_id
        self.clear_chat_history()
        self.add_message("assistant", f"🔍 Diagnosing job {job_id}...")
        
        try:
            result = await self.llm_manager.diagnose_failed_job(job_id)
            await self._handle_llm_response(result)
        except Exception as e:
            self.add_message("assistant", f"❌ Error diagnosing job: {e}")

    def set_yolo_mode(self, enabled: bool) -> None:
        """Set YOLO mode on the LLM manager."""
        self.llm_manager.yolo = enabled


def create_enhanced_chat_widget(llm_manager: LLMManager) -> ipyw.VBox:
    """Create an enhanced chat widget with job selection and controls."""
    from .widgets import _add_title
    
    # Create the main chat widget
    chat_widget = LLMChatWidget(llm_manager)
    
    # Create control widgets
    yolo_checkbox = ipyw.Checkbox(description="YOLO mode", value=False, indent=False)
    
    # Extract failed job IDs (same as original working implementation)
    try:
        failed_jobs_data = llm_manager.db_manager.failed
        print(f"Debug: Found {len(failed_jobs_data)} failed jobs")
        
        if failed_jobs_data:
            print(f"Debug: First 3 jobs:")
            for i, job in enumerate(failed_jobs_data[:3]):
                print(f"  Job {i}: job_id={job.get('job_id')!r}, type={type(job.get('job_id'))}")
        
        # Use EXACT same logic as original working implementation - no filtering
        job_ids = [job["job_id"] for job in failed_jobs_data]
        _debug_print(f"Debug: Extracted job IDs: {job_ids}")
        
        # Additional validation
        if not job_ids:
            print("Debug: No job IDs extracted - checking why...")
            for i, job in enumerate(failed_jobs_data[:3]):  # Check first 3 jobs
                print(f"Debug: Job {i}: {job}")
                if isinstance(job, dict):
                    if "job_id" in job:
                        print(f"Debug: Job {i} has job_id: {job['job_id']}")
                    else:
                        print(f"Debug: Job {i} missing 'job_id' key, has: {list(job.keys())}")
                else:
                    print(f"Debug: Job {i} is not a dict: {type(job)}")
        
    except Exception as e:
        print(f"Debug: Error extracting job IDs: {e}")
        import traceback
        traceback.print_exc()
        job_ids = []
    
    failed_job_dropdown = ipyw.Dropdown(
        options=job_ids,
        description="Failed Job:",
        disabled=False,
        value=None
    )
    refresh_button = ipyw.Button(description="Refresh Failed Jobs")
    
    def on_yolo_change(change):
        chat_widget.set_yolo_mode(change["new"])
    
    def on_job_change(change):
        try:
            print(f"Debug: Job selection changed - old: {change.get('old')!r}, new: {change.get('new')!r}")
            print(f"Debug: Change object keys: {list(change.keys())}")
            print(f"Debug: Full change object: {change}")
            
            if change["new"] is not None:  # More explicit check
                # Create task to diagnose job
                print(f"Debug: Starting diagnosis for job: {change['new']!r}")
                chat_widget._create_task(chat_widget.diagnose_job(str(change["new"])))
            else:
                print("Debug: No job selected (value is None)")
        except Exception as e:
            print(f"Debug: Error in job change handler: {e}")
            import traceback
            traceback.print_exc()
    
    def on_refresh(_):
        try:
            failed_jobs_data = llm_manager.db_manager.failed
            print(f"Debug: Refreshing - found {len(failed_jobs_data)} failed jobs")
            
            # Use EXACT same logic as original - no filtering
            job_ids = [job["job_id"] for job in failed_jobs_data]
            _debug_print(f"Debug: Refreshed job IDs: {job_ids}")
            
            failed_job_dropdown.unobserve(on_job_change, names="value")
            failed_job_dropdown.options = job_ids
            failed_job_dropdown.value = None
            failed_job_dropdown.observe(on_job_change, names="value")
            
        except Exception as e:
            print(f"Debug: Error during refresh: {e}")
            import traceback
            traceback.print_exc()
    
    # Register event handlers
    yolo_checkbox.observe(on_yolo_change, names="value")
    failed_job_dropdown.observe(on_job_change, names="value")
    refresh_button.on_click(on_refresh)
    
    # Assemble the widget
    widget = ipyw.VBox([
        refresh_button,
        failed_job_dropdown,
        yolo_checkbox,
        chat_widget,
    ])
    
    _add_title("adaptive_scheduler.widgets.LLMChatWidget", widget)
    return widget


def chat_widget(llm_manager: LLMManager) -> ipyw.VBox:
    """Creates and returns an AnyWidget-based chat widget for interacting with the LLM.

    This widget provides a modern React-based user interface for chatting with a 
    language model, diagnosing failed jobs, and approving or denying operations 
    that require user intervention.

    Parameters
    ----------
    llm_manager
        An instance of `LLMManager` that the widget will use to communicate
        with the language model and the database.

    Returns
    -------
    ipyw.VBox
        A VBox widget containing the entire chat interface.

    """
    return create_enhanced_chat_widget(llm_manager)