from __future__ import annotations

import asyncio
import html
from typing import TYPE_CHECKING, Any

import ipywidgets as ipyw
import mistune
import rich
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

from adaptive_scheduler._server_support.llm_manager import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Coroutine

    from langchain_core.messages import BaseMessage

    from adaptive_scheduler.server_support import LLMManager

console = rich.get_console()


def _render_markdown(text: str) -> str:
    """Render markdown to HTML with syntax highlighting."""
    try:
        renderer = mistune.HTMLRenderer(escape=False)
        markdown = mistune.Markdown(renderer=renderer)
        return markdown(text)
    except (TypeError, AttributeError, ValueError):
        return html.escape(text).replace("\n", "<br>")


def _render_chat_message(role: str, message: str) -> str:
    """Render a chat message with a role-specific style."""
    style = {
        "user": "background-color: #dcf8c6; text-align: right; margin-left: auto;",
        "llm": "background-color: #f1f0f0; text-align: left;",
        "tool": "background-color: #c6d8f8; text-align: center; margin-left: auto; margin-right: auto;",
    }[role]
    return f'<div style="padding: 5px; border-radius: 5px; margin: 5px; max-width: 80%; {style}">{message}</div>'


class ChatWidget:
    """A chat widget for interacting with the LLM."""

    def __init__(self, llm_manager: LLMManager) -> None:
        """Initialize the chat widget."""
        self.llm_manager = llm_manager
        self.thread_id = "1"  # Default thread
        self.waiting_for_approval = False
        self._tasks: set[asyncio.Task] = set()

        self._build_widget()

    def _build_widget(self) -> None:
        """Build the widget components."""
        from .widgets import _add_title

        self.text_input = ipyw.Text(
            value="",
            placeholder="Ask a question...",
            description="You:",
            disabled=False,
        )
        self.chat_history = ipyw.HTML(
            value=_render_chat_message(
                "llm",
                _render_markdown("👋 Hello! Select a failed job to diagnose or ask me a question."),
            ),
            layout={
                "width": "auto",
                "height": "300px",
                "border": "1px solid black",
                "word-wrap": "break-word",
                "overflow-y": "auto",
            },
        )
        self.yolo_checkbox = ipyw.Checkbox(description="YOLO mode", value=False, indent=False)
        self.failed_job_dropdown = ipyw.Dropdown(
            options=[job["job_id"] for job in self.llm_manager.db_manager.failed],
            description="Failed Job:",
            disabled=False,
        )
        self.refresh_button = ipyw.Button(description="Refresh Failed Jobs")

        # Register event handlers
        self.text_input.on_submit(self._on_submit)
        self.failed_job_dropdown.observe(self._on_failed_job_change, names="value")
        self.refresh_button.on_click(self._refresh_failed_jobs)

        # Assemble the widget
        self.widget = ipyw.VBox(
            [
                self.refresh_button,
                self.failed_job_dropdown,
                self.yolo_checkbox,
                self.chat_history,
                self.text_input,
            ],
        )
        _add_title("adaptive_scheduler.widgets.ChatWidget", self.widget)

        # Trigger diagnosis if a job is already selected
        if self.failed_job_dropdown.value:
            self._on_failed_job_change({"new": self.failed_job_dropdown.value})

    def _create_task(self, awaitable: Coroutine) -> None:
        """Run an async function as a background task."""
        task: asyncio.Task = asyncio.create_task(awaitable)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.remove)

    def _on_submit(self, sender: ipyw.Text) -> None:
        self._create_task(self._handle_submission(sender.value))
        sender.value = ""

    def _on_failed_job_change(self, change: dict[str, Any]) -> None:
        self._create_task(self._handle_job_selection(change["new"]))

    async def _handle_submission(self, message: str) -> None:
        """Handle logic for when the user submits text."""
        if not message:
            return

        self.llm_manager.yolo = self.yolo_checkbox.value

        if self.waiting_for_approval and message.lower() in [
            "approve",
            "approved",
            "deny",
            "denied",
        ]:
            approval_data = "approved" if message.lower() in ["approve", "approved"] else "denied"
            await self._run_llm_interaction(
                self.llm_manager.resume_chat(approval_data, thread_id=self.thread_id),
            )
            self.waiting_for_approval = False
        else:
            await self._run_llm_interaction(
                self.llm_manager.chat(message, thread_id=self.thread_id),
            )

    async def _handle_job_selection(self, job_id: str | None) -> None:
        """Handle logic for when a failed job is selected."""
        if job_id is None:
            self.thread_id = "1"
            return

        self.thread_id = job_id
        self.chat_history.value = _render_chat_message(
            "llm",
            _render_markdown(f"🔍 Diagnosing job {job_id}..."),
        )
        await self._run_llm_interaction(
            self.llm_manager.diagnose_failed_job(job_id),
            is_diagnosis=True,
        )

    async def _run_llm_interaction(
        self,
        coro: Awaitable,
        *,
        is_diagnosis: bool = False,
    ) -> None:
        """Central method to run LLM interactions and handle UI updates."""
        self.text_input.disabled = True
        self.failed_job_dropdown.disabled = True

        thinking_message = _render_chat_message("llm", _render_markdown("🤔 Thinking..."))
        if not is_diagnosis:
            # Show thinking message but don't add to history
            self.chat_history.value = self._render_history() + thinking_message

        try:
            result = await coro
            self.chat_history.value = self._render_history()

            if result.interrupted:
                self.waiting_for_approval = True
                interrupt_msg = f"🤖 {result.interrupt_message}\n\n*Reply with 'approve' or 'deny' to continue.*"
                self._add_message("llm", interrupt_msg)
            else:
                self.waiting_for_approval = False

        except Exception as e:  # noqa: BLE001
            console.print_exception(show_locals=True)
            self.chat_history.value = self.chat_history.value.replace(thinking_message, "")
            self._add_message("llm", f"❌ Error: {e}")
        finally:
            self.text_input.disabled = False
            self.failed_job_dropdown.disabled = False

    def _add_message(self, role: str, message: str) -> None:
        """Add a message to the chat history display."""
        self.chat_history.value += _render_chat_message(role, _render_markdown(message))

    def _render_history(self) -> str:
        """Render the entire chat history."""
        history = self.llm_manager.get_history(self.thread_id)
        messages = history.get("messages", [])
        return "".join(self._render_message(msg) for msg in messages)

    def _render_message(self, msg: BaseMessage) -> str:
        """Render a single message."""
        if isinstance(msg, HumanMessage):
            return _render_chat_message("user", _render_markdown(msg.content))
        if isinstance(msg, AIMessage):
            return _render_chat_message("llm", _render_markdown(msg.content))
        if isinstance(msg, ToolMessage):
            return _render_chat_message("tool", _render_markdown(msg.content))
        return ""

    def _refresh_failed_jobs(self, _: ipyw.Button) -> None:
        """Refresh the list of failed jobs in the dropdown."""
        failed_jobs = [job["job_id"] for job in self.llm_manager.db_manager.failed]
        self.failed_job_dropdown.unobserve(self._on_failed_job_change, names="value")
        self.failed_job_dropdown.options = failed_jobs
        self.failed_job_dropdown.value = None
        self.failed_job_dropdown.observe(self._on_failed_job_change, names="value")


def chat_widget(llm_manager: LLMManager) -> ipyw.VBox:
    """Creates and returns a chat widget for interacting with the LLM.

    This widget provides a user interface for chatting with a language model,
    diagnosing failed jobs, and approving or denying operations that
    require user intervention.

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
    return ChatWidget(llm_manager).widget


def _highlight_code(code: str, lang: str, _: str) -> str:
    """Highlight code blocks with pygments."""
    try:
        lexer = get_lexer_by_name(lang, stripall=True)
    except ValueError:
        lexer = get_lexer_by_name("text", stripall=True)
    formatter = HtmlFormatter(style="monokai", nowrap=True)
    return f"<div>{highlight(code, lexer, formatter)}</div>"
