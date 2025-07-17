import asyncio
from collections.abc import Callable
from typing import Any

import ipywidgets as ipyw
import mistune
import rich
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from rich.console import get_console

from adaptive_scheduler.server_support import RunManager

console = get_console()


def _render_markdown(text: str) -> str:
    """Render markdown to HTML with syntax highlighting."""
    try:
        renderer = mistune.HTMLRenderer(escape=False)
        markdown = mistune.Markdown(renderer=renderer)
        return markdown(text)
    except (TypeError, AttributeError, ValueError):
        # Fallback to simple HTML formatting
        import html

        return html.escape(text).replace("\n", "<br>")


def _render_chat_message(role: str, message: str) -> str:
    """Render a chat message with a role-specific style."""
    style = {
        "user": "background-color: #dcf8c6; text-align: right; margin-left: auto;",
        "llm": "background-color: #f1f0f0; text-align: left;",
    }[role]
    return f'<div style="padding: 5px; border-radius: 5px; margin: 5px; max-width: 80%; {style}">{message}</div>'


def _create_task_wrapper(
    widget: ipyw.Widget,
) -> Callable[[Callable[[Any], Any]], Callable[[Any], None]]:
    """Create a wrapper that runs an async function when a widget changes."""

    def wrapper(func: Callable[[Any], Any]) -> Callable[[Any], None]:
        def on_change(change: Any) -> None:
            task = asyncio.create_task(func(change))
            if not hasattr(widget, "_tasks"):
                widget._tasks = set()
            widget._tasks.add(task)
            task.add_done_callback(widget._tasks.remove)

        return on_change

    return wrapper


console = rich.get_console()


def chat_widget(run_manager: RunManager) -> ipyw.VBox:  # noqa: PLR0915
    """Chat widget for interacting with the LLM."""
    import ipywidgets as ipyw

    from adaptive_scheduler.widgets import _add_title

    text_input = ipyw.Text(
        value="",
        placeholder="Ask a question...",
        description="You:",
        disabled=False,
    )
    chat_history = ipyw.HTML(
        value=_render_chat_message(
            "llm",
            _render_markdown("👋 Hello! Select a failed job to diagnose or ask me a question."),
        ),
        placeholder="",
        description="Chat:",
        layout={
            "width": "auto",
            "height": "300px",
            "border": "1px solid black",
            "word-wrap": "break-word",
            "overflow-y": "auto",
        },
    )
    yolo_checkbox = ipyw.Checkbox(description="YOLO mode", value=False, indent=False)

    def ask_approval(message: str) -> None:
        chat_history.value += _render_chat_message("llm", _render_markdown(message))

    # Track the current thread context for chat messages
    current_thread_id = "1"  # Default thread
    waiting_for_approval = False  # Track if we're waiting for approval

    @_create_task_wrapper(text_input)
    async def on_submit(sender: ipyw.Text) -> None:
        nonlocal current_thread_id, waiting_for_approval
        message = sender.value
        sender.value = ""
        if run_manager.llm_manager is None:
            chat_history.value += _render_chat_message(
                "llm",
                _render_markdown("⚠️ No LLM manager available."),
            )
            return

        # Disable input while processing
        sender.disabled = True

        run_manager.llm_manager.yolo = yolo_checkbox.value
        chat_history.value += _render_chat_message("user", _render_markdown(message))

        # Show thinking indicator
        thinking_message = _render_chat_message("llm", _render_markdown("🤔 Thinking..."))
        chat_history.value += thinking_message

        try:
            # Check if we're responding to an approval request
            if waiting_for_approval and message.lower() in [
                "approve",
                "approved",
                "deny",
                "denied",
            ]:
                approval_data = (
                    "approved" if message.lower() in ["approve", "approved"] else "denied"
                )
                result = await run_manager.llm_manager.resume_chat(
                    approval_data,
                    thread_id=current_thread_id,
                )
                waiting_for_approval = False
            else:
                # Regular chat
                result = await run_manager.llm_manager.chat(message, thread_id=current_thread_id)

            # Remove thinking indicator
            chat_history.value = chat_history.value.replace(thinking_message, "")

            # Handle the result
            if result.interrupted:
                # First show the LLM's content if there is any (the explanation)
                if result.content:
                    chat_history.value += _render_chat_message(
                        "llm",
                        _render_markdown(result.content),
                    )

                # Then add the interruption message
                chat_history.value += _render_chat_message(
                    "llm",
                    _render_markdown(
                        f"🤖 {result.interrupt_message}\n\n*Reply with 'approve' or 'deny' to continue.*",
                    ),
                )
                waiting_for_approval = True
            else:
                # Add regular response
                chat_history.value += _render_chat_message("llm", _render_markdown(result.content))
        except (ValueError, TypeError, RuntimeError) as e:
            # Remove thinking indicator and add error
            chat_history.value = chat_history.value.replace(thinking_message, "")
            console.print_exception(show_locals=True)
            chat_history.value += _render_chat_message("llm", _render_markdown(f"❌ Error: {e}"))
        finally:
            # Re-enable input
            sender.disabled = False

    text_input.on_submit(on_submit)

    # Add a dropdown to select a failed job
    failed_jobs = [job["job_id"] for job in run_manager.database_manager.failed]
    failed_job_dropdown = ipyw.Dropdown(
        options=failed_jobs,
        description="Failed Job:",
        disabled=False,
    )

    @_create_task_wrapper(failed_job_dropdown)
    async def on_failed_job_change(change: dict[str, Any]) -> None:
        nonlocal current_thread_id
        job_id = change["new"]
        if job_id is None:
            # No job selected, reset to default thread
            current_thread_id = "1"
            return

        if run_manager.llm_manager is None:
            chat_history.value = _render_chat_message(
                "llm",
                _render_markdown("⚠️ No LLM manager available."),
            )
            return

        # Set the current thread context to this job's thread
        current_thread_id = job_id

        # Disable dropdown while processing
        failed_job_dropdown.disabled = True

        # Show diagnosing indicator
        diagnosing_message = _render_chat_message(
            "llm",
            _render_markdown(f"🔍 Diagnosing job {job_id}..."),
        )
        chat_history.value = diagnosing_message

        try:
            result = await run_manager.llm_manager.diagnose_failed_job(job_id)
            # Replace diagnosing indicator with result
            if result.interrupted:
                # First show the LLM's analysis if there is any
                if result.content:
                    chat_history.value = _render_chat_message(
                        "llm",
                        _render_markdown(f"**Diagnosis for job {job_id}:**\n{result.content}"),
                    )

                # Then add the interruption message
                chat_history.value += _render_chat_message(
                    "llm",
                    _render_markdown(
                        f"🤖 {result.interrupt_message}\n\n*Reply with 'approve' or 'deny' to continue.*",
                    ),
                )
            else:
                chat_history.value = _render_chat_message(
                    "llm",
                    _render_markdown(f"**Diagnosis for job {job_id}:**\n{result.content}"),
                )
        except Exception as e:  # noqa: BLE001
            # Replace diagnosing indicator with error
            console.print_exception(show_locals=True)
            chat_history.value = _render_chat_message("llm", _render_markdown(f"❌ Error: {e}"))
        finally:
            # Re-enable dropdown after processing
            failed_job_dropdown.disabled = False

    failed_job_dropdown.observe(on_failed_job_change, names="value")
    refresh_button = ipyw.Button(description="Refresh Failed Jobs")

    def refresh_failed_jobs(_: Any) -> None:
        failed_jobs = [job["job_id"] for job in run_manager.database_manager.failed]
        # Temporarily remove the observer to prevent triggering during refresh
        failed_job_dropdown.unobserve(on_failed_job_change, names="value")
        failed_job_dropdown.options = failed_jobs
        if failed_jobs:
            # Set the value to trigger the observe callback automatically
            failed_job_dropdown.value = failed_jobs[0]
        else:
            # Clear the value when there are no failed jobs
            failed_job_dropdown.value = None
        # Re-add the observer
        failed_job_dropdown.observe(on_failed_job_change, names="value")

    refresh_button.on_click(refresh_failed_jobs)

    vbox = ipyw.VBox(
        [
            refresh_button,
            failed_job_dropdown,
            yolo_checkbox,
            chat_history,
            text_input,
        ],
    )
    _add_title("adaptive_scheduler.widgets.chat_widget", vbox)
    return vbox


def _highlight_code(code: str, lang: str, _: str) -> str:
    """Highlight code blocks with pygments."""
    try:
        lexer = get_lexer_by_name(lang, stripall=True)
    except ValueError:
        lexer = get_lexer_by_name("text", stripall=True)
    formatter = HtmlFormatter(style="monokai", nowrap=True)
    return f"<div>{highlight(code, lexer, formatter)}</div>"
