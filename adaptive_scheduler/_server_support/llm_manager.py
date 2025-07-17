from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import aiofiles
from langchain_community.agent_toolkits.file_management.toolkit import (
    FileManagementToolkit,
)
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from .base_manager import BaseManager

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.messages import BaseMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph

    from .database_manager import DatabaseManager


# Constants
CONTENT_PREVIEW_LENGTH = 100


@dataclass
class ChatResult:
    """Result of a chat operation."""

    content: str
    interrupted: bool = False
    interrupt_message: str | None = None
    thread_id: str | None = None

    @property
    def needs_approval(self) -> bool:
        """Check if this result requires human approval."""
        return self.interrupted and self.interrupt_message is not None


def _format_response_content(content: Any) -> str:
    """Format response content, handling both strings and lists."""
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return content or ""


async def _execute_agent(
    agent_executor: CompiledStateGraph,
    input_data: Any,
    thread_id: str = "1",
    run_metadata: dict[str, Any] | None = None,
    run_name: str = "LLM Manager",
) -> ChatResult:
    """Execute the agent with input data and return a ChatResult."""
    metadata = run_metadata or {}
    metadata["thread_id"] = thread_id
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "run_name": run_name,
        "run_id": uuid.uuid4(),
        "tags": ["llm_manager"],
        "metadata": metadata,
    }

    response = await agent_executor.ainvoke(input_data, config)

    # Check if the graph is in an interrupted state (LangGraph official way)
    if "__interrupt__" in response:
        interrupt_info = response["__interrupt__"][0]  # Get first interrupt
        interrupt_msg = interrupt_info.value.get("message", "Approval needed")
        return ChatResult(
            content="",
            interrupted=True,
            interrupt_message=interrupt_msg,
            thread_id=thread_id,
        )

    # Get the last message content
    content = ""
    if response["messages"]:
        content = response["messages"][-1].content
        content = _format_response_content(content)

    return ChatResult(content=content, thread_id=thread_id)


async def chat(
    agent_executor: CompiledStateGraph,
    message: str | list[ToolMessage] | bool,
    thread_id: str = "1",
    run_metadata: dict[str, Any] | None = None,
) -> ChatResult:
    """Handle a chat message and return a response."""
    if isinstance(message, str):
        payload = {"messages": [HumanMessage(content=message)]}
    elif isinstance(message, bool):
        payload = Command(resume="approved" if message else "denied")
    else:
        payload = {"messages": message}

    return await _execute_agent(
        agent_executor,
        payload,
        thread_id,
        run_metadata,
        "LLM Manager Chat",
    )


def create_agent_graph(llm: BaseLanguageModel, tools: list[Any], *, yolo: bool) -> StateGraph:
    """Create the LangGraph agent with approval flow."""
    graph = StateGraph(MessagesState)

    async def call_model(state: MessagesState) -> dict[str, list]:
        messages = state["messages"]
        response = await llm.bind_tools(tools).ainvoke(messages)
        return {"messages": [response]}

    # Add nodes
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("human_approval", _human_approval_node)

    # Add edges
    graph.add_conditional_edges("agent", _create_should_continue_function(yolo=yolo))
    graph.add_edge("tools", "agent")
    graph.add_edge("human_approval", "tools")
    graph.set_entry_point("agent")

    return graph


class LLMManagerKwargs(TypedDict, total=False):
    """Type for LLMManager keyword arguments."""

    model_name: str
    model_provider: str
    working_dir: str | Path
    yolo: bool


class LLMManager(BaseManager):
    """A manager for handling interactions with a language model."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        model_name: str = "gpt-4",
        model_provider: str = "openai",
        move_old_logs_to: Path | None = None,
        working_dir: str | Path = ".",
        *,
        yolo: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.db_manager = db_manager
        self.move_old_logs_to = move_old_logs_to
        if model_provider == "openai":
            self.llm = ChatOpenAI(model_name=model_name, **kwargs)
        elif model_provider == "google":
            self.llm = ChatGoogleGenerativeAI(model=model_name, **kwargs)
        else:
            msg = f"Unknown model provider: {model_provider}"
            raise ValueError(msg)
        self._diagnoses_cache: dict[str, str] = {}
        self.toolkit = FileManagementToolkit(
            root_dir=str(working_dir),
            selected_tools=["read_file", "write_file", "list_directory", "move_file"],
        )
        tools = self.toolkit.get_tools()

        self.memory = MemorySaver()

        # Create the agent graph using extracted functions
        graph = create_agent_graph(self.llm, tools, yolo=yolo)
        self.agent_executor: CompiledStateGraph = graph.compile(checkpointer=self.memory)
        self.yolo = yolo

    async def _manage(self) -> None:
        """The main loop for the manager."""
        if self.task is None:
            return
        while not self.task.done():  # noqa: ASYNC110
            await asyncio.sleep(1)

    async def diagnose_failed_job(self, job_id: str) -> ChatResult:
        """Analyzes the log file of a failed job and returns a diagnosis result."""
        if job_id in self._diagnoses_cache:
            return ChatResult(content=self._diagnoses_cache[job_id], thread_id=job_id)

        log_paths = _get_log_file_paths(self.db_manager, job_id, self.move_old_logs_to)
        if not log_paths:
            return ChatResult(
                content=f"Could not find log files for job {job_id}",
                thread_id=job_id,
            )

        log_content = await _read_log_files_async(log_paths)
        if "Log file not found" in log_content and not any(
            "Log file not found" not in c for c in log_content.splitlines()
        ):
            return ChatResult(content=log_content, thread_id=job_id)

        initial_message = _create_diagnosis_prompt(log_content)
        # Use job_id as thread_id and pass job_id in metadata for better tracking
        run_metadata = {"job_id": job_id}
        result = await chat(
            self.agent_executor,
            initial_message,
            thread_id=job_id,
            run_metadata=run_metadata,
        )

        # Only cache completed diagnoses, not interrupted ones
        if not result.interrupted:
            self._diagnoses_cache[job_id] = result.content

        return result

    async def chat(
        self,
        message: str | list[ToolMessage] | bool,
        thread_id: str = "1",
        run_metadata: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Handle a chat message and return a response."""
        return await chat(self.agent_executor, message, thread_id, run_metadata)

    def get_history(self, thread_id: str) -> MessagesState:
        """Get the message history for a thread."""
        return self.agent_executor.get_state({"configurable": {"thread_id": thread_id}}).values


def _extract_write_operations(last_message: ToolMessage) -> list[str]:
    """Extract write operations from tool calls for approval context."""
    write_ops = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        if tool_name in ["write_file", "move_file"]:
            args = tool_call["args"]
            if tool_name == "write_file":
                file_path = args["file_path"]
                content = args["text"]
                # Show preview of content for context
                content_preview = (
                    content[:CONTENT_PREVIEW_LENGTH] + "..."
                    if len(content) > CONTENT_PREVIEW_LENGTH
                    else content
                )
                write_ops.append(f"write to {file_path}:\n```\n{content_preview}\n```")
            elif tool_name == "move_file":
                src = args["src_path"]
                dst = args["new_path"]
                write_ops.append(f"move {src} to {dst}")
    return write_ops


def _needs_approval(last_message: BaseMessage, *, yolo: bool) -> bool:
    """Check if the message contains tool calls that need approval."""
    if yolo or not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return False

    return any(
        tool_call.get("name", "") in ["write_file", "move_file"]
        for tool_call in last_message.tool_calls
    )


def _human_approval_node(state: MessagesState) -> dict[str, list]:
    """Node that requests human approval for write operations."""
    messages = state["messages"]
    last_message = messages[-1]

    # Extract write operations for detailed approval message
    assert isinstance(last_message, AIMessage), last_message
    write_ops = _extract_write_operations(last_message)

    # Create approval message with operation details
    approval_message = "Approve these operations?\n\n" + "\n\n".join(write_ops)

    # Interrupt for human approval
    decision = interrupt({"message": approval_message, "operations": write_ops})

    if decision != "approved":
        # Add denial message if not approved
        denial_msg = HumanMessage(content="Operations denied by user.")
        return {"messages": [denial_msg]}

    # If approved, don't add any messages, just return empty dict
    return {}


def _create_should_continue_function(*, yolo: bool) -> Callable[[MessagesState], str]:
    """Create the conditional edge function for the graph."""

    def should_continue(state: MessagesState) -> str:
        last_message = state["messages"][-1]

        # Only AI messages can have tool calls
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return END

        # Check if any tool call needs approval
        if _needs_approval(last_message, yolo=yolo):
            return "human_approval"

        return "tools"

    return should_continue


def _create_diagnosis_prompt(log_content: str) -> str:
    """Create the initial prompt for job failure diagnosis."""
    return (
        "Analyze this job failure log and provide a diagnosis with a fix.\n\n"
        "If you can identify the problem from the log alone, provide the corrected code.\n"
        "You can freely read files without asking for permission. For write operations, proceed directly with the file tools - approval will be handled automatically.\n\n"
        f"Log file(s):\n```\n{log_content}\n```\n\n"
        "What caused this failure and how can it be fixed?"
    )


async def _read_log_files_async(log_paths: list[Path]) -> str:
    """Read and combine the content of multiple log files asynchronously."""
    log_contents = []
    for log_path in log_paths:
        try:
            async with aiofiles.open(log_path) as f:
                log_contents.append(await f.read())
        except FileNotFoundError:  # noqa: PERF203
            log_contents.append(f"Log file not found: {log_path}")
    return "\n".join(log_contents)


def _get_log_file_paths(
    db_manager: DatabaseManager,
    job_id: str,
    move_old_logs_to: Path | None = None,
) -> list[Path]:
    """Get the log file paths from the database."""
    for job in db_manager.failed:
        if job["job_id"] == job_id:
            output_logs = [Path(log) for log in job["output_logs"]]
            log_paths = []
            for log_path in output_logs:
                if log_path.exists():
                    log_paths.append(log_path)
                elif move_old_logs_to:
                    log_path_alt = move_old_logs_to / log_path.name
                    if log_path_alt.exists():
                        log_paths.append(log_path_alt)
            return log_paths
    return []
