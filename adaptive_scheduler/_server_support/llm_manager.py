from __future__ import annotations

import asyncio
import uuid
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
from langgraph.types import interrupt

from .base_manager import BaseManager

if TYPE_CHECKING:
    from .database_manager import DatabaseManager


class InterruptedException(Exception):  # noqa: N818
    """Exception raised when the LLM execution is interrupted for human input."""


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

        # Define the graph
        graph = StateGraph(MessagesState)

        async def call_model(state: MessagesState) -> dict[str, list]:
            messages = state["messages"]
            response = await self.llm.bind_tools(tools).ainvoke(messages)
            return {"messages": [response]}

        def human_approval_node(state: MessagesState):
            """Node that handles human approval requests for write operations."""
            from langgraph.types import Command

            messages = state["messages"]

            # Find the last AI message with tool calls
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    # Check if any tool call is a write operation that needs approval
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get("name", "")
                        if tool_name in ["write_file", "move_file"]:
                            # Get the action description from the tool call args
                            args = tool_call.get("args", {})
                            if tool_name == "write_file":
                                file_path = args.get("file_path", "unknown file")
                                action_description = f"write to file {file_path}"
                            elif tool_name == "move_file":
                                src = args.get("src_path", "unknown source")
                                dst = args.get("new_path", "unknown destination")
                                action_description = f"move file from {src} to {dst}"
                            else:
                                action_description = f"perform {tool_name} operation"

                            # Interrupt for human approval
                            decision = interrupt(
                                {
                                    "action": action_description,
                                    "message": f"Do you approve of this action: {action_description}",
                                },
                            )

                            if decision == "approved":
                                return Command(goto="tools")
                            # If denied, go back to agent with denial message
                            from langchain_core.messages import HumanMessage

                            denial_msg = HumanMessage(
                                content=f"Action denied by user: {action_description}",
                            )
                            return Command(goto="agent", update={"messages": [denial_msg]})

            # No write operations found, proceed to tools
            return Command(goto="tools")

        def check_approval_needed(state: MessagesState) -> str:
            """Check if human approval is needed before tool execution."""
            if yolo:
                return "tools"  # Skip approval in YOLO mode

            messages = state["messages"]

            # Check if the last AI message has tool calls that need approval
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get("name", "")
                        # Write operations need approval (read operations are auto-approved)
                        if tool_name in ["write_file", "move_file"]:
                            return "human_approval"
                    break

            return "tools"

        graph.add_node("agent", call_model)
        graph.add_node("tools", ToolNode(tools))
        graph.add_node("human_approval", human_approval_node)
        graph.add_node("check_approval", lambda state: {})

        def should_continue(state: MessagesState) -> str:
            last_message = state["messages"][-1]
            if not last_message.tool_calls:
                return END
            # Check if approval is needed before executing tools
            return "check_approval"

        graph.add_conditional_edges("agent", should_continue)
        graph.add_conditional_edges("check_approval", check_approval_needed)
        graph.add_edge("tools", "agent")
        # human_approval uses Command(goto=...) so no explicit edges needed
        graph.set_entry_point("agent")
        self.agent_executor = graph.compile(checkpointer=self.memory)
        self.yolo = yolo

    async def _manage(self) -> None:
        """The main loop for the manager."""
        if self.task is None:
            return
        while not self.task.done():  # noqa: ASYNC110
            await asyncio.sleep(1)

    def _get_log_file_paths(self, job_id: str) -> list[Path]:
        """Get the log file paths from the database."""
        for job in self.db_manager.failed:
            if job["job_id"] == job_id:
                output_logs = [Path(log) for log in job["output_logs"]]
                log_paths = []
                for log_path in output_logs:
                    if log_path.exists():
                        log_paths.append(log_path)
                    elif self.move_old_logs_to:
                        log_path_alt = self.move_old_logs_to / log_path.name
                        if log_path_alt.exists():
                            log_paths.append(log_path_alt)
                return log_paths
        return []

    async def _read_log_files(self, log_paths: list[Path]) -> str:
        """Read and combine the content of multiple log files."""
        log_contents = []
        for log_path in log_paths:
            try:
                async with aiofiles.open(log_path) as f:
                    log_contents.append(await f.read())
            except FileNotFoundError:  # noqa: PERF203
                log_contents.append(f"Log file not found: {log_path}")
        return "\n".join(log_contents)

    async def diagnose_failed_job(self, job_id: str) -> str:
        """Analyzes the log file of a failed job and returns a diagnosis."""
        if job_id in self._diagnoses_cache:
            return self._diagnoses_cache[job_id]

        log_paths = self._get_log_file_paths(job_id)
        if not log_paths:
            return f"Could not find log files for job {job_id}"

        log_content = await self._read_log_files(log_paths)
        if "Log file not found" in log_content and not any(
            "Log file not found" not in c for c in log_content.splitlines()
        ):
            return log_content

        initial_message = (
            "Analyze this job failure log and provide a diagnosis with a fix.\n\n"
            "If you can identify the problem from the log alone, provide the corrected code.\n"
            "You can freely read files without asking for permission. For write operations, ask for permission first.\n\n"
            f"Log file(s):\n```\n{log_content}\n```\n\n"
            "What caused this failure and how can it be fixed?"
        )
        # Use job_id as thread_id and pass job_id in metadata for better tracking
        run_metadata = {"job_id": job_id}
        diagnosis = await self.chat(
            initial_message,
            thread_id=job_id,
            run_metadata=run_metadata,
        )
        self._diagnoses_cache[job_id] = diagnosis
        return diagnosis

    async def chat(
        self,
        message: str | list[ToolMessage],
        thread_id: str = "1",
        run_metadata: dict | None = None,
    ) -> str:
        """Handles a chat message and returns a response."""
        metadata = run_metadata or {}
        metadata["thread_id"] = thread_id
        config = {
            "configurable": {"thread_id": thread_id},
            "run_name": "LLM Manager Chat",
            "run_id": uuid.uuid4(),
            "tags": ["llm_manager"],
            "metadata": metadata,
        }

        if isinstance(message, str):
            payload = {"messages": [HumanMessage(content=message)]}
        else:
            payload = {"messages": message}

        try:
            response = await self.agent_executor.ainvoke(
                payload,
                config,
            )
            content = response["messages"][-1].content

            # Handle case where content is a list (structured output)
            if isinstance(content, list):
                return "\n".join(str(item) for item in content)

            return content  # noqa: TRY300
        except Exception as e:
            # Check if this is an interruption that we need to handle
            if "interrupt" in str(e).lower() or "Interrupted" in str(e):
                # Re-raise as a custom exception that the UI can handle
                raise InterruptedException(str(e)) from e
            raise

    async def resume_chat(
        self,
        approval_data: dict,
        thread_id: str = "1",
        run_metadata: dict | None = None,
    ) -> str:
        """Resume an interrupted chat session with human approval."""
        from langgraph.types import Command

        metadata = run_metadata or {}
        metadata["thread_id"] = thread_id
        config = {
            "configurable": {"thread_id": thread_id},
            "run_name": "LLM Manager Resume Chat",
            "run_id": uuid.uuid4(),
            "tags": ["llm_manager"],
            "metadata": metadata,
        }

        # Resume with the approval data
        command = Command(resume=approval_data)

        response = await self.agent_executor.ainvoke(
            command,
            config,
        )
        return response["messages"][-1].content
