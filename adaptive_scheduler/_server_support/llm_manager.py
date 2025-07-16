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
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from .base_manager import BaseManager

if TYPE_CHECKING:
    from collections.abc import Callable

    from .database_manager import DatabaseManager


class ApprovalTool(BaseTool):
    """A tool that requires approval before running."""

    tool: BaseTool
    llm_manager: LLMManager = Field(exclude=True)

    name: str = ""
    description: str = ""
    args_schema: type[BaseModel] | None = None

    def __init__(self, tool: BaseTool, llm_manager: LLMManager, **kwargs: Any) -> None:
        super().__init__(tool=tool, llm_manager=llm_manager, **kwargs)
        self.name = tool.name
        self.description = tool.description
        self.args_schema = tool.args_schema

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Run the tool with approval."""
        if not self.llm_manager.yolo:
            await self.llm_manager.approval_event.wait()  # Wait for UI to be ready
            if self.llm_manager.ask_approval:
                tool_input = args or kwargs
                msg = (
                    f"The AI wants to run the tool `{self.name}` with input"
                    f" `{tool_input}`. Type 'approve' to allow."
                )
                self.llm_manager.ask_approval(msg)
                approval = await self.llm_manager.approval_queue.get()
                if approval.lower() != "approve":
                    return "Action cancelled by user."

        return await self.tool._arun(*args, **kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the tool with approval."""
        # This tool is async only
        msg = "This tool does not support sync execution."
        raise NotImplementedError(msg)


class LLMManagerKwargs(TypedDict, total=False):
    """Type for LLMManager keyword arguments."""

    model_name: str
    model_provider: str
    working_dir: str | Path
    yolo: bool
    ask_approval: Callable[[str], None] | None


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
        ask_approval: Callable[[str], None] | None = None,
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
        tools = [ApprovalTool(tool, self) for tool in self.toolkit.get_tools()]
        self.memory = MemorySaver()
        system_message = (
            "You are an expert software developer assistant. Your purpose is to help"
            " users analyze and debug failed jobs. You have access to tools for file"
            " management. When you identify a fix, you should use the `write_file`"
            " tool to apply it."
        )
        if not yolo:
            system_message += (
                " The user will be asked for approval before the tool is run."
                " After proposing a fix, end your message with 'Type \"approve\" to apply this fix.'."
            )
        else:
            system_message += " You are in YOLO mode. Apply any fixes without asking for approval."
        system_message += " The working directory is the root of the `adaptive-scheduler` repo."

        self.agent_executor = create_react_agent(
            self.llm,
            tools,
            checkpointer=self.memory,
        )
        self.agent_executor.system_message = system_message
        self.yolo = yolo
        self.ask_approval = ask_approval
        self.approval_queue: asyncio.Queue[str] = asyncio.Queue()
        self.approval_event = asyncio.Event()

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
            "Analyze the following job log to determine the cause of failure. If the"
            " cause is a code error, provide the corrected code.\n\nLog"
            f" file(s):\n```\n{log_content}\n```"
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
        message: str,
        thread_id: str = "1",
        run_metadata: dict | None = None,
    ) -> str:
        """Handles a chat message and returns a response."""
        config = {
            "configurable": {"thread_id": thread_id},
            "run_name": "LLM Manager Chat",
            "run_id": uuid.uuid4(),
            "tags": ["llm_manager"],
            "metadata": run_metadata or {},
        }
        config["metadata"]["thread_id"] = thread_id  # type: ignore[index]
        response = await self.agent_executor.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config,
        )
        return response["messages"][-1].content

    def provide_approval(self, message: str) -> None:
        """Provide approval for a tool to run."""
        self.approval_queue.put_nowait(message)
