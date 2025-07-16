from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiofiles
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from .base_manager import BaseManager

if TYPE_CHECKING:
    from langchain.schema import BaseMessage

    from .database_manager import DatabaseManager


class LLMManager(BaseManager):
    """A manager for handling interactions with a language model."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        model_name: str = "gpt-4",
        model_provider: str = "openai",
        move_old_logs_to: Path | None = None,
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
        self._chat_history: list[BaseMessage] = []

    async def _manage(self) -> None:
        """The main loop for the manager."""
        if self.task is None:
            return
        while not self.task.done():  # noqa: ASYNC110
            await asyncio.sleep(1)

    def _get_log_file_paths(self, job_id: str) -> list[str]:
        """Get the log file paths from the database."""
        # First check the failed jobs, then the active jobs.
        search_in = self.db_manager.failed + self.db_manager.as_dicts()
        for job in search_in:
            if job["job_id"] == job_id:
                output_logs = job["output_logs"]
                log_paths = []
                for log in output_logs:
                    if os.path.exists(log):  # noqa: PTH110
                        log_paths.append(log)
                    elif self.move_old_logs_to:
                        log_path_alt = self.move_old_logs_to / Path(log).name
                        if os.path.exists(log_path_alt):  # noqa: PTH110
                            log_paths.append(str(log_path_alt))
                return log_paths
        return []

    async def _read_log_files(self, log_paths: list[str]) -> str:
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

        job_in_db = self.db_manager._db.get(lambda j: j.job_id == job_id)
        if job_in_db is None:
            # Job is not in the main database, so it must be in the failed list
            job_in_db = next(j for j in self.db_manager.failed if j["job_id"] == job_id)
        options = self.db_manager.scheduler._multi_job_script_options(job_in_db["index"])
        job_script = self.db_manager.scheduler.job_script(options=options)

        messages: list[BaseMessage] = [
            SystemMessage(
                content="You are a helpful assistant that analyzes job failure logs.",
            ),
            HumanMessage(
                content=(
                    "Analyze the following job script and log file to determine the"
                    f" cause of failure.\n\nJob script:\n```\n{job_script}\n```\n\nLog"
                    f" file(s):\n```\n{log_content}\n```"
                ),
            ),
        ]
        response = await self.llm.agenerate([messages])
        diagnosis = response.generations[0][0].text
        self._diagnoses_cache[job_id] = diagnosis
        return diagnosis

    async def chat(self, message: str) -> str:
        """Handles a chat message and returns a response."""
        self._chat_history.append(HumanMessage(content=message))
        response = await self.llm.agenerate([self._chat_history])
        result = response.generations[0][0].text
        self._chat_history.append(SystemMessage(content=result))
        return result
