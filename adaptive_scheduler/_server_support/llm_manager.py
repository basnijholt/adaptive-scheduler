from __future__ import annotations

import asyncio
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
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.db_manager = db_manager
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

    def _get_log_file_path(self, job_id: str) -> str | None:
        """Get the log file path from the database."""
        for job in self.db_manager.as_dicts():
            if job["job_id"] == job_id:
                return job["log_fname"]
        return None

    async def _read_log_file(self, log_path: str) -> str:
        # In a real implementation, this would read the content of the log file
        try:
            async with aiofiles.open(log_path) as f:
                return await f.read()
        except FileNotFoundError:
            return "Log file not found."

    async def diagnose_failed_job(self, job_id: str) -> str:
        """Analyzes the log file of a failed job and returns a diagnosis."""
        if job_id in self._diagnoses_cache:
            return self._diagnoses_cache[job_id]

        log_path = self._get_log_file_path(job_id)
        if not log_path:
            return f"Could not find log file for job {job_id}"

        log_content = await self._read_log_file(log_path)
        if log_content == "Log file not found.":
            return log_content

        messages: list[BaseMessage] = [
            SystemMessage(
                content="You are a helpful assistant that analyzes job failure logs.",
            ),
            HumanMessage(
                content=f"Analyze the following log and determine the cause of failure:\n\n{log_content}",
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
