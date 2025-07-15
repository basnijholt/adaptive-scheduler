from __future__ import annotations

import asyncio
import random

import aiofiles

from .base_manager import BaseManager


class LLMManager(BaseManager):
    """A manager for handling interactions with a language model."""

    def __init__(self) -> None:
        super().__init__()
        self._diagnoses_cache: dict[str, str] = {}
        self._chat_history: list[dict[str, str]] = []

    async def _manage(self) -> None:
        """The main loop for the manager."""
        while not self.task.done():
            await asyncio.sleep(1)

    def _get_log_file_path(self, job_id: str) -> str | None:
        # In a real implementation, this would look up the log file path from the database
        return f"logs/job_{job_id}.log"

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
        # In a real implementation, this would be an API call to an LLM
        diagnosis = await self._simulate_llm_call(
            f"Analyze the following log and determine the cause of failure:\n\n{log_content}",
        )
        self._diagnoses_cache[job_id] = diagnosis
        return diagnosis

    async def chat(self, message: str) -> str:
        """Handles a chat message and returns a response."""
        self._chat_history.append({"role": "user", "content": message})
        # In a real implementation, this would be an API call to an LLM
        response = await self._simulate_llm_call(str(self._chat_history))
        self._chat_history.append({"role": "assistant", "content": response})
        return response

    async def _simulate_llm_call(self, prompt: str) -> str:
        """Simulates a call to a language model."""
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Simulate network latency
        responses = [
            "It seems like there was a `FileNotFoundError`. Check if the input files are correctly specified.",
            "The job failed due to a `MemoryError`. Try requesting more memory for your job.",
            "I see a `ValueError` in the logs. It seems like an invalid argument was passed to a function.",
            "The simulation diverged. You might want to adjust the simulation parameters.",
            "I'm not sure what went wrong. Could you provide more details?",
        ]
        return random.choice(responses)
