"""Tests for conftest module."""

import textwrap
from typing import Any

from adaptive_scheduler.scheduler import BaseScheduler


class MockScheduler(BaseScheduler):
    """A mock scheduler for testing purposes."""

    _ext = ".mock"
    _submit_cmd = "echo"
    _options_flag = "#MOCK"
    _cancel_cmd = "echo"

    def __init__(self, **kw: Any) -> None:
        """Initialize the mock scheduler."""
        super().__init__(**kw)
        self._queue_info = {}
        self._started_jobs = []
        self._job_id = 0

    def queue(
        self,
        me_only: bool = True,  # noqa: FBT001, FBT002, ARG002
    ) -> dict[str, dict]:
        """Return a fake queue for demonstration purposes."""
        print("Mock queue:", self._queue_info)
        return self._queue_info

    def job_script(self) -> str:
        """Return a job script for the mock scheduler."""
        return textwrap.dedent(
            f"""\
            {self.extra_scheduler}
            {self.extra_env_vars}

            {self.extra_script}

            {self._executor_specific("MOCK_JOB")}
            """,
        )

    def start_job(self, name: str) -> None:
        """Start a mock job."""
        print("Starting a mock job:", name)
        self._started_jobs.append(name)
        self._queue_info[str(self._job_id)] = {
            "job_name": name,
            "status": "R",
            "state": "RUNNING",
        }
        self._job_id += 1

    def cancel(
        self,
        job_names: list[str],
        with_progress_bar: bool = True,  # noqa: FBT001, ARG002, FBT002
        max_tries: int = 5,  # noqa: ARG002
    ) -> None:
        """Cancel mock jobs."""
        print("Canceling mock jobs:", job_names)
        for job_name in job_names:
            self._queue_info.pop(job_name, None)
            self._started_jobs.remove(job_name)

    def update_queue(self, job_name: str, status: str) -> None:
        """Update the queue with the given job_name and status."""
        self._queue_info[job_name] = {"job_name": job_name, "status": status}
