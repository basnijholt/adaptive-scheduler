"""Tests for conftest module."""
from __future__ import annotations

import os
import textwrap
from contextlib import contextmanager
from typing import TYPE_CHECKING

from adaptive_scheduler.scheduler import BaseScheduler
from adaptive_scheduler.utils import _deserialize, _serialize

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from pathlib import Path
    from typing import Any, ClassVar

    import zmq.asyncio


@contextmanager
def temporary_working_directory(path: Path) -> Generator[None, None, None]:
    """Context manager for temporarily changing the working directory."""
    original_cwd = os.getcwd()  # noqa: PTH109
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_cwd)


class MockScheduler(BaseScheduler):
    """A mock scheduler for testing purposes."""

    _ext: ClassVar[str] = ".mock"
    _submit_cmd: ClassVar[str] = "echo"
    _options_flag: ClassVar[str] = "#MOCK"
    _cancel_cmd: ClassVar[str] = "echo"

    def __init__(self, **kw: Any) -> None:
        """Initialize the mock scheduler."""
        super().__init__(**kw)
        self._queue_info: dict[str, dict[str, Any]] = {}
        self._started_jobs: list[str] = []
        self._job_id = 0

    def queue(
        self,
        *,
        me_only: bool = True,  # noqa: ARG002
    ) -> dict[str, dict]:
        """Return a fake queue for demonstration purposes."""
        print("Mock queue:", self._queue_info)
        return self._queue_info

    def job_script(self) -> str:
        """Return a job script for the mock scheduler."""
        job_script = textwrap.dedent(
            f"""\
            #!/bin/bash
            #MOCK --cores {self.cores}
            {{extra_scheduler}}

            export MKL_NUM_THREADS={self.num_threads}
            export OPENBLAS_NUM_THREADS={self.num_threads}
            export OMP_NUM_THREADS={self.num_threads}
            export NUMEXPR_NUM_THREADS={self.num_threads}

            {{extra_env_vars}}

            {{extra_script}}

            {{executor_specific}}
            """,
        )
        job_script = job_script.format(
            extra_scheduler=self.extra_scheduler,
            extra_env_vars=self.extra_env_vars,
            extra_script=self.extra_script,
            executor_specific=self._executor_specific("${NAME}"),
        )
        return job_script

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

        def to_cancel(job_names: Iterable[str]) -> list[str]:
            return [
                job_id
                for job_id, info in self.queue().items()
                if info["job_name"] in job_names
            ]

        for job_id in to_cancel(job_names):
            self._queue_info.pop(job_id, None)

        for job_name in job_names:
            if job_name in self._started_jobs:
                self._started_jobs.remove(job_name)

    def update_queue(self, job_name: str, status: str) -> None:
        """Update the queue with the given job_name and status."""
        self._queue_info[job_name] = {"job_name": job_name, "status": status}


PARTITIONS = {
    "hb120v2-low": 120,
    "hb60-high": 60,
    "nc24-low": 24,
    "nd40v2-mpi": 40,
}


async def send_message(socket: zmq.asyncio.Socket, message: Any) -> Any:
    """Send a message to the socket and return the response."""
    await socket.send_serialized(message, _serialize)
    return await socket.recv_serialized(_deserialize)
