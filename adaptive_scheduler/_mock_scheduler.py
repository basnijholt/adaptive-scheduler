#!/usr/bin/env python
from __future__ import annotations

import asyncio
import datetime
import logging
import os
import signal
import subprocess
from typing import TYPE_CHECKING

import structlog
import zmq
import zmq.asyncio
from toolz.dicttoolz import dissoc

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from typing import Any

ctx = zmq.asyncio.Context()

logger = logging.getLogger("adaptive_scheduler._mock_scheduler")
logger.setLevel(logging.INFO)
log = structlog.wrap_logger(logger)

DEFAULT_URL = "tcp://127.0.0.1:60547"

_RequestSubmitType = tuple[str, str, str | list[str]]
_RequestCancelType = tuple[str, str]
_RequestQueueType = tuple[str]


class MockScheduler:
    """Emulates a HPC-like scheduler.

    Start an instance of `MockScheduler` and then you are able to do
    ```
    python _mock_scheduler.py --queue
    python _mock_scheduler.py --submit JOB_NAME_HERE script_here.sh  # returns JOB_ID
    python _mock_scheduler.py --cancel JOB_ID
    ```

    Parameters
    ----------
    startup_delay
        The waiting before starting the process.
    max_running_jobs
        Maximum number of simultaneously running jobs.
    refresh_interval
        Refresh interval of checking whether proccesses are still running.
    bash
        ``bash`` executable.
    url
        The URL of the socket. Defaults to {DEFAULT_URL}.

    """

    def __init__(
        self,
        *,
        startup_delay: int = 3,
        max_running_jobs: int = 4,
        refresh_interval: float = 0.1,
        bash: str = "bash",
        url: str | None = None,
    ) -> None:
        self._current_queue: dict[str, dict[str, Any]] = {}
        self._job_id = 0
        self.max_running_jobs = max_running_jobs
        self.startup_delay = startup_delay
        self.refresh_interval = refresh_interval
        self.bash = bash
        self.ioloop = asyncio.get_event_loop()
        self.refresh_task = self.ioloop.create_task(self._refresh_coro())
        self.url = url or DEFAULT_URL
        self.command_listener_task = self.ioloop.create_task(self._command_listener())

    def queue(
        self,
        *,
        me_only: bool = True,  # noqa: ARG002
    ) -> dict[str, dict[str, Any]]:
        """Return the current queue."""
        # me_only doesn't do anything, but the argument is there
        # because it is in the other schedulers.

        # remove the "proc" entries because they aren't pickable
        return {job_id: dissoc(info, "proc") for job_id, info in self._current_queue.items()}

    def _queue_is_full(self) -> bool:
        n_running = sum(info["state"] == "R" for info in self._current_queue.values())
        return n_running >= self.max_running_jobs

    def _get_new_job_id(self) -> str:
        job_id = self._job_id
        self._job_id += 1
        return str(job_id)

    async def _submit_coro(self, job_name: str, job_id: str, fname: str) -> None:
        await asyncio.sleep(self.startup_delay)
        while self._queue_is_full():
            await asyncio.sleep(self.refresh_interval)
        self._submit(job_name, job_id, fname)

    def _submit(self, job_name: str, job_id: str, fname: str) -> None:
        if job_id in self._current_queue:
            # job_id could be cancelled before it started
            cmd = f"{self.bash} {fname}"
            proc = subprocess.Popen(
                cmd.split(),
                stdout=subprocess.PIPE,
                env=dict(os.environ, JOB_ID=job_id, NAME=job_name),
                # Set a new process group for the process
                preexec_fn=os.setpgrp,  # noqa: PLW1509
            )
            info = self._current_queue[job_id]
            info["proc"] = proc
            info["state"] = "R"

    def submit(self, job_name: str, fname: str) -> str:
        job_id = self._get_new_job_id()
        self._current_queue[job_id] = {
            "job_name": job_name,
            "proc": None,
            "state": "P",
            "timestamp": str(datetime.datetime.now()),  # noqa: DTZ005
        }
        self.ioloop.create_task(self._submit_coro(job_name, job_id, fname))
        return job_id

    def cancel(self, job_id: str) -> None:
        job_id = str(job_id)
        info = self._current_queue.pop(job_id)
        if info["proc"] is not None:
            os.killpg(
                os.getpgid(info["proc"].pid),
                signal.SIGTERM,
            )  # Kill the process group

    async def _refresh_coro(self) -> Coroutine[None, None, None]:
        while True:
            try:
                await asyncio.sleep(self.refresh_interval)
                self._refresh()
            except Exception as e:  # noqa: BLE001, PERF203
                print(e)

    def _refresh(self) -> None:
        for info in self._current_queue.values():
            if info["state"] == "R" and info["proc"].poll() is not None:
                info["state"] = "F"

    async def _command_listener(self) -> Coroutine[None, None, None]:
        log.debug("started _command_listener")
        socket = ctx.socket(zmq.REP)
        socket.bind(self.url)
        try:
            while True:
                request = await socket.recv_pyobj()
                reply = self._dispatch(request)
                await socket.send_pyobj(reply)
        finally:
            socket.close()

    def _dispatch(
        self,
        request: _RequestSubmitType | _RequestCancelType | _RequestQueueType,
    ) -> str | None | dict[str, dict[str, Any]] | Exception:
        log.debug("got a request", request=request)
        request_type, *request_arg = request
        try:
            if request_type == "submit":
                job_name, fname = request_arg
                log.debug("submitting a task", fname=fname, job_name=job_name)
                return self.submit(job_name, fname)  # type: ignore[arg-type]
            if request_type == "cancel":
                job_id = request_arg[0]  # type: ignore[assignment]
                log.debug("got a cancel request", job_id=job_id)
                self.cancel(job_id)  # type: ignore[arg-type]
                return None
            if request_type == "queue":
                log.debug("got a queue request")
                return self._current_queue
            log.debug("got unknown request")
        except Exception as e:  # noqa: BLE001
            return e
        msg = f"unknown request_type: {request_type}"
        raise ValueError(msg)


def _external_command(command: tuple[str, ...], url: str) -> Any:
    async def _coro(command: tuple[str, ...], url: str) -> None:
        with ctx.socket(zmq.REQ) as socket:
            socket.setsockopt(zmq.RCVTIMEO, 2000)
            socket.connect(url)
            await socket.send_pyobj(command)
            return await socket.recv_pyobj()

    coro = _coro(command, url)
    ioloop = asyncio.get_event_loop()
    task = ioloop.create_task(coro)
    return ioloop.run_until_complete(task)


def queue(url: str = DEFAULT_URL) -> dict[str, dict[str, Any]]:
    return _external_command(("queue",), url)


def submit(job_name: str, fname: str, url: str = DEFAULT_URL) -> Any:
    return _external_command(("submit", job_name, fname), url)


def cancel(job_id: str, url: str = DEFAULT_URL) -> None:
    return _external_command(("cancel", job_id), url)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", action="store_true")
    parser.add_argument("--submit", action="store", nargs=2, type=str)
    parser.add_argument("--cancel", action="store", type=str)
    parser.add_argument("--url", action="store", type=str, default=DEFAULT_URL)
    args = parser.parse_args()

    if args.queue:
        print(queue(args.url))
    elif args.submit:
        job_name, fname = args.submit
        print(submit(job_name, fname, args.url))
    elif args.cancel:
        job_id = args.cancel
        cancel(job_id, args.url)
        print("Cancelled")
