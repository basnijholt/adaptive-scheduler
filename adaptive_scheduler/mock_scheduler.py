import asyncio
import datetime
import logging
import os
import subprocess
from typing import Coroutine, Tuple

import structlog
import zmq
import zmq.asyncio

logger = logging.getLogger("adaptive_scheduler.mock_scheduler")
logger.setLevel(logging.INFO)
log = structlog.wrap_logger(logger)

DEFAULT_URL = "tcp://127.0.0.1:60547"


class MockScheduler:
    def __init__(
        self,
        startup_delay=3,
        max_running_jobs=4,
        refresh_interval=0.1,
        python_executable="python",
        url=None,
    ):
        self._current_queue = {}
        self._job_id = 0
        self.max_running_jobs = max_running_jobs
        self.startup_delay = startup_delay
        self.refresh_interval = refresh_interval
        self.python_executable = python_executable
        self.ioloop = asyncio.get_event_loop()
        self.refresh_task = self.ioloop.create_task(self._refresh_coro())
        self.url = url or DEFAULT_URL
        self.command_listener_task = self.ioloop.create_task(self._command_listener())

    def queue(self, only_me: bool = True):
        # only_me doesn't do anything, but the argument is there
        # because it is in the other schedulers.
        return self._current_queue

    def _queue_is_full(self):
        n_running = sum(info["status"] == "R" for info in self._current_queue.values())
        return n_running >= self.max_running_jobs

    def _get_new_job_id(self):
        job_id = self._job_id
        self._job_id += 1
        return str(job_id)

    async def _submit_coro(self, job_id: str, fname: str):
        await asyncio.sleep(self.startup_delay)
        while self._queue_is_full():
            await asyncio.sleep(self.refresh_interval)
        self._submit(job_id, fname)

    def _submit(self, job_id: str, fname: str):
        if job_id in self._current_queue:
            # job_id could be cancelled before it started
            proc = subprocess.Popen(
                [self.python_executable, fname],
                stdout=subprocess.PIPE,
                env=dict(os.environ, JOB_ID=job_id),
            )
            info = self._current_queue[job_id]
            info["proc"] = proc
            info["status"] = "R"

    def submit(self, job_name: str, fname: str):
        job_id = self._get_new_job_id()
        self._current_queue[job_id] = {
            "job_name": job_name,
            "proc": None,
            "status": "P",
            "timestamp": str(datetime.datetime.now()),
        }
        self.ioloop.create_task(self._submit_coro(job_id, fname))
        return job_id

    def cancel(self, job_id: str):
        job_id = str(job_id)
        info = self._current_queue.pop(job_id)
        if info["proc"] is not None:
            info["proc"].kill()

    async def _refresh_coro(self):
        while True:
            try:
                await asyncio.sleep(self.refresh_interval)
                self.refresh()
            except Exception as e:
                print(e)

    def refresh(self):
        for job_id, info in self._current_queue.items():
            if info["status"] == "R" and info["proc"].poll() is not None:
                info["status"] = "F"

    async def _command_listener(self) -> Coroutine:
        log.debug("started _command_listener")
        ctx = zmq.asyncio.Context()
        socket = ctx.socket(zmq.REP)
        socket.bind(self.url)
        try:
            while True:
                request = await socket.recv_pyobj()
                reply = self._dispatch(request)
                await socket.send_pyobj(reply)
        finally:
            socket.close()

    def _dispatch(self, request: Tuple[str, ...]):
        log.debug("got a request", request=request)
        request_type, *request_arg = request
        try:
            if request_type == "submit":
                job_name, fname = request_arg
                log.debug("submitting a task", fname=fname, job_name=job_name)
                job_id = self.submit(job_name, fname)
                return job_id
            elif request_type == "cancel":
                job_id = request_arg[0]
                log.debug("got a cancel request", job_id=job_id)
                self.cancel(job_id)
                return None
            elif request_type == "queue":
                log.debug("got a queue request")
                return self.queue()
            else:
                log.debug("got unknown request")
        except Exception as e:
            return e


def submit(job_name: str, fname: str, url: str) -> None:
    with zmq.Context().socket(zmq.REQ) as socket:
        socket.connect(url)
        socket.send_pyobj(("submit", job_name, fname))
        job_id = socket.recv_pyobj()
        print(job_id)


def cancel(job_id: str, url: str) -> None:
    with zmq.Context().socket(zmq.REQ) as socket:
        socket.connect(url)
        socket.send_pyobj(("cancel", job_id))
        socket.recv_pyobj()
        print("Cancelled")


def queue(url: str) -> None:
    with zmq.Context().socket(zmq.REQ) as socket:
        socket.connect(url)
        socket.send_pyobj(("queue",))
        queue = socket.recv_pyobj()
        print(queue)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", action="store_true")
    parser.add_argument("--submit", action="store", nargs=2, type=str)
    parser.add_argument("--cancel", action="store", type=str)
    parser.add_argument("--url", action="store", type=str, default=DEFAULT_URL)
    args = parser.parse_args()

    if args.queue:
        queue(args.url)
    elif args.submit:
        job_name, fname = args.submit
        submit(job_name, fname, args.url)
    elif args.cancel:
        job_id = args.cancel
        cancel(job_id, args.url)
