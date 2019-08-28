import asyncio
import logging
import subprocess
from typing import Any, Coroutine, Tuple

import structlog
import zmq

from adaptive_scheduler.server_support import get_allowed_url

logger = logging.getLogger("adaptive_scheduler.mock_scheduler")
logger.setLevel(logging.INFO)
log = structlog.wrap_logger(logger)

ctx = zmq.Context()


class MockScheduler:
    def __init__(
        self,
        startup_delay=3,
        max_running_jobs=4,
        refresh_interval=0.1,
        python_executable="python",
    ):
        self.current_queue = {}
        self._job_id = 0
        self.max_running_jobs = max_running_jobs
        self.startup_delay = startup_delay
        self.refresh_interval = refresh_interval
        self.python_executable = python_executable
        self.ioloop = asyncio.get_event_loop()
        self.refresh_task = self.ioloop.create_task(self._refresh_coro())
        self.url = get_allowed_url()
        self.command_listener_task = self.ioloop.create_task(self._command_listener())

    def queue(self, only_me=True):
        return self.current_queue

    def _queue_is_full(self):
        n_running = sum(info["status"] == "R" for info in self.current_queue.values())
        return n_running >= self.max_running_jobs

    def _get_new_job_id(self):
        job_id = self._job_id
        self._job_id += 1
        return job_id

    async def _submit_coro(self, job_id, fname):
        await asyncio.sleep(self.startup_delay)
        while self._queue_is_full():
            await asyncio.sleep(self.refresh_interval)
        self._submit(job_id, fname)

    def _submit(self, job_id, fname):
        proc = subprocess.Popen([self.python_executable, fname], stdout=subprocess.PIPE)
        info = self.current_queue[job_id]
        info["proc"] = proc
        info["status"] = "R"

    def submit(self, job_name, fname):
        job_id = self._get_new_job_id()
        self.current_queue[job_id] = {"job_name": job_name, "proc": None, "status": "P"}
        self.ioloop.create_task(self._submit_coro(job_id, fname))
        return job_id

    def cancel(self, job_id):
        info = self.current_queue.pop("job_id")
        info["proc"].kill()

    async def _refresh_coro(self):
        while True:
            try:
                await asyncio.sleep(self.refresh_interval)
                self.refresh()
            except Exception as e:
                print(e)

    def refresh(self):
        for job_id, info in self.current_queue.items():
            if info["status"] == "R" and info["proc"].poll() is not None:
                info["status"] = "F"

    async def _command_listener(self) -> Coroutine:
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

    def _dispatch(self, request: Tuple[str, Any]):
        log.debug("got a request", request=request)
        request_type, *request_arg = request
        try:
            if request_type == "submit":
                job_name, fname = request_arg
                log.debug("submitting a task", fname=fname, job_name=job_name)
                job_id = self.submit(fname)
                return job_id
            elif request_type == "cancel":
                job_id = request_arg[0]
                log.debug("got a cancel request", job_id=job_id)
                self.cancel(job_id)
            elif request_type == "queue":
                log.debug("got a queue request")
                return self.queue()
        except Exception as e:
            return e


def submit(job_name: str, fname: str, url: str) -> None:
    with ctx.socket(zmq.REQ) as socket:
        socket.connect(url)
        socket.send_pyobj(("submit", job_name, fname))
        job_id = socket.recv_pyobj()
        print(job_id)


def cancel(job_id: str, url: str) -> None:
    with ctx.socket(zmq.REQ) as socket:
        socket.connect(url)
        socket.send_pyobj(("cancel", job_id))
        socket.recv_pyobj()
        print("Cancelled")


def queue(url: str) -> None:
    with ctx.socket(zmq.REQ) as socket:
        socket.connect(url)
        socket.send_pyobj(("queue",))
        queue = socket.recv_pyobj()
        print(queue)
