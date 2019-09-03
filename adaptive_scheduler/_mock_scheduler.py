import asyncio
import datetime
import logging
import os
import subprocess
from typing import Coroutine, Tuple

import structlog
import zmq
import zmq.asyncio
from toolz.dicttoolz import dissoc

ctx = zmq.asyncio.Context()

logger = logging.getLogger("adaptive_scheduler._mock_scheduler")
logger.setLevel(logging.INFO)
log = structlog.wrap_logger(logger)

DEFAULT_URL = "tcp://127.0.0.1:60547"


class MockScheduler:
    f"""Emulates a HPC-like scheduler.

    Start an instance of `MockScheduler` and then you are able to do
    ```
    python _mock_scheduler.py --queue
    python _mock_scheduler.py --submit JOB_NAME_HERE script_here.sh  # returns JOB_ID
    python _mock_scheduler.py --cancel JOB_ID
    ```

    Parameters
    ----------
    startup_delay : int
        The waiting before starting the process.
    max_running_jobs : int
        Maximum number of simultaneously running jobs.
    refresh_interval : int
        Refresh interval of checking whether proccesses are still running.
    bash : str, default: "bash"
        ``bash`` executable.
    url : str, optional
        The URL of the socket. Defaults to {DEFAULT_URL}.
    """

    def __init__(
        self,
        startup_delay=3,
        max_running_jobs=4,
        refresh_interval=0.1,
        bash="bash",
        url=None,
    ):
        self._current_queue = {}
        self._job_id = 0
        self.max_running_jobs = max_running_jobs
        self.startup_delay = startup_delay
        self.refresh_interval = refresh_interval
        self.bash = bash
        self.ioloop = asyncio.get_event_loop()
        self.refresh_task = self.ioloop.create_task(self._refresh_coro())
        self.url = url or DEFAULT_URL
        self.command_listener_task = self.ioloop.create_task(self._command_listener())

    def queue(self, only_me: bool = True):
        # only_me doesn't do anything, but the argument is there
        # because it is in the other schedulers.

        # remove the "proc" entries because they aren't pickable
        return {
            job_id: dissoc(info, "proc") for job_id, info in self._current_queue.items()
        }

    def _queue_is_full(self):
        n_running = sum(info["state"] == "R" for info in self._current_queue.values())
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
            cmd = f"{self.bash} {fname}"
            proc = subprocess.Popen(
                cmd.split(), stdout=subprocess.PIPE, env=dict(os.environ, JOB_ID=job_id)
            )
            info = self._current_queue[job_id]
            info["proc"] = proc
            info["state"] = "R"

    def submit(self, job_name: str, fname: str):
        job_id = self._get_new_job_id()
        self._current_queue[job_id] = {
            "job_name": job_name,
            "proc": None,
            "state": "P",
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
                self._refresh()
            except Exception as e:
                print(e)

    def _refresh(self):
        for job_id, info in self._current_queue.items():
            if info["state"] == "R" and info["proc"].poll() is not None:
                info["state"] = "F"

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
                return self._current_queue
            else:
                log.debug("got unknown request")
        except Exception as e:
            return e


def _external_command(command: Tuple[str, ...], url: str):
    async def _coro(command, url) -> None:
        with ctx.socket(zmq.REQ) as socket:
            socket.setsockopt(zmq.RCVTIMEO, 2000)
            socket.connect(url)
            await socket.send_pyobj(command)
            reply = await socket.recv_pyobj()
            return reply

    coro = _coro(command, url)
    ioloop = asyncio.get_event_loop()
    task = ioloop.create_task(coro)
    return ioloop.run_until_complete(task)


def queue(url: str = DEFAULT_URL):
    return _external_command(("queue",), url)


def submit(job_name: str, fname: str, url: str = DEFAULT_URL) -> None:
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
