import asyncio
import subprocess


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

    async def _submit_coro(self, job_id, fname):
        await asyncio.sleep(self.startup_delay)
        while self.queue_is_full():
            await asyncio.sleep(self.refresh_interval)
        self._submit(job_id, fname)

    def queue_is_full(self):
        n_running = sum(info["status"] == "R" for info in self.current_queue.values())
        return n_running >= self.max_running_jobs

    def get_new_job_id(self):
        job_id = self._job_id
        self._job_id += 1
        return job_id

    def submit(self, fname):
        job_id = self.get_new_job_id()
        self.current_queue[job_id] = {"proc": None, "status": "P"}
        self.ioloop.create_task(self._submit_coro(job_id, fname))

    def _submit(self, job_id, fname):
        proc = subprocess.Popen([self.python_executable, fname], stdout=subprocess.PIPE)
        info = self.current_queue[job_id]
        info["proc"] = proc
        info["status"] = "R"
        self.current_queue[job_id] = {"proc": proc, "status": "R"}

    def refresh(self):
        for job_id, info in self.current_queue.items():
            if info["status"] == "R" and info["proc"].poll() is not None:
                info["status"] = "F"

    async def _refresh_coro(self):
        while True:
            try:
                await asyncio.sleep(self.refresh_interval)
                self.refresh()
            except Exception as e:
                print(e)
