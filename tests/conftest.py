import textwrap

import pytest
import zmq.asyncio
from adaptive import Learner1D

from adaptive_scheduler.scheduler import BaseScheduler
from adaptive_scheduler.server_support import DatabaseManager, get_allowed_url


class MockScheduler(BaseScheduler):
    _ext = ".mock"
    _submit_cmd = "echo"
    _options_flag = "#MOCK"
    _cancel_cmd = "echo"

    def __init__(self, **kw):
        super().__init__(**kw)
        self._queue_info = {
            "1": {"job_name": "MOCK_JOB-1", "state": "R"},
            "2": {"job_name": "MOCK_JOB-2", "state": "Q"},
        }
        self._started_jobs = []

    def queue(self, me_only: bool = True) -> dict[str, dict]:
        # Return a fake queue for demonstration purposes
        return self._queue_info

    def job_script(self) -> str:
        return textwrap.dedent(
            f"""\
            {self.extra_scheduler}
            {self.extra_env_vars}

            {self.extra_script}

            {self._executor_specific("MOCK_JOB")}
            """
        )

    def start_job(self, name: str) -> None:
        print("Starting a mock job:", name)
        self._started_jobs.append(name)

    def cancel(
        self, job_names: list[str], with_progress_bar: bool = True, max_tries: int = 5
    ) -> None:
        print("Canceling mock jobs:", job_names)
        for job_name in job_names:
            self._queue_info.pop(job_name, None)
            self._started_jobs.remove(job_name)

    def update_queue(self, job_name, status):
        self._queue_info[job_name] = {"job_name": job_name, "status": status}


@pytest.fixture
def mock_scheduler():
    return MockScheduler(cores=8)


@pytest.fixture
def db_manager(mock_scheduler, learners_and_fnames):
    url = get_allowed_url()
    db_fname = "test_db.json"
    learners, fnames = learners_and_fnames
    return DatabaseManager(url, mock_scheduler, db_fname, learners, fnames)


@pytest.fixture
def learners_and_fnames():
    def func(x):
        return x**2

    learner1 = Learner1D(func, bounds=(-1, 1))
    learner2 = Learner1D(func, bounds=(-1, 1))
    learners = [learner1, learner2]
    fnames = ["learner1.pkl", "learner2.pkl"]
    return learners, fnames


@pytest.fixture(scope="function")
def socket(db_manager):
    ctx = zmq.asyncio.Context.instance()
    socket = ctx.socket(zmq.REQ)
    socket.connect(db_manager.url)
    yield socket
    socket.close()
