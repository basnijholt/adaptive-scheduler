import textwrap

import pytest
import zmq.asyncio
from adaptive import Learner1D

from adaptive_scheduler.scheduler import BaseScheduler
from adaptive_scheduler.server_support import (
    DatabaseManager,
    JobManager,
    get_allowed_url,
)


class MockScheduler(BaseScheduler):
    _ext = ".mock"
    _submit_cmd = "echo"
    _options_flag = "#MOCK"
    _cancel_cmd = "echo"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._queue_info = {}
        self._started_jobs = []
        self._job_id = 0

    def queue(self, me_only: bool = True) -> dict[str, dict]:
        # Return a fake queue for demonstration purposes
        print("Mock queue:", self._queue_info)
        return self._queue_info

    def job_script(self) -> str:
        return textwrap.dedent(
            f"""\
            {self.extra_scheduler}
            {self.extra_env_vars}

            {self.extra_script}

            {self._executor_specific("MOCK_JOB")}
            """,
        )

    def start_job(self, name: str) -> None:
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
        with_progress_bar: bool = True,  # noqa: FBT001
        max_tries: int = 5,
    ) -> None:
        print("Canceling mock jobs:", job_names)
        for job_name in job_names:
            self._queue_info.pop(job_name, None)
            self._started_jobs.remove(job_name)

    def update_queue(self, job_name: str, status: str) -> None:
        self._queue_info[job_name] = {"job_name": job_name, "status": status}


@pytest.fixture()
def mock_scheduler(tmpdir) -> MockScheduler:
    return MockScheduler(log_folder=str(tmpdir), cores=8)


@pytest.fixture()
def db_manager(
    mock_scheduler: MockScheduler,
    learners: list[Learner1D],
    fnames: list[str],
    tmp_path,
):
    url = get_allowed_url()
    db_fname = str(tmp_path / "test_db.json")
    return DatabaseManager(url, mock_scheduler, db_fname, learners, fnames)


def func(x):
    return x**2


@pytest.fixture()
def learners() -> list[Learner1D]:
    learner1 = Learner1D(func, bounds=(-1, 1))
    learner2 = Learner1D(func, bounds=(-1, 1))
    learners = [learner1, learner2]
    return learners


@pytest.fixture()
def fnames(learners, tmpdir) -> list[str]:
    fnames = [str(tmpdir / f"learner{i}.pkl") for i, _ in enumerate(learners)]
    return fnames


@pytest.fixture()
def socket(db_manager: DatabaseManager) -> zmq.asyncio.Socket:
    ctx = zmq.asyncio.Context.instance()
    socket = ctx.socket(zmq.REQ)
    socket.connect(db_manager.url)
    yield socket
    socket.close()


@pytest.fixture()
def job_manager(
    db_manager: DatabaseManager,
    mock_scheduler: MockScheduler,
) -> JobManager:
    job_names = ["job1", "job2"]
    return JobManager(job_names, db_manager, mock_scheduler, interval=0.05)
