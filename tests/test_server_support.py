import asyncio
import os

import pytest
import zmq.asyncio
from adaptive import Learner1D
from tinydb import Query, TinyDB

from adaptive_scheduler._mock_scheduler import MockScheduler
from adaptive_scheduler.server_support import DatabaseManager
from adaptive_scheduler.utils import _deserialize, _serialize, smart_goal


@pytest.fixture
def mock_scheduler():
    return MockScheduler()


@pytest.fixture
def db_manager(mock_scheduler, learners_and_fnames):
    url = "tcp://127.0.0.1:5555"
    db_fname = "test_db.json"
    learners, fnames = learners_and_fnames
    return DatabaseManager(url, mock_scheduler, db_fname, learners, fnames)


@pytest.mark.asyncio
async def test_database_manager_start_and_cancel(db_manager):
    db_manager.start()
    await asyncio.sleep(0.1)  # Give it some time to start
    assert db_manager.is_started
    result = db_manager.cancel()
    assert result is not None
    with pytest.raises(asyncio.InvalidStateError):
        assert db_manager.task.result()


def test_database_manager_n_done(db_manager):
    assert db_manager.n_done() == 0


def test_smart_goal(learners_and_fnames):
    """Test empty learners didn't reach the goal."""
    learners, fnames = learners_and_fnames
    goal = smart_goal(100, learners)
    assert not goal(learners[0])
    assert not goal(learners[1])
    goal = smart_goal(0, learners)
    assert goal(learners[0])


@pytest.fixture
def learners_and_fnames():
    def func(x):
        return x**2

    learner1 = Learner1D(func, bounds=(-1, 1))
    learner2 = Learner1D(func, bounds=(-1, 1))
    learners = [learner1, learner2]
    fnames = ["learner1.pkl", "learner2.pkl"]
    return learners, fnames


def test_database_manager_create_empty_db(db_manager):
    db_manager.create_empty_db()
    assert os.path.exists(db_manager.db_fname)

    with TinyDB(db_manager.db_fname) as db:
        assert len(db.all()) == 2


def test_database_manager_as_dicts(db_manager):
    db_manager.create_empty_db()
    assert db_manager.as_dicts() == [
        {
            "fname": "learner1.pkl",
            "is_done": False,
            "job_id": None,
            "job_name": None,
            "log_fname": None,
            "output_logs": [],
        },
        {
            "fname": "learner2.pkl",
            "is_done": False,
            "job_id": None,
            "job_name": None,
            "log_fname": None,
            "output_logs": [],
        },
    ]


@pytest.mark.asyncio
async def test_database_manager_dispatch_start_stop(db_manager, learners_and_fnames):
    db_manager.learners, db_manager.fnames = learners_and_fnames
    db_manager.create_empty_db()

    start_request = ("start", "1000", "log_1000.txt", "test_job")
    fname = db_manager._dispatch(start_request)
    assert fname in db_manager.fnames

    stop_request = ("stop", fname)
    db_manager._dispatch(stop_request)

    with TinyDB(db_manager.db_fname) as db:
        Entry = Query()
        entry = db.get(Entry.fname == fname)
        assert entry["job_id"] is None
        assert entry["is_done"] is True


async def send_message(db_manager, message):
    ctx = zmq.asyncio.Context.instance()
    socket = ctx.socket(zmq.REQ)
    socket.connect(db_manager.url)
    await socket.send_serialized(message, _serialize)
    response = await socket.recv_serialized(_deserialize)
    socket.close()
    return response


@pytest.mark.asyncio
async def test_database_manager_update(
    db_manager: DatabaseManager, mock_scheduler, learners_and_fnames
):
    db_manager.create_empty_db()

    queue = {"1000": {"job_id": "1000"}}
    db_manager.update(queue)

    # with TinyDB(db_manager.db_fname) as db:
    #     Entry = Query()
    #     entry = db.get(Entry.fname == "learner1.pkl")
    #     assert entry["job_id"] == "1000", entry

    # Start the DatabaseManager
    asyncio.create_task(db_manager._manage())

    # Send a start message to the DatabaseManager
    job_id, log_fname, job_name = "1000", "log.log", "job_name"
    start_message = ("start", job_id, log_fname, job_name)
    fname = await send_message(db_manager, start_message)

    # Check if the correct fname is returned
    assert fname == "learner1.pkl", fname

    # Send a stop message to the DatabaseManager
    stop_message = ("stop", fname)
    await send_message(db_manager, stop_message)

    with TinyDB(db_manager.db_fname) as db:
        Entry = Query()
        entry = db.get(Entry.fname == "learner1.pkl")
        assert entry["job_id"] is None
