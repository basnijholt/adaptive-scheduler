import asyncio
import os

import pytest
from tinydb import Query, TinyDB

from adaptive_scheduler.server_support import DatabaseManager
from adaptive_scheduler.utils import _deserialize, _serialize, smart_goal


@pytest.mark.asyncio()
async def test_database_manager_start_and_cancel(db_manager) -> None:
    db_manager.start()
    await asyncio.sleep(0.1)  # Give it some time to start
    assert db_manager.is_started
    with pytest.raises(Exception, match="already started"):
        db_manager.start()
    result = db_manager.cancel()
    assert result is not None
    with pytest.raises(asyncio.InvalidStateError):
        assert db_manager.task.result()


def test_database_manager_n_done(db_manager) -> None:
    assert db_manager.n_done() == 0


def test_smart_goal(learners, fnames) -> None:
    """Test empty learners didn't reach the goal."""
    goal = smart_goal(100, learners)
    assert not goal(learners[0])
    assert not goal(learners[1])
    goal = smart_goal(0, learners)
    assert goal(learners[0])


def test_database_manager_create_empty_db(db_manager) -> None:
    db_manager.create_empty_db()
    assert os.path.exists(db_manager.db_fname)

    with TinyDB(db_manager.db_fname) as db:
        assert len(db.all()) == 2


def test_database_manager_as_dicts(db_manager, fnames) -> None:
    db_manager.create_empty_db()
    assert db_manager.as_dicts() == [
        {
            "fname": fnames[0],
            "is_done": False,
            "job_id": None,
            "job_name": None,
            "log_fname": None,
            "output_logs": [],
        },
        {
            "fname": fnames[1],
            "is_done": False,
            "job_id": None,
            "job_name": None,
            "log_fname": None,
            "output_logs": [],
        },
    ]


@pytest.mark.asyncio()
async def test_database_manager_dispatch_start_stop(
    db_manager,
    learners,
    fnames,
) -> None:
    db_manager.learners, db_manager.fnames = learners, fnames
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


async def send_message(socket, message) -> None:
    await socket.send_serialized(message, _serialize)
    response = await socket.recv_serialized(_deserialize)
    return response


@pytest.mark.asyncio()
async def test_database_manager_start_and_update(
    socket,
    db_manager: DatabaseManager,
    fnames: list[str],
) -> None:
    db_manager.create_empty_db()
    db_manager.start()
    await asyncio.sleep(0.1)  # Give it some time to start

    # Send a start message to the DatabaseManager
    job_id, log_fname, job_name = "1000", "log.log", "job_name"
    start_message = ("start", job_id, log_fname, job_name)
    fname = await send_message(socket, start_message)

    # Check if the correct fname is returned
    assert fname == fnames[0], fname

    # Check that the database is updated correctly
    with TinyDB(db_manager.db_fname) as db:
        Entry = Query()
        entry = db.get(Entry.fname == fname)
        assert entry["job_id"] == job_id
        assert entry["log_fname"] == log_fname
        assert entry["job_name"] == job_name

    # Say that the job is still running
    queue = {"1000": {"job_id": "1000"}}
    db_manager.update(queue)

    # Check that the database is the same
    with TinyDB(db_manager.db_fname) as db:
        Entry = Query()
        entry = db.get(Entry.fname == fname)
        assert entry["job_id"] == job_id
        assert entry["log_fname"] == log_fname
        assert entry["job_name"] == job_name

    # Say that the job is died
    queue = {}
    db_manager.update(queue)

    # Check that the database is updated correctly
    with TinyDB(db_manager.db_fname) as db:
        Entry = Query()
        entry = db.get(Entry.fname == fname)
        assert entry["job_id"] is None


@pytest.mark.asyncio()
async def test_database_manager_start_stop(
    socket,
    db_manager: DatabaseManager,
    fnames: list[str],
) -> None:
    db_manager.create_empty_db()
    # Start the DatabaseManager
    db_manager.start()
    await asyncio.sleep(0.1)  # Give it some time to start
    assert db_manager.task is not None

    # Send a start message to the DatabaseManager
    job_id, log_fname, job_name = "1000", "log.log", "job_name"
    start_message = ("start", job_id, log_fname, job_name)
    fname = await send_message(socket, start_message)
    # Try starting again:
    exception = await send_message(socket, start_message)
    with pytest.raises(
        Exception,
        match="The job_id 1000 already exists in the database and runs",
    ):
        raise exception

    # Check if the correct fname is returned
    assert fname == fnames[0], fname

    # Check that the database is updated correctly
    with TinyDB(db_manager.db_fname) as db:
        Entry = Query()
        entry = db.get(Entry.fname == fname)
        assert entry["job_id"] == job_id
        assert entry["log_fname"] == log_fname
        assert entry["job_name"] == job_name

    # Check that task is still running
    assert db_manager.task is not None
    assert not db_manager.task.done()

    # Send a stop message to the DatabaseManager
    stop_message = ("stop", fname)
    reply = await send_message(socket, stop_message)
    assert reply is None

    with TinyDB(db_manager.db_fname) as db:
        Entry = Query()
        entry = db.get(Entry.fname == fnames[0])
        assert entry["job_id"] is None

    # Start and stop the learner2
    fname = await send_message(socket, start_message)
    assert fname == fnames[1]

    # Send a stop message to the DatabaseManager
    stop_message = ("stop", fname)
    reply = await send_message(socket, stop_message)
    assert reply is None

    exception = await send_message(socket, start_message)
    with pytest.raises(Exception, match="No more learners to run in the database"):
        raise exception


@pytest.mark.asyncio()
async def test_database_manager_stop_request_and_requests(
    socket,
    db_manager: DatabaseManager,
    fnames: list[str],
) -> None:
    db_manager.create_empty_db()
    # Start the DatabaseManager
    db_manager.start()
    await asyncio.sleep(0.1)  # Give it some time to start
    assert db_manager.task is not None

    # Start a job for learner1
    job_id1, log_fname1, job_name1 = "1000", "log1.log", "job_name1"
    start_message1 = ("start", job_id1, log_fname1, job_name1)
    fname1 = await send_message(socket, start_message1)
    assert fname1 == fnames[0], fname1

    # Start a job for learner2
    job_id2, log_fname2, job_name2 = "1001", "log2.log", "job_name2"
    start_message2 = ("start", job_id2, log_fname2, job_name2)
    fname2 = await send_message(socket, start_message2)
    assert fname2 == fnames[1], fname2

    # Stop the job for learner1 using _stop_request
    db_manager._stop_request(fname1)

    with TinyDB(db_manager.db_fname) as db:
        Entry = Query()
        entry = db.get(Entry.fname == fname1)
        assert entry["job_id"] is None
        assert entry["is_done"] is True
        assert entry["job_name"] is None

    # Stop the job for learner2 using _stop_requests
    db_manager._stop_requests([fname2])

    with TinyDB(db_manager.db_fname) as db:
        Entry = Query()
        entry = db.get(Entry.fname == fname2)
        assert entry["job_id"] is None
        assert entry["is_done"] is True
        assert entry["job_name"] is None
