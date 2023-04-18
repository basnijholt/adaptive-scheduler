"""Tests for DatabaseManager."""
from __future__ import annotations

import asyncio
from pathlib import Path

import adaptive
import pytest
import zmq
from tinydb import Query, TinyDB

from adaptive_scheduler._server_support.database_manager import (
    DatabaseManager,
    _ensure_str,
)
from adaptive_scheduler.utils import smart_goal

from .helpers import send_message


@pytest.mark.asyncio()
async def test_database_manager_start_and_cancel(db_manager: DatabaseManager) -> None:
    """Test starting and canceling the DatabaseManager."""
    db_manager.start()
    await asyncio.sleep(0.1)  # Give it some time to start
    assert db_manager.is_started
    with pytest.raises(Exception, match="already started"):
        db_manager.start()
    result = db_manager.cancel()
    assert result is not None
    with pytest.raises(asyncio.InvalidStateError):  # noqa: PT012
        assert db_manager.task is not None
        assert db_manager.task.result()


def test_database_manager_n_done(db_manager: DatabaseManager) -> None:
    """Test the number of done jobs."""
    assert db_manager.n_done() == 0


def test_smart_goal(learners: list) -> None:
    """Test empty learners didn't reach the goal."""
    goal = smart_goal(100, learners)
    assert not goal(learners[0])
    assert not goal(learners[1])
    goal = smart_goal(0, learners)
    assert goal(learners[0])


def test_database_manager_create_empty_db(db_manager: DatabaseManager) -> None:
    """Test creating an empty database."""
    db_manager.create_empty_db()
    assert Path(db_manager.db_fname).exists()
    n_learners = 2
    with TinyDB(db_manager.db_fname) as db:
        assert len(db.all()) == n_learners


def test_database_manager_as_dicts(
    db_manager: DatabaseManager,
    fnames: list[Path] | list[str],
) -> None:
    """Test getting the database as a list of dictionaries."""
    db_manager.create_empty_db()
    assert db_manager.as_dicts() == [
        {
            "fname": _ensure_str(fnames[0]),
            "is_done": False,
            "job_id": None,
            "job_name": None,
            "log_fname": None,
            "output_logs": [],
            "start_time": None,
        },
        {
            "fname": _ensure_str(fnames[1]),
            "is_done": False,
            "job_id": None,
            "job_name": None,
            "log_fname": None,
            "output_logs": [],
            "start_time": None,
        },
    ]


@pytest.mark.asyncio()
async def test_database_manager_dispatch_start_stop(
    db_manager: DatabaseManager,
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
) -> None:
    """Test starting and stopping jobs using the dispatch method."""
    db_manager.learners, db_manager.fnames = learners, fnames
    db_manager.create_empty_db()

    start_request = ("start", "1000", "log_1000.txt", "test_job")
    fname = db_manager._dispatch(start_request)
    assert fname in _ensure_str(db_manager.fnames)
    if isinstance(learners[0], adaptive.BalancingLearner):
        assert isinstance(fname, list)
        assert isinstance(fname[0], str)
    else:
        assert isinstance(fname, str)

    stop_request = ("stop", fname)
    db_manager._dispatch(stop_request)

    with TinyDB(db_manager.db_fname) as db:
        entry = Query()
        entry = db.get(entry.fname == fname)
        assert entry["job_id"] is None
        assert entry["is_done"] is True


@pytest.mark.asyncio()
async def test_database_manager_start_and_update(
    socket: zmq.asyncio.Socket,
    db_manager: DatabaseManager,
    fnames: list[str] | list[Path],
) -> None:
    """Test starting and updating jobs."""
    db_manager.create_empty_db()
    db_manager.start()
    await asyncio.sleep(0.1)  # Give it some time to start

    # Send a start message to the DatabaseManager
    job_id, log_fname, job_name = "1000", "log.log", "job_name"
    start_message = ("start", job_id, log_fname, job_name)
    fname = await send_message(socket, start_message)

    # Check if the correct fname is returned
    assert fname == _ensure_str(fnames[0]), fname

    # Check that the database is updated correctly
    with TinyDB(db_manager.db_fname) as db:
        entry = Query()
        entry = db.get(entry.fname == fname)
        assert entry["job_id"] == job_id
        assert entry["log_fname"] == log_fname
        assert entry["job_name"] == job_name

    # Say that the job is still running
    queue = {"1000": {"job_id": "1000"}}
    db_manager.update(queue)

    # Check that the database is the same
    with TinyDB(db_manager.db_fname) as db:
        entry = Query()
        entry = db.get(entry.fname == fname)
        assert entry["job_id"] == job_id
        assert entry["log_fname"] == log_fname
        assert entry["job_name"] == job_name

    # Say that the job is died
    queue = {}
    db_manager.update(queue)

    # Check that the database is updated correctly
    with TinyDB(db_manager.db_fname) as db:
        entry = Query()
        entry = db.get(entry.fname == fname)
        assert entry["job_id"] is None


@pytest.mark.asyncio()
async def test_database_manager_start_stop(
    socket: zmq.asyncio.Socket,
    db_manager: DatabaseManager,
    fnames: list[str] | list[Path],
) -> None:
    """Test starting and stopping jobs."""
    db_manager.create_empty_db()
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
    assert fname == _ensure_str(fnames[0]), fname

    # Check that the database is updated correctly
    with TinyDB(db_manager.db_fname) as db:
        query = Query()
        entry = db.get(query.fname == fname)
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
        query = Query()
        entry = db.get(query.fname == _ensure_str(fnames[0]))
        assert entry["job_id"] is None

    # Start and stop the learner2
    fname = await send_message(socket, start_message)
    assert fname == _ensure_str(fnames[1])

    # Send a stop message to the DatabaseManager
    stop_message = ("stop", fname)
    reply = await send_message(socket, stop_message)
    assert reply is None

    with pytest.raises(zmq.error.Again, match="Resource temporarily unavailable"):
        await send_message(socket, start_message)


@pytest.mark.asyncio()
async def test_database_manager_stop_request_and_requests(
    socket: zmq.asyncio.Socket,
    db_manager: DatabaseManager,
    fnames: list[str] | list[Path],
) -> None:
    """Test stopping jobs using stop_request and stop_requests methods."""
    db_manager.create_empty_db()
    db_manager.start()
    await asyncio.sleep(0.1)  # Give it some time to start
    assert db_manager.task is not None

    # Start a job for learner1
    job_id1, log_fname1, job_name1 = "1000", "log1.log", "job_name1"
    start_message1 = ("start", job_id1, log_fname1, job_name1)
    fname1 = await send_message(socket, start_message1)
    assert fname1 == _ensure_str(fnames[0]), fname1

    # Start a job for learner2
    job_id2, log_fname2, job_name2 = "1001", "log2.log", "job_name2"
    start_message2 = ("start", job_id2, log_fname2, job_name2)
    fname2 = await send_message(socket, start_message2)
    assert fname2 == _ensure_str(fnames[1]), fname2

    # Stop the job for learner1 using _stop_request
    db_manager._stop_request(fname1)

    with TinyDB(db_manager.db_fname) as db:
        entry = Query()
        entry = db.get(entry.fname == fname1)
        assert entry["job_id"] is None
        assert entry["is_done"] is True
        assert entry["job_name"] is None

    # Stop the job for learner2 using _stop_requests
    db_manager._stop_requests([fname2])

    with TinyDB(db_manager.db_fname) as db:
        entry = Query()
        entry = db.get(entry.fname == fname2)
        assert entry["job_id"] is None
        assert entry["is_done"] is True
        assert entry["job_name"] is None


@pytest.mark.parametrize(
    ("input_list", "expected_output"),
    [
        # Test with a list of strings
        (["path1", "path2", "path3", "path4"], ["path1", "path2", "path3", "path4"]),
        # Test with a list of Path objects
        (
            [Path("path1"), Path("path2"), Path("path3"), Path("path4")],
            ["path1", "path2", "path3", "path4"],
        ),
        # Test with a list of lists of strings
        (
            [["path1", "path2"], ["path3", "path4"]],
            [["path1", "path2"], ["path3", "path4"]],
        ),
        # Test with a list of lists of Path objects
        (
            [[Path("path1"), Path("path2")], [Path("path3"), Path("path4")]],
            [["path1", "path2"], ["path3", "path4"]],
        ),
        # Test empty
        ([], []),
    ],
)
def test_ensure_str(
    input_list: list[str] | list[list[str]] | list[Path] | list[list[Path]],
    expected_output: list[str] | list[list[str]],
) -> None:
    """Test the _ensure_str function."""
    output_list = _ensure_str(input_list)
    assert output_list == expected_output


@pytest.mark.parametrize(
    "invalid_input",
    [
        # Test with an invalid input
        {1, 2},
        10,
        10.0,
    ],
)
def test_ensure_str_invalid_input(invalid_input: list[str]) -> None:
    """Test the _ensure_str function with an invalid input."""
    with pytest.raises(ValueError, match="Invalid input:"):
        _ensure_str(invalid_input)  # type: ignore[arg-type]
