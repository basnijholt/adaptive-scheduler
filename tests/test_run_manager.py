"""Test the RunManager class."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import adaptive
import pytest
from ipywidgets import VBox

from adaptive_scheduler._server_support.run_manager import (
    RunManager,
    start_one_by_one,
)

from .helpers import (
    MockScheduler,
    get_socket,
    send_message,
    temporary_working_directory,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_run_manager_init(
    mock_scheduler: MockScheduler,
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
) -> None:
    """Test the initialization of RunManager."""
    rm = RunManager(mock_scheduler, learners, fnames)
    assert isinstance(rm, RunManager)


@pytest.mark.asyncio()
async def test_run_manager_start_and_cancel(
    mock_scheduler: MockScheduler,
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
) -> None:
    """Test starting the RunManager."""
    rm = RunManager(mock_scheduler, learners, fnames)
    rm.start()
    assert rm.status() == "running"
    rm.cancel()
    await asyncio.sleep(0.1)
    assert rm.status() == "cancelled"


def test_run_manager_cleanup(
    mock_scheduler: MockScheduler,
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
    tmp_path: Path,
) -> None:
    """Test the cleanup method of RunManager."""
    with temporary_working_directory(tmp_path):
        rm = RunManager(mock_scheduler, learners, fnames)
        assert rm.move_old_logs_to is not None
        rm.move_old_logs_to.mkdir(parents=True, exist_ok=True)
        rm.cleanup(remove_old_logs_folder=True)
        assert not rm.move_old_logs_to.exists()


def test_run_manager_parse_log_files(
    mock_scheduler: MockScheduler,
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
) -> None:
    """Test parsing log files in RunManager."""
    rm = RunManager(mock_scheduler, learners, fnames)
    rm.start()
    df = rm.parse_log_files(only_last=True)
    assert df.empty  # nothing has been run yet


def test_run_manager_load_learners(
    mock_scheduler: MockScheduler,
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
) -> None:
    """Test loading learners in RunManager."""
    rm = RunManager(mock_scheduler, learners, fnames)
    for lrn, fn in zip(learners, fnames, strict=True):
        adaptive.runner.simple(lrn, npoints_goal=10)
        lrn.save(fn)
    rm.load_learners()
    for learner in learners:
        assert learner.data


@pytest.mark.asyncio()
async def test_run_manager_elapsed_time(
    mock_scheduler: MockScheduler,
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
) -> None:
    """Test elapsed time in RunManager."""
    rm = RunManager(mock_scheduler, learners, fnames)
    assert rm.elapsed_time() == 0
    rm.start()
    await asyncio.sleep(0.05)
    assert rm.elapsed_time() > 0
    rm.cancel()
    await asyncio.sleep(0.05)
    assert rm.elapsed_time() > 0


@pytest.mark.asyncio()
async def test_run_manager_status(
    mock_scheduler: MockScheduler,
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
) -> None:
    """Test the status of RunManager."""
    rm = RunManager(mock_scheduler, learners, fnames)
    assert rm.status() == "not yet started"
    rm.start()
    await asyncio.sleep(0.05)
    assert rm.status() == "running"
    rm.cancel()
    await asyncio.sleep(0.05)
    assert rm.status() == "cancelled"


def test_run_manager_repr_html(
    mock_scheduler: MockScheduler,
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
) -> None:
    """Test the _repr_html_ method of RunManager."""
    rm = RunManager(mock_scheduler, learners, fnames)
    rm.start()
    with patch("IPython.display.display") as mocked_display:
        rm._repr_html_()
        assert mocked_display.called


def test_run_manager_info(
    mock_scheduler: MockScheduler,
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
) -> None:
    """Test the info method of RunManager."""
    rm = RunManager(mock_scheduler, learners, fnames)
    rm.start()
    with patch("IPython.display.display") as mocked_display:
        rm.info()
        assert mocked_display.called
        display_arg = mocked_display.call_args[0]
        # Check the content of the display
        assert isinstance(display_arg, tuple)
        assert isinstance(display_arg[0], VBox)


def test_run_manager_load_dataframes(
    mock_scheduler: MockScheduler,
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
) -> None:
    """Test loading dataframes in RunManager."""
    rm = RunManager(mock_scheduler, learners, fnames, save_dataframe=False)
    with pytest.raises(
        ValueError,
        match="The `save_dataframe` option was not set to True.",
    ):
        rm.load_dataframes()


@pytest.mark.asyncio()
async def test_start_one_by_one(
    mock_scheduler: MockScheduler,
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
) -> None:
    """Test starting RunManagers one by one."""

    def goal(rm: RunManager) -> bool:
        return rm.elapsed_time() >= 2

    rm1 = RunManager(mock_scheduler, learners[:1], fnames[:1], job_name="rm1")
    rm2 = RunManager(mock_scheduler, learners[1:], fnames[1:], job_name="rm2")

    rm1.start()
    tasks = start_one_by_one(rm1, rm2, goal=goal)
    await asyncio.sleep(0.2)
    assert isinstance(tasks, tuple)
    assert len(tasks) == 2
    assert isinstance(tasks[0], asyncio.Future)
    assert isinstance(tasks[1], list)
    assert isinstance(tasks[1][0], asyncio.Future)

    # Cancel tasks
    tasks[0].cancel()
    tasks[1][0].cancel()
    rm1.cancel()
    rm2.cancel()


@pytest.mark.asyncio()
async def test_run_manager_auto_restart(
    mock_scheduler: MockScheduler,
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
) -> None:
    """Test starting the RunManager."""
    rm = RunManager(mock_scheduler, learners, fnames, job_manager_interval=0.1)
    rm.start()
    await asyncio.sleep(0.1)
    assert rm.task is not None
    assert rm.status() == "running"
    q = rm.scheduler.queue()
    assert "0" in q
    assert "1" in q
    # Note: we don't know whether it is:
    # This: "0": {"job_name": "adaptive-scheduler-0"}, "1": {"job_name": "adaptive-scheduler-1"}
    # or:   "1": {"job_name": "adaptive-scheduler-0"}, "0": {"job_name": "adaptive-scheduler-1"}
    job_names = job_name0, job_name1 = "adaptive-scheduler-0", "adaptive-scheduler-1"
    job_id0, job_id1 = rm.scheduler.job_names_to_job_ids(job_name0, job_name1)
    log_fnames = ("log0.log", "log1.log")

    db = rm.database_manager.as_dicts()
    # jobs were all assigned...
    assert all(x["job_name"] is not None for x in db)
    # but they have not yet reported that they started.
    assert all(x["job_id"] is None for x in db)

    # Send a start message to the DatabaseManager
    # This is coming from the client
    with get_socket(rm.database_manager) as socket:
        for i, (entry, log_fname) in enumerate(zip(db, log_fnames, strict=True)):
            job_id = str(i)
            job_name = entry["job_name"]
            start_message = ("start", job_id, log_fname, job_name)
            await send_message(socket, start_message)

    db = rm.database_manager.as_dicts()
    assert len(db) == 2
    for entry in db:
        assert entry["job_id"] in ("0", "1")
        assert entry["job_name"] in job_names
        assert not entry["is_done"]

    # Mark the first job as cancelled
    rm.scheduler.update_queue(job_name0, "C")  # type: ignore[attr-defined]
    assert rm.scheduler.queue() == {
        job_id0: {"job_name": job_name0, "status": "C"},
        job_id1: {
            "job_name": "adaptive-scheduler-1",
            "status": "R",
            "state": "RUNNING",
        },
    }
    cast(MockScheduler, rm.scheduler)._queue_info.pop(job_id0)
    await asyncio.sleep(0.15)
    # Check that the job is restarted automatically with a new job_id:
    q = rm.scheduler.queue()
    assert q["2"] == {"job_name": job_name0, "state": "RUNNING", "status": "R"}

    # Start the job "from the client"
    with get_socket(rm.database_manager) as socket:
        start_message = ("start", "2", "log2.log", job_name0)
        await send_message(socket, start_message)

    await asyncio.sleep(0.1)

    # Check if the new job is started in the database
    db = rm.database_manager.as_dicts()
    assert len(db) == 2
    for entry in db:
        assert entry["job_id"] in ("0", "1", "2")
        assert entry["job_name"] in job_names
        assert not entry["is_done"]
        assert entry["log_fname"] == f"log{entry['job_id']}.log"

    # Now mark the 2 jobs as done
    with get_socket(rm.database_manager) as socket:
        for entry in db:
            stop_message = ("stop", entry["fname"])
            await send_message(socket, stop_message)

    await asyncio.sleep(0.15)

    # Check that the jobs are now done
    db = rm.database_manager.as_dicts()
    assert len(db) == 2
    for entry in db:
        assert entry["job_id"] is None
        assert entry["job_name"] is None
        assert entry["is_done"]
        assert entry["log_fname"].endswith(".log")

    # Check that RunManager is done
    assert rm.task.done()
