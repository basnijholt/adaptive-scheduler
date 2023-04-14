"""Test the RunManager class."""
import asyncio
from pathlib import Path
from unittest.mock import patch

import adaptive
import pytest
from adaptive import Learner1D
from ipywidgets import VBox

from adaptive_scheduler._server_support.run_manager import (
    RunManager,
    start_one_by_one,
)

from .helpers import MockScheduler, temporary_working_directory


def test_run_manager_init(
    mock_scheduler: MockScheduler,
    learners: list[Learner1D],
    fnames: list[str],
) -> None:
    """Test the initialization of RunManager."""
    rm = RunManager(mock_scheduler, learners, fnames)
    assert isinstance(rm, RunManager)


@pytest.mark.asyncio()
async def test_run_manager_start_and_cancel(
    mock_scheduler: MockScheduler,
    learners: list[Learner1D],
    fnames: list[str],
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
    learners: list[Learner1D],
    fnames: list[str],
    tmp_path: Path,
) -> None:
    """Test the cleanup method of RunManager."""
    with temporary_working_directory(tmp_path):
        rm = RunManager(mock_scheduler, learners, fnames)
        rm.move_old_logs_to.mkdir(parents=True, exist_ok=True)
        rm.cleanup(remove_old_logs_folder=True)
        assert not rm.move_old_logs_to.exists()


def test_run_manager_parse_log_files(
    mock_scheduler: MockScheduler,
    learners: list[Learner1D],
    fnames: list[str],
) -> None:
    """Test parsing log files in RunManager."""
    rm = RunManager(mock_scheduler, learners, fnames)
    df = rm.parse_log_files(only_last=True)
    assert df.empty  # nothing has been run yet


def test_run_manager_load_learners(
    mock_scheduler: MockScheduler,
    learners: list[Learner1D],
    fnames: list[str],
) -> None:
    """Test loading learners in RunManager."""
    rm = RunManager(mock_scheduler, learners, fnames)
    for lrn, fn in zip(learners, fnames):
        adaptive.runner.simple(lrn, npoints_goal=10)
        lrn.save(fn)
    rm.load_learners()
    for learner in learners:
        assert learner.data


@pytest.mark.asyncio()
async def test_run_manager_elapsed_time(
    mock_scheduler: MockScheduler,
    learners: list[Learner1D],
    fnames: list[str],
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
    learners: list[Learner1D],
    fnames: list[str],
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
    learners: list[Learner1D],
    fnames: list[str],
) -> None:
    """Test the _repr_html_ method of RunManager."""
    rm = RunManager(mock_scheduler, learners, fnames)
    with patch("IPython.display.display") as mocked_display:
        rm._repr_html_()
        assert mocked_display.called


def test_run_manager_info(
    mock_scheduler: MockScheduler,
    learners: list[Learner1D],
    fnames: list[str],
) -> None:
    """Test the info method of RunManager."""
    rm = RunManager(mock_scheduler, learners, fnames)
    with patch("IPython.display.display") as mocked_display:
        rm.info()
        assert mocked_display.called
        display_arg = mocked_display.call_args[0]
        # Check the content of the display
        assert isinstance(display_arg, tuple)
        assert isinstance(display_arg[0], VBox)


def test_run_manager_load_dataframes(
    mock_scheduler: MockScheduler,
    learners: list[Learner1D],
    fnames: list[str],
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
    learners: list[Learner1D],
    fnames: list[str],
) -> None:
    """Test starting RunManagers one by one."""

    def goal(rm: RunManager) -> bool:
        return rm.elapsed_time() >= 2  # noqa: PLR2004

    rm1 = RunManager(mock_scheduler, learners[:1], fnames[:1], job_name="rm1")
    rm2 = RunManager(mock_scheduler, learners[1:], fnames[1:], job_name="rm2")

    rm1.start()
    tasks = start_one_by_one(rm1, rm2, goal=goal)
    await asyncio.sleep(0.2)
    assert isinstance(tasks, tuple)
    assert len(tasks) == 2  # noqa: PLR2004
    assert isinstance(tasks[0], asyncio.Future)
    assert isinstance(tasks[1], list)
    assert isinstance(tasks[1][0], asyncio.Future)

    # Cancel tasks
    tasks[0].cancel()
    tasks[1][0].cancel()
    rm1.cancel()
    rm2.cancel()
