"""Tests for the KillManager module."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from adaptive_scheduler.server_support import (
    DatabaseManager,
    KillManager,
    logs_with_string_or_condition,
)

if TYPE_CHECKING:
    from pathlib import Path


INTERVAL = 0.05


@pytest.fixture()
def kill_manager(db_manager: DatabaseManager) -> KillManager:
    """Fixture for creating a KillManager instance."""
    return KillManager(
        scheduler=db_manager.scheduler,
        database_manager=db_manager,
        error="srun: error:",
        interval=INTERVAL,
        max_cancel_tries=1,
        move_to=None,
    )


@pytest.mark.asyncio()
async def test_kill_manager_init(kill_manager: KillManager) -> None:
    """Test KillManager initialization."""
    assert kill_manager.scheduler is not None
    assert kill_manager.database_manager is not None
    assert kill_manager.error == "srun: error:"
    assert kill_manager.interval == INTERVAL
    assert kill_manager.max_cancel_tries == 1
    assert kill_manager.move_to is None
    assert kill_manager.cancelled == []
    assert kill_manager.deleted == []


def test_logs_with_string_or_condition_invalid_error() -> None:
    """Test logs_with_string_or_condition with invalid error type."""
    database_manager = MagicMock(spec=DatabaseManager)
    with pytest.raises(TypeError, match="`error` can only be a `str` or `callable`."):
        logs_with_string_or_condition(123, database_manager)  # type: ignore[arg-type]


def test_logs_with_string_or_condition_string_error(tmp_path: Path) -> None:
    """Test logs_with_string_or_condition with string error."""
    database_manager = MagicMock(spec=DatabaseManager)
    logs_file = tmp_path / "logs.txt"
    logs_file.write_text("Error: Something went wrong.")

    database_manager.as_dicts.return_value = [
        {
            "job_id": 1,
            "job_name": "test_job",
            "output_logs": [str(logs_file)],
            "log_fname": "log_file.log",
        },
    ]

    error = "Something went wrong"
    result = logs_with_string_or_condition(error, database_manager)
    assert len(result) == 1
    assert result[0][0] == "test_job"
    assert result[0][1][0] == str(logs_file)


def test_logs_with_string_or_condition_callable_error(tmp_path: Path) -> None:
    """Test logs_with_string_or_condition with callable error."""
    database_manager = MagicMock(spec=DatabaseManager)
    logs_file = tmp_path / "logs.txt"
    logs_file.write_text("Error: Something went wrong.")

    database_manager.as_dicts.return_value = [
        {
            "job_id": 1,
            "job_name": "test_job",
            "output_logs": [str(logs_file)],
            "log_fname": "log_file.log",
        },
    ]

    def custom_error(lines: list[str]) -> bool:
        return "Error" in "".join(lines)

    result = logs_with_string_or_condition(custom_error, database_manager)
    assert len(result) == 1
    assert result[0][0] == "test_job"
    assert result[0][1][0] == str(logs_file)


def test_logs_with_string_or_condition_no_error(tmp_path: Path) -> None:
    """Test logs_with_string_or_condition with no error."""
    database_manager = MagicMock(spec=DatabaseManager)
    logs_file = tmp_path / "logs.txt"
    logs_file.write_text("Info: Everything is fine.")

    database_manager.as_dicts.return_value = [
        {
            "job_id": 1,
            "job_name": "test_job",
            "output_logs": [str(logs_file)],
            "log_fname": "log_file.log",
        },
    ]

    error = "Something went wrong"
    result = logs_with_string_or_condition(error, database_manager)
    assert len(result) == 0


def test_logs_with_string_or_condition_missing_file() -> None:
    """Test logs_with_string_or_condition with missing file."""
    database_manager = MagicMock(spec=DatabaseManager)

    database_manager.as_dicts.return_value = [
        {
            "job_id": 1,
            "job_name": "test_job",
            "output_logs": ["non_existent_file.txt"],
            "log_fname": "log_file.log",
        },
    ]

    error = "Something went wrong"
    result = logs_with_string_or_condition(error, database_manager)
    assert len(result) == 0


@pytest.mark.asyncio()
async def test_kill_manager_manage(kill_manager: KillManager) -> None:
    """Test KillManager.manage method."""
    # The KillManager will read from the .out files, which are determined
    # from the scheduler.
    output_file_path = kill_manager.database_manager._output_logs("0", "test_job")[0]
    with output_file_path.open("w") as f:
        f.write("srun: error: GPU on fire!\n")

    kill_manager.database_manager.start()  # creates empty db
    kill_manager.start()
    assert kill_manager.task is not None
    # Marks the job as running, and sets the job_id, job_name, and log_fname
    kill_manager.database_manager._choose_fname("test_job")
    kill_manager.database_manager._start_request("0", "log_fname.log", "test_job")
    kill_manager.scheduler.start_job("test_job")
    await asyncio.sleep(0.1)  # Give it some time to start and cancel the job
    assert "test_job" in kill_manager.cancelled
    assert str(output_file_path) in kill_manager.deleted


@pytest.mark.asyncio()
async def test_kill_manager_manage_exception(
    kill_manager: KillManager,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test KillManager.manage method with an exception."""
    # This will cause a ValueError when calling `_manage`
    kill_manager.error = 12345  # type: ignore[assignment]
    kill_manager.database_manager.start()  # creates empty db
    kill_manager.start()
    assert kill_manager.task is not None
    await asyncio.sleep(0.2)  # Give it some time to start and raise the exception

    timeout = 0.1
    try:
        await asyncio.wait_for(kill_manager.task, timeout=timeout)
    except asyncio.TimeoutError:
        kill_manager.task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await kill_manager.task

    assert "got exception in kill manager" in caplog.text
    assert "TypeError" in caplog.text


@pytest.mark.asyncio()
async def test_kill_manager_manage_canceled(
    kill_manager: KillManager,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test KillManager.manage method when canceled."""
    caplog.set_level(logging.INFO)
    kill_manager.error = "never going to happen"
    kill_manager.database_manager.start()  # creates empty db
    kill_manager.start()
    assert kill_manager.task is not None
    await asyncio.sleep(0.1)  # Give it some time to start and raise the exception
    kill_manager.task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await kill_manager.task

    assert "task was cancelled because of a CancelledError" in caplog.text
