import asyncio
import logging
import os
import shutil
from unittest.mock import MagicMock

import pytest
from tinydb import Query, TinyDB

from adaptive_scheduler.server_support import (
    DatabaseManager,
    JobManager,
    KillManager,
    MaxRestartsReached,
    logs_with_string_or_condition,
)


@pytest.fixture
def kill_manager(mock_scheduler, db_manager: DatabaseManager) -> KillManager:
    return KillManager(
        scheduler=mock_scheduler,
        database_manager=db_manager,
        error="srun: error:",
        interval=0.05,
        max_cancel_tries=1,
        move_to=None,
    )


@pytest.mark.asyncio
async def test_kill_manager_init(kill_manager: KillManager) -> None:
    assert kill_manager.scheduler is not None
    assert kill_manager.database_manager is not None
    assert kill_manager.error == "srun: error:"
    assert kill_manager.interval == 0.05
    assert kill_manager.max_cancel_tries == 1
    assert kill_manager.move_to is None
    assert kill_manager.cancelled == []
    assert kill_manager.deleted == []


@pytest.mark.asyncio
async def test_kill_manager_manage(kill_manager: KillManager, tmpdir) -> None:
    test_log_dir = "test_logs"
    kill_manager.interval = 0.1

    log_file_path = tmpdir / "test_log.txt"

    with log_file_path.open("w") as f:
        f.write("srun: error: GPU on fire!\n")

    # test_entry = {
    #     "fname": "learner1.pkl",
    #     "is_done": False,
    #     "job_id": "1",
    #     "job_name": "test_job",
    #     "log_fname": str(log_file_path),
    #     "output_logs": [],
    # }

    # with TinyDB(kill_manager.database_manager.db_fname) as db:
    #     db.insert(test_entry)

    # kill_manager.scheduler._queue_info["1"] = {"job_name": "test_job", "state": "R"}

    kill_manager.database_manager.start()
    kill_manager.database_manager._start_request("1", str(log_file_path), "test_job")
    kill_manager.scheduler.start_job("test_job")
    kill_manager.start()
    await asyncio.sleep(1)  # Give it some time to start and cancel the job
    kill_manager.cancel()

    assert "test_job" in kill_manager.cancelled
    assert log_file_path in kill_manager.deleted

    shutil.rmtree(test_log_dir)


@pytest.mark.asyncio
async def test_kill_manager_manage_exception(kill_manager: KillManager, caplog) -> None:
    kill_manager.error = 12345  # This will cause a ValueError when calling `_manage`
    kill_manager.start()
    await asyncio.sleep(0.2)  # Give it some time to start and raise the exception
    kill_manager.cancel()

    assert "got exception in kill manager" in caplog.text
    assert "ValueError" in caplog.text


# @pytest.fixture
# def error_log_file(tmpdir) -> str:
#     log_file = tmpdir.join("error.log")
#     log_file.write("srun: error: GPU is on fire\n")
#     return str(log_file)


# @pytest.fixture
# def normal_log_file(tmpdir) -> str:
#     log_file = tmpdir.join("normal.log")
#     log_file.write("This is a normal log file.\n")
#     return str(log_file)


# @pytest.fixture
# def kill_manager(db_manager: DatabaseManager, mock_scheduler) -> KillManager:
#     return KillManager(mock_scheduler, db_manager, interval=0.05)


# def test_logs_with_string_or_condition_with_string(
#     error_log_file, normal_log_file, db_manager
# ):
#     db_manager.start()
#     db_manager.add(1, "job1", [error_log_file])

#     result = logs_with_string_or_condition("srun: error:", db_manager)
#     assert len(result) == 1
#     assert result[0][0] == "job1"
#     assert error_log_file in result[0][1]

#     db_manager.add(2, "job2", [normal_log_file])

#     result = logs_with_string_or_condition("srun: error:", db_manager)
#     assert len(result) == 1

#     db_manager.stop()


# def test_logs_with_string_or_condition_with_callable(
#     error_log_file, normal_log_file, db_manager
# ):
#     def custom_error_condition(lines: list[str]) -> bool:
#         return "srun: error: GPU is on fire" in "".join(lines)

#     db_manager.start()
#     db_manager.add(1, "job1", [error_log_file])

#     result = logs_with_string_or_condition(custom_error_condition, db_manager)
#     assert len(result) == 1
#     assert result[0][0] == "job1"
#     assert error_log_file in result[0][1]

#     db_manager.add(2, "job2", [normal_log_file])

#     result = logs_with_string_or_condition(custom_error_condition, db_manager)
#     assert len(result) == 1

#     db_manager.stop()


# def test_logs_with_string_or_condition_invalid_input(db_manager):
#     with pytest.raises(ValueError, match="`error` can only be a `str` or `callable`."):
#         logs_with_string_or_condition(42, db_manager)


# @pytest.mark.asyncio
# async def test_kill_manager_normal_execution(
#     kill_manager: KillManager, error_log_file, db_manager
# ):
#     db_manager.start()
#     db_manager.add(1, "job1", [error_log_file])

#     kill_manager.start()

#     await asyncio.sleep(0.1)
#     kill_manager.task.cancel()

#     with pytest.raises(asyncio.CancelledError):
#         await kill_manager.task

#     assert "job1" in kill_manager.cancelled
#     assert error_log_file in kill_manager.deleted

#     db_manager.stop()


# @pytest.mark.asyncio
# async def test_kill_manager_exception_handling(
#     kill_manager: KillManager, caplog, db_manager
# ):
#     caplog.set_level(logging.ERROR)

#     async def bad_update():
#         raise ValueError("Simulated error in update")

#     db_manager.update = bad_update
#     db_manager.start()

#     kill_manager.start()

#     await asyncio.sleep(0.1)
#     kill_manager.task.cancel()

#     with pytest.raises(asyncio.CancelledError):
#         await kill_manager.task

#     assert "got exception in kill manager" in caplog.text
#     assert "Simulated error in update" in caplog.text

#     db_manager.stop()
