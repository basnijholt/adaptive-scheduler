"""Tests for the SlurmExecutor class."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import adaptive
import pytest

from adaptive_scheduler import SlurmExecutor
from adaptive_scheduler._server_support.run_manager import RunManager

if TYPE_CHECKING:
    from pathlib import Path


def example_func(x: float) -> float:
    """Example function that returns its input."""
    return x


@pytest.fixture()
def executor(tmp_path: Path) -> SlurmExecutor:
    """Create a SlurmExecutor instance."""
    return SlurmExecutor(
        name="test",
        folder=tmp_path,
        save_interval=1,
        log_interval=1,
        job_manager_interval=1,
    )


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_submit_single_task(executor: SlurmExecutor) -> None:
    """Test submitting a single task."""
    task = executor.submit(example_func, 1.0, 2.0)
    assert task.task_id.learner_index == 0
    assert task.task_id.sequence_index == 0
    assert executor._sequence_mapping[example_func] == 0
    assert executor._sequences[example_func] == [(1.0, 2.0)]


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_submit_multiple_tasks_same_function(executor: SlurmExecutor) -> None:
    """Test submitting multiple tasks with the same function."""
    tasks = [executor.submit(example_func, x) for x in [1.0, 2.0, 3.0]]
    assert all(task.task_id.learner_index == 0 for task in tasks)
    assert [task.task_id.sequence_index for task in tasks] == [0, 1, 2]
    assert executor._sequence_mapping[example_func] == 0
    assert executor._sequences[example_func] == [(1.0,), (2.0,), (3.0,)]


def another_func(x: float) -> float:
    """Another example function."""
    return x * 2


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_submit_multiple_functions(executor: SlurmExecutor) -> None:
    """Test submitting tasks with different functions."""
    task1 = executor.submit(example_func, 1.0)
    task2 = executor.submit(another_func, 2.0)

    assert task1.task_id.learner_index == 0
    assert task2.task_id.learner_index == 1
    assert len(executor._sequence_mapping) == 2
    assert len(executor._sequences) == 2


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_submit_with_kwargs_raises(executor: SlurmExecutor) -> None:
    """Test that submitting with kwargs raises ValueError."""
    with pytest.raises(ValueError, match="Keyword arguments are not supported"):
        executor.submit(example_func, x=1.0)


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_finalize_creates_run_manager(executor: SlurmExecutor) -> None:
    """Test that finalize creates a RunManager."""
    executor.submit(example_func, 1.0)
    rm = executor.finalize(start=False)
    assert isinstance(rm, RunManager)
    assert len(rm.learners) == 1
    assert len(rm.fnames) == 1
    assert isinstance(rm, RunManager)


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_finalize_twice_raises(executor: SlurmExecutor) -> None:
    """Test that calling finalize twice raises RuntimeError."""
    executor.submit(example_func, 1.0)
    executor.finalize(start=False)
    with pytest.raises(RuntimeError, match="RunManager already initialized"):
        executor.finalize()


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_map(executor: SlurmExecutor) -> None:
    """Test the map method."""
    values = [1.0, 2.0, 3.0]
    tasks = executor.map(example_func, values)
    assert len(tasks) == len(values)
    assert all(task.task_id.learner_index == 0 for task in tasks)  # type: ignore[attr-defined]
    assert [task.task_id.sequence_index for task in tasks] == [0, 1, 2]  # type: ignore[attr-defined]


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_map_with_timeout_raises(executor: SlurmExecutor) -> None:
    """Test that map with timeout raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Timeout not implemented"):
        executor.map(example_func, [1.0], timeout=1.0)


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_map_with_chunksize_raises(executor: SlurmExecutor) -> None:
    """Test that map with chunksize raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Chunksize not implemented"):
        executor.map(example_func, [1.0], chunksize=2)


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_new_executor(executor: SlurmExecutor) -> None:
    """Test creating a new executor with the same parameters."""
    executor.submit(example_func, 1.0)
    executor.finalize(start=False)

    new_executor = executor.new()
    assert new_executor._run_manager is None
    assert not new_executor._sequences
    assert not new_executor._sequence_mapping
    assert new_executor.name == executor.name
    assert new_executor.folder == executor.folder


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_folder_creation(tmp_path: Path) -> None:
    """Test that the folder is created correctly."""
    folder = tmp_path / "test_folder"
    executor = SlurmExecutor(folder=folder)
    assert executor.folder == folder

    # Test default folder creation
    executor = SlurmExecutor()
    assert executor.folder.parent.name == ".adaptive_scheduler"  # type: ignore[union-attr]
    assert len(executor.folder.name.split("-")) == 3  # type: ignore[union-attr]
    # date-time-uuid format


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_cleanup(executor: SlurmExecutor) -> None:
    """Test the cleanup method."""
    executor.submit(example_func, 1.0)
    executor.finalize(start=False)

    executor.cleanup()


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_task_get_before_finalize(executor: SlurmExecutor) -> None:
    """Test that _get before finalize returns None."""
    task = executor.submit(example_func, 1.0)
    with pytest.raises(AssertionError, match="RunManager not initialized"):
        task._get()


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_task_get_with_missing_file(executor: SlurmExecutor) -> None:
    """Test that _get with missing file returns None."""
    task = executor.submit(example_func, 1.0)
    executor.finalize(start=False)
    assert task._get() is None  # File doesn't exist yet


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_task_get(executor: SlurmExecutor) -> None:
    """Test that _get gets the data."""
    task = executor.submit(example_func, 1.0)
    executor.finalize(start=False)

    # First _get should try to load
    assert task._get() is None

    # Create a dummy file
    learner, fname = task._learner_and_fname
    adaptive.runner.simple(learner)
    learner.save(fname)

    # Does not respect min_load_interval because data is already in memory
    assert task._get() == 1.0


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_task_get_adapts_load_interval(executor: SlurmExecutor) -> None:
    """Test that _get adapts min_load_interval based on load time."""
    task = executor.submit(example_func, 1.0)
    executor.finalize(start=False)

    # Mock a slow load operation
    def slow_load(*args: Any) -> None:  # noqa: ARG001
        time.sleep(0.1)  # Simulate slow load

    learner, fname = task._learner_and_fname
    with patch.object(learner, "load", side_effect=slow_load):
        # Create a dummy file
        with open(fname, "wb") as f:  # noqa: PTH123
            f.write(b"dummy")

        task._get()
        assert task.min_load_interval >= 2.0  # Should be at least 20x the load time


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_task_result_timeout_not_implemented(executor: SlurmExecutor) -> None:
    """Test that result() with timeout raises NotImplementedError."""
    task = executor.submit(example_func, 1.0)
    with pytest.raises(NotImplementedError, match="Timeout not implemented"):
        task.result(timeout=1.0)


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_task_repr_triggers_get(executor: SlurmExecutor) -> None:
    """Test that repr triggers _get."""
    task = executor.submit(example_func, 1.0)
    executor.finalize(start=False)

    assert "PENDING" in repr(task)

    # Mock learner.done() to return True
    learner, _ = task._learner_and_fname
    with patch.object(learner, "done", return_value=True), patch.dict(learner.data, {0: 42}):
        assert "FINISHED" in repr(task)


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
async def test_task_await(executor: SlurmExecutor) -> None:
    """Test awaiting a task."""
    task = executor.submit(example_func, 1.0)
    executor.finalize(start=False)

    # Create a background task to simulate result appearing
    async def simulate_result() -> None:
        await asyncio.sleep(0.1)
        learner, fname = task._learner_and_fname
        learner.data[0] = 42
        with patch.object(learner, "done", return_value=True):
            task._get()

    asyncio.create_task(simulate_result())  # noqa: RUF006
    result = await task
    assert result == 42
