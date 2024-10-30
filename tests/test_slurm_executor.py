"""Tests for the SlurmExecutor class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from adaptive_scheduler import SlurmExecutor
from adaptive_scheduler._server_support.run_manager import RunManager

if TYPE_CHECKING:
    from pathlib import Path


def example_func(x: float) -> float:
    """Example function that returns its input."""
    return x


@pytest.fixture()
def executor() -> SlurmExecutor:
    """Create a SlurmExecutor instance."""
    return SlurmExecutor(
        name="test",
        folder=None,  # Will create a temporary folder
        save_interval=1,
        log_interval=1,
        job_manager_interval=1,
    )


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_submit_single_task(executor: SlurmExecutor) -> None:
    """Test submitting a single task."""
    task = executor.submit(example_func, 1.0)
    assert task.task_id.learner_index == 0
    assert task.task_id.sequence_index == 0
    assert executor._sequence_mapping[example_func] == 0
    assert executor._sequences[example_func] == [1.0]


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_submit_multiple_tasks_same_function(executor: SlurmExecutor) -> None:
    """Test submitting multiple tasks with the same function."""
    tasks = [executor.submit(example_func, x) for x in [1.0, 2.0, 3.0]]
    assert all(task.task_id.learner_index == 0 for task in tasks)
    assert [task.task_id.sequence_index for task in tasks] == [0, 1, 2]
    assert executor._sequence_mapping[example_func] == 0
    assert executor._sequences[example_func] == [1.0, 2.0, 3.0]


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
def test_submit_multiple_args_raises(executor: SlurmExecutor) -> None:
    """Test that submitting with multiple args raises ValueError."""
    with pytest.raises(ValueError, match="Exactly one argument is required"):
        executor.submit(example_func, 1.0, 2.0)


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
