"""Tests for the slurm_run function."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from adaptive_scheduler._server_support.run_manager import RunManager
from adaptive_scheduler._server_support.slurm_run import slurm_run
from adaptive_scheduler.scheduler import SLURM

if TYPE_CHECKING:
    from pathlib import Path

    import adaptive

    from adaptive_scheduler.utils import _DATAFRAME_FORMATS


@pytest.fixture()
def extra_run_manager_kwargs() -> dict[str, Any]:
    """Fixture for creating extra run manager keyword arguments."""
    return {"kill_on_error": "GPU on fire", "loky_start_method": "fork"}


@pytest.fixture()
def extra_scheduler_kwargs() -> dict[str, Any]:
    """Fixture for creating extra scheduler keyword arguments."""
    return {"mpiexec_executable": "mpiexec"}


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_slurm_run_with_default_arguments(
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
) -> None:
    """Test slurm_run function with default arguments."""
    rm = slurm_run(learners, fnames)
    assert isinstance(rm, RunManager)
    assert isinstance(rm.scheduler, SLURM)
    assert rm.save_interval == 300
    assert rm.log_interval == 300
    assert rm.learners == learners
    assert rm.fnames == fnames
    assert rm.scheduler.exclusive is False


def goal_example(learner: adaptive.Learner1D) -> bool:
    """Example goal function for testing."""
    return len(learner.data) >= 10


@pytest.mark.parametrize(
    ("partition", "nodes", "cores_per_node"),
    [("hb120v2-low", 2, 120), ("hb60-high", 3, 30)],
)
@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_slurm_run_with_custom_partition_nodes_and_cores(
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
    partition: str,
    nodes: int,
    cores_per_node: int,
) -> None:
    """Test slurm_run function with custom partition, nodes, and cores_per_node."""
    rm = slurm_run(
        learners,
        fnames,
        partition=partition,
        nodes=nodes,
        cores_per_node=cores_per_node,
        executor_type="ipyparallel",
    )
    assert isinstance(rm, RunManager)
    assert isinstance(rm.scheduler, SLURM)
    assert rm.scheduler.partition == partition
    assert rm.scheduler.nodes == nodes
    assert rm.scheduler.cores_per_node == cores_per_node


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_slurm_run_with_custom_goal(
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
) -> None:
    """Test slurm_run function with custom goal."""
    rm = slurm_run(learners, fnames, goal=goal_example)
    assert isinstance(rm, RunManager)
    assert rm.goal == goal_example


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_slurm_run_with_custom_folder_and_name(
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
    tmp_path: Path,
) -> None:
    """Test slurm_run function with custom folder and name."""
    folder = tmp_path / "test_folder"
    name = "custom_name"
    rm = slurm_run(learners, fnames, folder=folder, name=name)
    assert isinstance(rm, RunManager)
    assert rm.scheduler.log_folder == folder / "logs"
    assert rm.job_name == name


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_slurm_run_with_custom_num_threads(
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
) -> None:
    """Test slurm_run function with custom num_threads."""
    num_threads = 4
    rm = slurm_run(learners, fnames, num_threads=num_threads)
    assert isinstance(rm, RunManager)
    assert rm.scheduler.num_threads == num_threads


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_slurm_run_with_extra_run_manager_kwargs(
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
    extra_run_manager_kwargs: dict[str, Any],
) -> None:
    """Test slurm_run function with extra_run_manager_kwargs."""
    rm = slurm_run(learners, fnames, extra_run_manager_kwargs=extra_run_manager_kwargs)
    assert isinstance(rm, RunManager)
    for key, value in extra_run_manager_kwargs.items():
        assert getattr(rm, key) == value


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
@pytest.mark.parametrize("dataframe_format", ["csv", "json", "pickle"])
def test_slurm_run_with_custom_dataframe_format(
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
    dataframe_format: _DATAFRAME_FORMATS,
) -> None:
    """Test slurm_run function with custom dataframe_format."""
    rm = slurm_run(learners, fnames, dataframe_format=dataframe_format)
    assert isinstance(rm, RunManager)
    assert rm.dataframe_format == dataframe_format


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_slurm_run_with_custom_max_fails_and_jobs(
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
) -> None:
    """Test slurm_run function with custom max_fails_per_job and max_simultaneous_jobs."""
    max_fails_per_job = 100
    max_simultaneous_jobs = 200
    rm = slurm_run(
        learners,
        fnames,
        max_fails_per_job=max_fails_per_job,
        max_simultaneous_jobs=max_simultaneous_jobs,
    )
    assert isinstance(rm, RunManager)
    assert rm.max_fails_per_job == max_fails_per_job
    assert rm.max_simultaneous_jobs == max_simultaneous_jobs


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_slurm_run_with_extra_scheduler_kwargs(
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
    extra_scheduler_kwargs: dict[str, Any],
) -> None:
    """Test slurm_run function with extra_scheduler_kwargs."""
    rm = slurm_run(learners, fnames, extra_scheduler_kwargs=extra_scheduler_kwargs)
    assert isinstance(rm, RunManager)
    for key, value in extra_scheduler_kwargs.items():
        assert getattr(rm.scheduler, key) == value


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
@pytest.mark.parametrize("executor_type", ["ipyparallel", "dask-mpi", "mpi4py"])
def test_slurm_run_with_custom_executor_type(
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
    executor_type: str,
) -> None:
    """Test slurm_run function with custom executor_type."""
    rm = slurm_run(
        learners,
        fnames,
        executor_type=executor_type,  # type: ignore[arg-type]
    )
    assert isinstance(rm, RunManager)
    assert rm.scheduler.executor_type == executor_type


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_slurm_run_with_invalid_nodes_and_executor_type(
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
) -> None:
    """Test slurm_run function with invalid nodes and executor_type."""
    with pytest.raises(
        ValueError,
        match="process-pool can maximally use a single node",
    ):
        slurm_run(learners, fnames, nodes=2, executor_type="process-pool")


@pytest.mark.usefixtures("_mock_slurm_partitions")
@pytest.mark.usefixtures("_mock_slurm_queue")
def test_slurm_run_with_multiple_partitions(
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
) -> None:
    """Test slurm_run function with multiple partitions."""
    rm = slurm_run(learners, fnames, partition=("hb120v2-low", "hb60-high"))
    assert isinstance(rm, RunManager)
    s = rm.scheduler
    assert isinstance(s, SLURM)
    assert s.cores == (120, 60)
    assert s.partition == ("hb120v2-low", "hb60-high")
    assert s.nodes == (1, 1)
    assert s.cores_per_node == (120, 60)
