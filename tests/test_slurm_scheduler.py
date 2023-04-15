"""Tests for the SLURM scheduler."""

import re

import pytest

from adaptive_scheduler._scheduler.slurm import SLURM


def test_init_cores() -> None:
    """Test that the cores are set correctly."""
    with pytest.raises(
        ValueError,
        match="Specify either `nodes` and `cores_per_node`, or only `cores`, not both.",
    ):
        SLURM(cores=4, nodes=2, cores_per_node=2)


def test_init_partition() -> None:
    """Test that the partition is set correctly."""
    with pytest.raises(ValueError, match="Invalid partition: nonexistent"):
        SLURM(cores=4, partition="nonexistent")


def test_init_nodes_cores_per_node() -> None:
    """Test that the nodes and cores_per_node are set correctly."""
    with pytest.raises(
        ValueError,
        match="Specify either `nodes` and `cores_per_node`, or only `cores`, not both.",
    ):
        SLURM(nodes=2)


def test_slurm_scheduler() -> None:
    """Test that the slurm scheduler is set correctly."""
    s = SLURM(cores=4)
    assert s._cores == 4
    assert s.nodes is None
    assert s.cores_per_node is None


def test_slurm_scheduler_nodes_cores_per_node() -> None:
    """Test that the nodes and cores_per_node are set correctly."""
    s = SLURM(nodes=2, cores_per_node=2)
    assert s._cores == 4
    assert s.nodes == 2
    assert s.cores_per_node == 2


def test_partition() -> None:
    """Test that the partition is set correctly."""
    with pytest.raises(RuntimeError, match="SLURM is not responding."):
        SLURM(cores=4, partition="nonexistent")


def test_getstate_setstate() -> None:
    """Test that the getstate and setstate methods work."""
    s = SLURM(cores=4)
    state = s.__getstate__()
    s2 = SLURM(cores=2)
    s2.__setstate__(state)
    assert s._cores == s2._cores


def test_ipyparallel() -> None:
    """Test that the ipyparallel raises."""
    s = SLURM(cores=1, executor_type="ipyparallel")
    with pytest.raises(
        ValueError,
        match="`ipyparalllel` uses 1 cores of the `adaptive.Runner`",
    ):
        s._executor_specific("")


def test_job_script() -> None:
    """Test the SLURM.job_script method."""
    s = SLURM(cores=4)
    job_script = s.job_script()
    assert "#SBATCH --ntasks 4" in job_script
    assert "#SBATCH --exclusive" in job_script
    assert "#SBATCH --no-requeue" in job_script
    assert "MKL_NUM_THREADS=1" in job_script
    assert "OPENBLAS_NUM_THREADS=1" in job_script
    assert "OMP_NUM_THREADS=1" in job_script
    assert "NUMEXPR_NUM_THREADS=1" in job_script


def test_start_job() -> None:
    """Test the SLURM.start_job method."""
    s = SLURM(cores=4)
    s.write_job_script = lambda name: None
    s.run_submit = lambda submit_cmd: None
    s.start_job("testjob")


def test_slurm_job_script_default() -> None:
    """Test the SLURM.job_script method with default arguments."""
    """Test the SLURM.job_script method with default arguments."""
    s = SLURM(cores=4)
    job_script = s.job_script()

    assert "#SBATCH --ntasks 4" in job_script
    assert "#SBATCH --no-requeue" in job_script
    assert "#SBATCH --exclusive" in job_script
    assert "export MKL_NUM_THREADS=1" in job_script
    assert "export OPENBLAS_NUM_THREADS=1" in job_script
    assert "export OMP_NUM_THREADS=1" in job_script
    assert "export NUMEXPR_NUM_THREADS=1" in job_script

    # Check executor_specific section
    executor_specific_section = re.search(
        r"srun --mpi=pmi2 -n 4 .* --log-fname .* --job-id \${SLURM_JOB_ID} --name \${NAME}",
        job_script,
    )
    assert executor_specific_section is not None


def test_slurm_job_script_custom() -> None:
    """Test the SLURM.job_script method with custom arguments."""
    extra_scheduler = ["--time=1", "--mem=4G"]
    extra_env_vars = ["TMPDIR='/scratch'", "PYTHONPATH='my_dir:$PYTHONPATH'"]
    extra_script = "echo 'test'"

    s = SLURM(
        cores=4,
        partition="debug",
        extra_scheduler=extra_scheduler,
        extra_env_vars=extra_env_vars,
        extra_script=extra_script,
    )
    job_script = s.job_script()

    # Check extra_scheduler
    for opt in extra_scheduler:
        assert f"#SBATCH {opt}" in job_script

    # Check extra_env_vars
    for var in extra_env_vars:
        assert f"export {var}" in job_script

    # Check extra_script
    assert extra_script in job_script

    # Check partition
    assert "#SBATCH --partition=debug" in job_script
