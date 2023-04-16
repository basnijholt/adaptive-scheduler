"""Tests for the SLURM scheduler."""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

import adaptive_scheduler
from adaptive_scheduler._scheduler.slurm import SLURM

from .helpers import PARTITIONS, temporary_working_directory


def test_init_cores() -> None:
    """Test that the cores are set correctly."""
    with pytest.raises(
        ValueError,
        match="Specify either `nodes` and `cores_per_node`, or only `cores`, not both.",
    ):
        SLURM(cores=4, nodes=2, cores_per_node=2)


@pytest.mark.usefixtures("_mock_slurm_partitions")
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
    assert s._cores == 4  # noqa: PLR2004
    assert s.nodes is None
    assert s.cores_per_node is None


@pytest.mark.usefixtures("_mock_slurm_partitions")
def test_slurm_scheduler_nodes_cores_per_node() -> None:
    """Test that the nodes and cores_per_node are set correctly."""
    s = SLURM(nodes=2, cores_per_node=2, partition="nc24-low")
    assert s.partition == "nc24-low"
    assert s._cores is None
    assert s.cores == 4  # noqa: PLR2004
    assert s.nodes == 2  # noqa: PLR2004
    assert s.cores_per_node == 2  # noqa: PLR2004


def test_getstate_setstate() -> None:
    """Test that the getstate and setstate methods work."""
    s = SLURM(cores=4)
    state = s.__getstate__()
    s2 = SLURM(cores=2)
    s2.__setstate__(state)
    assert s._cores == s2._cores


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


@pytest.mark.usefixtures("_mock_slurm_partitions")
def test_start_job(tmp_path: Path) -> None:
    """Test the SLURM.start_job method."""
    s = SLURM(cores=4, partition="nc24-low")
    with temporary_working_directory(tmp_path), patch(
        "adaptive_scheduler._scheduler.slurm.run_submit",
    ) as mock_submit:
        s.start_job("testjob")
        mock_submit.assert_called_once_with(
            f"sbatch --job-name testjob --output {tmp_path}/testjob-%A.out testjob.sbatch",
            "testjob",
        )
        sbatch_file = tmp_path / "testjob.sbatch"
        assert Path.cwd() / s.batch_fname("testjob") == sbatch_file
        assert sbatch_file.exists()
        lines = sbatch_file.read_text()
        assert "#SBATCH --ntasks 4" in lines


@pytest.mark.usefixtures("_mock_slurm_partitions")
def test_slurm_job_script_default() -> None:
    """Test the SLURM.job_script method with default arguments."""
    """Test the SLURM.job_script method with default arguments."""
    s = SLURM(cores=4, partition="nc24-low")
    job_script = s.job_script()

    assert "#SBATCH --ntasks 4" in job_script
    assert "#SBATCH --no-requeue" in job_script
    assert "#SBATCH --exclusive" in job_script
    assert "export MKL_NUM_THREADS=1" in job_script
    assert "export OPENBLAS_NUM_THREADS=1" in job_script
    assert "export OMP_NUM_THREADS=1" in job_script
    assert "export NUMEXPR_NUM_THREADS=1" in job_script


@pytest.mark.usefixtures("_mock_slurm_partitions")
def test_slurm_job_script_custom() -> None:
    """Test the SLURM.job_script method with custom arguments."""
    extra_scheduler = ["--time=1", "--mem=4G"]
    extra_env_vars = ["TMPDIR='/scratch'", "PYTHONPATH='my_dir:$PYTHONPATH'"]
    extra_script = "echo 'test'"

    s = SLURM(
        cores=4,
        partition="nc24-low",
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
    assert "#SBATCH --partition=nc24-low" in job_script


@pytest.mark.usefixtures("_mock_slurm_partitions_output")
def test_slurm_partitions() -> None:
    """Test slurm_partitions function."""
    partitions = adaptive_scheduler._scheduler.slurm.slurm_partitions(with_ncores=False)
    assert partitions == [
        "nc24-low",
        "hb120v2-low",
        "hb60-high",
        "nd40v2-mpi",
    ]
    partitions = adaptive_scheduler._scheduler.slurm.slurm_partitions(with_ncores=True)
    assert partitions == PARTITIONS


@pytest.mark.usefixtures("_mock_slurm_partitions")
def test_slurm_partitions_mock() -> None:
    """Test slurm_partitions function."""
    assert adaptive_scheduler._scheduler.slurm.slurm_partitions() == PARTITIONS


def test_base_scheduler_job_script_ipyparallel() -> None:
    """Test the BaseScheduler.job_script method."""
    s = SLURM(
        cores=4,
        extra_scheduler=["--exclusive=user", "--time=1"],
        extra_env_vars=["TMPDIR='/scratch'", "PYTHONPATH='my_dir:$PYTHONPATH'"],
        extra_script="echo 'YOLO'",
        executor_type="ipyparallel",
    )
    job_script = s.job_script()
    log_fname = s.log_fname("${NAME}")
    assert (
        job_script.strip()
        == textwrap.dedent(
            f"""\
        #!/bin/bash
        #SBATCH --ntasks 4
        #SBATCH --no-requeue
        #SBATCH --exclusive=user
        #SBATCH --time=1
        #SBATCH --exclusive

        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export OMP_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        export TMPDIR='/scratch'
        export PYTHONPATH='my_dir:$PYTHONPATH'

        echo 'YOLO'

        profile=adaptive_scheduler_${{JOB_ID}}

        echo "Creating profile ${{profile}}"
        ipython profile create ${{profile}}

        echo "Launching controller"
        ipcontroller --ip="*" --profile=${{profile}} --log-to-file &
        sleep 10

        echo "Launching engines"
        srun --ntasks 3 ipengine \\
            --profile=${{profile}} \\
            --cluster-id='' \\
            --log-to-file &

        echo "Starting the Python script"
        srun --ntasks 1 {s.python_executable} run_learner.py \\
            --profile ${{profile}} \\
            --n 3 \\
            --log-fname {log_fname} \\
            --job-id ${{JOB_ID}} \\
            --name ${{NAME}}
        """,
        ).strip()
    )


@pytest.mark.usefixtures("_mock_slurm_partitions")
def test_base_scheduler_ipyparallel() -> None:
    """Test the BaseScheduler.job_script method."""
    s = SLURM(
        extra_scheduler=["--exclusive=user", "--time=1"],
        extra_env_vars=["TMPDIR='/scratch'", "PYTHONPATH='my_dir:$PYTHONPATH'"],
        extra_script="echo 'YOLO'",
        executor_type="ipyparallel",
        nodes=999,
        cores_per_node=24,
        partition="nc24-low",
        exclusive=True,
    )
    assert s.cores == 999 * 24
    ipy = s._ipyparallel("TEST")
    log_fname = s.log_fname("TEST")
    print(ipy)
    assert (
        ipy.strip()
        == textwrap.dedent(
            f"""\
        profile=adaptive_scheduler_${{JOB_ID}}

        echo "Creating profile ${{profile}}"
        ipython profile create ${{profile}}

        echo "Launching controller"
        ipcontroller --ip="*" --profile=${{profile}} --log-to-file &
        sleep 10

        echo "Launching engines"
        srun --ntasks 23975 ipengine \\
            --profile=${{profile}} \\
            --cluster-id='' \\
            --log-to-file &

        echo "Starting the Python script"
        srun --ntasks 1 {s.python_executable} run_learner.py \\
            --profile ${{profile}} \\
            --n 23975 \\
            --log-fname {log_fname} \\
            --job-id ${{JOB_ID}} \\
            --name TEST
        """,
        ).strip()
    )
