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
    assert s._cores == 4
    assert s.nodes is None
    assert s.cores_per_node is None


@pytest.mark.usefixtures("_mock_slurm_partitions")
def test_slurm_scheduler_nodes_cores_per_node() -> None:
    """Test that the nodes and cores_per_node are set correctly."""
    s = SLURM(nodes=2, cores_per_node=2, partition="nc24-low")
    assert s.partition == "nc24-low"
    assert s._cores is None
    assert s.cores == 4
    assert s.nodes == 2
    assert s.cores_per_node == 2


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
    job_script = s.job_script(options={})
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
        s.write_job_script("testjob", {})
        s.start_job("testjob")
        mock_submit.assert_called_once_with(
            f"sbatch --job-name testjob --output {tmp_path}/testjob-%A.out {tmp_path}/testjob.sbatch",
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
    job_script = s.job_script(options={})

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
    job_script = s.job_script(options={})

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


def test_slurm_scheduler_job_script_ipyparallel() -> None:
    """Test the SLURM.job_script method."""
    s = SLURM(
        cores=4,
        extra_scheduler=["--exclusive=user", "--time=1"],
        extra_env_vars=["TMPDIR='/scratch'", "PYTHONPATH='my_dir:$PYTHONPATH'"],
        extra_script="echo 'YOLO'",
        executor_type="ipyparallel",
    )
    job_script = s.job_script(options={"--n": 3})
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

        export TMPDIR='/scratch'
        export PYTHONPATH='my_dir:$PYTHONPATH'
        export EXECUTOR_TYPE=ipyparallel
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export OMP_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1

        echo 'YOLO'

        profile=adaptive_scheduler_${{SLURM_JOB_ID}}

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
        srun --ntasks 1 {s.python_executable} {s.launcher} \\
            --profile ${{profile}} \\
            --log-fname {log_fname} \\
            --job-id ${{SLURM_JOB_ID}} \\
            --name ${{NAME}} \\
            --n 3
        """,
        ).strip()
    )


@pytest.mark.usefixtures("_mock_slurm_partitions")
def test_slurm_scheduler_ipyparallel() -> None:
    """Test the SLURM.job_script method."""
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
    ipy = s._executor_specific("TEST", {"--n": 23975})
    log_fname = s.log_fname("TEST")
    print(ipy)
    assert (
        ipy.strip()
        == textwrap.dedent(
            f"""\
        profile=adaptive_scheduler_${{SLURM_JOB_ID}}

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
        srun --ntasks 1 {s.python_executable} {s.launcher} \\
            --profile ${{profile}} \\
            --log-fname {log_fname} \\
            --job-id ${{SLURM_JOB_ID}} \\
            --name TEST \\
            --n 23975
        """,
        ).strip()
    )


def test_multiple_jobs() -> None:  # noqa: PLR0915
    """Test that multiple jobs can be started."""
    cores = (3, 4, 5)
    s = SLURM(cores=cores)
    for i, n in enumerate(cores):
        js = s.job_script(options={}, index=i)
        assert f"#SBATCH --ntasks {n}" in js
        assert js.count("--ntasks") == 1
    assert isinstance(s._extra_scheduler, tuple)
    assert len(s._extra_scheduler) == 3

    s = SLURM(cores_per_node=cores, nodes=2)
    assert isinstance(s.nodes, tuple)
    assert s.cores == tuple(2 * n for n in cores)

    partitions = ("nc24-low", "hb120v2-low")
    with patch("adaptive_scheduler._scheduler.slurm.slurm_partitions") as mock:
        mock.return_value = {"nc24-low": 24, "hb120v2-low": 120}

        s = SLURM(partition=partitions, cores=1)
        assert s.cores == (1, 1)
        for i, p in enumerate(partitions):
            js = s.job_script(options={}, index=i)
            assert f"#SBATCH --partition={p}" in js
            assert js.count("--partition") == 1

    s = SLURM(cores=cores, extra_scheduler=["--time=1"], executor_type="ipyparallel")
    js = s.job_script(options={}, index=0)
    assert "#SBATCH --time=1" in js
    assert "--ntasks 2" in js

    # Check with incorrect types
    with pytest.raises(TypeError, match=r"Expected `<class 'int'>`"):
        SLURM(cores="4")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="All tuples should have the same length."):
        SLURM(cores_per_node=(4,), nodes=(2, 2))  # type: ignore[arg-type]

    s = SLURM(cores=1, exclusive=(True, False))
    js = s.job_script(options={}, index=0)
    opt = "#SBATCH --exclusive"
    assert opt in js
    js = s.job_script(options={}, index=1)
    assert opt not in js

    s = SLURM(cores=1, exclusive=True)
    js = s.job_script(options={})
    assert opt in js

    s = SLURM(cores=1, extra_env_vars=(["YOLO=1"], []))
    js = s.job_script(options={}, index=0)
    assert "export YOLO=1" in js
    js = s.job_script(options={}, index=1)
    assert "export YOLO=1" not in js

    s = SLURM(cores=1, extra_script=("echo 'YOLO'", ""))
    js = s.job_script(options={}, index=0)
    print(js)
    assert "echo 'YOLO'" in js
    js = s.job_script(options={}, index=1)
    assert "echo 'YOLO'" not in js

    s = SLURM(cores=1, num_threads=(1, 2))
    js = s.job_script(options={}, index=0)
    print(js)
    assert "export OPENBLAS_NUM_THREADS=1" in js
    js = s.job_script(options={}, index=1)
    assert "export OPENBLAS_NUM_THREADS=2" in js


def test_multi_job_script_options() -> None:
    """Test the SLURM.job_script method."""
    s = SLURM(cores=2, executor_type=("ipyparallel", "sequential"))
    assert not s.single_job_script
    s._command_line_options = {"--n": 2}

    # Test ipyparallel
    with patch("adaptive_scheduler._scheduler.slurm.run_submit") as mock_submit:
        s.start_job("testjob", index=0)
        mock_submit.assert_called_once()
        args, _ = mock_submit.call_args
        assert args[0].startswith("sbatch")
    assert "--executor-type ipyparallel" in s.batch_fname("testjob").read_text()

    # Test sequential
    with patch("adaptive_scheduler._scheduler.slurm.run_submit") as mock_submit:
        s.start_job("testjob", index=1)
        mock_submit.assert_called_once()
        args, _ = mock_submit.call_args
        assert args[0].startswith("sbatch")
    assert "--executor-type sequential" in s.batch_fname("testjob").read_text()
