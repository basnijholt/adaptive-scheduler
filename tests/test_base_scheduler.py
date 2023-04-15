"""Tests for the BaseScheduler class."""
import textwrap

from .helpers import MockScheduler


def test_base_scheduler_job_script() -> None:
    """Test the BaseScheduler.job_script method."""
    self = MockScheduler(
        cores=4,
        extra_scheduler=["--exclusive=user", "--time=1"],
        extra_env_vars=["TMPDIR='/scratch'", "PYTHONPATH='my_dir:$PYTHONPATH'"],
        extra_script="echo 'YOLO'",
    )
    job_script = self.job_script()
    log_fname = self.log_fname("${NAME}")
    assert job_script == textwrap.dedent(
        f"""\
        #!/bin/bash
        #MOCK --cores 4
        ##MOCK --exclusive=user
        ##MOCK --time=1

        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export OMP_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1

        export TMPDIR='/scratch'
        export PYTHONPATH='my_dir:$PYTHONPATH'

        echo 'YOLO'

        {self.mpiexec_executable} \\
            -n 4 {self.python_executable} \\
            -m mpi4py.futures run_learner.py \\
            --log-fname {log_fname} \\
            --job-id ${{JOB_ID}} \\
            --name ${{NAME}}
        """,
    )


def test_base_scheduler() -> None:
    """Test that the base scheduler is set correctly."""
    s = MockScheduler(cores=4)
    assert s.cores == 4  # noqa: PLR2004


def test_queue() -> None:
    """Test the queue method of MockScheduler."""
    scheduler = MockScheduler(cores=1)
    scheduler.start_job("test_job1")
    scheduler.start_job("test_job2")

    queue_info = scheduler.queue()
    assert len(queue_info) == 2  # noqa: PLR2004
    assert queue_info["0"]["job_name"] == "test_job1"
    assert queue_info["1"]["job_name"] == "test_job2"


def test_cancel() -> None:
    """Test the cancel method of MockScheduler."""
    scheduler = MockScheduler(cores=2)
    scheduler.start_job("test_job1")
    scheduler.start_job("test_job2")
    assert len(scheduler.queue()) == 2  # noqa: PLR2004
    scheduler.cancel(["test_job1"])

    queue_info = scheduler.queue()
    assert len(queue_info) == 1
    assert queue_info["1"]["job_name"] == "test_job2"


def test_update_queue() -> None:
    """Test the update_queue method of MockScheduler."""
    scheduler = MockScheduler(cores=2)
    scheduler.start_job("test_job1")
    scheduler.start_job("test_job2")
    scheduler.update_queue("1", "COMPLETED")

    queue_info = scheduler.queue()
    assert len(queue_info) == 2  # noqa: PLR2004
    assert queue_info["1"]["status"] == "COMPLETED"
