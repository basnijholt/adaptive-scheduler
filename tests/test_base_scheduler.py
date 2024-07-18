"""Tests for the BaseScheduler class."""

from __future__ import annotations

import textwrap

import pytest

from .helpers import MockScheduler


def test_base_scheduler_job_script() -> None:
    """Test the BaseScheduler.job_script method."""
    s = MockScheduler(
        cores=4,
        extra_scheduler=["--exclusive=user", "--time=1"],
        extra_env_vars=["TMPDIR='/scratch'", "PYTHONPATH='my_dir:$PYTHONPATH'"],
        extra_script="echo 'YOLO'",
        executor_type="mpi4py",
    )
    job_script = s.job_script(options={})
    log_fname = s.log_fname("${NAME}")
    assert job_script == textwrap.dedent(
        f"""\
        #!/bin/bash
        #MOCK --cores 4
        ##MOCK --exclusive=user
        ##MOCK --time=1

        export TMPDIR='/scratch'
        export PYTHONPATH='my_dir:$PYTHONPATH'
        export EXECUTOR_TYPE=mpi4py
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export OMP_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1

        echo 'YOLO'

        {s.mpiexec_executable} \\
            -n 4 {s.python_executable} \\
            -m mpi4py.futures {s.launcher} \\
            --log-fname {log_fname} \\
            --job-id ${{JOB_ID}} \\
            --name ${{NAME}}
        """,
    )


def test_base_scheduler() -> None:
    """Test that the base scheduler is set correctly."""
    s = MockScheduler(cores=4)
    assert s.cores == 4


def test_queue() -> None:
    """Test the queue method of MockScheduler."""
    scheduler = MockScheduler(cores=1)
    scheduler.start_job("test_job1")
    scheduler.start_job("test_job2")

    queue_info = scheduler.queue()
    assert len(queue_info) == 2
    assert queue_info["0"]["job_name"] == "test_job1"
    assert queue_info["1"]["job_name"] == "test_job2"


def test_cancel() -> None:
    """Test the cancel method of MockScheduler."""
    scheduler = MockScheduler(cores=2)
    scheduler.start_job("test_job1")
    scheduler.start_job("test_job2")
    assert len(scheduler.queue()) == 2
    scheduler.cancel(["test_job1"])

    queue_info = scheduler.queue()
    assert len(queue_info) == 1
    assert queue_info["1"]["job_name"] == "test_job2"


def test_update_queue() -> None:
    """Test the update_queue method of MockScheduler."""
    scheduler = MockScheduler(cores=2)
    scheduler.start_job("test_job1")
    scheduler.start_job("test_job2")
    j1, _ = scheduler.job_names_to_job_ids("test_job1", "test_job2")
    scheduler.update_queue("test_job1", "COMPLETED")

    queue_info = scheduler.queue()
    assert len(queue_info) == 2
    assert queue_info[j1]["status"] == "COMPLETED"


def test_ipyparallel() -> None:
    """Test that the ipyparallel raises."""
    s = MockScheduler(cores=1, executor_type="ipyparallel")
    with pytest.raises(
        ValueError,
        match="`ipyparalllel` uses 1 cores of the `adaptive.Runner`",
    ):
        s._executor_specific("NAME", {})


def test_getstate_setstate() -> None:
    """Test that the getstate and setstate methods work."""
    s = MockScheduler(cores=4)
    state = s.__getstate__()
    s2 = MockScheduler(cores=2)
    s2.__setstate__(state)
    assert s.cores == s2.cores


def test_base_scheduler_ipyparallel() -> None:
    """Test the BaseScheduler.job_script method."""
    s = MockScheduler(
        cores=4,
        extra_scheduler=["--exclusive=user", "--time=1"],
        extra_env_vars=["TMPDIR='/scratch'", "PYTHONPATH='my_dir:$PYTHONPATH'"],
        extra_script="echo 'YOLO'",
        executor_type="ipyparallel",
    )
    ipy = s._executor_specific("TEST", {"--save-dataframe": None, "--log-interval": 10})
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
            mpiexec \\
                -n 3 \\
                ipengine \\
                --profile=${{profile}} \\
                --mpi \\
                --cluster-id='' \\
                --log-to-file &

            echo "Starting the Python script"
            {s.python_executable} {s.launcher} \\
                --profile ${{profile}} \\
                --n 3 \\
                --log-fname {log_fname} \\
                --job-id ${{JOB_ID}} \\
                --name TEST \\
                --save-dataframe \\
                --log-interval 10
            """,
        ).strip()
    )


def test_callable_scheduler_arguments() -> None:
    """Test that the scheduler arguments can be callables."""
    s = MockScheduler(
        cores=(4, lambda: 2),
        executor_type=(lambda: "ipyparallel", "mpi4py"),
        num_threads=(lambda: 2, 1),
        extra_scheduler=(
            ["--exclusive=user", "--time=1"],
            lambda: ["--exclusive=user", "--time=2"],
        ),
        extra_env_vars=(lambda: ["from=func"], ["from=static"]),
        extra_script=(lambda: "echo 'func'", "echo 'static'"),
    )

    js0 = s.job_script(options={}, index=0)
    js1 = s.job_script(options={}, index=1)
    assert js0 != js1
    extra_scheduler0 = s.extra_scheduler(index=0)
    extra_scheduler1 = s.extra_scheduler(index=1)
    assert extra_scheduler0 == "##MOCK --exclusive=user\n##MOCK --time=1"
    assert extra_scheduler1 == "##MOCK --exclusive=user\n##MOCK --time=2"
    extra_env_vars0 = s.extra_env_vars(index=0)
    extra_env_vars1 = s.extra_env_vars(index=1)
    assert extra_env_vars0.startswith("export from=func\n")
    assert extra_env_vars1.startswith("export from=static\n")
    extra_script0 = s.extra_script(index=0)
    extra_script1 = s.extra_script(index=1)
    assert extra_script0 == "echo 'func'"
    assert extra_script1 == "echo 'static'"
    cores0 = s._get_cores(index=0)
    cores1 = s._get_cores(index=1)
    assert cores0 == 4
    assert cores1 == 2
    executor_type0 = s._get_executor_type(index=0)
    executor_type1 = s._get_executor_type(index=1)
    assert executor_type0 == "ipyparallel"
    assert executor_type1 == "mpi4py"
