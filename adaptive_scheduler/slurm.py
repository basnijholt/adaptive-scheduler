import getpass
import os
import subprocess

from adaptive_scheduler.utils import _cancel_function

ext = ".sbatch"
submit_cmd = "sbatch"


def make_job_script(
    name,
    cores,
    run_script="run_learner.py",
    python_executable=None,
    *,
    mpiexec_executable=None,
    executor_type="mpi4py",
    extra_sbatch=None,
    extra_env_vars=None,
    num_threads=1,
):
    """Get a jobscript in string form.

    Parameters
    ----------
    name : str
        Name of the job.
    cores : int
        Number of cores per job (so per learner.)
    run_script : str, default: "run_learner.py"
        Filename of the script that is run on the nodes. Inside this script we
        query the database and run the learner.
    python_executable : str, default: sys.executable
        The Python executable that should run the `run_script`. By default
        it uses the same Python as where this function is called.
    mpiexec_executable : str, optional
        ``mpiexec`` executable. By default ``srun --mpi=pmi2`` will be
        used, you can also use ``mpiexec`` (which is probably from ``conda``).
    executor_type : str, default: "mpi4py"
        The executor that is used, by default `mpi4py.futures.MPIPoolExecutor` is used.
        One can use ``"ipyparallel"`` too.
    extra_sbatch : list, optional
        Extra ``#SBATCH`` arguments, e.g. ``["--exclusive=user", "--time=1"]``.
    extra_env_vars : list, optional
        Extra environment variables that are exported in the job
        script. e.g. ``["TMPDIR='/scratch'", "PYTHONPATH='my_dir:$PYTHONPATH'"]``.
    num_threads : int, default 1
        ``MKL_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``, and ``OMP_NUM_THREADS``
        will be set to this number.

    Returns
    -------
    job_script : str
        A job script that can be submitted to the scheduler system.
    """
    import sys
    import textwrap

    python_executable = python_executable or sys.executable
    extra_sbatch = extra_sbatch or []
    extra_sbatch = "\n".join(f"#SBATCH {arg}" for arg in extra_sbatch)
    extra_env_vars = extra_env_vars or []
    extra_env_vars = "\n".join(f"export {arg}" for arg in extra_env_vars)

    mpiexec_executable = mpiexec_executable or "srun --mpi=pmi2"
    if executor_type == "mpi4py":
        executor_specific = f"{mpiexec_executable} -n {cores} {python_executable} -m mpi4py.futures {run_script}"
    elif executor_type == "dask-mpi":
        executor_specific = (
            f"{mpiexec_executable} -n {cores} {python_executable} {run_script}"
        )
    elif executor_type == "ipyparallel":
        job_id = "${SLURM_JOB_ID}"
        profile = "${profile}"
        executor_specific = textwrap.dedent(
            f"""\
            profile=job_{job_id}_$(hostname)

            echo "Creating profile {profile}"
            ipython profile create {profile}

            echo "Launching controller"
            ipcontroller --ip="*" --profile={profile} --log-to-file &
            sleep 10

            echo "Launching engines"
            srun --ntasks {cores-1} ipengine --profile={profile} --cluster-id='' --log-to-file &

            echo "Starting the Python script"
            srun --ntasks 1 {python_executable} {run_script} {profile} {cores-1}
            """
        )
    else:
        raise NotImplementedError("Use 'ipyparallel', 'dask-mpi' or 'mpi4py'.")

    job_script = textwrap.dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name {name}
        #SBATCH --ntasks {cores}
        #SBATCH --output {name}-%A.out
        #SBATCH --no-requeue
        {{extra_sbatch}}

        export MKL_NUM_THREADS={num_threads}
        export OPENBLAS_NUM_THREADS={num_threads}
        export OMP_NUM_THREADS={num_threads}
        {{extra_env_vars}}

        {{executor_specific}}
        """
    )

    job_script = job_script.format(
        extra_sbatch=extra_sbatch,
        extra_env_vars=extra_env_vars,
        executor_specific=executor_specific,
    )

    return job_script


def queue(me_only=True):
    """Get the current running and pending jobs.

    Parameters
    ----------
    me_only : bool, default: True
        Only see your jobs.

    Returns
    -------
    dictionary of `job_id` -> dict with `name` and `state`, for
    example ``{job_id: {"name": "TEST_JOB-1", "state": "RUNNING" or "PENDING"}}``.

    Notes
    -----
    This function returns extra information about the job, however this is not
    used elsewhere in this package.
    """
    python_format = {
        "jobid": 100,
        "name": 100,
        "state": 100,
        "numnodes": 100,
        "reasonlist": 4000,
    }  # (key -> length) mapping

    slurm_format = ",".join(f"{k}:{v}" for k, v in python_format.items())
    cmd = ["/usr/bin/squeue", rf'--Format=",{slurm_format},"', "--noheader", "--array"]
    if me_only:
        username = getpass.getuser()
        cmd.append(f"--user={username}")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    output = proc.stdout

    if (
        "squeue: error" in output
        or "slurm_load_jobs error" in output
        or proc.returncode != 0
    ):
        raise RuntimeError("SLURM is not responding.")

    def line_to_dict(line):
        line = list(line)
        info = {}
        for k, v in python_format.items():
            info[k] = "".join(line[:v]).strip()
            line = line[v:]
        return info

    squeue = [line_to_dict(line) for line in output.split("\n")]
    squeue = [info for info in squeue if info["state"] in ("PENDING", "RUNNING")]
    running = {info.pop("jobid"): info for info in squeue}
    return running


def get_job_id():
    """Get the job_id from the current job's environment."""
    return os.environ.get("SLURM_JOB_ID", "UNKNOWN")


cancel = _cancel_function("scancel", queue)
