import getpass
import os
import subprocess

from adaptive_scheduler.utils import _cancel_function

ext = ".batch"

# "-k oe" writes the log output to files directly instead of
# at the end of the job. The downside is that the logfiles
# are put in the homefolder.
submit_cmd = "qsub -k oe"


def make_job_script(
    name,
    cores,
    run_script="run_learner.py",
    python_executable=None,
    mpiexec_executable=None,
    executor_type="mpi4py",
    *,
    extra_pbs=None,
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
        ``mpiexec`` executable. By default `mpiexec` will be
        used (so probably from ``conda``).
    executor_type : str, default: "mpi4py"
        The executor that is used, by default `mpi4py.futures.MPIPoolExecutor` is used.
        One can use ``"ipyparallel"`` too.
    extra_pbs : list, optional
        Extra ``#PBS`` arguments, e.g. ``["--exclusive=user", "--time=1"]``.
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
    extra_pbs = extra_pbs or []
    extra_pbs = "\n".join(f"#PBS {arg}" for arg in extra_pbs)
    extra_env_vars = extra_env_vars or []
    extra_env_vars = "\n".join(f"export {arg}" for arg in extra_env_vars)

    if executor_type == "mpi4py":
        mpiexec_executable = mpiexec_executable or "mpiexec"
        executor_specific = f"{mpiexec_executable} -n {cores} {python_executable} -m mpi4py.futures {run_script}"
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
            {mpiexec_executable} -n {cores-1} ipengine --profile={profile} --cluster-id='' --log-to-file &

            echo "Starting the Python script"
            {python_executable} {run_script} {profile} {cores-1}
            """
        )

    job_script = textwrap.dedent(
        f"""\
        #!/bin/sh
        #PBS -t 1-{cores}
        #PBS -V
        #PBS -N {name}
        #PBS -o {name}.out
        {{extra_pbs}}

        export MKL_NUM_THREADS={num_threads}
        export OPENBLAS_NUM_THREADS={num_threads}
        export OMP_NUM_THREADS={num_threads}
        {{extra_env_vars}}

        cd $PBS_O_WORKDIR

        {{executor_specific}}
        """
    )

    job_script = job_script.format(
        extra_pbs=extra_pbs,
        extra_env_vars=extra_env_vars,
        executor_specific=executor_specific,
    )

    return job_script


def _fix_line_cuts(raw_info):
    info = []
    for line in raw_info:
        if " = " in line:
            info.append(line)
        else:
            info[-1] += line
    return info


def _split_by_job(lines):
    jobs = [[]]
    for line in lines:
        line = line.strip()
        if line:
            jobs[-1].append(line)
        else:
            jobs.append([])
    return [j for j in jobs if j]


def queue(me_only=True):
    """Get the current running and pending jobs.

    Parameters
    ----------
    me_only : bool, default: True
        Only see your jobs.

    Returns
    -------
    dictionary of `job_id` -> dict with `name` and `state`, for
    example ``{job_id: {"name": "TEST_JOB-1", "state": "R" or "Q"}}``.

    Notes
    -----
    This function returns extra information about the job, however this is not
    used elsewhere in this package.
    """
    cmd = ["qstat", "-f"]
    if me_only:
        username = getpass.getuser()
        cmd.extend(["-u", username])
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=dict(os.environ, SGE_LONG_QNAMES="1000"),
    )
    output = proc.stdout

    if proc.returncode != 0:
        raise RuntimeError("qstat is not responding.")

    jobs = _split_by_job(output.replace("\n\t", "").split("\n"))

    running = {}
    for header, *raw_info in jobs:
        jobid = header.split("Job Id: ")[1]
        info = dict([line.split(" = ") for line in _fix_line_cuts(raw_info)])
        if info["job_state"] in ["R", "Q"]:
            info["name"] = info["Job_Name"]  # used in `server_support.manage_jobs`
            info["state"] = info["job_state"]  # used in `RunManager.live`
            running[jobid] = info
    return running


def get_job_id():
    """Get the job_id from the current job's environment."""
    return os.environ.get("PBS_JOBID", "UNKNOWN")


cancel = _cancel_function("qdel", queue)
