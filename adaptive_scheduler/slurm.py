import getpass
import os
import subprocess
import sys
import textwrap

ext = ".sbatch"
submit_cmd = "sbatch"


def make_job_script(name, cores, run_script="run_learner.py", python_executable=None):
    """Get a jobscript in string form.

    Parameters
    ----------
    name : str
        Name of the job.
    cores : int
        Number of cores per job (so per learner.)
    job_script_function : callable, default: `adaptive_scheduler.slurm.make_job_script` or `adaptive_scheduler.pbs.make_job_script`
        A function with the following signature:
        ``job_script(name, cores, run_script, python_executable)`` that returns
        a job script in string form. See ``adaptive_scheduler/slurm.py`` or
        ``adaptive_scheduler/pbs.py`` for an example.
    run_script : str
        Filename of the script that is run on the nodes. Inside this script we
        query the database and run the learner.
    python_executable : str, default: sys.executable
        The Python executable that should run the `run_script`. By default
        it uses the same Python as where this function is called.

    Returns
    -------
    job_script : str
        A job script that can be submitted to the scheduler system.
    """
    if python_executable is None:
        python_executable = sys.executable
    job_script = textwrap.dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name {name}
        #SBATCH --ntasks {cores}
        #SBATCH --output {name}-%A.out
        #SBATCH --no-requeue

        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export OMP_NUM_THREADS=1

        export MPI4PY_MAX_WORKERS=$SLURM_NTASKS
        srun -n $SLURM_NTASKS --mpi=pmi2 {python_executable} -m mpi4py.futures {run_script}
        """
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
    dictionary of `job_id` -> dict with `name and `state`.
    e.g. ``{job_id: {'name': 'TEST_JOB-1', 'state': "PENDING" or "RUNNING"}}``.

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
