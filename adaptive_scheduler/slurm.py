import getpass
import os
import subprocess
import sys
import textwrap


def make_sbatch(name, cores, run_script="run_learner.py", python_executable=None):
    if python_executable is None:
        python_executable = sys.executable
    job_script = textwrap.dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name {name}
        #SBATCH --ntasks {cores}
        #SBATCH --output {name}.out
        #SBATCH --no-requeue

        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export OMP_NUM_THREADS=1

        export MPI4PY_MAX_WORKERS=$SLURM_NTASKS
        srun -n $SLURM_NTASKS --mpi=pmi2 {python_executable} -m mpi4py.futures {run_script}
        """
    )
    return job_script


def check_running(me_only=True):
    cmd = [
        "/usr/bin/squeue",
        r'--Format=",jobid:100,name:100,state:100,numnodes:100,reasonlist:400,"',
        "--noheader",
        "--array",
    ]
    if me_only:
        username = getpass.getuser()
        cmd.append(f"--user={username}")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    squeue = proc.stdout

    if (
        "squeue: error" in squeue
        or "slurm_load_jobs error" in squeue
        or proc.returncode != 0
    ):
        raise RuntimeError("SLURM is too busy.")

    squeue = [line.split() for line in squeue.split("\n")]
    squeue = [line for line in squeue if line]
    allowed = ("PENDING", "RUNNING")
    running = {
        job_id: dict(
            job_name=job_name,
            state=state,
            n_nodes=int(n_nodes),
            reason_list=reason_list,
        )
        for job_id, job_name, state, n_nodes, reason_list in squeue
        if state in allowed
    }
    return running


def get_job_id():
    return os.environ.get("SLURM_JOB_ID", "UNKNOWN")
