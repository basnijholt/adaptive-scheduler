import getpass
import os
import subprocess
import sys
import textwrap


def make_job_script(name, cores, run_script="run_learner.py", python_executable=None):
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
    python_format = {
        "jobid": 100,
        "name": 100,
        "state": 100,
        "numnodes": 100,
        "reasonlist": 1000,
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
    return os.environ.get("SLURM_JOB_ID", "UNKNOWN")
