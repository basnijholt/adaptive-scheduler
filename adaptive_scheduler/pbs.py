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
        #!/bin/sh
        #PBS -t 1-{cores}
        #PBS -V
        #PBS -N {name}
        #PBS -o {name}.out

        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export OMP_NUM_THREADS=1

        export MPI4PY_MAX_WORKERS={cores}
        mpiexec -n {cores} {python_executable} -m mpi4py.futures {run_script}
        """
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


def queue(me_only=False):
    cmd = ["qstat", "-f"]
    if me_only:
        username = getpass.getuser()
        cmd.append(f"-u={username}")
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=dict(os.environ, SGE_LONG_QNAMES="1000"),
    )
    output = proc.stdout

    if proc.returncode != 0:
        raise RuntimeError("qstat is not responding.")

    jobs = _split_by_job(output.split("\n"))

    running = {}
    for header, *raw_info in jobs:
        jobid = header.split("Job Id: ")[1]
        info = dict([line.split(" = ") for line in _fix_line_cuts(raw_info)])
        if info["job_state"] in ["R", "Q"]:
            info["name"] = info["Job_Name"]  # used in `server_support.manage_jobs`
            running[jobid] = info
    return running


def get_job_id():
    return os.environ.get("PBS_JOBID", "UNKNOWN")
