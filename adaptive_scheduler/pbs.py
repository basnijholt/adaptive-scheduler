import collections
import getpass
import math
import os
import subprocess
import warnings

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
    *,
    mpiexec_executable=None,
    executor_type="mpi4py",
    cores_per_node=None,
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
    cores_per_node : int, optional
        Number of cores per node. By default the number will be guessed using the
        ``qnodes`` command.
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

    if cores_per_node is None:
        partial_msg = (
            " Use `functools.partial(make_job_script, cores_per_node=...)` before"
            " passing `make_job_script` to the `job_script_function` argument."
        )
        try:
            max_cores_per_node = _guess_cores_per_node()
            nnodes = math.ceil(cores / max_cores_per_node)
            cores_per_node = round(cores / nnodes)
            msg = (
                f"`#PBS -l nodes={nnodes}:ppn={cores_per_node}` is guessed"
                f" using the `qnodes` command. You might want to change this. {partial_msg}"
            )
            warnings.warn(msg)
            cores = nnodes * cores_per_node
        except Exception as e:
            msg = f"Couldn't guess `cores_per_node`, this argument is required for PBS. {partial_msg}"
            raise Exception(msg) from e
    else:
        nnodes = cores / cores_per_node
        if not float(nnodes).is_integer():
            raise ValueError("cores / cores_per_node must be an integer!")
        else:
            nnodes = int(nnodes)

    python_executable = python_executable or sys.executable
    extra_pbs = extra_pbs or []
    extra_pbs = "\n".join(f"#PBS {arg}" for arg in extra_pbs)
    extra_env_vars = extra_env_vars or []
    extra_env_vars = "\n".join(f"export {arg}" for arg in extra_env_vars)

    mpiexec_executable = mpiexec_executable or "mpiexec"
    if executor_type == "mpi4py":
        executor_specific = f"{mpiexec_executable} -n {cores} {python_executable} -m mpi4py.futures {run_script}"
    elif "dask-mpi":
        executor_specific = (
            f"{mpiexec_executable} -n {cores} {python_executable} {run_script}"
        )
    elif executor_type == "ipyparallel":
        raise NotImplementedError(
            "See https://github.com/ipython/ipyparallel/issues/370"
        )
        # This does not really work yet.
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
    else:
        raise NotImplementedError("Use 'ipyparallel', 'dask-mpi' or 'mpi4py'.")

    job_script = textwrap.dedent(
        f"""\
        #!/bin/sh
        #PBS -l nodes={nnodes}:ppn={cores_per_node}
        #PBS -V
        #PBS -N {name}
        #PBS -o $PBS_O_WORKDIR/{name}.out
        #PBS -e $PBS_O_WORKDIR/{name}.err
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


def _qnodes():
    proc = subprocess.run(["qnodes"], text=True, capture_output=True)
    output = proc.stdout

    if proc.returncode != 0:
        raise RuntimeError("qnodes is not responding.")

    jobs = _split_by_job(output.replace("\n\t", "").split("\n"))

    nodes = {
        node: dict([line.split(" = ") for line in _fix_line_cuts(raw_info)])
        for node, *raw_info in jobs
    }
    return nodes


def _guess_cores_per_node():
    nodes = _qnodes()
    cntr = collections.Counter([int(info["np"]) for info in nodes.values()])
    ncores, freq = cntr.most_common(1)[0]
    return ncores


cancel = _cancel_function("qdel", queue)
