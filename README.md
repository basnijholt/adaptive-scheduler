# Asynchronous Job Scheduler for Adaptive :rocket:

[![PyPI](https://img.shields.io/pypi/v/adaptive-scheduler.svg)](https://pypi.python.org/pypi/adaptive-scheduler)
[![Conda](https://img.shields.io/conda/v/conda-forge/adaptive-scheduler.svg?label=conda-forge)](https://anaconda.org/conda-forge/adaptive-scheduler)
[![Downloads](https://anaconda.org/conda-forge/adaptive-scheduler/badges/downloads.svg)](https://anaconda.org/conda-forge/adaptive-scheduler)
[![Build Status](https://github.com/basnijholt/adaptive-scheduler/actions/workflows/pytest.yml/badge.svg)](https://github.com/basnijholt/adaptive-scheduler/actions/workflows/pytest.yml)
[![Documentation Status](https://readthedocs.org/projects/adaptive-scheduler/badge/?version=latest)](https://adaptive-scheduler.readthedocs.io/en/latest/?badge=latest)
[![CodeCov](https://codecov.io/gh/basnijholt/adaptive-scheduler/branch/main/graph/badge.svg)](https://codecov.io/gh/basnijholt/adaptive-scheduler)

This is an asynchronous job scheduler for [`Adaptive`](https://github.com/python-adaptive/adaptive/), designed to run many `adaptive.Learner`s on many cores (>10k) using `mpi4py.futures`, `ipyparallel`, or `distributed`.

<!-- toc-start -->
## :books: Table of Contents
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [🤔 What is this?](#thinking-what-is-this)
- [🎯 Design Goals](#dart-design-goals)
- [🧪 How does it work?](#test_tube-how-does-it-work)
- [🔍 But how does it *really* work?](#mag-but-how-does-it-really-work)
- [📓 Jupyter Notebook Example](#notebook-jupyter-notebook-example)
- [💻 Installation](#computer-installation)
- [🛠️ Development](#hammer_and_wrench-development)
- [⚠️ Limitations](#warning-limitations)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->
<!-- toc-end -->

## :thinking: What is this?

The Adaptive scheduler is designed to solve the problem of running more learners than you can run with a single runner and/or can use >1k cores.

`ipyparallel` and `distributed` provide powerful engines for interactive sessions. However, when you want to connect to >1k cores, they start to struggle. Furthermore, on a shared cluster, there is often the problem of starting an interactive session with enough space available.

Our approach is to schedule a different job for each `adaptive.Learner`. The creation and running of these jobs are managed by `adaptive-scheduler`. This means that your calculation will definitely run, even though the cluster might be fully occupied at the moment. Because of this approach, there is almost no limit to how many cores you want to use. You can either use 10 nodes for 1 job (`learner`) or 1 core for 1 job (`learner`) while scheduling hundreds of jobs.

The computation is designed to be maximally local. This means that if one of the jobs crashes, there is no problem and it will automatically schedule a new one and continue the calculation where it left off (thanks to Adaptive's periodic saving functionality). Even if the central "job manager" dies, the jobs will continue to run (although no new jobs will be scheduled).

## :dart: Design Goals

1. Needs to be able to run efficiently on >30k cores.
2. Works seamlessly with the Adaptive package.
3. Minimal load on the file system.
4. Removes all boilerplate of working with a scheduler:
   - Writes job script.
   - (Re)submits job scripts.
5. Handles random crashes (or node evictions) with minimal data loss.
6. Preserves Python kernel and variables inside a job (in contrast to submitting jobs for every parameter).
7. Separates the simulation definition code from the code that runs the simulation.
8. Maximizes computation locality, jobs continue to run when the main process dies.

## :test_tube: How does it work?

You create a bunch of `learners` and corresponding `fnames` so they can be loaded, like:

```python
import adaptive
from functools import partial

def h(x, pow, a):
    return a * x**pow

combos = adaptive.utils.named_product(
    pow=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    a=[0.1, 0.5],
)  # returns list of dicts, cartesian product of all values

learners = [adaptive.Learner1D(partial(h, **combo),
            bounds=(-1, 1)) for combo in combos]
fnames = [f"data/{combo}" for combo in combos]
```

Then you start a process that creates and submits as many job-scripts as there are learners, like:

```python
import adaptive_scheduler

def goal(learner):
    return learner.npoints > 200

scheduler = adaptive_scheduler.scheduler.SLURM(cores=10)  # every learner gets this many cores

run_manager = adaptive_scheduler.server_support.RunManager(
    scheduler,
    learners,
    fnames,
    goal=goal,
    log_interval=30,  # write info such as npoints, cpu_usage, time, etc. to the job log file
    save_interval=300,  # save the data every 300 seconds
)
run_manager.start()
```

That's it! You can run `run_manager.info()` which will display an interactive `ipywidget` that shows the amount of running, pending, and finished jobs, buttons to cancel your job, and other useful information.

![Widget demo](http://files.nijho.lt/info.gif)

## :mag: But how does it *really* work?

The `adaptive_scheduler.server_support.RunManager` basically does the following:

- *You* need to create `N` `learners` and `fnames` (like in the section above).
- Then a "job manager" writes and submits `max(N, max_simultaneous_jobs)` job scripts but *doesn't know* which learner it is going to run!
- This is the responsibility of the "database manager", which keeps a database of `job_id <--> learner`.
- The job script starts a Python file `run_learner.py` in which the learner is run.

In a Jupyter notebook, you can start the "job manager" and the "database manager", and create the `run_learner.py` like:

```python
import adaptive_scheduler
from adaptive_scheduler import server_support

# create a scheduler
scheduler = adaptive_scheduler.scheduler.SLURM(cores=10, run_script="run_learner.py",)

# create a new database that keeps track of job <-> learner
db_fname = "running.json"
url = (
   server_support.get_allowed_url()
)  # get a url where we can run the database_manager
database_manager = server_support.DatabaseManager(
   url, scheduler, db_fname, learners, fnames
)
database_manager.start()

# create the Python script that runs a learner (run_learner.py)
server_support._make_default_run_script(
   url=url,
   save_interval=300,
   log_interval=30,
   goal=None,
   executor_type=scheduler.executor_type,
   run_script_fname=scheduler.run_script,
)

# create unique names for the jobs
n_jobs = len(learners)
job_names = [f"test-job-{i}" for i in range(n_jobs)]

job_manager = server_support.JobManager(job_names, database_manager, scheduler)
job_manager.start()
```

Then, when the jobs have been running for a while, you can check `server_support.parse_log_files(database_manager, scheduler)`.

And use `scheduler.cancel(job_names)` to cancel the jobs.

You don't actually ever have to leave the Jupyter notebook; take a look at the [`example notebook`](https://github.com/basnijholt/adaptive-scheduler/blob/master/example.ipynb).

## :notebook: Jupyter Notebook Example

See [`example.ipynb`](https://github.com/basnijholt/adaptive-scheduler/blob/master/example.ipynb).

## :computer: Installation

**WARNING:** This is still in the pre-alpha development stage.

Install the **latest stable** version from conda (recommended):

```bash
conda install adaptive-scheduler
```

or from PyPI:

```bash
pip install adaptive_scheduler
```

or install **master** with:

```bash
pip install -U https://github.com/basnijholt/adaptive-scheduler/archive/master.zip
```

or clone the repository and do a dev install (recommended for dev):

```bash
git clone git@github.com:basnijholt/adaptive-scheduler.git
cd adaptive-scheduler
pip install -e .
```

## :hammer_and_wrench: Development

In order not to pollute the history with the output of the notebooks, please set up the git filter by executing:

```bash
python ipynb_filter.py
```

in the repository.

We also use `pre-commit`, so `pip install pre_commit` and run:

```bash
pre-commit install
```

in the repository.

## :warning: Limitations

Currently, `adaptive_scheduler` only works for SLURM and PBS.
However, only a class like [`adaptive_scheduler/scheduler.py`](https://github.com/basnijholt/adaptive-scheduler/blob/master/adaptive_scheduler/scheduler.py#L471) would have to be implemented for another type of scheduler.
