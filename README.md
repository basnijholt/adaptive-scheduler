# An asynchronous scheduler using MPI for [Adaptive](https://github.com/python-adaptive/adaptive/)

[![PyPI](https://img.shields.io/pypi/v/adaptive-scheduler.svg)](https://pypi.python.org/pypi/adaptive-scheduler)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Conda](https://anaconda.org/conda-forge/adaptive-scheduler/badges/installer/conda.svg)](https://anaconda.org/conda-forge/adaptive-scheduler)
[![Downloads](https://anaconda.org/conda-forge/adaptive-scheduler/badges/downloads.svg)](https://anaconda.org/conda-forge/adaptive-scheduler)

Run many learners on many cores (>10k) using MPI.


## What is this?

The Adaptive scheduler solves the following problem, you need to run a few 100 learners and can use >1k cores.
 
You can't use a centrally managed place that is responsible for all the workers (like with `dask` or `ipyparallel`) because >1k cores is too many for them to handle.
 
You also don't want to use `dask` or `ipyparallel` inside a job script because they write job scripts on their own. Having a job script that runs code that creates job scripts...

With `adaptive_scheduler` you only need to define the learners and then it takes care of the running (and restarting) of the jobs on the cluster.


## How does it work?

You create a file where you define a bunch of learners such that they can be imported, like:
```python
# learners_file.py
import adaptive
from functools import partial

def h(x, pow, a):
    return a * x**pow

combos = adaptive.utils.named_product(
    pow=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    a=[0.1, 0.5],
)  # returns list of dicts, cartesian product of all values

learners = []
fnames = []

for combo in combos:
    f = partial(h, **combo)
    learner = adaptive.Learner1D(f, bounds=(-1, 1))
    fnames.append(f"data/{combo}")
    learners.append(learner)
```

Then a "job manager" writes and submits as many jobs as there are learners but _doesn't know_ which learner it is going to run!
This is the responsibility of the "database manager", which keeps a database of `job_id <--> learner`.

In another Python file (the file that is run on the nodes) we do something like:
```python
# run_learner.py
import adaptive
from adaptive_scheduler import client_support
from mpi4py.futures import MPIPoolExecutor

# the file that defines the learners we created above
from learners_file import learners, fnames

# the address of the "database manager"
url = "tcp://10.75.0.5:37371"

# ask the database for a learner that we can run
learner, fname = client_support.get_learner(url, learners, fnames)

# load the data
learner.load(fname)

# run until `some_goal` is reached with an `MPIPoolExecutor`
runner = adaptive.Runner(
    learner, executor=MPIPoolExecutor(), shutdown_executor=True, goal=some_goal
)

# periodically save the data (in case the job dies)
runner.start_periodic_saving(dict(fname=fname), interval=600)

# block until runner goal reached
runner.ioloop.run_until_complete(runner.task)

# tell the database that this learner has reached its goal
client_support.is_done(url, fname)
```

In a Jupyter notebook we can start the "job manager" and the "database manager" like:
```python
from adaptive_scheduler import server_support
from learners_file import learners, fnames

# create a new database
db_fname = "running.tinydb"
server_support.create_empty_db(db_fname, fnames)

# create unique names for the jobs
n_jobs = len(learners)
job_names = [f"test-job-{i}" for i in range(n_jobs)]

# start the "job manager" and the "database manager"
ioloop = asyncio.get_event_loop()

database_task = ioloop.create_task(
    server_support.manage_database("tcp://10.75.0.5:37371", db_fname)
)

job_task = ioloop.create_task(
    server_support.manage_jobs(
        job_names,
        db_fname=db_fname,
        ioloop=ioloop,
        cores=200,  # number of cores per job
        interval=60,
        run_script="run_learner.py",
        python_executable="~/miniconda3/envs/python37/bin/python",
    )
)
```

So in summary, you have three files:
1. `learners_file.py` which defines the learners and its filenames
2. `run_learner.py` which picks a learner and runs it
3. a Jupyter notebook where you run the "database manager" and the "job manager"

You don't actually ever have to leave the Jupter notebook, take a look at the [example notebook](example.ipynb).


## Jupyter notebook example

See [`example.ipynb`](example.ipynb).


## Installation

**WARNING:** This is still the pre-alpha development stage. No error-handling is done and its stability is uncertain.

Install the **latest stable** version from conda with (recommended)
```bash
conda install adaptive-scheduler
```

or from PyPI with
```bash
pip install adaptive_scheduler
```

or install **master** with
```bash
pip install -U https://github.com/basnijholt/adaptive-scheduler/archive/master.zip
```
or clone the repository and do a dev install (recommended for dev)
```
git clone git@github.com:basnijholt/adaptive-scheduler.git
cd adaptive-scheduler
pip install -e .
```


## Development

In order to not pollute the history with the output of the notebooks, please setup the git filter by executing
```
python ipynb_filter.py
```
in the repository.

We also use [pre-commit](https://pre-commit.com), so `pip install pre_commit` and run
```
pre-commit install
```
in the repository.


## Limitations

Right now `adaptive_scheduler` is only working for SLURM and PBS, however only the functions in [`adaptive_scheduler/slurm.py`](adaptive_scheduler/slurm.py) would have to be implemented for another type of scheduler.
