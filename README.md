# An asynchronous scheduler using MPI for Adaptive

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

def h(x, power=0):
    return x**power

combos = adaptive.utils.named_product(power=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

learners = []
fnames = []

for i, combo in enumerate(combos):
    f = partial(h, offset=combo["offset"])
    learner = adaptive.Learner1D(f, bounds=(-1, 1))
    fnames.append(f"data/{combo}")
    learners.append(learner)

learner = adaptive.BalancingLearner(learners)
```

Then a "job manager" writes and submits as many jobs as there are learners but _doesn't know_ which learner it is going to run!
This is the responsibility of the "database manager", which keeps a database of `job_id <--> learner`.

In another Python file (the file that is run on the nodes) we do something like:
```python
# run_learner.py
import adaptive
from adaptive_scheduler import client_support
from mpi4py.futures import MPIPoolExecutor

# the file that defines the learners
from learners_file import learners, combos

# the address of the "database manager"
url = "tcp://10.75.0.5:37371"

# ask the database for a learner that we can run
learner, fname = client_support.get_learner(url, learners, combos)  

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

# tell the database that we are done
client_support.is_done(url, fname)
```

In a Jupyter notebook we can start the "job manager" and the "database manager" like:
```python
from adaptive_scheduler import server_support
from learners_file import learners, combos, fname

# create a new database
db_fname = "running.tinydb"
server_support.create_empty_db(db_fname, fnames, combos)

# create unique names for the jobs
job_names = [f"test-job-{i}" for i in range(len(learners))]

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
        cores=200,
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

**WARNING:** This is still the pre-alpha development stage. No error-handling is done, an its stability is uncertain.

Install master with
```bash
pip install -U https://github.com/basnijholt/adaptive-scheduler/archive/master.zip
```
or clone the repository and do a dev install (recommended)
```
git clone git@github.com:basnijholt/adaptive-scheduler.git
cd adaptive-scheduler
pip install -e .
```
