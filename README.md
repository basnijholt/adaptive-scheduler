# An asynchronous scheduler using MPI for Adaptive

Run many learners on many cores (>10k) using MPI.

## Example

### Define the learners in a Python file

We need the following variables:
* `learners` a list of learners
* `combos` a list of dicts of parameters that describe each learner
* `fnames` a list of filenames of each learner

Create `_learners.py`:
```python
import adaptive
from functools import partial


def h(x, offset=0):
    import numpy as np
    import random

    # Burn some CPU time just because
    for _ in range(10):
        np.linalg.eig(np.random.rand(1000, 1000))

    a = 0.01
    return x + a ** 2 / (a ** 2 + (x - offset) ** 2)


offset = [i / 100 for i in range(100)]

combos = adaptive.utils.named_product(offset=offset)

learners = []
fnames = []

folder = "data/"

for i, combo in enumerate(combos):
    f = partial(h, offset=combo["offset"])
    learner = adaptive.Learner1D(f, bounds=(-1, 1))
    fnames.append(f"{folder}{combo}")
    learners.append(learner)

learner = adaptive.BalancingLearner(learners
```


### Run the learners

Create `run_learner.py`:
```python
import adaptive
from adaptive_scheduler import client_support
from mpi4py.futures import MPIPoolExecutor

from _learners import learners, combos


url = "tcp://10.76.0.5:57681"  # ip of the headnode

if __name__ == "__main__":
    learner, fname = client_support.get_learner(url, learners, combos)
    learner.load(fname)
    ex = MPIPoolExecutor()
    runner = adaptive.Runner(
        learner,
        executor=ex,
        goal=None,  # run forever
        shutdown_executor=True,
        retries=10,
        raise_if_retries_exceeded=False,
    )
    runner.start_periodic_saving(dict(fname=fname), interval=600)
    runner.ioloop.run_until_complete(runner.task)  # wait until runner goal reached
    client_support.is_done(url, fname)
```


### Run the database and job manager

One can do this interactively in a Jupyter notebook on the cluster head node:
```python
import asyncio
from pprint import pprint

from adaptive_scheduler import client_support
from tinydb import TinyDB

import _learners

# Create a new database that keeps track of (learner -> job_id, is_done)
db_fname = 'running.tinydb'
server_support.create_empty_db(db_fname, _learners.fnames, _learners.combos)

## Check the running learners
# All the onces that are `None` are still `PENDING` or are not scheduled.
with TinyDB(db_fname) as db:
    pprint(db.all())


## Start the job scripts

# Get some unique names for the jobs
job_names = [f"test-{i}" for i in range(len(_learners.learners))]

ioloop = asyncio.get_event_loop()

database_task = ioloop.create_task(
    server_support.manage_database("tcp://*:57681", db_fname)
)

job_task = ioloop.create_task(
    server_support.manage_jobs(job_names, db_fname, ioloop, cores=50*8, interval=60)
)
```
