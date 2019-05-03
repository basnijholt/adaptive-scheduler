# A asynchronous scheduler using MPI for Adaptive

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
    return x + a**2 / (a**2 + (x - offset)**2)

offset = list(range(50))

combos = adaptive.utils.named_product(offset=offset)

learners = []
fnames = []

folder = "data/test/"

for i, combo in enumerate(combos):
    f = partial(h, offset=combo['offset'])
    learner = adaptive.Learner1D(f, bounds=(0, 1))
    fnames.append(f"{folder}{combo}")
    learners.append(learner)

learner = adaptive.BalancingLearner(learners)
```

### Run the learners

Create `run_learner.py`:
```python
import adaptive
from adaptive.scheduler import client_support
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
