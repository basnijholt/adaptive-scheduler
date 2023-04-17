# FAQ

Here is a list of questions we have either been asked by users or potential pitfalls we hope to help users avoid:

## **Q: It doesn't work, what now?**

**A:** Check the log-files that are created and look for an error message. If you suspect a bug in `adaptive_scheduler` check out `run_manager.task_status()` and if that doesn't reveal anything, open an issue on [`GitHub`](https://github.com/basnijholt/adaptive-scheduler/issues).

## **Q: What if I have more learners than cores?**

**A:** You can distribute all learners in a certain amount of `adaptive.BalancingLearner`s. Like so

```python
from functools import partial

import adaptive
import numpy as np
from adaptive_scheduler.utils import split_in_balancing_learners, shuffle_list

def jacobi(x, n, alpha, beta):
    from scipy.special import eval_jacobi
    return eval_jacobi(n, alpha, beta, x)

combos = adaptive.utils.named_product(
        n=[1, 2, 4, 8],
    alpha=np.linspace(0, 2, 21),
    beta=np.linspace(0, 1, 21),
)

learners = [adaptive.Learner1D(partial(jacobi, **combo), bounds=(0, 1)) for combo in combos]
fnames = [f"data/jacobi/{combo}" for combo in combos]

# shuffle the learners (and fnames in the same order) because
# some learners might be slower than others (not in this example).
unshuffled = learners, fnames  # to have a handle to the unshuffled list
learners, fnames = shuffle_list(*unshuffled)

# split in many new BalancingLearners
# `learners` will be a list of BalancingLeaners
# `fnames` will be a list of lists with fnames
learners, fnames = split_in_balancing_learners(
        learners,
    fnames,
    n_parts=100,  # split into 100 BalancingLeaners
    strategy="npoints"
)
```

## **Q: Why aren't my jobs dying when I cancel the job manager?**

**A:** The job manager just starts the jobs and you want the job to keep running
in case the job manager somehow dies. So you still need to `scancel` or `qdel` them
in case you want to really cancel the jobs or call `adaptive_scheduler.cancel_jobs` with
`job_names` from your Python environment.

## **Q: How do I set extra SBATCH/PBS arguments or environment variables in my job script?**

**A:** You can change this in the `scheduler` object.
For example modifying a job script for SLURM:

```python
from adaptive_scheduler.scheduler import SLURM
scheduler = SLURM(
        cores=10,
    extra_scheduler=["--exclusive=user", "--time=1"],
    extra_env_vars=["TMPDIR='/scratch'", "PYTHONPATH='my_dir:$PYTHONPATH'"],
    mpiexec_executable="srun --mpi=pmi2",
)  # pass this to `server_support.start_job_manager` or `RunManager`

# see the job script with
print(scheduler.job_script('this_will_be_the_job_name'))
```

## **Q: My code uses MPI so the `~mpi4py.futures.MPIPoolExecutor` won't work for me, I want to use `ipyparallel`, how?**

**A:** You just have to pass `executor_type="ipyparallel"` to `~adaptive_scheduler.scheduler.SLURM` or `~adaptive_scheduler.scheduler.PBS`.
For example:

```python
from adaptive_scheduler.scheduler import SLURM

scheduler = SLURM(
    cores=48,
    executor_type="ipyparallel",
)

run_manager = adaptive_scheduler.server_support.RunManager(
        scheduler=scheduler,
    learners=learners,
    fnames=fnames,

)
run_manager.start()
```

## **Q: `ipyparallel` doesn't work for me, I want to use `process-pool`, how?**

**A:** Sometimes `ipyparallel` doesn't import modules correctly on its workers. In this case you can use `process-pool`. You just have to pass `executor_type="process-pool"` to `~adaptive_scheduler.scheduler.SLURM` or `~adaptive_scheduler.scheduler.PBS`. Note the `process-pool` uses Python's `~concurrent.futures.ProcessPoolExecutor` for parallelism and cannot be used beyond a single machine (for one learner).

## **Q: Cool! What else should I check out?**

**A:** There are a bunch of things that are not present in the example notebook, I recommend to take a look at:

- `adaptive_scheduler.utils.combo_to_fname`
- `adaptive_scheduler.server_support.cleanup`
- `adaptive_scheduler.server_support.parse_log_files`
- `adaptive_scheduler.utils.load_parallel` and `adaptive_scheduler.utils.save_parallel`
