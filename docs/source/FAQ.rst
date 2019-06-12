
FAQ
===

Here is a list of questions we have either been asked by users or potential pitfalls we hope to help users avoid:

Q: What if I have more learners than cores?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**A:** You can distribute all learners in a certain amount of `adaptive.BalancingLearner`\ s. Like so

.. code-block:: python

    %%writefile learners_file.py

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
    # `learners` will be a list of BalancingLearners
    # `fnames` will be a list of lists with fnames
    learners, fnames = split_in_balancing_learners(
        learners,
        fnames,
        n_parts=100,  # split into 100 BalancingLeaners
        strategy="npoints"
    )

Q: Why aren't my jobs dying when I cancel the job manager?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**A:** The job manager just starts the jobs and you want the job to keep running
in case the job manager somehow dies. So you still need to ``scancel`` or ``qdel`` them
in case you want to really cancel the jobs or call `adaptive_scheduler.cancel_jobs` with
``job_names`` from your Python environment.

Q: How do I set extra SBATCH/PBS arguments or environment variables in my job script?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**A:** The job_manager expects a function, so you need to modify the ``make_job_script`` function using ``functools.partial``.
For example modifying a job script for SLURM:

.. code-block:: python

    from functools import partial
    from adaptive_scheduler.slurm import make_job_script
    job_script_function = partial(
        make_job_script,
        extra_sbatch=["--exclusive=user", "--time=1"],
        extra_env_vars=["TMPDIR='/scratch'", "PYTHONPATH='my_dir:$PYTHONPATH'"],
        mpiexec_executable="srun --mpi=pmi2",
    )  # pass this to `server_support.start_job_manager`

Q: My code uses MPI so the `MPIPoolExecutor` won't work for me, I want to use `ipyparallel`, how?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**A:** You just have to pass `executor_type="ipyparallel"` to the `job_script_function` and `RunManager`.
For example:

.. code-block:: python

    from functools import partial

    job_script_function = partial(
        adaptive_scheduler.slurm.make_job_script,
        executor_type="ipyparallel",
    )

    run_manager = adaptive_scheduler.server_support.RunManager(
        learners_file="learners_file.py",
        executor_type="ipyparallel",
        cores_per_job=48,
        job_script_function=job_script_function,

    )
    run_manager.start()

Q: Cool! What else should I check out?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**A:** There are a bunch of things that are not present in the example notebook, I recommend to take a look at:

* `adaptive_scheduler.utils.combo_to_fname`
* `adaptive_scheduler.utils.cleanup_files`
* `adaptive_scheduler.utils.load_parallel` and `adaptive_scheduler.utils.save_parallel`
* `adaptive_scheduler.utils.parse_log_files`
