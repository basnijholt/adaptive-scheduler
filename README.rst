
An asynchronous job scheduler for `Adaptive <https://github.com/python-adaptive/adaptive/>`_
============================================================================================

|PyPI|  |Conda|  |Downloads|  |Build Status| |Documentation Status|

Run many ``adaptive.Learner``\ s on many cores (>10k) using `mpi4py.futures`, `ipyparallel`, or `dask.distributed`.

What is this?
-------------

The Adaptive scheduler solves the following problem, you need to run a few 100 learners and can use >1k cores.
 
`ipyparallel` and `dask.distributed` provide very powerful engines for interactive sessions. However, when you want to connect to >1k cores it starts to struggle. Besides that, on a shared cluster there is often the problem of starting an interactive session with ample space available.

Our approach is to schedule a different job for each ``adaptive.Learner``. The creation and running of these jobs are managed by ``adaptive-scheduler``. This means that your calculation will definitely run, even though the cluster might be fully occupied at the moment. Because of this approach, there is almost no limit to how many cores you want to use. You can either use 10 nodes for 1 job (\ ``learner``\ ) or 1 core for 1 job (\ ``learner``\ ) while scheduling hundreds of jobs.

Everything is written such that the computation is maximally local. This means that is one of the jobs crashes, there is no problem and it will automatically schedule a new one and continue the calculation where it left off (because of Adaptive's periodic saving functionality). Even if the central "job manager" dies, the jobs will continue to run (although no new jobs will be scheduled.)

How does it work?
-----------------

You create a file where you define a bunch of ``learners`` and corresponding ``fnames`` such that they can be imported, like:

.. code-block:: python

   # learners_file.py
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


Then you start a process that creates and submits as many job-scripts as there are learners. Like:

.. code-block:: python

   import adaptive_scheduler

   def goal(learner):
       return learner.npoints > 200

   run_manager = adaptive_scheduler.server_support.RunManager(
       learners_file="learners_file.py",
       goal=goal,
       cores_per_job=12,  # every learner is one job
       log_interval=30,  #  write info such as npoints, cpu_usage, time, etc. to the job log file
       save_interval=300,  # save the data every 300 seconds
   )
   run_manager.start()


That's it! You can run ``run_manager.info()`` which will display an interactive ``ipywidget`` that shows the amount of running, pending, and finished jobs, buttons to cancel your job, and other useful information.

.. image:: http://files.nijho.lt/info.gif
   :target: http://files.nijho.lt/info.gif
   :alt: Widget demo



But how does *really* it work?
------------------------------

The `~adaptive_scheduler.server_support.RunManager` basically does what is written below.
So, you need to create a ``learners_file.py`` that defines ``learners`` and ``fnames`` (like in the section above).
Then a "job manager" writes and submits as many jobs as there are learners but *doesn't know* which learner it is going to run!
This is the responsibility of the "database manager", which keeps a database of ``job_id <--> learner``.

In another Python file (the file that is run on the nodes) we do something like:

.. code-block:: python

   # run_learner.py
   import adaptive
   from adaptive_scheduler import client_support
   from mpi4py.futures import MPIPoolExecutor

   # the file that defines the learners we created above
   from learners_file import learners, fnames


   if __name__ == "__main__":  # ← use this, see warning @ https://bit.ly/2HAk0GG
       # the address of the "database manager"
       url = "tcp://10.75.0.5:37371"

       # ask the database for a learner that we can run
       learner, fname = client_support.get_learner(url, learners, fnames)

       # load the data
       learner.load(fname)

       # run until `some_goal` is reached with an `MPIPoolExecutor`
       # you can also use a ipyparallel.Client, or dask.distributed.Client
       runner = adaptive.Runner(
           learner, executor=MPIPoolExecutor(), shutdown_executor=True, goal=some_goal
       )

       # periodically save the data (in case the job dies)
       runner.start_periodic_saving(dict(fname=fname), interval=600)

       # log progress info in the job output script, optional
       client_support.log_info(runner, interval=600)

       # block until runner goal reached
       runner.ioloop.run_until_complete(runner.task)

       # tell the database that this learner has reached its goal
       client_support.tell_done(url, fname)


In a Jupyter notebook we can start the "job manager" and the "database manager" like:

.. code-block:: python

   from adaptive_scheduler import server_support
   from learners_file import learners, fnames

   # create a new database
   db_fname = "running.json"
   server_support.create_empty_db(db_fname, fnames)

   # create unique names for the jobs
   n_jobs = len(learners)
   job_names = [f"test-job-{i}" for i in range(n_jobs)]

   # start the "job manager" and the "database manager"
   database_task = server_support.start_database_manager("tcp://10.75.0.5:37371", db_fname)

   job_task = server_support.start_job_manager(
       job_names,
       db_fname,
       cores=200,  # number of cores per job
       run_script="run_learner.py",
   )


So in summary, you have three files:

- ``learners_file.py`` which defines the learners and its filenames
- ``run_learner.py`` which picks a learner and runs it
- a Jupyter notebook where you run the "database manager" and the "job manager"

You don't actually ever have to leave the Jupter notebook, take a look at the `example notebook <https://github.com/basnijholt/adaptive-scheduler/blob/master/example.ipynb>`_.

Jupyter notebook example
------------------------

See `example.ipynb <https://github.com/basnijholt/adaptive-scheduler/blob/master/example.ipynb>`_.

Installation
------------

**WARNING:** This is still the pre-alpha development stage.

Install the **latest stable** version from conda with (recommended)

.. code-block:: bash

   conda install adaptive-scheduler


or from PyPI with

.. code-block:: bash

   pip install adaptive_scheduler


or install **master** with

.. code-block:: bash

   pip install -U https://github.com/basnijholt/adaptive-scheduler/archive/master.zip


or clone the repository and do a dev install (recommended for dev)

.. code-block::

   git clone git@github.com:basnijholt/adaptive-scheduler.git
   cd adaptive-scheduler
   pip install -e .


Development
-----------

In order to not pollute the history with the output of the notebooks, please setup the git filter by executing

.. code-block:: bash

   python ipynb_filter.py


in the repository.

We also use `pre-commit <https://pre-commit.com>`_\ , so ``pip install pre_commit`` and run

.. code-block:: bash

   pre-commit install


in the repository.

Limitations
-----------

Right now ``adaptive_scheduler`` is only working for SLURM and PBS, however only the functions in `adaptive_scheduler/slurm.py <https://github.com/basnijholt/adaptive-scheduler/blob/master/adaptive_scheduler/slurm.py>`_ would have to be implemented for another type of scheduler. Also there are **no tests** at all!

.. references-start
.. |PyPI| image:: https://img.shields.io/pypi/v/adaptive-scheduler.svg
   :target: https://pypi.python.org/pypi/adaptive-scheduler
   :alt: PyPI
.. |Conda| image:: https://anaconda.org/conda-forge/adaptive-scheduler/badges/installer/conda.svg
   :target: https://anaconda.org/conda-forge/adaptive-scheduler
   :alt: Conda
.. |Downloads| image:: https://anaconda.org/conda-forge/adaptive-scheduler/badges/downloads.svg
   :target: https://anaconda.org/conda-forge/adaptive-scheduler
   :alt: Downloads
.. |Build Status| image:: https://dev.azure.com/basnijholt/adaptive-scheduler/_apis/build/status/basnijholt.adaptive-scheduler?branchName=master
   :target: https://dev.azure.com/basnijholt/adaptive-scheduler/_build/latest?definitionId=1&branchName=master
   :alt: Build Status
.. |Documentation Status| image:: https://readthedocs.org/projects/adaptive-scheduler/badge/?version=latest
   :target: https://adaptive-scheduler.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. references-end
