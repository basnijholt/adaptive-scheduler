An asynchronous job scheduler for `Adaptive <https://github.com/python-adaptive/adaptive/>`_
============================================================================================

|PyPI|  |Conda|  |Downloads|  |Build Status| |Documentation Status|

Run many ``adaptive.Learner``\ s on many cores (>10k) using `mpi4py.futures`, `ipyparallel`, or `dask.distributed`.

What is this?
-------------

The Adaptive scheduler solves the following problem, you need to run more learners than you can run with a single runner and/or can use >1k cores.
 
`ipyparallel` and `dask.distributed` provide very powerful engines for interactive sessions. However, when you want to connect to >1k cores it starts to struggle. Besides that, on a shared cluster there is often the problem of starting an interactive session with ample space available.

Our approach is to schedule a different job for each ``adaptive.Learner``. The creation and running of these jobs are managed by ``adaptive-scheduler``. This means that your calculation will definitely run, even though the cluster might be fully occupied at the moment. Because of this approach, there is almost no limit to how many cores you want to use. You can either use 10 nodes for 1 job (\ ``learner``\ ) or 1 core for 1 job (\ ``learner``\ ) while scheduling hundreds of jobs.

Everything is written such that the computation is maximally local. This means that is one of the jobs crashes, there is no problem and it will automatically schedule a new one and continue the calculation where it left off (because of Adaptive's periodic saving functionality). Even if the central "job manager" dies, the jobs will continue to run (although no new jobs will be scheduled.)


Design goals
------------

#. Needs to be able to run on efficiently >30k cores
#. Works seamlessly with the Adaptive package
#. Minimal load on the file system
#. Removes all boilerplate of working with a scheduler

   #. writes job script
   #. (re)submits job scripts

#. Handles random crashes (or node evictions) with minimal data loss
#. Preserves Python kernel and variables inside a job (in contrast to submitting jobs for every parameter)
#. Separates the simulation definition code from the code that runs the simulation
#. Maximizes computation locality, jobs continue to run when the main process dies

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

   scheduler = adaptive_scheduler.scheduler.SLURM(cores=10)  # every learner get this many cores

   run_manager = adaptive_scheduler.server_support.RunManager(
       scheduler=scheduler,
       learners_file="learners_file.py",
       goal=goal,
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

The `~adaptive_scheduler.server_support.RunManager` basically does the following.
So, *you* need to create a ``learners_file.py`` that defines ``N`` ``learners`` and ``fnames`` (like in the section above).
Then a "job manager" writes and submits ``max(N, max_simultaneous_jobs)`` job scripts but *doesn't know* which learner it is going to run!
This is the responsibility of the "database manager", which keeps a database of ``job_id <--> learner``.
The job script starts a Python file ``run_learner.py`` in which the learner is run.


In a Jupyter notebook we can start the "job manager" and the "database manager", and create the ``run_learner.py`` like:

.. code-block:: python

   import adaptive_scheduler
   from adaptive_scheduler import server_support
   from learners_file import learners, fnames

   # create a new database
   db_fname = "running.json"
   server_support.create_empty_db(db_fname, fnames)

   # get a url where we can run the database manager
   url = server_support.get_allowed_url()

   # start the "database manager"
   database_task = server_support.start_database_manager(url, db_fname)

   # create a scheduler
   scheduler = adaptive_scheduler.scheduler.PBS(
       cores=10,
       run_script="run_learner.py",
   )

   # create the Python script that runs a learner (run_learner.py)
   server_support._make_default_run_script(
       url=url,
       learners_file="learners_file.py",
       save_interval=300,
       log_interval=30,
       goal=None,
       executor_type=scheduler.executor_type,
       run_script_fname=scheduler.run_script,
   )

   # create unique names for the jobs
   n_jobs = len(learners)
   job_names = [f"test-job-{i}" for i in range(n_jobs)]

   # start the "database manager"
   job_task = server_support.start_job_manager(
       job_names,
       db_fname,
       scheduler,
   )


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

.. code-block:: bash

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
