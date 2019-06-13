import ast
import collections
import glob
import math
import os
import random
import shutil
import subprocess
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import datetime
from typing import Any, Dict, Tuple, Sequence, List, Optional, Callable

import adaptive
import toolz
from adaptive.notebook_integration import in_ipynb
from ipyparallel import Client
from tqdm import tqdm, tqdm_notebook

MAX_LINE_LENGTH = 100


def shuffle_list(*lists, seed=0):
    """Shuffle multiple lists in the same order."""
    combined = list(zip(*lists))
    random.Random(seed).shuffle(combined)
    return zip(*combined)


def _split(seq, n_parts):
    lst = list(seq)
    n = math.ceil(len(lst) / n_parts)
    return toolz.partition_all(n, lst)


def split_in_balancing_learners(
    learners: List[adaptive.BaseLearner],
    fnames: List[str],
    n_parts: int,
    strategy: str = "npoints",
) -> Tuple[List[adaptive.BaseLearner], List[str]]:
    r"""Split a list of learners and fnames into `adaptive.BalancingLearner`\s.

    Parameters
    ----------
    learners : list
        List of learners.
    fnames : list
        List of filenames.
    n_parts : int
        Total number of `~adaptive.BalancingLearner`\s.
    strategy : str
        Learning strategy of the `~adaptive.BalancingLearner`.

    Returns
    -------
    new_learners, new_fnames
    """
    new_learners = []
    new_fnames = []
    for x in _split(zip(learners, fnames), n_parts):
        learners_part, fnames_part = zip(*x)
        learner = adaptive.BalancingLearner(learners_part, strategy=strategy)
        new_learners.append(learner)
        new_fnames.append(fnames_part)
    return new_learners, new_fnames


def _progress(seq: Sequence, with_progress_bar: bool = True, desc: str = ""):
    if not with_progress_bar:
        return seq
    else:
        if in_ipynb():
            return tqdm_notebook(list(seq), desc=desc)
        else:
            return tqdm(list(seq), desc=desc)


def _cancel_function(cancel_cmd: str, queue_function: Callable) -> Callable:
    def cancel(
        job_names: List[str], with_progress_bar: bool = True, max_tries: int = 5
    ) -> Callable:
        """Cancel all jobs in `job_names`.

        Parameters
        ----------
        job_names : list
            List of job names.
        with_progress_bar : bool, default: True
            Display a progress bar using `tqdm`.
        max_tries : int, default: 5
            Maximum number of attempts to cancel a job.
        """

        def to_cancel(job_names):
            return [
                job_id
                for job_id, info in queue_function().items()
                if info["name"] in job_names
            ]

        def cancel_jobs(job_ids):
            for job_id in _progress(job_ids, with_progress_bar, "Canceling jobs"):
                cmd = f"{cancel_cmd} {job_id}".split()
                returncode = subprocess.run(cmd, stderr=subprocess.PIPE).returncode
                if returncode != 0:
                    warnings.warn(f"Couldn't cancel '{job_id}'.", UserWarning)

        job_names = set(job_names)
        for _ in range(max_tries):
            job_ids = to_cancel(job_names)
            if not job_ids:
                # no more running jobs
                break
            cancel_jobs(job_ids)

    return cancel


def combo_to_fname(combo: Dict[str, Any], folder: Optional[str] = None) -> str:
    """Converts a dict into a human readable filename."""
    fname = "__".join(f"{k}_{v}" for k, v in combo.items()) + ".pickle"
    if folder is None:
        return fname
    return os.path.join(folder, fname)


def cleanup_files(
    job_names: List[str],
    extensions: List[str] = ("sbatch", "out", "batch", "e*", "o*"),
    with_progress_bar: bool = True,
    move_to: Optional[str] = None,
) -> None:
    """Cleanup the scheduler log-files files.

    Parameters
    ----------
    job_names : list
        List of job names.
    extensions : list
        List of file extensions to be removed.
    with_progress_bar : bool, default: True
        Display a progress bar using `tqdm`.
    move_to : str, default None
        Move the file to a different directory.
        If None the file is removed.
    """
    # Finding the files
    fnames = []
    for job in job_names:
        for ext in extensions:
            fnames += glob.glob(f"{job}*.{ext}")

    _remove_or_move_files(fnames, with_progress_bar, move_to)


def _remove_or_move_files(
    fnames: List[str], with_progress_bar: bool = True, move_to: Optional[str] = None
) -> None:
    """Remove files by filename.

    Parameters
    ----------
    fnames : list
        List of filenames.
    with_progress_bar : bool, default: True
        Display a progress bar using `tqdm`.
    move_to : str, default None
        Move the file to a different directory.
        If None the file is removed.
    """
    n_failed = 0
    for fname in _progress(fnames, with_progress_bar, "Removing files"):
        try:
            if move_to is None:
                os.remove(fname)
            else:
                os.makedirs(move_to, exist_ok=True)
                shutil.move(fname, move_to)
        except Exception:
            n_failed += 1

    if n_failed:
        warnings.warn(f"Failed to remove {n_failed} files.")


def load_parallel(
    learners: List[adaptive.BaseLearner],
    fnames: List[str],
    *,
    with_progress_bar: bool = True,
) -> None:
    r"""Load a sequence of learners in parallel.

    Parameters
    ----------
    learners : sequence of `adaptive.BaseLearner`\s
        The learners to be loaded.
    fnames : sequence of str
        A list of filenames corresponding to `learners`.
    with_progress_bar : bool, default True
        Display a progress bar using `tqdm`.
    """

    def load(learner, fname):
        learner.load(fname)

    with ThreadPoolExecutor() as ex:
        futs = []
        iterator = zip(learners, fnames)
        pbar = _progress(iterator, with_progress_bar, "Submitting loading tasks")
        futs = [ex.submit(load, *args) for args in pbar]
        for fut in _progress(futs, with_progress_bar, "Finishing loading"):
            fut.result()


def save_parallel(
    learners: List[adaptive.BaseLearner],
    fnames: List[str],
    *,
    with_progress_bar: bool = True,
) -> None:
    r"""Save a sequence of learners in parallel.

    Parameters
    ----------
    learners : sequence of `adaptive.BaseLearner`\s
        The learners to be saved.
    fnames : sequence of str
        A list of filenames corresponding to `learners`.
    with_progress_bar : bool, default True
        Display a progress bar using `tqdm`.
    """

    def save(learner, fname):
        learner.save(fname)

    with ThreadPoolExecutor() as ex:
        futs = []
        iterator = zip(learners, fnames)
        pbar = _progress(iterator, with_progress_bar, "Submitting saving tasks")
        futs = [ex.submit(save, *args) for args in pbar]
        for fut in _progress(futs, with_progress_bar, "Finishing saving"):
            fut.result()


def _get_status_prints(fname: str, only_last: bool = True):
    status_lines = []
    with open(fname) as f:
        lines = f.readlines()
        if not lines:
            return status_lines
        for line in reversed(lines):
            if "current status" in line:
                status_lines.append(line)
                if only_last:
                    return status_lines
    return status_lines


def parse_log_files(
    job_names: List[str], only_last: bool = True, db_fname: Optional[str] = None
):
    """Parse the log-files and convert it to a `~pandas.core.frame.DataFrame`.

    This only works if you use `adaptive_scheduler.client_support.log_info`
    inside your ``run_script``.

    Parameters
    ----------
    job_names : list
        List of job names.
    only_last : bool, default: True
        Only look use the last printed status message.
    db_fname : str, optional
        The database filename. If passed, ``fname`` will be populated.

    Returns
    -------
    `~pandas.core.frame.DataFrame`
    """
    # XXX: it could be that the job_id and the logfile don't match up ATM! This
    # probably happens when a job got canceled and is pending now.
    try:
        import pandas as pd

        with_pandas = True
    except ImportError:
        with_pandas = False
        warnings.warn("`pandas` is not installed, a list of dicts will be returned.")

    # import here to avoid circular imports
    from adaptive_scheduler.server_support import queue, get_database

    def convert_type(k, v):
        if k == "elapsed_time":
            return pd.to_timedelta(v)
        elif k == "overhead":
            return float(v[:-1])
        else:
            return ast.literal_eval(v)

    def join_str(info):
        """Turns an incorrectly split string
        ["elapsed_time=1", "day,", "0:20:57.330515", "nlearners=31"]
        back the correct thing
        ['elapsed_time=1 day, 0:20:57.330515', 'nlearners=31']
        """
        _info = []
        for x in info:
            if "=" in x:
                _info.append(x)
            else:
                _info[-1] += f" {x}"
        return _info

    infos = []
    for job in job_names:
        fnames = glob.glob(f"{job}-*.out")
        if not fnames:
            continue
        fname = fnames[-1]  # take the last file
        statuses = _get_status_prints(fname, only_last)
        if statuses is None:
            continue
        for status in statuses:
            time, info = status.split("current status")
            info = join_str(info.strip().split(" "))
            info = dict([x.split("=") for x in info])
            info = {k: convert_type(k, v) for k, v in info.items()}
            info["job"] = job
            info["time"] = datetime.strptime(time.strip(), "%Y-%m-%d %H:%M.%S")
            info["log_file"] = fname
            infos.append(info)

    # Polulate state and job_id from the queue
    mapping = {
        info["name"]: (job_id, info["state"]) for job_id, info in queue().items()
    }

    for info in infos:
        info["job_id"], info["state"] = mapping.get(info["job"], (None, None))

    if db_fname is not None:
        # populate job_id
        db = get_database(db_fname)
        fnames = {info["job_id"]: info["fname"] for info in db}
        id_done = {info["job_id"]: info["is_done"] for info in db}
        for info in infos:
            info["fname"] = fnames.get(info["job_id"], "UNKNOWN")
            info["is_done"] = id_done.get(info["job_id"], "UNKNOWN")

    return pd.DataFrame(infos) if with_pandas else infos


def _is_string_inside_file(fname, string):
    with open(fname) as f:
        lines = f.readlines()
    return string in "".join(lines)


def logs_with_string(job_names: List[str], string: str) -> Dict[str, list]:
    """Get jobs that have `string` inside their log-file.

    Parameters
    ----------
    job_names : list
        List of job names.
    string : str
        String that is searched for.

    Returns
    -------
    has_string : list
        List with jobs that have the string inside their log-file.
    """
    has_string = collections.defaultdict(list)
    for job in job_names:
        fnames = glob.glob(f"{job}-*.out")
        if not fnames:
            continue
        for fname in fnames:
            job_id = fname.split(f"{job}-")[1].split(".out")[0]
            if _is_string_inside_file(fname, string):
                has_string[job].append(job_id)
    return dict(has_string)


def _print_same_line(msg, new_line_end=False):
    msg = msg.strip()
    global MAX_LINE_LENGTH
    MAX_LINE_LENGTH = max(len(msg), MAX_LINE_LENGTH)
    empty_space = max(MAX_LINE_LENGTH - len(msg), 0) * " "
    print(msg + empty_space, end="\r" if not new_line_end else "\n")


def _wait_for_successful_ipyparallel_client_start(client, n, timeout):
    from ipyparallel.error import NoEnginesRegistered

    n_engines_old = 0
    for t in range(timeout):
        n_engines = len(client)
        with suppress(NoEnginesRegistered):
            # This can happen, we just need to wait a little longer.
            dview = client[:]
        msg = f"Connected to {n_engines} out of {n} engines after {t} seconds."
        _print_same_line(msg, new_line_end=(n_engines_old != n_engines))
        if n_engines >= n:
            return dview
        n_engines_old = n_engines
        time.sleep(1)

    raise Exception(f"Not all ({n_engines}/{n}) connected after {timeout} seconds.")


def connect_to_ipyparallel(
    n: int,
    profile: str,
    timeout: int = 300,
    folder: Optional[str] = None,
    client_kwargs=None,
):
    """Connect to an `ipcluster` on the cluster headnode.

    Parameters
    ----------
    n : int
        Number of engines to be started.
    profile : str
        Profile name of IPython profile.
    timeout : int
        Time for which we try to connect to get all the engines.
    folder : str, optional
        Folder that is added to the path of the engines, e.g. "~/Work/my_current_project".

    Returns
    -------
    client : ipython.Client object
        An IPyparallel client.
    """
    client = Client(profile=profile, **(client_kwargs or {}))
    dview = _wait_for_successful_ipyparallel_client_start(client, n, timeout)
    dview.use_dill()

    if folder is not None:
        print(f"Adding {folder} to path.")
        cmd = f"import sys, os; sys.path.append(os.path.expanduser('{folder}'))"
        dview.execute(cmd).result()

    return client
