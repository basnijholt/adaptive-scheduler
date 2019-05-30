import ast
import glob
import math
import os
import random
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, Tuple, Sequence, List, Optional, Callable

import adaptive
import toolz
from adaptive.notebook_integration import in_ipynb
from tqdm import tqdm, tqdm_notebook


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


def _progress(seq: Sequence, with_progress_bar: bool, desc: str = ""):
    if not with_progress_bar:
        return seq
    else:
        if in_ipynb():
            return tqdm_notebook(list(seq), desc=desc)
        else:
            return tqdm(list(seq), desc=desc)


def _cancel_function(cancel_cmd: str, queue_function: Callable) -> Callable:
    def cancel(job_names: List[str], with_progress_bar: bool = True) -> Callable:
        """Cancel all jobs in `job_names`.

        Parameters
        ----------
        job_names : list
            List of job names.
        with_progress_bar : bool, default True
            Display a progress bar using `tqdm`.
        """
        job_names = set(job_names)
        to_cancel = [
            job_id
            for job_id, info in queue_function().items()
            if info["name"] in job_names
        ]
        for job_id in _progress(to_cancel, with_progress_bar, "Canceling jobs"):
            cmd = f"{cancel_cmd} {job_id}"
            returncode = subprocess.run(cmd.split(), stderr=subprocess.PIPE).returncode
            if returncode != 0:
                warnings.warn("Couldn't cancel '{job_id}'.", UserWarning)

    return cancel


def combo_to_fname(combo: Dict[str, Any], folder: Optional[str] = None) -> str:
    """Converts a dict into a human readable filename."""
    fname = "__".join(f"{k}_{v}" for k, v in combo.items()) + ".pickle"
    if folder is None:
        return fname
    return os.path.join(folder, fname)


def cleanup_files(
    job_names: List[str],
    extensions: List[str] = ("sbatch", "out", "batch"),
    with_progress_bar: bool = True,
) -> None:
    """Cleanup scheduler output files.

    Parameters
    ----------
    job_names : list
        List of job names.
    extensions : list
        List of file extensions to be removed.
    with_progress_bar : bool, default True
        Display a progress bar using `tqdm`.
    """
    # Finding the files
    fnames = []
    for job in job_names:
        for ext in extensions:
            fnames += glob.glob(f"{job}*.{ext}")

    # Removing the files
    n_failed = 0
    for fname in _progress(fnames, with_progress_bar, "Removing files"):
        try:
            os.remove(fname)
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


def _get_status_prints(fname, only_last=True):
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


def parse_log_files(job_names: List[str], only_last: bool = True):
    """Parse the log-files and convert it to a `~pandas.core.frame.DataFrame`.

    This only works if you use `adaptive_scheduler.client_support.log_info`
    inside your ``run_script``.

    Parameters
    ----------
    job_names : list
        List of job names.
    only_last : bool, default: True
        Only look use the last printed status message.

    Returns
    -------
    `~pandas.core.frame.DataFrame`
    """
    import pandas as pd

    def convert_type(k, v):
        if k == "elapsed_time":
            return pd.to_timedelta(v)
        elif k == "overhead":
            return float(v[:-1])
        else:
            return ast.literal_eval(v)

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
            info = info.strip().split(" ")
            info = dict([x.split("=") for x in info])
            info = {k: convert_type(k, v) for k, v in info.items()}
            info["job"] = job
            info["time"] = datetime.strptime(time.strip(), "%Y-%m-%d %H:%M.%S")
            infos.append(info)
    return pd.DataFrame(infos)
