import abc
import collections.abc
import inspect
import math
import os
import random
import shutil
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import Any, Callable, Dict, List, Optional, Tuple

import adaptive
import toolz
from adaptive.notebook_integration import in_ipynb
from ipyparallel import Client
from tqdm import tqdm, tqdm_notebook

MAX_LINE_LENGTH = 100


class _RequireAttrsABCMeta(abc.ABCMeta):
    required_attributes = []

    def __call__(self, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        for name in obj.required_attributes:
            if not hasattr(obj, name):
                raise ValueError(f"Required attribute {name} not set in __init__.")
        return obj


def shuffle_list(*lists, seed=0):
    """Shuffle multiple lists in the same order."""
    combined = list(zip(*lists))
    random.Random(seed).shuffle(combined)
    return zip(*combined)


def _split(seq: collections.abc.Iterable, n_parts: int):
    # TODO: remove this in v1.0.0
    s = "adaptive_scheduler.utils."
    raise Exception(f"`{s}_split` is renamed to {s}split`.")


def split(seq: collections.abc.Iterable, n_parts: int):
    """Split up a sequence into ``n_parts``.

    Parameters
    ----------
    seq : sequence
        A list or other iterable that has to be split up.
    n_parts : int
        The sequence will be split up in this many parts.

    Returns
    -------
    iterable of tuples"""
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
    for x in split(zip(learners, fnames), n_parts):
        learners_part, fnames_part = zip(*x)
        learner = adaptive.BalancingLearner(learners_part, strategy=strategy)
        new_learners.append(learner)
        new_fnames.append(fnames_part)
    return new_learners, new_fnames


def _get_npoints(learner: adaptive.BaseLearner) -> Optional[int]:
    with suppress(AttributeError):
        return learner.npoints
    with suppress(AttributeError):
        # If the Learner is a BalancingLearner
        return sum(l.npoints for l in learner.learners)


def _progress(
    seq: collections.abc.Iterable, with_progress_bar: bool = True, desc: str = ""
):
    if not with_progress_bar:
        return seq
    else:
        if in_ipynb():
            return tqdm_notebook(list(seq), desc=desc)
        else:
            return tqdm(list(seq), desc=desc)


def combo_to_fname(combo: Dict[str, Any], folder: Optional[str] = None) -> str:
    """Converts a dict into a human readable filename."""
    fname = "__".join(f"{k}_{v}" for k, v in combo.items()) + ".pickle"
    if folder is None:
        return fname
    return os.path.join(folder, fname)


def _remove_or_move_files(
    fnames: List[str],
    with_progress_bar: bool = True,
    move_to: Optional[str] = None,
    desc: Optional[str] = None,
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
    desc : str, default: None
        Description of the progressbar.
    """
    n_failed = 0
    for fname in _progress(fnames, with_progress_bar, desc or "Removing files"):
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
    max_workers: Optional[int] = None,
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
    max_workers : int, optional
        The maximum number of parallel threads when loading the data.
        If ``None``, use the maximum number of threads that is possible.
    """

    def load(learner, fname):
        learner.load(fname)

    with ThreadPoolExecutor(max_workers) as ex:
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
        iterator = zip(learners, fnames)
        pbar = _progress(iterator, with_progress_bar, "Submitting saving tasks")
        futs = [ex.submit(save, *args) for args in pbar]
        for fut in _progress(futs, with_progress_bar, "Finishing saving"):
            fut.result()


def _print_same_line(msg: str, new_line_end: bool = False):
    msg = msg.strip()
    global MAX_LINE_LENGTH
    MAX_LINE_LENGTH = max(len(msg), MAX_LINE_LENGTH)
    empty_space = max(MAX_LINE_LENGTH - len(msg), 0) * " "
    print(msg + empty_space, end="\r" if not new_line_end else "\n")


def _wait_for_successful_ipyparallel_client_start(client, n: int, timeout: int):
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
        Folder that is added to the path of the engines, e.g. ``"~/Work/my_current_project"``.

    Returns
    -------
    client : `ipyparallel.Client` object
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


def _get_default_args(func: Callable) -> Dict[str, str]:
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def log_exception(log, msg, exception):
    try:
        raise exception
    except Exception:
        log.exception(msg, exc_info=True)
