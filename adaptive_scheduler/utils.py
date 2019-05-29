import math
import random
import os.path
import subprocess
import warnings

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


def split_in_balancing_learners(learners, fnames, n_parts, strategy="npoints"):
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


def _progress(seq, with_progress_bar, desc=""):
    if not with_progress_bar:
        return seq
    else:
        if in_ipynb():
            return tqdm_notebook(list(seq), desc=desc)
        else:
            return tqdm(list(seq), desc=desc)


def _cancel_function(cancel_cmd, queue_function):
    def cancel(job_names, with_progress_bar=True):
        """Cancel all jobs in `job_names`.

        Parameters
        ----------
        job_names : str
            List of job names.
        with_progress_bar : bool, default False
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


def combo_to_fname(combo, folder=None):
    """Converts a dict into a human readable filename."""
    fname = "__".join(f"{k}_{v}" for k, v in combo.items()) + ".pickle"
    if folder is None:
        return fname
    return os.path.join(folder, fname)
