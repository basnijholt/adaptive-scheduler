import math
import random
import subprocess
import warnings

import adaptive
import toolz


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


def _cancel_function(cancel_cmd, queue_function):
    def cancel(job_names):
        """Cancel all jobs in `job_names`."""
        job_names = set(job_names)
        to_cancel = [
            job_id
            for job_id, info in queue_function().items()
            if info["name"] in job_names
        ]
        for job_id in to_cancel:
            cmd = f"{cancel_cmd} {job_id}"
            returncode = subprocess.run(cmd.split(), stderr=subprocess.PIPE).returncode
            if returncode != 0:
                warnings.warn("Couldn't cancel '{job_id}'.", UserWarning)

    return cancel
