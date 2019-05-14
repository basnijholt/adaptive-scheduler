import math
import random

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
    """Split a list of learners and fnames into `adaptive.BalancingLearners`.

    Parameters
    ----------
    learners : list
        List of learners.
    fnames : list
        List of filenames.
    n_parts : int
        Total number of `~adaptive.BalancingLeaner`s.
    strategy : str
        Learning strategy of the `~adaptive.BalancingLeaner`.

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
