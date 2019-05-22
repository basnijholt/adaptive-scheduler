# FAQ
Here is a list of questions we have either been asked by users or potential pitfalls we hope to help users avoid:

#### Q: What if I have more learners than cores?
**A:** You can distribute all learners in a certain amount of `adaptive.BalancingLearner`s. Like so

```python
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
```
