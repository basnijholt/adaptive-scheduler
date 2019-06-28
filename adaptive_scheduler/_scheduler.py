"""In this file it is determined which scheduler system is being used.

It tries to determine it by running both PBS and SLURM commands.

If both are available then one needs to set an environment variable
called 'SCHEDULER_SYSTEM' which is either 'PBS' or 'SLURM'.

For example add the following to your `.bashrc`

```bash
export SCHEDULER_SYSTEM="PBS"
```

By default it is "SLURM".
"""

import os
import types
import warnings
from distutils.spawn import find_executable

from adaptive_scheduler import pbs
from adaptive_scheduler import slurm

DEFAULT = "SLURM"
DEFAULT_MODULE = slurm

has_pbs = bool(find_executable("qsub")) and bool(find_executable("qstat"))
has_slurm = bool(find_executable("sbatch")) and bool(find_executable("squeue"))

if has_slurm and has_pbs:
    scheduler_system = os.environ.get("SCHEDULER_SYSTEM", DEFAULT)
    if scheduler_system not in ("PBS", "SLURM"):
        raise NotImplementedError(
            f"SCHEDULER_SYSTEM={scheduler_system} is not implemented. Use SLURM or PBS."
        )
elif has_pbs:
    scheduler_system = "pbs"
    scheduler_module: types.ModuleType = pbs
elif has_slurm:
    scheduler_system = "slurm"
    scheduler_module: types.ModuleType = slurm
elif not has_slurm and not has_pbs:
    scheduler_system = DEFAULT
    scheduler_module: types.ModuleType = DEFAULT_MODULE
    msg = f"No scheduler system could be detected. We set it to '{scheduler_system}'."
    warnings.warn(msg)

from scheduler_module import (
    ext,
    get_job_id,
    make_job_script,
    queue,
    submit_cmd,
    cancel,
)  # noqa: E402

__all__ = ["ext", "get_job_id", "make_job_script", "queue", "submit_cmd", "cancel"]
