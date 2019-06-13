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
from distutils.spawn import find_executable

has_pbs = find_executable("qsub") and find_executable("qstat")
has_slurm = find_executable("sbatch") and find_executable("squeue")

if has_slurm and has_pbs:
    DEFAULT = "SLURM"
    scheduler_system = os.environ.get("SCHEDULER_SYSTEM", DEFAULT)
    if scheduler_system not in ("PBS", "SLURM"):
        raise NotImplementedError(
            f"SCHEDULER_SYSTEM={scheduler_system} is not implemented. Use SLURM or PBS."
        )
elif has_pbs:
    scheduler_system = "pbs"
elif has_slurm:
    scheduler_system = "slurm"

names = ["ext", "get_job_id", "make_job_script", "queue", "submit_cmd", "cancel"]

for name in names:
    module = __import__(
        f"adaptive_scheduler.{scheduler_system.lower()}", fromlist=[name]
    )
    globals()[name] = getattr(module, name)

__all__ = names
