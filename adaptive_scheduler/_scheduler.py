"""In this file it is determined which scheduler system is being used.

One needs to set an environment variable called 'SCHEDULER_SYSTEM' which
is either 'PBS' or 'SLURM'.

For example add the following to your `.bashrc`

```bash
export SCHEDULER_SYSTEM="PBS"
```

By default it is "SLURM".
"""

import os

DEFAULT = "SLURM"

scheduler_system = os.environ.get("SCHEDULER_SYSTEM", DEFAULT)
if scheduler_system == "SLURM":
    from adaptive_scheduler.slurm import (
        get_job_id,
        make_job_script,
        queue,
        ext,
        submit_cmd,
    )
elif scheduler_system == "PBS":
    from adaptive_scheduler.pbs import (
        get_job_id,
        make_job_script,
        queue,
        ext,
        submit_cmd,
    )
else:
    raise NotImplementedError(
        f"SCHEDULER_SYSTEM={scheduler_system} is not implemented. Use SLURM or PBS."
    )

__all__ = ["ext", "get_job_id", "make_job_script", "queue", "submit_cmd"]
