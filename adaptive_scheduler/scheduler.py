"""Scheduler classes for Adaptive Scheduler."""

from __future__ import annotations

import logging
import os
import os.path
import shutil
import warnings

from adaptive_scheduler._scheduler.base_scheduler import BaseScheduler
from adaptive_scheduler._scheduler.local import LocalMockScheduler
from adaptive_scheduler._scheduler.pbs import PBS
from adaptive_scheduler._scheduler.slurm import SLURM, slurm_partitions

__all__ = [
    "PBS",
    "SLURM",
    "BaseScheduler",
    "DefaultScheduler",
    "LocalMockScheduler",
    "slurm_partitions",
]


logger = logging.getLogger(__name__)


def _get_default_scheduler() -> type[BaseScheduler]:
    """Determine which scheduler system is being used.

    It tries to determine it by running both PBS and SLURM commands.

    If both are available then one needs to set an environment variable
    called 'SCHEDULER_SYSTEM' which is either 'PBS' or 'SLURM'.

    For example add the following to your `.bashrc`

    ```bash
    export SCHEDULER_SYSTEM="PBS"
    ```

    By default it is "SLURM".
    """
    has_pbs = bool(shutil.which("qsub")) and bool(shutil.which("qstat"))
    has_slurm = bool(shutil.which("sbatch")) and bool(shutil.which("squeue"))

    default = SLURM
    default_msg = f"We set DefaultScheduler to '{default}'."
    scheduler_system = os.environ.get("SCHEDULER_SYSTEM", "").upper()
    if scheduler_system:
        if scheduler_system not in ("PBS", "SLURM"):
            warnings.warn(
                f"SCHEDULER_SYSTEM={scheduler_system} is not implemented."
                f"Use SLURM or PBS. {default_msg}",
                stacklevel=2,
            )
            return default
        return {"SLURM": SLURM, "PBS": PBS}[scheduler_system]  # type: ignore[return-value]
    if has_slurm and has_pbs:
        msg = f"Both SLURM and PBS are detected. {default_msg}"
        warnings.warn(msg, stacklevel=2)
        return default
    if has_pbs:
        return PBS
    if has_slurm:
        return SLURM
    msg = f"No scheduler system could be detected. {default_msg}"
    logger.info(msg)
    return default


DefaultScheduler = _get_default_scheduler()
