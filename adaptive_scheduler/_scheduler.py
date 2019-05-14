import os

DEFAULT = "SLURM"

scheduler_system = os.environ.get("SCHEDULER_SYSTEM", DEFAULT)
if scheduler_system == "SLURM":
    from adaptive_scheduler.slurm import get_job_id, make_job_script, queue

    ext = ".sbatch"
    submit_cmd = "sbatch"
elif scheduler_system == "PBS":
    from adaptive_scheduler.pbs import get_job_id, make_job_script, queue

    ext = ".batch"
    submit_cmd = "qsub"
else:
    raise NotImplementedError(
        f"SCHEDULER_SYSTEM={scheduler_system} is not implemented. Use SLURM or PBS."
    )

__all__ = ["ext", "get_job_id", "make_job_script", "queue", "submit_cmd"]
