"""Cross-cluster SLURM executor for SLURM multi-cluster.

A ``concurrent.futures.Executor`` that runs each submitted callable as an
independent SLURM job on a (possibly remote) cluster via ``sbatch -M`` and
``sacct -M``.  File transfer between clusters is handled by SLURM's built-in
CWD sync — ``sbatch`` is run from the task directory so SLURM syncs it to
the remote cluster before the job starts and back after it completes.
"""

from __future__ import annotations

import concurrent.futures
import logging
import pickle
import subprocess
import textwrap
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Self

import cloudpickle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RemoteJobError(RuntimeError):
    """Raised when a SLURM job fails on the remote cluster."""

    def __init__(
        self,
        message: str,
        *,
        job_id: str | None = None,
        exit_code: str | None = None,
        state: str | None = None,
    ) -> None:
        super().__init__(message)
        self.job_id = job_id
        self.exit_code = exit_code
        self.state = state


# ---------------------------------------------------------------------------
# CrossClusterFuture
# ---------------------------------------------------------------------------


class CrossClusterFuture(concurrent.futures.Future):
    """A Future whose result is produced by a SLURM job.

    Adds ``job_id`` and ``cluster`` attributes, and overrides ``cancel()``
    to issue ``scancel [-M cluster]``.
    """

    def __init__(self, *, job_id: str, cluster: str | None) -> None:
        super().__init__()
        self.job_id: str = job_id
        self.cluster: str | None = cluster

    def cancel(self) -> bool:
        """Attempt to cancel the SLURM job via ``scancel``."""
        if self.done():
            return False
        try:
            cmd: list[str] = ["scancel"]
            if self.cluster is not None:
                cmd.append(f"-M{self.cluster}")
            cmd.append(self.job_id)
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            logger.warning("scancel failed for job %s", self.job_id)
            return False
        return super().cancel()


# ---------------------------------------------------------------------------
# CrossClusterSlurmExecutor
# ---------------------------------------------------------------------------


class CrossClusterSlurmExecutor(concurrent.futures.Executor):
    """Execute callables as SLURM jobs on a (possibly remote) cluster.

    Parameters
    ----------
    cluster
        SLURM cluster name passed to ``sbatch -M`` / ``sacct -M``.
        ``None`` means the local cluster (no ``-M`` flag).
    partition
        SLURM partition for job submission.
    python_cmd
        Python interpreter on the compute nodes.
    base_dir
        Base directory for job files (local to the submitting host).
        ``~`` is expanded via ``Path.expanduser()``.  SLURM's CWD sync
        transfers each task directory to/from the remote cluster.
    extra_script
        Shell snippet injected into every wrapper script (e.g. env activation).
    extra_sbatch
        Extra ``#SBATCH`` flags (e.g. ``["--comment=gcp-consent", "--time=10:00"]``).
    poll_interval
        Seconds between ``sacct`` polls.

    """

    def __init__(
        self,
        cluster: str | None = None,
        partition: str = "short",
        python_cmd: str = "python",
        base_dir: str | Path = "~/cc-jobs",
        extra_script: str = "",
        extra_sbatch: list[str] | None = None,
        poll_interval: float = 5.0,
    ) -> None:
        self.cluster = cluster
        self.partition = partition
        self.python_cmd = python_cmd
        self.base_dir = Path(base_dir).expanduser()
        self.extra_script = extra_script
        self.extra_sbatch: list[str] = extra_sbatch or []
        self.poll_interval = poll_interval

        self._run_id = uuid.uuid4().hex[:12]
        self._lock = threading.Lock()
        self._pending: dict[
            str,
            tuple[CrossClusterFuture, Path],
        ] = {}  # job_id -> (future, task_dir)
        self._shutdown = False

        # Start the monitor thread lazily on first submit
        self._monitor_thread: threading.Thread | None = None

    # -- public API --------------------------------------------------------

    def submit(  # type: ignore[override]
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> CrossClusterFuture:
        """Pickle *fn(args, kwargs)*, write to task dir, and ``sbatch``."""
        with self._lock:
            if self._shutdown:
                msg = "Cannot submit to a shut-down executor"
                raise RuntimeError(msg)

        task_id = uuid.uuid4().hex[:12]
        task_dir = self.base_dir / self._run_id / task_id

        # 1. Create task directory on shared filesystem
        task_dir.mkdir(parents=True, exist_ok=True)

        # 2. Write pickled callable + arguments
        payload = cloudpickle.dumps((fn, args, kwargs))
        (task_dir / "input.pkl").write_bytes(payload)

        # 3. Generate and write wrapper script
        wrapper = self._make_wrapper(task_id)
        (task_dir / "wrapper.sh").write_text(wrapper)

        # 4. Submit via sbatch (from the task dir so SLURM syncs it)
        job_id = self._sbatch(task_dir)

        # 5. Create future and register with monitor
        future = CrossClusterFuture(job_id=job_id, cluster=self.cluster)
        with self._lock:
            self._pending[job_id] = (future, task_dir)

        cluster_name = self.cluster or "local"
        logger.info("Submitted job %s (task %s) on %s", job_id, task_id, cluster_name)

        # Start monitor thread if not running
        self._ensure_monitor()

        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:  # noqa: FBT001, FBT002
        """Shut down the executor.

        Parameters
        ----------
        wait
            Block until all futures are resolved.
        cancel_futures
            If True, cancel all pending jobs via ``scancel``.

        """
        with self._lock:
            self._shutdown = True

        if cancel_futures:
            with self._lock:
                pending_items = list(self._pending.items())
            for _job_id, (future, _task_dir) in pending_items:
                if not future.done():
                    future.cancel()

        if wait and self._monitor_thread is not None:
            self._monitor_thread.join()

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown(wait=True)

    # -- internal ----------------------------------------------------------

    def _make_wrapper(self, task_id: str) -> str:
        """Generate the SBATCH wrapper script for a single task.

        All paths are relative — sbatch is run from the task directory, and
        SLURM's CWD sync ensures the files are available on the remote cluster.
        """
        lines = [
            "#!/bin/bash",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --job-name=cc-{task_id[:8]}",
            "#SBATCH --output=stdout.log",
            "#SBATCH --error=stderr.log",
        ]
        lines.extend(f"#SBATCH {flag}" for flag in self.extra_sbatch)
        lines.append("")
        if self.extra_script:
            lines.append(self.extra_script)
            lines.append("")
        lines.append(f"{self.python_cmd} << 'PYTHON_EOF'")
        lines.append(
            textwrap.dedent("""\
            import cloudpickle, pickle, sys, traceback
            with open('input.pkl', 'rb') as f:
                fn, args, kwargs = cloudpickle.load(f)
            try:
                result = fn(*args, **kwargs)
                payload = {'status': 'ok', 'result': result}
            except Exception as e:
                payload = {'status': 'error', 'error': str(e), 'tb': traceback.format_exc()}
            with open('output.pkl', 'wb') as f:
                pickle.dump(payload, f)
        """).rstrip(),
        )
        lines.append("PYTHON_EOF")
        lines.append("")
        return "\n".join(lines) + "\n"

    def _sbatch(self, task_dir: Path) -> str:
        """Submit ``wrapper.sh`` via ``sbatch`` from *task_dir*.

        Running sbatch from the task directory ensures SLURM's CWD sync
        transfers the directory contents to the remote cluster.
        """
        cmd: list[str] = ["sbatch"]
        if self.cluster is not None:
            cmd.append(f"-M{self.cluster}")
        cmd.append("wrapper.sh")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=task_dir,
        )
        # Parse job ID from "Submitted batch job 12345 [on cluster ...]"
        stdout = result.stdout.strip()
        parts = stdout.split()
        for i, part in enumerate(parts):
            if part == "job" and i + 1 < len(parts):
                return parts[i + 1]
        msg = f"Could not parse job ID from sbatch output: {stdout}"
        raise RuntimeError(msg)

    def _ensure_monitor(self) -> None:
        """Start the monitor thread if it isn't running yet."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._monitor_thread = threading.Thread(
                target=self._monitor,
                daemon=True,
                name="cc-monitor",
            )
            self._monitor_thread.start()

    def _monitor(self) -> None:
        """Background thread: poll sacct and resolve futures."""
        while True:
            with self._lock:
                if not self._pending:
                    if self._shutdown:
                        return
                    pending_snapshot = {}
                else:
                    pending_snapshot = dict(self._pending)

            if not pending_snapshot:
                time.sleep(self.poll_interval)
                continue

            # Query sacct for all tracked jobs in one call
            job_ids = list(pending_snapshot.keys())
            states = self._query_sacct_batch(job_ids)

            resolved: list[str] = []
            for job_id, (future, task_dir) in pending_snapshot.items():
                if future.done() or self._resolve_job(
                    job_id,
                    future,
                    task_dir,
                    states,
                ):
                    resolved.append(job_id)

            # Remove resolved jobs from pending
            if resolved:
                with self._lock:
                    for jid in resolved:
                        self._pending.pop(jid, None)

            # Check if we're done
            with self._lock:
                if not self._pending and self._shutdown:
                    return

            time.sleep(self.poll_interval)

    def _resolve_job(
        self,
        job_id: str,
        future: CrossClusterFuture,
        task_dir: Path,
        states: dict[str, tuple[str, str]],
    ) -> bool:
        """Try to resolve a single job. Return True if the job is resolved."""
        state, exit_code = states.get(job_id, ("UNKNOWN", "?"))

        if state in ("RUNNING", "PENDING", "UNKNOWN"):
            return False

        if state == "COMPLETED" and exit_code == "0:0":
            self._resolve_completed_job(job_id, future, task_dir, exit_code, state)
        else:
            future.set_exception(
                RemoteJobError(
                    f"Job {job_id} failed with state={state}, exit_code={exit_code}. "
                    f"Check logs at {task_dir}/stderr.log.",
                    job_id=job_id,
                    exit_code=exit_code,
                    state=state,
                ),
            )
        return True

    def _resolve_completed_job(
        self,
        job_id: str,
        future: CrossClusterFuture,
        task_dir: Path,
        exit_code: str,
        state: str,
    ) -> None:
        """Resolve a COMPLETED job by fetching its result."""
        try:
            result = self._fetch_result(task_dir)
        except (OSError, pickle.UnpicklingError) as exc:
            future.set_exception(
                RemoteJobError(
                    f"Job {job_id} completed but failed to fetch result: {exc}",
                    job_id=job_id,
                    exit_code=exit_code,
                    state=state,
                ),
            )
            return

        if result["status"] == "ok":
            future.set_result(result["result"])
        else:
            future.set_exception(
                RemoteJobError(
                    f"Job {job_id} raised an exception:\n"
                    f"{result.get('tb', result.get('error', 'unknown error'))}",
                    job_id=job_id,
                    exit_code=exit_code,
                    state=state,
                ),
            )

    def _query_sacct_batch(self, job_ids: list[str]) -> dict[str, tuple[str, str]]:
        """Return {job_id: (State, ExitCode)} for all *job_ids* via a single ``sacct`` call."""
        cmd: list[str] = ["sacct"]
        if self.cluster is not None:
            cmd.append(f"-M{self.cluster}")
        cmd.extend(["-j", ",".join(job_ids), "-n", "--format=JobID,State,ExitCode", "-P"])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning("sacct query failed: %s", exc)
            return {}

        states: dict[str, tuple[str, str]] = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) >= 3:  # noqa: PLR2004
                raw_id = parts[0].strip()
                # Skip sub-steps like "12345.batch" — use parent job line only
                if "." in raw_id:
                    continue
                state = parts[1].strip()
                exit_code = parts[2].strip()
                states[raw_id] = (state, exit_code)
        return states

    def _fetch_result(
        self,
        task_dir: Path,
        retries: int = 12,
        delay: float = 5.0,
    ) -> dict[str, Any]:
        """Read ``output.pkl`` from *task_dir*, waiting for SLURM sync.

        After ``sacct`` reports COMPLETED, the CWD sync-back may still be
        in progress.  Retry up to *retries* times with *delay* seconds
        between attempts.
        """
        output_path = task_dir / "output.pkl"
        for attempt in range(retries):
            if output_path.exists():
                with output_path.open("rb") as f:
                    return pickle.load(f)  # noqa: S301
            logger.debug(
                "output.pkl not yet available (attempt %d/%d), waiting for sync...",
                attempt + 1,
                retries,
            )
            time.sleep(delay)
        # Final attempt — raise if still missing
        with output_path.open("rb") as f:
            return pickle.load(f)  # noqa: S301
