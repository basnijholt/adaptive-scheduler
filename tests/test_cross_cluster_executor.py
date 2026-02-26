"""Tests for CrossClusterSlurmExecutor."""

from __future__ import annotations

import pickle
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from adaptive_scheduler._cross_cluster_executor import (
    CrossClusterFuture,
    CrossClusterSlurmExecutor,
    RemoteJobError,
)

if TYPE_CHECKING:
    from collections.abc import Generator

MODULE = "adaptive_scheduler._cross_cluster_executor"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sbatch_result(job_id: str = "12345", cluster: str | None = None) -> MagicMock:
    """Build a mock CompletedProcess for a successful sbatch call."""
    stdout = f"Submitted batch job {job_id}"
    if cluster:
        stdout += f" on cluster {cluster}"
    result = MagicMock()
    result.stdout = stdout
    return result


def _sacct_result(rows: list[str]) -> MagicMock:
    """Build a mock CompletedProcess for a sacct call.

    Each row should be a pipe-separated string like "12345|COMPLETED|0:0".
    """
    result = MagicMock()
    result.stdout = "\n".join(rows) + "\n"
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def executor(tmp_path: Path) -> CrossClusterSlurmExecutor:
    """Local-mode executor with fast polling."""
    return CrossClusterSlurmExecutor(
        base_dir=tmp_path / "cc-jobs",
        poll_interval=0.01,
    )


@pytest.fixture
def cluster_executor(tmp_path: Path) -> CrossClusterSlurmExecutor:
    """Remote-cluster executor with extras."""
    return CrossClusterSlurmExecutor(
        cluster="ionqgcp",
        partition="gpu",
        python_cmd="/opt/conda/bin/python",
        base_dir=tmp_path / "cc-jobs",
        extra_script="source /opt/env/bin/activate",
        extra_sbatch=["--comment=gcp-consent", "--time=10:00"],
        poll_interval=0.01,
    )


@pytest.fixture
def _mock_sbatch() -> Generator[MagicMock, None, None]:
    """Patch subprocess.run to return a successful sbatch result."""
    with patch(f"{MODULE}.subprocess.run", return_value=_sbatch_result()) as m:
        yield m


# ---------------------------------------------------------------------------
# 1. RemoteJobError
# ---------------------------------------------------------------------------


class TestRemoteJobError:
    """Tests for the RemoteJobError exception class."""

    def test_attributes(self) -> None:
        """Test that RemoteJobError stores job_id, exit_code, and state."""
        err = RemoteJobError(
            "boom",
            job_id="111",
            exit_code="1:0",
            state="FAILED",
        )
        assert str(err) == "boom"
        assert err.job_id == "111"
        assert err.exit_code == "1:0"
        assert err.state == "FAILED"


# ---------------------------------------------------------------------------
# 2. __init__
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for CrossClusterSlurmExecutor.__init__."""

    def test_defaults(self, tmp_path: Path) -> None:
        """Test that default parameters are set correctly."""
        ex = CrossClusterSlurmExecutor(base_dir=tmp_path)
        assert ex.cluster is None
        assert ex.partition == "short"
        assert ex.python_cmd == "python"
        assert ex.extra_sbatch == []
        assert ex.extra_script == ""
        assert ex.poll_interval == 5.0
        assert ex._pending == {}
        assert ex._shutdown is False
        assert ex._monitor_thread is None

    def test_custom_params(self, cluster_executor: CrossClusterSlurmExecutor) -> None:
        """Test that custom parameters are stored correctly."""
        ex = cluster_executor
        assert ex.cluster == "ionqgcp"
        assert ex.partition == "gpu"
        assert ex.python_cmd == "/opt/conda/bin/python"
        assert ex.extra_sbatch == ["--comment=gcp-consent", "--time=10:00"]
        assert "activate" in ex.extra_script

    def test_tilde_expansion(self) -> None:
        """Test that ~ in base_dir is expanded to the home directory."""
        ex = CrossClusterSlurmExecutor(base_dir="~/cc-jobs")
        assert "~" not in str(ex.base_dir)
        assert ex.base_dir == Path.home() / "cc-jobs"


# ---------------------------------------------------------------------------
# 3. _make_wrapper
# ---------------------------------------------------------------------------


class TestMakeWrapper:
    """Tests for the SBATCH wrapper script generation."""

    def test_basic_directives(self, executor: CrossClusterSlurmExecutor) -> None:
        """Test that the wrapper contains required SBATCH directives."""
        wrapper = executor._make_wrapper("abcd1234efgh")
        assert "#!/bin/bash" in wrapper
        assert "#SBATCH --partition=short" in wrapper
        assert "#SBATCH --job-name=cc-abcd1234" in wrapper
        assert "#SBATCH --output=stdout.log" in wrapper
        assert "#SBATCH --error=stderr.log" in wrapper

    def test_relative_paths(self, executor: CrossClusterSlurmExecutor) -> None:
        """Test that the wrapper uses relative paths for pkl files."""
        wrapper = executor._make_wrapper("task0001")
        assert "input.pkl" in wrapper
        assert "output.pkl" in wrapper
        # Should NOT contain absolute paths to the base_dir
        assert str(executor.base_dir) not in wrapper

    def test_extra_sbatch(self, cluster_executor: CrossClusterSlurmExecutor) -> None:
        """Test that extra_sbatch flags appear in the wrapper."""
        wrapper = cluster_executor._make_wrapper("task0001")
        assert "#SBATCH --comment=gcp-consent" in wrapper
        assert "#SBATCH --time=10:00" in wrapper

    def test_extra_script(self, cluster_executor: CrossClusterSlurmExecutor) -> None:
        """Test that extra_script content is injected into the wrapper."""
        wrapper = cluster_executor._make_wrapper("task0001")
        assert "source /opt/env/bin/activate" in wrapper


# ---------------------------------------------------------------------------
# 4. _sbatch
# ---------------------------------------------------------------------------


class TestSbatch:
    """Tests for sbatch submission and output parsing."""

    def test_parse_local_output(self, executor: CrossClusterSlurmExecutor, tmp_path: Path) -> None:
        """Test parsing job ID from local sbatch output."""
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        with patch(f"{MODULE}.subprocess.run", return_value=_sbatch_result("99999")):
            job_id = executor._sbatch(task_dir)
        assert job_id == "99999"

    def test_parse_multicluster_output(
        self,
        cluster_executor: CrossClusterSlurmExecutor,
        tmp_path: Path,
    ) -> None:
        """Test parsing job ID from multi-cluster sbatch output."""
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        with patch(
            f"{MODULE}.subprocess.run",
            return_value=_sbatch_result("55555", cluster="ionqgcp"),
        ):
            job_id = cluster_executor._sbatch(task_dir)
        assert job_id == "55555"

    def test_unparseable_raises(self, executor: CrossClusterSlurmExecutor, tmp_path: Path) -> None:
        """Test that unparseable sbatch output raises RuntimeError."""
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        bad = MagicMock()
        bad.stdout = "Something unexpected"
        with (
            patch(f"{MODULE}.subprocess.run", return_value=bad),
            pytest.raises(RuntimeError, match="Could not parse"),
        ):
            executor._sbatch(task_dir)

    def test_subprocess_error_propagates(
        self,
        executor: CrossClusterSlurmExecutor,
        tmp_path: Path,
    ) -> None:
        """Test that CalledProcessError from sbatch propagates."""
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        with (
            patch(
                f"{MODULE}.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "sbatch"),
            ),
            pytest.raises(subprocess.CalledProcessError),
        ):
            executor._sbatch(task_dir)


# ---------------------------------------------------------------------------
# 5. submit
# ---------------------------------------------------------------------------


class TestSubmit:
    """Tests for the submit method."""

    @pytest.mark.usefixtures("_mock_sbatch")
    def test_creates_files(
        self,
        executor: CrossClusterSlurmExecutor,
    ) -> None:
        """Test that submit creates input.pkl and wrapper.sh in the task directory."""
        with patch.object(executor, "_ensure_monitor"):
            future = executor.submit(lambda x: x, 42)
        assert isinstance(future, CrossClusterFuture)
        # Check that input.pkl and wrapper.sh were created
        task_dir = next((executor.base_dir / executor._run_id).iterdir())
        assert (task_dir / "input.pkl").exists()
        assert (task_dir / "wrapper.sh").exists()

    def test_returns_future_with_job_id(self, executor: CrossClusterSlurmExecutor) -> None:
        """Test that submit returns a CrossClusterFuture with the correct job_id."""
        with (
            patch(f"{MODULE}.subprocess.run", return_value=_sbatch_result("77777")),
            patch.object(executor, "_ensure_monitor"),
        ):
            future = executor.submit(lambda x: x, 1)
        assert future.job_id == "77777"
        assert future.cluster is None

    @pytest.mark.usefixtures("_mock_sbatch")
    def test_registers_pending(
        self,
        executor: CrossClusterSlurmExecutor,
    ) -> None:
        """Test that submit registers the job in the _pending dict."""
        with patch.object(executor, "_ensure_monitor"):
            future = executor.submit(lambda: None)
        assert "12345" in executor._pending
        assert executor._pending["12345"][0] is future

    def test_raises_after_shutdown(self, executor: CrossClusterSlurmExecutor) -> None:
        """Test that submit raises RuntimeError after shutdown."""
        executor._shutdown = True
        with pytest.raises(RuntimeError, match="shut-down"):
            executor.submit(lambda: None)


# ---------------------------------------------------------------------------
# 6. _query_sacct_batch
# ---------------------------------------------------------------------------


class TestQuerySacctBatch:
    """Tests for sacct batch querying and output parsing."""

    def test_parse_output(self, executor: CrossClusterSlurmExecutor) -> None:
        """Test parsing job states from sacct output."""
        sacct_out = _sacct_result(["12345|COMPLETED|0:0", "67890|FAILED|1:0"])
        with patch(f"{MODULE}.subprocess.run", return_value=sacct_out):
            states = executor._query_sacct_batch(["12345", "67890"])
        assert states == {
            "12345": ("COMPLETED", "0:0"),
            "67890": ("FAILED", "1:0"),
        }

    def test_skip_substeps(self, executor: CrossClusterSlurmExecutor) -> None:
        """Test that sub-step lines like 12345.batch are skipped."""
        sacct_out = _sacct_result(
            [
                "12345|COMPLETED|0:0",
                "12345.batch|COMPLETED|0:0",
                "12345.extern|COMPLETED|0:0",
            ],
        )
        with patch(f"{MODULE}.subprocess.run", return_value=sacct_out):
            states = executor._query_sacct_batch(["12345"])
        assert len(states) == 1
        assert "12345" in states

    def test_empty_on_failure(self, executor: CrossClusterSlurmExecutor) -> None:
        """Test that CalledProcessError returns an empty dict."""
        with patch(
            f"{MODULE}.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "sacct"),
        ):
            states = executor._query_sacct_batch(["12345"])
        assert states == {}

    def test_empty_on_file_not_found(self, executor: CrossClusterSlurmExecutor) -> None:
        """Test that FileNotFoundError returns an empty dict."""
        with patch(f"{MODULE}.subprocess.run", side_effect=FileNotFoundError):
            states = executor._query_sacct_batch(["12345"])
        assert states == {}

    def test_cluster_flag(self, cluster_executor: CrossClusterSlurmExecutor) -> None:
        """Test that the -M flag is passed when cluster is set."""
        sacct_out = _sacct_result(["12345|RUNNING|0:0"])
        with patch(f"{MODULE}.subprocess.run", return_value=sacct_out) as mock_run:
            cluster_executor._query_sacct_batch(["12345"])
        cmd = mock_run.call_args[0][0]
        assert "-Mionqgcp" in cmd


# ---------------------------------------------------------------------------
# 7. _fetch_result
# ---------------------------------------------------------------------------


class TestFetchResult:
    """Tests for reading output.pkl with retry logic."""

    def test_reads_pkl(self, executor: CrossClusterSlurmExecutor, tmp_path: Path) -> None:
        """Test reading an existing output.pkl file."""
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        payload = {"status": "ok", "result": 42}
        (task_dir / "output.pkl").write_bytes(pickle.dumps(payload))
        result = executor._fetch_result(task_dir, retries=1, delay=0)
        assert result == payload

    def test_retries_until_file_appears(
        self,
        executor: CrossClusterSlurmExecutor,
        tmp_path: Path,
    ) -> None:
        """Test that _fetch_result retries and succeeds when file appears later."""
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        output_path = task_dir / "output.pkl"
        payload = {"status": "ok", "result": 99}

        call_count = 0

        def delayed_write(_seconds: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                output_path.write_bytes(pickle.dumps(payload))

        with patch(f"{MODULE}.time.sleep", side_effect=delayed_write):
            result = executor._fetch_result(task_dir, retries=5, delay=0.01)
        assert result["result"] == 99

    def test_raises_after_retries_exhausted(
        self,
        executor: CrossClusterSlurmExecutor,
        tmp_path: Path,
    ) -> None:
        """Test that FileNotFoundError is raised when retries are exhausted."""
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        with (
            patch(f"{MODULE}.time.sleep"),
            pytest.raises(FileNotFoundError),
        ):
            executor._fetch_result(task_dir, retries=2, delay=0)


# ---------------------------------------------------------------------------
# 8. _monitor
# ---------------------------------------------------------------------------


class TestMonitor:
    """Tests for the background monitor thread logic."""

    def _run_monitor_once(self, executor: CrossClusterSlurmExecutor) -> None:
        """Set _shutdown so the monitor exits after processing pending jobs."""
        executor._shutdown = True
        with patch(f"{MODULE}.time.sleep"):
            executor._monitor()

    def test_resolves_completed(self, executor: CrossClusterSlurmExecutor, tmp_path: Path) -> None:
        """Test that a COMPLETED job resolves the future with the result."""
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        payload = {"status": "ok", "result": 42}
        (task_dir / "output.pkl").write_bytes(pickle.dumps(payload))

        future = CrossClusterFuture(job_id="100", cluster=None)
        executor._pending["100"] = (future, task_dir)

        sacct_out = _sacct_result(["100|COMPLETED|0:0"])
        with patch(f"{MODULE}.subprocess.run", return_value=sacct_out):
            self._run_monitor_once(executor)

        assert future.result() == 42

    def test_sets_error_on_failed(
        self,
        executor: CrossClusterSlurmExecutor,
        tmp_path: Path,
    ) -> None:
        """Test that a FAILED job sets RemoteJobError on the future."""
        task_dir = tmp_path / "task"
        task_dir.mkdir()

        future = CrossClusterFuture(job_id="200", cluster=None)
        executor._pending["200"] = (future, task_dir)

        sacct_out = _sacct_result(["200|FAILED|1:0"])
        with patch(f"{MODULE}.subprocess.run", return_value=sacct_out):
            self._run_monitor_once(executor)

        with pytest.raises(RemoteJobError, match="state=FAILED"):
            future.result()
        exc = future.exception()
        assert isinstance(exc, RemoteJobError)
        assert exc.job_id == "200"
        assert exc.state == "FAILED"

    def test_handles_remote_python_error(
        self,
        executor: CrossClusterSlurmExecutor,
        tmp_path: Path,
    ) -> None:
        """Test that a remote Python exception is wrapped in RemoteJobError."""
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        payload = {"status": "error", "error": "ZeroDivisionError", "tb": "Traceback..."}
        (task_dir / "output.pkl").write_bytes(pickle.dumps(payload))

        future = CrossClusterFuture(job_id="300", cluster=None)
        executor._pending["300"] = (future, task_dir)

        sacct_out = _sacct_result(["300|COMPLETED|0:0"])
        with patch(f"{MODULE}.subprocess.run", return_value=sacct_out):
            self._run_monitor_once(executor)

        with pytest.raises(RemoteJobError, match="Traceback"):
            future.result()

    def test_skips_running_and_pending(
        self,
        executor: CrossClusterSlurmExecutor,
        tmp_path: Path,
    ) -> None:
        """Test that RUNNING and PENDING jobs are not resolved."""
        task_dir = tmp_path / "task"
        task_dir.mkdir()

        future_r = CrossClusterFuture(job_id="400", cluster=None)
        future_p = CrossClusterFuture(job_id="401", cluster=None)
        executor._pending["400"] = (future_r, task_dir)
        executor._pending["401"] = (future_p, task_dir)

        sacct_out = _sacct_result(["400|RUNNING|0:0", "401|PENDING|0:0"])
        # Monitor must exit even with pending jobs when _shutdown=True and
        # pending remains. We simulate two iterations: first returns RUNNING/PENDING,
        # second we let shutdown take effect by clearing pending.
        call_count = 0

        def fake_sleep(_: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                # After first sleep, remove pending so monitor exits
                with executor._lock:
                    executor._pending.clear()

        executor._shutdown = True
        with (
            patch(f"{MODULE}.subprocess.run", return_value=sacct_out),
            patch(f"{MODULE}.time.sleep", side_effect=fake_sleep),
        ):
            executor._monitor()

        # Futures should still be unresolved
        assert not future_r.done()
        assert not future_p.done()

    def test_exits_on_shutdown_no_pending(self, executor: CrossClusterSlurmExecutor) -> None:
        """Test that the monitor exits immediately when shutdown with no pending jobs."""
        executor._shutdown = True
        # No pending jobs — monitor should return immediately
        with patch(f"{MODULE}.time.sleep"):
            executor._monitor()

    def test_handles_fetch_failure(
        self,
        executor: CrossClusterSlurmExecutor,
        tmp_path: Path,
    ) -> None:
        """Test that a fetch failure on COMPLETED job sets RemoteJobError."""
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        # No output.pkl — _fetch_result will fail

        future = CrossClusterFuture(job_id="500", cluster=None)
        executor._pending["500"] = (future, task_dir)

        sacct_out = _sacct_result(["500|COMPLETED|0:0"])
        with (
            patch(f"{MODULE}.subprocess.run", return_value=sacct_out),
            patch(f"{MODULE}.time.sleep"),
        ):
            # _fetch_result will raise since no output.pkl; monitor should catch it
            executor._shutdown = True
            executor._monitor()

        with pytest.raises(RemoteJobError, match="failed to fetch result"):
            future.result()


# ---------------------------------------------------------------------------
# 9. CrossClusterFuture.cancel
# ---------------------------------------------------------------------------


class TestCrossClusterFutureCancel:
    """Tests for CrossClusterFuture.cancel via scancel."""

    def test_cancel_with_cluster(self) -> None:
        """Test that cancel passes -M flag when cluster is set."""
        future = CrossClusterFuture(job_id="111", cluster="ionqgcp")
        with patch(f"{MODULE}.subprocess.run") as mock_run:
            result = future.cancel()
        assert result is True
        cmd = mock_run.call_args[0][0]
        assert cmd == ["scancel", "-Mionqgcp", "111"]

    def test_cancel_without_cluster(self) -> None:
        """Test that cancel omits -M flag for local cluster."""
        future = CrossClusterFuture(job_id="222", cluster=None)
        with patch(f"{MODULE}.subprocess.run") as mock_run:
            result = future.cancel()
        assert result is True
        cmd = mock_run.call_args[0][0]
        assert cmd == ["scancel", "222"]

    def test_returns_false_when_done(self) -> None:
        """Test that cancel returns False when the future is already done."""
        future = CrossClusterFuture(job_id="333", cluster=None)
        future.set_result(42)
        result = future.cancel()
        assert result is False

    def test_returns_false_on_subprocess_failure(self) -> None:
        """Test that cancel returns False when scancel fails."""
        future = CrossClusterFuture(job_id="444", cluster=None)
        with patch(
            f"{MODULE}.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "scancel"),
        ):
            result = future.cancel()
        assert result is False


# ---------------------------------------------------------------------------
# 10. shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    """Tests for executor shutdown behavior."""

    def test_sets_flag(self, executor: CrossClusterSlurmExecutor) -> None:
        """Test that shutdown sets the _shutdown flag."""
        executor.shutdown(wait=False)
        assert executor._shutdown is True

    def test_cancel_futures_calls_cancel(
        self,
        executor: CrossClusterSlurmExecutor,
        tmp_path: Path,
    ) -> None:
        """Test that shutdown with cancel_futures cancels pending jobs."""
        future = CrossClusterFuture(job_id="555", cluster=None)
        executor._pending["555"] = (future, tmp_path)
        with patch(f"{MODULE}.subprocess.run"):
            executor.shutdown(wait=False, cancel_futures=True)
        assert future.cancelled()

    def test_wait_joins_thread(self, executor: CrossClusterSlurmExecutor) -> None:
        """Test that shutdown with wait=True joins the monitor thread."""
        mock_thread = MagicMock(spec=threading.Thread)
        executor._monitor_thread = mock_thread
        executor.shutdown(wait=True)
        mock_thread.join.assert_called_once()


# ---------------------------------------------------------------------------
# 11. Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    """Tests for the context manager protocol."""

    def test_enter_returns_self(self, executor: CrossClusterSlurmExecutor) -> None:
        """Test that __enter__ returns the executor instance."""
        with patch.object(executor, "shutdown"), executor as ex:
            assert ex is executor

    def test_exit_calls_shutdown(self, executor: CrossClusterSlurmExecutor) -> None:
        """Test that __exit__ calls shutdown(wait=True)."""
        with patch.object(executor, "shutdown") as mock_shutdown, executor:
            pass
        mock_shutdown.assert_called_once_with(wait=True)


# ---------------------------------------------------------------------------
# 12. End-to-end (mocked subprocess, real threads)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end tests with real monitor threads and mocked subprocess calls."""

    def test_submit_then_resolve_local(self, executor: CrossClusterSlurmExecutor) -> None:
        """Submit a job, have sacct report COMPLETED, and verify the future resolves."""

        def fake_subprocess_run(cmd: list[str], **kwargs: Any) -> MagicMock:
            if cmd[0] == "sbatch":
                # Write output.pkl atomically in the sbatch mock to avoid
                # a race with the monitor thread that starts immediately.
                cwd = kwargs.get("cwd")
                if cwd is not None:
                    payload = {"status": "ok", "result": 42}
                    Path(cwd, "output.pkl").write_bytes(pickle.dumps(payload))
                return _sbatch_result("90001")
            if cmd[0] == "sacct":
                return _sacct_result(["90001|COMPLETED|0:0"])
            msg = f"Unexpected command: {cmd}"
            raise ValueError(msg)

        with (
            patch(f"{MODULE}.subprocess.run", side_effect=fake_subprocess_run),
            patch(f"{MODULE}.time.sleep"),
        ):
            future = executor.submit(lambda x: x * 2, 21)
            result = future.result(timeout=5)

        assert result == 42
        executor.shutdown(wait=False)

    def test_submit_then_resolve_cluster(
        self,
        cluster_executor: CrossClusterSlurmExecutor,
    ) -> None:
        """Same as above but with cluster mode — verifies -M flags are passed."""
        cmds_seen: list[list[str]] = []

        def fake_subprocess_run(cmd: list[str], **kwargs: Any) -> MagicMock:
            cmds_seen.append(list(cmd))
            if cmd[0] == "sbatch":
                cwd = kwargs.get("cwd")
                if cwd is not None:
                    payload = {"status": "ok", "result": 10}
                    Path(cwd, "output.pkl").write_bytes(pickle.dumps(payload))
                return _sbatch_result("90002", cluster="ionqgcp")
            if cmd[0] == "sacct":
                return _sacct_result(["90002|COMPLETED|0:0"])
            msg = f"Unexpected command: {cmd}"
            raise ValueError(msg)

        with (
            patch(f"{MODULE}.subprocess.run", side_effect=fake_subprocess_run),
            patch(f"{MODULE}.time.sleep"),
        ):
            future = cluster_executor.submit(lambda x: x + 1, 9)
            result = future.result(timeout=5)

        assert result == 10

        # Verify -M flag was used for both sbatch and sacct
        sbatch_cmds = [c for c in cmds_seen if c[0] == "sbatch"]
        sacct_cmds = [c for c in cmds_seen if c[0] == "sacct"]
        assert any("-Mionqgcp" in c for c in sbatch_cmds)
        assert any("-Mionqgcp" in c for c in sacct_cmds)

        cluster_executor.shutdown(wait=False)
