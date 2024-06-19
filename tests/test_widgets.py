"""Test the widgets module."""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import timedelta
from pathlib import Path
from random import randint
from typing import TYPE_CHECKING
from unittest.mock import patch

import pandas as pd

from adaptive_scheduler.widgets import (
    _bytes_to_human_readable,
    _failed_job_logs,
    _files_that_contain,
    _get_fnames,
    _interp_red_green,
    _sort_fnames,
    _timedelta_to_human_readable,
    _total_size,
    info,
    log_explorer,
    queue_widget,
    results_widget,
)

if TYPE_CHECKING:
    from typing import Any

    import pytest


def create_temp_files(num_files: int) -> list[str]:
    """Create a list of temporary files."""
    temp_files = []
    for _ in range(num_files):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("A" * randint(1, 100))  # noqa: S311
            temp_files.append(f.name)
    return temp_files


def _sum_size(files: list[str]) -> int:
    return sum(os.path.getsize(f) for f in files)  # noqa: PTH202


def test_total_size_single_list() -> None:
    """Test the _total_size function with a single list."""
    temp_files = create_temp_files(5)
    total_size = _sum_size(temp_files)
    assert _total_size(temp_files) == total_size


def test_total_size_nested_list() -> None:
    """Test the _total_size function with nested lists."""
    temp_files = create_temp_files(5)
    nested_files = [temp_files[:2], [temp_files[2]], temp_files[3:]]
    total_size = _sum_size(temp_files)
    assert _total_size(nested_files) == total_size


def test_total_size_mixed_types() -> None:
    """Test the _total_size function with mixed types in the list."""
    temp_files = create_temp_files(5)
    temp_paths = [Path(f) for f in temp_files[1:4]]
    mixed_files = [temp_files[0], temp_paths, temp_files[4]]
    total_size = _sum_size(temp_files)
    assert _total_size(mixed_files) == total_size  # type: ignore[arg-type]


def test_total_size_non_existing() -> None:
    """Test the _total_size function with non-existing files."""
    assert _total_size(["non_existing1", Path("Non_existing2")]) == 0  # type: ignore[arg-type]


def test_bytes_to_human_readable() -> None:
    """Test the _bytes_to_human_readable function."""
    size_in_bytes = 1234567890
    assert _bytes_to_human_readable(size_in_bytes) == "1.15 GB"


class MockDatabaseManager:
    """Mock DatabaseManager for testing."""

    def __init__(self, tmp_folder: Path) -> None:
        """Initialize the MockDatabaseManager."""
        self.failed: list[dict[str, Any]] = []
        self._total_learner_size = None
        self.fnames: list[str | Path] = []

        self.tmp_folder = tmp_folder

    def as_dicts(self) -> list[dict[str, Any]]:
        """Return a list of dictionaries representing the database entries."""
        return [
            {
                "fname": f"data/{i}.pickle",
                "job_id": f"100{i}",
                "is_done": False,
                "log_fname": f"{self.tmp_folder}/adaptive-{i}-100{i}.log",
                "job_name": f"adaptive-{i}",
                "output_logs": [f"{self.tmp_folder}/adaptive-{i}-100{i}.out"],
                "start_time": "2023-04-18T23:47:45.167896+00:00",
            }
            for i in range(3)
        ]

    def update(self, queue: dict) -> None:
        """Update the database with the given queue."""


class MockRunManager:
    """Mock RunManager for testing."""

    def __init__(self, tmp_folder: Path) -> None:
        """Initialize the MockRunManager."""
        self.job_name = "adaptive"
        self.scheduler = self
        self.log_folder = tmp_folder
        self.ext = ".log"
        self.move_old_logs_to = None
        self.max_log_lines = 500
        self._folder = tmp_folder
        self.database_manager = MockDatabaseManager(tmp_folder)
        self.job_names = ["adaptive-{i}" for i in range(3)]

    def status(self) -> str:
        """Return the status of the RunManager."""
        return "running"

    def elapsed_time(self) -> float:
        """Return the elapsed time of the RunManager."""
        return 123.0

    def job_starting_times(self) -> list[float]:
        """Return the starting times of the jobs."""
        return [float(i) for i in range(30)]

    def queue(
        self,
        me_only: bool = True,  # noqa: FBT001, ARG002, FBT002
    ) -> dict[str, dict[str, str]]:
        """Return a dictionary representing the current queue."""
        return {
            f"{i}": {
                "state": "RUNNING",
                "NumNodes": "1",
                "NumTasks": "1",
                "ReasonList": "barb8-nc24-high-1",
                "SubmitTime": "2023-04-16T22:09:53",
                "StartTime": "2023-04-16T22:09:54",
                "UserName": "basnijholt",
                "Partition": "nc24-high",
                "job_name": f"adaptive-{i}",
            }
            for i in range(3)
        }


def test_files_that_contain2(tmp_path: Path) -> None:
    """Test the _files_that_contain function."""
    d = tmp_path / "logs"
    d.mkdir()
    p1 = d / "test1.log"
    p1.write_text("Hello, world!")
    p2 = d / "test2.log"
    p2.write_text("Hello, everyone!")
    fnames = [p1, p2]
    result = _files_that_contain(fnames, "world")
    assert result == [p1]


def test_get_fnames(tmp_path: Path) -> None:
    """Test the _get_fnames function."""
    d = tmp_path / "logs"
    d.mkdir()
    p1 = d / "test_job-1.log"
    p1.write_text("Log 1")
    p2 = d / "test_job-2.log"
    p2.write_text("Log 2")

    run_manager = MockRunManager(tmp_path)

    fnames = _get_fnames(run_manager, only_running=False)  # type: ignore[arg-type]
    assert len(fnames) == 2

    fnames = _get_fnames(run_manager, only_running=True)  # type: ignore[arg-type]
    assert len(fnames) == 6


def test_failed_job_logs(tmp_path: Path) -> None:
    """Test the _failed_job_logs function."""
    d = tmp_path / "logs"
    d.mkdir()
    p1 = d / "test_job-1.log"
    p1.write_text("Log 1")
    p2 = d / "test_job-2.log"
    p2.write_text("Log 2")

    run_manager = MockRunManager(tmp_path)

    fnames = [p1, p2]
    failed_fnames = _failed_job_logs(fnames, run_manager, only_running=False)  # type: ignore[arg-type]
    assert failed_fnames == []


def test_timedelta_to_human_readable_int() -> None:
    """Test the _timedelta_to_human_readable function with an integer."""
    seconds = 3666
    assert _timedelta_to_human_readable(seconds) == "1 h, 1 m, 6 s"
    assert (
        _timedelta_to_human_readable(seconds, short_format=False) == "1 hour, 1 minute, 6 seconds"
    )


def test_log_explorer(tmp_path: Path) -> None:
    """Test the log_explorer function."""
    run_manager = MockRunManager(tmp_path)

    async def mock_get_running_loop():  # noqa: ANN202
        return asyncio.get_event_loop()

    with patch("asyncio.get_running_loop", side_effect=mock_get_running_loop):
        widget = log_explorer(run_manager)  # type: ignore[arg-type]

    assert widget is not None


def test_queue_widget(tmp_path: Path) -> None:
    """Test the queue_widget function."""
    run_manager = MockRunManager(tmp_path)
    widget = queue_widget(run_manager)  # type: ignore[arg-type]
    assert widget is not None


def test_info(capfd: pytest.CaptureFixture, tmp_path: Path) -> None:
    """Test the info function."""
    run_manager = MockRunManager(tmp_path)
    info(run_manager)  # type: ignore[arg-type]
    captured = capfd.readouterr()
    assert "status" in captured.out


def test_timedelta_to_human_readable_short_format() -> None:
    """Test the _timedelta_to_human_readable function with short_format."""
    time_input = timedelta(days=2, hours=3, minutes=4, seconds=5)
    assert _timedelta_to_human_readable(time_input, short_format=True) == "2 d, 3 h, 4 m, 5 s"


def test_files_that_contain(tmp_path: Path) -> None:
    """Test the _files_that_contain function."""
    file1 = tmp_path / "file1.txt"
    file1.write_text("This file contains the keyword: example")

    file2 = tmp_path / "file2.txt"
    file2.write_text("This file does not contain the keyword.")

    fnames = [file1, file2]
    filtered_fnames = _files_that_contain(fnames, "example")
    assert len(filtered_fnames) == 1
    assert filtered_fnames[0] == file1


def test_get_fnames_only_running(tmp_path: Path) -> None:
    """Test the _get_fnames function with only_running=True."""
    run_manager = MockRunManager(tmp_path)
    fnames = _get_fnames(run_manager, only_running=True)  # type: ignore[arg-type]
    assert len(fnames) == 6


def test_get_fnames_only_running_false(tmp_path: Path) -> None:
    """Test the _get_fnames function with only_running=False."""
    run_manager = MockRunManager(tmp_path)
    log_fname = run_manager.database_manager.as_dicts()[0]["log_fname"]
    with open(log_fname, "w") as f:  # noqa: PTH123
        f.write("This is a test log file.")
    fnames = _get_fnames(run_manager, only_running=False)  # type: ignore[arg-type]
    assert len(fnames) == 3
    # TODO: it is picking up the database file and scheduler file
    # we should probably filter those out?


def test_sort_fnames(tmp_path: Path) -> None:
    """Test the _sort_fnames function."""
    run_manager = MockRunManager(tmp_path)
    fnames = [Path(f"logs/{run_manager.job_name}-{i}-{i * 100}.log") for i in range(3)]

    sorted_fnames = _sort_fnames("Alphabetical", run_manager, fnames)  # type: ignore[arg-type]
    assert sorted_fnames == fnames  # In this case, they should be the same


def test_results_widget(tmp_path: Path) -> None:
    """Test the results_widget function."""
    # Create some sample data files in DataFrame pickle format
    sample_data = [
        pd.DataFrame({"col1": range(10), "col2": range(10)}),
        pd.DataFrame({"col1": range(10, 20), "col2": range(10, 20)}),
    ]

    data_files = []
    for idx, data in enumerate(sample_data):
        file_path = tmp_path / f"sample_data_{idx}.pickle"
        data.to_pickle(file_path)
        data_files.append(file_path)

    # Create the widget
    widget = results_widget(data_files, "pickle")

    # Test the widget's configuration
    assert widget is not None


def test_interp_red_green() -> None:
    """Test _interp_red_green."""
    assert _interp_red_green(100, pct_red=10, pct_green=90) == (0, 255, 0)
    assert _interp_red_green(0, pct_red=10, pct_green=90) == (255, 0, 0)
    assert _interp_red_green(50, pct_red=10, pct_green=90) == (127, 127, 0)
    assert _interp_red_green(50, pct_red=90, pct_green=10) == (127, 127, 0)
    assert _interp_red_green(100, pct_green=10, pct_red=90) == (255, 0, 0)
    assert _interp_red_green(0, pct_green=10, pct_red=90) == (0, 255, 0)
