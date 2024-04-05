"""Test the file creation progress tracking utilities."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from rich.progress import Progress, TaskID

from adaptive_scheduler.utils import (
    _remove_completed_paths,
    _track_file_creation_progress,
    _update_progress_for_paths,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_update_progress_for_paths(tmp_path: Path) -> None:
    """Test the update progress for paths function."""
    # Create test files
    create_test_files(tmp_path, ["file1", "file3", "file4"])

    paths_dict: dict[str, set[Path | tuple[Path, ...]]] = {
        "category1": {tmp_path / "file1", tmp_path / "file2"},
        "category2": {(tmp_path / "file3", tmp_path / "file4")},
    }

    progress_mock = Mock(spec=Progress)
    task_id_mock = Mock(spec=TaskID)
    task_ids = {"category1": task_id_mock, "category2": task_id_mock}

    processed = _update_progress_for_paths(
        paths_dict,
        progress_mock,
        task_id_mock,
        task_ids,
    )

    assert processed == 2  # Only "file1" and the tuple ("file3", "file4") exist
    assert len(paths_dict["category1"]) == 1  # "file2" does not exist and should remain
    assert len(paths_dict["category2"]) == 0  # Tuple paths exist and should be removed
    progress_mock.update.assert_called()


def create_test_files(tmp_path: Path, file_names: list[str]) -> None:
    """Create test files in the given directory."""
    for name in file_names:
        (tmp_path / name).touch()


def test_remove_completed_paths(tmp_path: Path) -> None:
    """Test the remove completed paths function."""
    # Create test files
    existing_files = ["file1", "file3", "file4"]
    create_test_files(tmp_path, existing_files)

    paths_dict: dict[str, set[Path | tuple[Path, ...]]] = {
        "category1": {tmp_path / "file1", tmp_path / "file2"},
        "category2": {(tmp_path / "file3", tmp_path / "file4")},
    }

    n_completed = _remove_completed_paths(paths_dict)

    assert n_completed == {"category1": 1, "category2": 1}
    assert paths_dict == {"category1": {tmp_path / "file2"}, "category2": set()}


@pytest.mark.asyncio()
async def test_track_file_creation_progress(tmp_path: Path) -> None:
    """Test the track file creation progress function."""
    # Create test files
    create_test_files(tmp_path, ["file1"])

    paths_dict: dict[str, set[Path | tuple[Path, ...]]] = {
        "category1": {tmp_path / "file1", tmp_path / "file2"},
        "category2": {(tmp_path / "file3", tmp_path / "file4")},
    }

    progress = Progress(auto_refresh=False)
    task = asyncio.create_task(
        _track_file_creation_progress(paths_dict, progress, interval=1e-3),
    )

    # Allow some time for the task to process
    await asyncio.sleep(0.1)

    progress.stop()
    assert "Total" in progress._tasks[0].description
    assert progress._tasks[0].total == 3
    assert progress._tasks[0].completed == 1

    assert "category1" in progress._tasks[1].description
    assert progress._tasks[1].total == 2
    assert progress._tasks[1].completed == 1

    assert "category2" in progress._tasks[2].description
    assert progress._tasks[2].total == 1
    assert progress._tasks[2].completed == 0

    # Create one of the files of category2, should still not be completed
    create_test_files(tmp_path, ["file3"])
    await asyncio.sleep(0.1)

    assert "category2" in progress._tasks[2].description
    assert progress._tasks[2].total == 1
    assert progress._tasks[2].completed == 0
    assert paths_dict["category2"] == {(tmp_path / "file4",)}

    # Create the other file of category2, should now be completed
    create_test_files(tmp_path, ["file4"])
    await asyncio.sleep(0.05)
    assert "category2" in progress._tasks[2].description
    assert progress._tasks[2].total == 1
    assert progress._tasks[2].completed == 1
    assert paths_dict["category2"] == set()

    # Create the other file of category1, should now be completed
    create_test_files(tmp_path, ["file2"])
    await asyncio.sleep(0.05)
    assert "category1" in progress._tasks[1].description
    assert progress._tasks[1].total == 2
    assert progress._tasks[1].completed == 2

    # Check the total progress
    assert "Total" in progress._tasks[0].description
    assert progress._tasks[0].total == 3
    assert progress._tasks[0].completed == 3

    # Stop the progress and the task
    progress.stop()
    task.cancel()
