"""Test the widgets module."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from random import randint

from adaptive_scheduler.widgets import _bytes_to_human_readable, _total_size


def create_temp_files(num_files: int) -> list[str]:
    """Create a list of temporary files."""
    temp_files = []
    for _ in range(num_files):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("A" * randint(1, 100))  # noqa: S311
            temp_files.append(f.name)
    return temp_files


def test_total_size_single_list() -> None:
    """Test the _total_size function with a single list."""
    temp_files = create_temp_files(5)
    total_size = sum(os.path.getsize(f) for f in temp_files)
    assert _total_size(temp_files) == total_size


def test_total_size_nested_list() -> None:
    """Test the _total_size function with nested lists."""
    temp_files = create_temp_files(5)
    nested_files = [temp_files[:2], [temp_files[2]], temp_files[3:]]
    total_size = sum(os.path.getsize(f) for f in temp_files)
    assert _total_size(nested_files) == total_size


def test_total_size_mixed_types() -> None:
    """Test the _total_size function with mixed types in the list."""
    temp_files = create_temp_files(5)
    temp_paths = [Path(f) for f in temp_files[1:4]]
    mixed_files = [temp_files[0], temp_paths, temp_files[4]]
    total_size = sum(os.path.getsize(f) for f in temp_files)
    assert _total_size(mixed_files) == total_size  # type: ignore[arg-type]


def test_total_size_non_existing() -> None:
    """Test the _total_size function with non-existing files."""
    assert _total_size(["non_existing1", Path("Non_existing2")]) == 0  # type: ignore[arg-type]


def test_bytes_to_human_readable() -> None:
    """Test the _bytes_to_human_readable function."""
    size_in_bytes = 1234567890
    assert _bytes_to_human_readable(size_in_bytes) == "1.15 GB"
