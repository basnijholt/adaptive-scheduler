"""Test the widgets module."""
from __future__ import annotations

from adaptive_scheduler.widgets import _bytes_to_human_readable


def test_bytes_to_human_readable() -> None:
    """Test the _bytes_to_human_readable function."""
    size_in_bytes = 1234567890
    assert _bytes_to_human_readable(size_in_bytes) == "1.15 GB"
