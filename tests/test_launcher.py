"""Test `adaptive-scheduler-launcher`."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# Import the functions you want to test from launcher.py
from adaptive_scheduler._server_support import launcher


# Test the _parse_args function
def test_parse_args() -> None:
    """Test that the _parse_args function returns the correct args."""
    test_args = [
        "launcher.py",
        "--n",
        "4",
        "--log-fname",
        "test.log",
        "--job-id",
        "123",
        "--name",
        "test",
        "--url",
        "http://example.com",
    ]

    # Store the original sys.argv
    original_argv = sys.argv

    # Replace sys.argv with the test arguments
    sys.argv = test_args

    # Call _parse_args and check the results
    args = launcher._parse_args()
    assert args.n == 4
    assert args.name == "test"

    # Restore the original sys.argv
    sys.argv = original_argv


# Test the _get_executor function
def test_get_executor() -> None:
    """Test that the _get_executor function returns the correct executor."""
    with patch("concurrent.futures.ProcessPoolExecutor") as mock_executor:
        mock_executor.return_value = MagicMock()
        executor = launcher._get_executor("process-pool", None, 4, None)
        assert executor is not None
        mock_executor.assert_called_with(max_workers=4)

    with pytest.raises(ValueError, match="Unknown executor_type"):
        launcher._get_executor("unknown", None, 4, "loky")  # type: ignore[arg-type]

    with patch("adaptive.runner.SequentialExecutor") as mock_executor:
        executor = launcher._get_executor("sequential", None, 4, None)
        assert executor is not None
        mock_executor.assert_called_once()
