"""Test for the log parsing functions."""

from __future__ import annotations

import json
import os
import tempfile

from adaptive_scheduler.server_support import (
    DatabaseManager,
    _get_infos,
    parse_log_files,
)


def test_get_infos() -> None:
    """Test for _get_infos function."""
    # Prepare sample data
    log_data = [
        {
            "event": "current status",
            "timestamp": "2023-04-13 10:30.00",
            "elapsed_time": "0 days 01:30:00",
        },
        {
            "event": "current status",
            "timestamp": "2023-04-13 11:30.00",
            "elapsed_time": "0 days 02:30:00",
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        for entry in log_data:
            f.write(json.dumps(entry) + "\n")
        f.flush()

        # Test only_last=True
        result = _get_infos(f.name, only_last=True)
        assert len(result) == 1
        assert result[0] == log_data[-1]

        # Test only_last=False
        result = _get_infos(f.name, only_last=False)
        assert len(result) == len(log_data)
        assert result == log_data[::-1]

    os.remove(f.name)  # noqa: PTH107


def test_parse_log_files(db_manager: DatabaseManager) -> None:
    """Test for parse_log_files function."""
    log_data = [
        {
            "event": "current status",
            "timestamp": "2023-04-13 10:30.00",
            "elapsed_time": "0 days 01:30:00",
        },
        {
            "event": "current status",
            "timestamp": "2023-04-13 11:30.00",
            "elapsed_time": "0 days 02:30:00",
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        for entry in log_data:
            f.write(json.dumps(entry) + "\n")
        f.flush()

        # Prepare the database manager, scheduler, and job names
        job_name = "test_job"
        db_manager.start()
        # Add an entry in the database manager
        index, _ = db_manager._choose_fname()
        db_manager._confirm_submitted(index, job_name)
        db_manager._start_request("0", f.name, job_name)
        db_manager.scheduler.start_job(job_name)

        # Test parse_log_files
        df_result = parse_log_files(
            db_manager,
            db_manager.scheduler,
            only_last=False,
        )
        # Check that the returned DataFrame has the expected columns
        expected_columns = [
            "timestamp",
            "elapsed_time",
            "job_id",
            "log_fname",
            "fname",
            "is_pending",
            "is_done",
            "output_logs",
            "start_time",
            "state",
            "job_name",
            "depends_on",
        ]
        assert sorted(df_result.columns) == sorted(expected_columns)

        print(df_result)
        # Check that the returned DataFrame has the expected number of rows
        assert len(df_result) == len(log_data)

    os.remove(f.name)  # noqa: PTH107
