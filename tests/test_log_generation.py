"""Test log generation functions."""
import asyncio
import json
import logging

import adaptive
import pytest

from adaptive_scheduler import client_support


@pytest.mark.asyncio()
async def test_get_log_entry(learners: list[adaptive.Learner1D]) -> None:
    """Test `client_support._get_log_entry`."""
    # Prepare the runner and the learner
    learner = learners[0]
    runner = adaptive.Runner(
        learner,
        npoints_goal=1000,
        executor=adaptive.runner.SequentialExecutor(),
    )
    min_points = 100
    while learner.npoints < min_points:
        learner.loss()  # populate cache
        await asyncio.sleep(0.05)
    result = client_support._get_log_entry(runner, 0)

    # Check if the result contains the expected keys
    expected_keys = [
        "elapsed_time",
        "overhead",
        "npoints",
        "latest_loss",
        "cpu_usage",
        "mem_usage",
    ]
    assert all(key in result for key in expected_keys)


@pytest.mark.asyncio()
async def test_log_info(
    learners: list[adaptive.Learner1D],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test `client_support.log_info`."""
    # Prepare the runner and the learner
    learner = learners[0]
    runner = adaptive.Runner(learner, npoints_goal=1000)
    learner.loss()  # populates loss cache and therefore latest_loss

    caplog.set_level(logging.INFO)
    interval = 0.1
    _ = client_support.log_info(runner, interval)

    # Wait for some time to let the logging happen
    await asyncio.sleep(2)

    # Filter the captured log records based on level and logger name
    filtered_records = [
        r
        for r in caplog.records
        if r.levelno == logging.INFO and r.name == "adaptive_scheduler.client"
    ]

    # Check if the captured log records contain expected log entries
    assert len(filtered_records) > 0
    current_status_entries = 0
    for record in filtered_records:
        msg = record.getMessage()
        log_entry = json.loads(msg)
        if "current status" in log_entry["event"]:
            current_status_entries += 1
            expected_keys = [
                "elapsed_time",
                "overhead",
                "npoints",
                "latest_loss",
                "cpu_usage",
                "mem_usage",
            ]
            assert all(key in log_entry for key in expected_keys)

    # Check if there were any "current status" log entries
    assert current_status_entries > 0
