import asyncio
import json
import logging

import adaptive
import pytest

from adaptive_scheduler import client_support


@pytest.mark.asyncio
async def test_get_log_entry(learners):
    # Prepare the runner and the learner
    learner = learners[0]
    runner = adaptive.Runner(
        learner, npoints_goal=1000, executor=adaptive.runner.SequentialExecutor()
    )
    while learner.npoints < 100:
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
    {
        "elapsed_time": "0:00:00.446113",
        "overhead": 0,
        "npoints": 0,
        "cpu_usage": 36.7,
        "mem_usage": 57.7,
    }
    print(result)
    assert all([key in result for key in expected_keys])


@pytest.mark.asyncio
async def test_log_info(learners, caplog):
    # Prepare the runner and the learner
    learner = learners[0]
    runner = adaptive.Runner(learner, npoints_goal=1000)
    learner.loss()  # populates loss cache and therefore latest_loss

    caplog.set_level(logging.INFO)
    interval = 0.1
    _ = client_support.log_info(runner, interval)

    # Wait for some time to let the logging happen
    await asyncio.sleep(2)

    # Check if the captured log records contain expected log entries
    assert len(caplog.records) > 0
    current_status_entries = 0
    for record in caplog.records:
        msg = record.getMessage()
        print(msg)
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
            print(log_entry)
            assert all([key in log_entry for key in expected_keys])

    # Check if there were any "current status" log entries
    assert current_status_entries > 0
