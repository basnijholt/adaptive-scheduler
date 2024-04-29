"""Common scheduler code for Adaptive Scheduler."""

from __future__ import annotations

import os
import os.path
import subprocess
import time

from rich.console import Console

console = Console()


def run_submit(cmd: str, name: str | None = None) -> None:
    """Run a submit command."""
    env = os.environ.copy()
    if name is not None:
        env["NAME"] = name
    for _ in range(10):
        proc = subprocess.run(cmd.split(), env=env, capture_output=True, check=False)
        if proc.returncode == 0:
            return
        stderr = proc.stderr.decode()
        if stderr:
            console.log(f"Error: {stderr}")
        time.sleep(0.5)
