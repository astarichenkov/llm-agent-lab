"""Runs the Node-based frontend DOM regression harness.

The application frontend is vanilla JS, so these checks execute the REAL
``app/static/js/day6.js`` / ``day7.js`` in a small fake-DOM sandbox and assert
on the resulting elements. Skipped automatically when Node.js is unavailable.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
HARNESS = Path(__file__).resolve().parent / "js" / "frontend_harness.js"


@pytest.mark.skipif(NODE is None, reason="Node.js is not available")
def test_frontend_dom_harness() -> None:
    result = subprocess.run(
        [NODE, str(HARNESS)],
        capture_output=True,
        text=True,
        cwd=str(HARNESS.parents[2]),
    )
    assert result.returncode == 0, (
        "frontend harness failed:\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "ALL FRONTEND CHECKS PASSED" in result.stdout
