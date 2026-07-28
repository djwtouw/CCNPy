"""Execute every script in examples/ and assert it runs without error.

This makes the example scripts double as smoke tests during development: if a
change breaks an example, this test fails.
"""

import runpy
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
EXAMPLE_SCRIPTS = sorted(EXAMPLES_DIR.glob("example_*.py"))


@pytest.mark.parametrize(
    "script", EXAMPLE_SCRIPTS, ids=[p.name for p in EXAMPLE_SCRIPTS])
def test_example_runs(script):
    # Runs the script as __main__ in a fresh namespace; any exception fails.
    runpy.run_path(str(script), run_name="__main__")
