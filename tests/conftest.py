"""Shared pytest setup.

Two things, both so the suite behaves the same on a laptop, in CI and on a compute node.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# `from src...` without an editable install, so the suite runs on a fresh clone. A suite that
# half-works depending on whether someone ran `pip install -e .` is a suite people stop trusting.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _headless_matplotlib() -> None:
    """Force the Agg backend. A compute node has no display, and the default backend either
    fails there or opens a window that blocks the run."""
    import matplotlib

    matplotlib.use("Agg", force=True)


@pytest.fixture
def isolated_output(tmp_path, monkeypatch):
    """Point every resolved path at `tmp_path` for the duration of one test.

    Setting `VSC_DATA` and the staging override is how the cluster's own two-tier layout is
    exercised off-cluster: `paths.on_vsc()` becomes true, so the test covers the branch that
    only ever runs in production, which is otherwise the branch nobody tests.
    """
    from src.utils import paths

    monkeypatch.setenv("VSC_DATA", str(tmp_path / "vsc_data"))
    monkeypatch.setenv(paths.STAGING_ENV_VARS[0], str(tmp_path / "staging"))
    return tmp_path
