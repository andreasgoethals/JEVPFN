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
def _isolate_routing_environment(monkeypatch):
    """Synthetic tests choose their own tiers; never inherit a real job's output roots.

    The Exp0 parent verifies actual host paths before invoking pytest. Unit tests
    must not mistake its experiment phase for exploration or write into real VSC
    allocations. Cluster-path tests explicitly set their own temporary variables.
    """
    for variable in (
        "JEVPFN_PHASE",
        "JEVPFN_STAGING_ROOT",
        "JEVPFN_PROJECT_ROOT",
        "VSC_PROJECT_ROOT",
        "JEVPFN_DURABLE_ROOT",
        "JEVPFN_STORAGE_MODE",
        "JEVPFN_SCRATCH_RUNTIME",
        "VSC_STAGING_ROOT",
        "JEVPFN_LOCAL_STORAGE_ROOT",
        "JEVPFN_NOTEBOOK_SCRATCH",
        "VSC_DATA",
        "VSC_SCRATCH",
        "VSC_SCRATCH_NODE",
        "VSC_SCRATCH_GPFS1",
        "VSC_PROJECT_LUSTRE1",
        "VSC_PROJECT_GPFS1",
        "SLURM_JOB_ID",
        "SLURM_CLUSTER_NAME",
        "SLURM_JOB_GPUS",
        "SLURM_JOB_PARTITION",
        "JEVPFN_CONTAINER_PROVIDER",
        "JEVPFN_CONTROL_TEST",
        "JEVPFN_CONTROL_LEGACY",
        "JEVPFN_CONTROL_IMAGE",
        "JEVPFN_CONTAINER_ID",
        "JEVPFN_RUNTIME_PROFILE",
        "JEVPFN_MONITOR_PYTHON",
        "JEVPFN_NVIDIA_SMI",
        "TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture(autouse=True)
def _isolate_local_credentials(tmp_path, monkeypatch):
    """Never let synthetic tests read the owner's real, ignored credential file."""
    from src.jev import auth

    monkeypatch.setattr(auth, "LOCAL_CREDENTIAL_PATH", tmp_path / "absent_credentials.json")


@pytest.fixture(autouse=True)
def _headless_matplotlib() -> None:
    """Force the Agg backend. A compute node has no display, and the default backend either
    fails there or opens a window that blocks the run."""
    import matplotlib

    matplotlib.use("Agg", force=True)


@pytest.fixture
def isolated_output(tmp_path, monkeypatch, _isolate_routing_environment):
    """Point every resolved path at `tmp_path` for the duration of one test.

    Setting `VSC_DATA` and the staging override is how the cluster's own two-tier layout is
    exercised off-cluster: `paths.on_vsc()` becomes true, so the test covers the branch that
    only ever runs in production, which is otherwise the branch nobody tests.
    """
    from src.utils import paths

    monkeypatch.setenv("VSC_DATA", str(tmp_path / "vsc_data"))
    monkeypatch.setenv(paths.STAGING_ENV_VARS[0], str(tmp_path / "staging"))
    monkeypatch.setenv("JEVPFN_LOCAL_STORAGE_ROOT", str(tmp_path / "local"))
    return tmp_path
