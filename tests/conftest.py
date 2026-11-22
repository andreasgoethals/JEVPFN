"""Shared pytest setup.

Two things, both so the suite behaves the same on a laptop, in CI and on a compute node.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Real trees no test may create, modify or delete a file in (checked over the whole session).
GUARDED_TREES = ("output_JEVPFN", "data")

# `from src...` without an editable install, so the suite runs on a fresh clone. A suite that
# half-works depending on whether someone ran `pip install -e .` is a suite people stop trusting.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _isolate_routing_environment(tmp_path, monkeypatch):
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
        "JEVPFN_SHARED_IMMUTABLES",
        "JEVPFN_STORAGE_CONFIG",
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
    monkeypatch.setenv("JEVPFN_STORAGE_CONFIG", str(tmp_path / "absent_storage.json"))


@pytest.fixture(autouse=True)
def _isolate_repository_outputs(request, tmp_path, monkeypatch, _isolate_routing_environment):
    """Off-cluster every tier collapses into the repository; tests collapse into `tmp_path`.

    Without this, any test that resolves `outputs_dir()`, `logs_dir()` or a notebook folder
    without setting its own roots writes build logs and stage manifests into the real
    `output_JEVPFN/`. Explicit `VSC_DATA`/`VSC_SCRATCH`/staging variables still take effect.
    Tests marked `real_repository_roots` only resolve (never write) the real local roots.
    """
    from src.utils import paths

    if request.node.get_closest_marker("real_repository_roots"):
        return None

    local = tmp_path / "repo_outputs" / paths.PROJECT_NAME
    monkeypatch.setattr(paths, "data_root", lambda: paths._env_path("VSC_DATA") or local)
    monkeypatch.setattr(paths, "scratch_root", lambda: paths._env_path("VSC_SCRATCH") or local)
    monkeypatch.setenv("JEVPFN_LOCAL_STORAGE_ROOT", str(local))
    monkeypatch.setenv("JEVPFN_NOTEBOOK_SCRATCH", str(tmp_path / "notebook_scratch"))
    return local


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
    # Tests may switch this fixture from VSC to local mode. Keep their compact
    # manifests local too, rather than reading the owner's real paid-call ledger.
    vsc_outputs = paths.outputs_dir
    monkeypatch.setattr(paths, "outputs_dir", lambda: vsc_outputs() if paths.on_vsc() else paths.notebook_outputs_dir())
    return tmp_path


# ---------------------------------------------------------------------------
# Session guard: the suite must leave the real output and data trees untouched.
# ---------------------------------------------------------------------------


def _tree_state() -> dict[str, tuple[int, int]]:
    state = {}
    for name in GUARDED_TREES:
        for directory, _, files in os.walk(REPO_ROOT / name):
            for file in files:
                path = Path(directory, file)
                try:
                    info = path.stat()
                except OSError:
                    continue
                state[path.relative_to(REPO_ROOT).as_posix()] = (info.st_mtime_ns, info.st_size)
    return state


def _local_interpreter_problems() -> list[str]:
    """Where the running interpreter violates pyproject's `requires-python`/`dependencies`.

    Local only (VSC images are checked by their own runtime contracts). A global Python with
    e.g. pandas 3 silently changes missing-text dtypes (None -> NaN) and fails dozens of
    unrelated synthetic contracts; one clear message names the project venv instead.
    """
    import importlib.metadata
    import platform
    import tomllib

    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet

    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    problems = []
    if platform.python_version() not in SpecifierSet(project["requires-python"]):
        problems.append(f"Python {platform.python_version()} (requires {project['requires-python']})")
    for dependency in project["dependencies"]:
        requirement = Requirement(dependency)
        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            continue  # A missing package fails loudly at import time.
        if version not in requirement.specifier:
            problems.append(f"{requirement.name} {version} (requires {requirement.specifier})")
    return problems


def pytest_sessionstart(session):
    if not os.environ.get("VSC_DATA") and (problems := _local_interpreter_problems()):
        raise pytest.UsageError(
            f"{sys.executable} does not satisfy pyproject.toml: {'; '.join(problems)}. "
            "Run the suite with the project venv: .venv\\Scripts\\python.exe -m pytest -q")
    if os.environ.get("VSC_DATA"):
        # On VSC the real output tree is shared with concurrent production jobs (token
        # shards, setup receipts, controller logs), so a before/after snapshot cannot tell
        # their writes from a test's. Tests stay redirected by the autouse fixture; the
        # snapshot guard is the local safety net only (CPU preflight 11839220, 21-11-2026).
        return
    session.config._jevpfn_tree_state = _tree_state()


def pytest_sessionfinish(session, exitstatus):
    before = getattr(session.config, "_jevpfn_tree_state", None)
    if before is None:
        return
    after = _tree_state()
    changed = sorted(
        [f"created  {p}" for p in after.keys() - before.keys()]
        + [f"deleted  {p}" for p in before.keys() - after.keys()]
        + [f"modified {p}" for p in before.keys() & after.keys() if before[p] != after[p]]
    )
    session.config._jevpfn_tree_changes = changed
    if changed:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    changed = getattr(config, "_jevpfn_tree_changes", None)
    if changed is None:
        return
    if changed:
        terminalreporter.write_sep("=", "REAL TREE WRITES DETECTED", red=True)
        for line in changed:
            terminalreporter.write_line(line)
    else:
        terminalreporter.write_line(
            f"Write guard: no file under {', '.join(t + '/' for t in GUARDED_TREES)} was created, modified or deleted.")


@pytest.fixture
def historical_workspace_limits(monkeypatch, request):
    """Admission scenarios written for the 2-workspace / 200 GiB guard (raised to 8 / 400 on
    20-11-2026). They test the guard logic at those fixed limits, not today's config values."""
    import sys

    import src.utils.config as config

    original = config.load_yaml

    def load(name, *args, **kwargs):
        value = original(name, *args, **kwargs)
        if name == "experiment_0/scheduling":
            value = {**value, "max_resident_workspaces": 2, "scratch_working_set_gib": 200}
        return value

    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith(("src.", "test_")) and getattr(module, "load_yaml", None) is original:
            monkeypatch.setattr(module, "load_yaml", load)
