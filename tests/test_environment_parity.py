"""Local venv and pyproject pins agree with the sealed VSC CPU control lock.

A test that passes locally must not fail in `control-v1` because a shared package differs.
Interpreter patch versions are reported, not enforced: Windows has no CPython 3.12 binary
release after 3.12.10, while the Linux image uses 3.12.15. Fix drift with the README setup.
"""

from __future__ import annotations

import os
import platform
import tomllib

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from src.cluster.preflight import runtime_parity
from src.utils import paths
from src.utils.config import load_yaml

# Checked at import: the autouse routing fixture clears VSC_DATA inside each test.
ON_VSC = bool(os.environ.get("VSC_DATA"))


def _lock():
    return load_yaml("0/control_runtime")


def test_control_extra_matches_lock():
    project = tomllib.loads((paths.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = {canonicalize_name(k): v for k, v in _lock()["packages"].items()}
    mismatched = []
    for spec in project["project"]["optional-dependencies"]["control"]:
        requirement = Requirement(spec)
        pins = [s.version for s in requirement.specifier if s.operator == "=="]
        if pins != [lock.get(canonicalize_name(requirement.name))]:
            mismatched.append(f"{spec} (lock: {lock.get(canonicalize_name(requirement.name))})")
    assert not mismatched, "pyproject control extra differs from the lock: " + ", ".join(mismatched)
    assert platform.python_version_tuple()[:2] == tuple(_lock()["python"].split(".")[:2])


@pytest.mark.skipif(ON_VSC, reason="VSC images are checked by preflight --runtime-parity")
def test_local_venv_matches_control_lock():
    parity = runtime_parity()
    diff = [
        f"  {name}: installed {parity['installed'][name] or 'missing'}, lock {expected}"
        for name, expected in parity["expected"].items()
        if parity["installed"][name] != expected
    ]
    assert not diff, (
        "Local environment differs from config/general/0/control_runtime.yaml:\n"
        + "\n".join(diff)
        + "\nSync it with the one-line command in README.md (Local environment)."
    )
