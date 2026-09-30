"""Lightweight fingerprints for uncommitted research code and its execution environment."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess

from src.utils import paths
from src.utils.serialization import digest, file_sha256


def code_identity() -> str:
    files = sorted((paths.REPO_ROOT / "src").rglob("*.py"))
    return digest({p.relative_to(paths.REPO_ROOT).as_posix(): file_sha256(p) for p in files})


def environment() -> dict:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "PyYAML", "matplotlib", "pyarrow"):
        versions[package] = importlib.metadata.version(package)
    return versions


def provenance() -> dict:
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=paths.REPO_ROOT, text=True, capture_output=True, check=False
        ).stdout.strip()

    return {
        "code_sha256": code_identity(),
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "tfm_library_pin": git("submodule", "status"),
        "environment": environment(),
    }
