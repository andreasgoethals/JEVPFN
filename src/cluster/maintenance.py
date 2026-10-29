"""Shared safety checks for explicit, quiescent VSC file maintenance."""

import os
import subprocess
from pathlib import Path

from src.cluster.preflight import storage_policy_failures
from src.utils import paths


def require_idle_vsc(*, allow_current=False):
    if not paths.on_vsc() or (os.environ.get("SLURM_JOB_ID") and not allow_current):
        raise RuntimeError("Run maintenance in the VSC login shell, outside a Slurm job")
    failures = storage_policy_failures("wice")
    if failures:
        raise ValueError("; ".join(failures))
    # Check both clusters; unrelated jobs such as CreditPFN do not block maintenance.
    listing = subprocess.check_output(
        ["squeue", "--me", "--clusters=wice,mindwell", "--noheader", "--format=%i|%j|%Z"],
        text=True,
        timeout=30,
    )
    checkout = paths.REPO_ROOT.resolve()
    active = []
    cluster = None
    for line in listing.splitlines():
        if line.startswith("CLUSTER:"):
            cluster = line.partition(":")[2].strip()
            continue
        if not line.strip():
            continue
        parts = line.strip().split("|", 2)
        if len(parts) != 3:
            raise ValueError("Cannot interpret the scheduler listing; maintenance refused")
        job, name, directory = (p.strip() for p in parts)
        if allow_current and job == os.environ.get("SLURM_JOB_ID") and cluster == os.environ.get("SLURM_CLUSTER_NAME"):
            continue  # Explicit single maintenance allocation; all other consumers still block it.
        in_checkout = directory.startswith("/") and Path(directory).resolve().is_relative_to(
            checkout
        )
        if name.lower().startswith("jevpfn") or in_checkout:
            active.append(job)
    if active:
        raise RuntimeError(
            "Wait for queued/running JEVPFN jobs before maintenance: " + ", ".join(active)
        )


def reject_links(path):
    """Refuse symlinks/junctions on existing path components before moving or replacing."""
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or getattr(part, "is_junction", lambda: False)():
            raise ValueError(f"Linked maintenance path is not allowed: {part}")
    return path
