"""Explicit hardware routes; never assume cross-mounted storage is suitable for compute."""

import os
import re
from pathlib import PurePosixPath

from src.utils.config import load_yaml


def gpu_profile(name=None, *, runtime=None):
    """Resolve a native route; scratch workspaces are isolated per model runtime.

    This is configuration, not proof that the route is staged or GPU-compatible.
    The launcher must validate its stage receipt before loading any model.
    """
    configuration = load_yaml("experiment_0/scheduling")
    profiles = configuration["profiles"]
    name = name or configuration["default_gpu_profile"]
    if name not in profiles:
        raise ValueError("Unknown GPU scheduling profile")
    profile = dict(profiles[name])
    allowed = {"wice": {"gpu_h100"}, "mindwell": {"gpu_b200"}}
    if profile["partition"] not in allowed.get(profile["cluster"], set()):
        raise ValueError("Unsupported cluster/partition combination")
    root = profile.get("staging_root")
    if profile.get("storage_mode") == "scratch_mirror":
        scratch = os.environ.get("VSC_SCRATCH_GPFS1") or profile.get("scratch_root")
        if not scratch:
            raise ValueError(
                "VSC_SCRATCH_GPFS1 or the configured native GPFS scratch root is required"
            )
        scratch_path = PurePosixPath(scratch)
        if (
            not scratch_path.is_absolute()
            or "gpfs" not in scratch.lower()
            or ".." in scratch_path.parts
        ):
            raise ValueError("The B200 working set must use an absolute native GPFS scratch path")
        if runtime is not None and not re.fullmatch(r"[a-z][a-z0-9_]*", runtime):
            raise ValueError("Invalid runtime workspace name")
        root = str(scratch_path / "jevpfn-work" / (runtime or "shared"))
        profile["staging_root"] = root
    native = "gpfs" if profile["cluster"] == "mindwell" else "lustre"
    if (
        not root
        or not PurePosixPath(root).is_absolute()
        or native not in root.lower()
        or ".." in PurePosixPath(root).parts
    ):
        raise ValueError("Supply the verified native GPFS/Lustre project allocation first")
    if PurePosixPath(root).name == "JEVPFN":
        raise ValueError("staging_root is the allocation parent, not its JEVPFN directory")
    return profile


def reject_a100_allocation():
    """Enforce the owner's exclusion even if old scripts/CLI flags override routing."""
    partition = os.environ.get("SLURM_JOB_PARTITION", "").lower()
    cluster = os.environ.get("SLURM_CLUSTER_NAME", "wice").lower()
    if cluster == "wice" and (
        partition == "gpu"
        or partition.startswith("gpu_a100")
        or (partition == "interactive" and os.environ.get("SLURM_JOB_GPUS"))
    ):
        raise RuntimeError("A100 allocations are disabled by owner decision; use Mindwell gpu_b200")


def require_cpu_staging():
    """Allow connected CPU allocations; never download/copy weights on a GPU job."""
    partition = os.environ.get("SLURM_JOB_PARTITION", "").lower()
    # GPU ID zero is an allocated GPU, whereas a zero *count* means no GPUs.
    ids = any(os.environ.get(key) for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"))
    counts = any(
        os.environ.get(key, "").strip() not in {"", "0"}
        for key in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE", "SLURM_GPUS_PER_TASK")
    )
    if partition.startswith("gpu") or ids or counts:
        raise RuntimeError("Stage downloads on a connected CPU setup allocation before acquiring a GPU")
