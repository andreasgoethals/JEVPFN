"""Explicit hardware routes; never assume cross-mounted storage is suitable for compute."""

from pathlib import PurePosixPath

from src.utils.config import load_yaml


def gpu_profile(name):
    profiles = load_yaml("experiment_0/scheduling")["profiles"]
    if name not in profiles:
        raise ValueError("Unknown GPU scheduling profile")
    profile = dict(profiles[name])
    allowed = {"wice": {"gpu_a100", "gpu_h100"}, "mindwell": {"gpu_b200"}}
    if profile["partition"] not in allowed.get(profile["cluster"], set()):
        raise ValueError("Unsupported cluster/partition combination")
    root = profile.get("staging_root")
    native = "gpfs" if profile["cluster"] == "mindwell" else "lustre"
    if not root or not PurePosixPath(root).is_absolute() or native not in root.lower():
        raise ValueError("Supply the verified native GPFS/Lustre project allocation first")
    if PurePosixPath(root).name == "JEVPFN":
        raise ValueError("staging_root is the allocation parent, not its JEVPFN directory")
    return profile
