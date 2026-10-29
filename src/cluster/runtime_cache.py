"""Owned, disposable per-allocation caches; host-Python bootstrap needs only stdlib."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

MARKER = "_JEVPFN_RUNTIME_CACHE.json"


def identity():
    job = os.environ.get("SLURM_JOB_ID", "")
    cluster = os.environ.get("SLURM_CLUSTER_NAME")
    if not job.isdecimal() or cluster not in {"wice", "mindwell"}:
        raise RuntimeError("Runtime caches require a recognized Slurm allocation")
    return dict(job_id=job, cluster=cluster, uid=getattr(os, "getuid", lambda: None)())


def native_root():
    current = identity()
    node = os.environ.get("VSC_SCRATCH_NODE")
    native = os.environ.get("VSC_SCRATCH_GPFS1" if current["cluster"] == "mindwell" else "VSC_SCRATCH_LUSTRE1")
    value = node or native
    if not value or not Path(value).is_absolute():
        raise ValueError("The allocation has no explicit native scratch directory")
    root = Path(value).resolve()
    if node:
        root.mkdir(parents=True, exist_ok=True)  # Some nodes supply an uncreated per-job prefix.
    else:
        root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Native scratch is not a directory")
    return root


def create():
    current = identity()
    root = native_root()
    target = Path(tempfile.mkdtemp(prefix="jevpfn-runtime-" + current["job_id"] + "-", dir=root))
    (target / MARKER).write_text(json.dumps(dict(**current, path=str(target))), encoding="utf-8")
    return target


def cleanup(target):
    current = identity()
    target = Path(target).absolute()
    if target.is_symlink() or target.parent != native_root() or not target.name.startswith("jevpfn-runtime-" + current["job_id"] + "-"):
        raise ValueError("Runtime cleanup escaped this allocation's native scratch leaf")
    if not target.exists():
        return
    saved = json.loads((target / MARKER).read_text(encoding="utf-8"))
    if saved != dict(**current, path=str(target)):
        raise ValueError("Runtime cache belongs to another allocation or owner")
    if any(p.is_mount() for p in target.rglob("*") if p.is_dir() and not p.is_symlink()):
        raise ValueError("Runtime cache contains a mount; preserve it for review")
    shutil.rmtree(target)  # Only the verified temporary leaf; never follow links into other trees.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "cleanup"))
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args()
    if args.action == "create":
        print(create())
    else:
        if args.path is None:
            parser.error("cleanup requires the exact owned cache path")
        cleanup(args.path)
        print("Runtime cache removed: " + str(args.path))


if __name__ == "__main__":
    main()
