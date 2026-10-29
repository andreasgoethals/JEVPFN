"""Bound GPFS working copies; publish results before releasing disposable inputs."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
from contextlib import contextmanager, suppress
from pathlib import Path

from src.cluster.maintenance import reject_links
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import write_json

GIB = 1024**3


def tree_bytes(root):
    """Conservative logical bytes, without following links into other allocations."""
    total = 0
    if root.exists():
        for directory, folders, files in os.walk(root, followlinks=False):
            base = Path(directory)
            folders[:] = [
                n
                for n in folders
                if not (base / n).is_symlink()
                and not getattr(base / n, "is_junction", lambda: False)()
            ]
            for name in files:
                p = base / name
                if not p.is_symlink():
                    # Other projects may rotate temporary files during the scan.
                    with suppress(FileNotFoundError):
                        total += p.stat().st_size
    return total


@contextmanager
def capacity_lease(work):
    parent = reject_links(work.parent.parent)
    parent.mkdir(parents=True, exist_ok=True)
    with file_lock(parent / ".capacity.lock", timeout=600):
        yield


def disposable_paths(work):
    """Only copied inputs/images: never response queues, embeddings or predictions."""
    relatives = ["data/raw", "data/decision_inputs", "checkpoints", "containers"]
    relatives += [f"output_JEVPFN/{paths.phase_folder('experiment_0')}/results/token_audit"]
    return [reject_links(work / name) for name in relatives if (work / name).exists()]


def check_capacity(work, candidates, source_container):
    """Check application and personal scratch budgets before the first bulk copy.

    Other projects may grow after this check; reserve headroom, never claim a quota
    reservation. Filesystem-wide statvfs free space alone is not a personal quota.
    """
    settings = load_yaml("experiment_0/scheduling")
    for key in (
        "max_resident_workspaces",
        "scratch_working_set_gib",
        "scratch_allocation_gib",
        "scratch_headroom_gib",
    ):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    root = work.parent.parent
    existing = [p / "JEVPFN" for p in root.iterdir() if p.is_dir()]
    resident = [p for p in existing if p != work and disposable_paths(p)]
    if len(resident) >= settings["max_resident_workspaces"]:
        raise OSError("GPFS working-set slots are full; publish/release idle JEVPFN copies first")
    sources = [(source, work / relative) for relative, source in candidates.items()]
    container_relative = source_container.relative_to(source_container.parents[2])
    # Container paths always live under JEVPFN/containers/<profile>/<runtime>.
    sources += [
        (p, work / container_relative / p.relative_to(source_container))
        for p in source_container.rglob("*")
        if p.is_file() and not p.is_symlink()
    ]
    growth = sum(
        max(0, source.stat().st_size - (dest.stat().st_size if dest.is_file() else 0))
        for source, dest in sources
    )
    largest = max((source.stat().st_size for source, _ in sources), default=0)
    extra = growth + 2 * largest + settings["scratch_headroom_gib"] * GIB
    project_bytes = tree_bytes(root)
    personal_bytes = tree_bytes(root.parent)
    if project_bytes + extra > settings["scratch_working_set_gib"] * GIB:
        raise OSError(
            "JEVPFN GPFS byte budget exceeded; publish/release idle copies before staging"
        )
    if personal_bytes + extra > settings["scratch_allocation_gib"] * GIB:
        raise OSError(
            "Personal GPFS scratch budget/headroom exhausted; check myquota and other projects"
        )
    if shutil.disk_usage(work).free < extra:
        raise OSError("GPFS filesystem has insufficient free capacity for staging")


def require_runtime_idle(runtime):
    """Protect queued consumers too: file leases alone only protect running workers."""
    if not paths.on_vsc():
        return
    from src.cluster.containers import RUNTIMES

    for cluster in ("wice", "mindwell"):
        listing = subprocess.check_output(
            ["squeue", "--clusters=" + cluster, "--me", "--noheader", "--format=%i|%j|%Z"],
            text=True,
            timeout=30,
        )
        for line in listing.splitlines():
            if not line.strip() or line.startswith("CLUSTER:"):
                continue
            fields = line.strip().split("|", 2)
            if len(fields) != 3:
                raise ValueError("Cannot verify queued scratch consumers")
            job, name, directory = fields
            if job == os.environ.get("SLURM_JOB_ID") and cluster == os.environ.get(
                "SLURM_CLUSTER_NAME"
            ):
                continue
            if not name.startswith("jevpfn") and "JEVPFN" not in directory:
                continue
            detail = subprocess.check_output(
                ["scontrol", "--clusters=" + cluster, "show", "job", job, "-o"],
                text=True,
                timeout=30,
            )
            tokens = shlex.split(detail)
            # These exact project entrypoints never consume a GPFS working set.
            # Unrelated CPU preparation can continue while a finished runtime releases.
            cpu_only = {
                "experiment_0_tokens.slurm",
                "experiment_0_cpu.slurm",
                "experiment_0_stage.slurm",
                "experiment_0_container_build.slurm",
                "experiment_0_embedding_stage.slurm",
                "stage_tabstar.slurm",
                "experiment_0_storage.slurm",
            }
            command = next(
                (t.removeprefix("Command=") for t in tokens if t.startswith("Command=")), ""
            )
            if Path(command).name in cpu_only and "/JEVPFN/scripts/slurm/" in command:
                continue
            owners = {r for r in RUNTIMES if r in tokens or f"/jevpfn-work/{r}/" in detail}
            # Unknown JEVPFN consumers fail closed, including manually queued scripts
            # whose omitted arguments select a default runtime.
            if runtime in owners or not owners:
                raise RuntimeError(f"Scratch release refused: JEVPFN job {job} may use {runtime}")


def release(runtime, *, apply=False):
    from src.cluster.scratch_stage import MARKER, _publish, roots, workspace_lease

    durable, work = roots(runtime)
    report = {"runtime": runtime, "working_root": str(work), "status": "absent", "applied": False}
    if not work.exists():
        return report
    with capacity_lease(work), workspace_lease(work):
        marker_path = work / MARKER
        if not marker_path.is_file():
            raise ValueError("No workspace identity: refusing to remove unknown scratch files")
        marker = json.loads(marker_path.read_text())
        if marker.get("runtime") != runtime or marker.get("durable_root") != str(durable):
            raise ValueError("Workspace identity/durable destination mismatch")
        targets = disposable_paths(work)
        # Require each asset tree's authoritative source. Failed staging is also
        # recoverable: GPU startup refuses a marker still marked 'staging'.
        for target in targets:
            source = reject_links(durable / target.relative_to(work))
            if not source.is_dir():
                raise ValueError(f"Missing durable input tree; preserve scratch: {source}")
            for copied in target.rglob("*"):
                relative = copied.relative_to(target)
                if any(
                    p in {".cache", "__pycache__"} or p.startswith(".stage-")
                    for p in relative.parts
                ):
                    continue  # Reproducible download/bytecode/interrupted-copy artifacts.
                if copied.name == "checkpoint_hashes.sqlite3" or copied.name.endswith(
                    ("-wal", "-shm", "-journal")
                ):
                    continue
                original = source / relative
                if (copied.is_file() or copied.is_symlink()) and not (
                    original.is_file() or original.is_symlink()
                ):
                    raise ValueError(f"Missing durable input file; preserve scratch: {original}")
        report.update(
            status="preview",
            removable_bytes=sum(tree_bytes(p) for p in targets),
            paths=[str(p) for p in targets],
            preserved=[
                "durable weights/images/inputs",
                "scratch response queues",
                "scratch embeddings/predictions/results",
                "DATA logs/manifests",
            ],
        )
        if not apply:
            return report
        require_runtime_idle(runtime)
        report["publication"] = _publish(durable, work)
        # A conflict or failed publication above leaves every input copy intact.
        write_json(marker_path, {**marker, "status": "releasing"})
        for target in targets:
            if not target.resolve().is_relative_to(work.resolve()):
                raise ValueError("Scratch cleanup target escaped the validated workspace")
            shutil.rmtree(target)
        write_json(marker_path, {**marker, "status": "released_requires_restage"})
        report.update(status="released_inputs_preserved_results", applied=True)
        write_json(
            paths.manifests_dir("experiment_0") / "scratch" / f"{runtime}_release.json", report
        )
        return report


def main():
    from src.cluster.containers import RUNTIMES
    from src.cluster.scheduling import require_cpu_staging

    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--runtime", choices=RUNTIMES)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    require_cpu_staging()
    if args.apply and paths.on_vsc() and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Publish/release large working sets in a CPU allocation")
    for runtime in RUNTIMES if args.all else [args.runtime]:
        print(json.dumps(release(runtime, apply=args.apply)), flush=True)


if __name__ == "__main__":
    main()
