"""Bound GPFS working copies; publish results before releasing disposable inputs."""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
from contextlib import closing, contextmanager, suppress
from pathlib import Path

from src.cluster.maintenance import reject_links
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import write_json

GIB = 1024**3
QUEUE_CHECK_UNAVAILABLE = 75
PUBLICATION_LOCK_UNAVAILABLE = 76
SCRATCH_CAPACITY_UNAVAILABLE = 77


class ScratchCapacityUnavailable(OSError):
    """Staging must wait for another working set to be released; nothing was copied."""


class QueueCheckUnavailable(RuntimeError):
    """The scheduler could not confirm that a scratch workspace is idle."""


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


def provider_cache_bound(source, provider):
    """Reserve for the selected provider's UTF-8 records, indexes and page slack.

    Count encoded bytes, not SQLite TEXT character lengths. Four times payload
    plus a page per row is deliberately generous for a freshly filtered store;
    check_capacity separately reserves two more copies of the largest store.
    """
    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        size = 16 * 2**20
        for table, predicate, columns in (
            ("responses", "json_extract(settings_json,'$.provider')=?",
             ("payload_json", "settings_json", "response_json")),
            ("origins", "request_key IN (SELECT request_key FROM responses WHERE "
             "json_extract(settings_json,'$.provider')=?)", ("provenance_json",)),
            ("failures", "json_extract(settings_json,'$.provider')=?",
             ("payload_json", "settings_json", "error_type")),
            ("feature_artifacts", "json_extract(manifest_json,'$.identity.provider')=?",
             ("manifest_json", "features_parquet", "requests_parquet")),
        ):
            if table in tables:
                payload_sql = "+".join(f"COALESCE(length(CAST({column} AS BLOB)),0)" for column in columns)
                count, payload = db.execute(
                    f"SELECT COUNT(*),COALESCE(SUM({payload_sql}),0) FROM {table} WHERE {predicate}",
                    (provider,),
                ).fetchone()
                size += 4 * (payload + 4096 * count)
    return min(source.stat().st_size, size)


def check_capacity(work, candidates, source_container, *, additional_containers=(), cache_provider=None,
                   shared_assets=False):
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
    def resident_workspace(p):
        if disposable_paths(p):
            return True
        from src.cluster.scratch_stage import MARKER

        marker = p / MARKER
        return marker.is_file() and json.loads(marker.read_text()).get("status") != "released_requires_restage"

    resident = [p for p in existing if p != work and resident_workspace(p)]
    if len(resident) >= settings["max_resident_workspaces"]:
        raise ScratchCapacityUnavailable("GPFS working-set slots are full; publish/release idle JEVPFN copies first")
    sources = [(source, work / relative) for relative, source in candidates.items()]
    container_relative = source_container.relative_to(source_container.parents[2])
    # Container paths always live under JEVPFN/containers/<profile>/<runtime>.
    sources += [] if shared_assets else [
        (p, work / container_relative / p.relative_to(source_container))
        for p in source_container.rglob("*")
        if p.is_file() and not p.is_symlink()
    ]
    for container in (() if shared_assets else additional_containers):
        sources += [(p, work / "containers/control-v1" / p.relative_to(container))
                    for p in container.rglob("*") if p.is_file() and not p.is_symlink()]
    sizes = [
        (provider_cache_bound(source, cache_provider)
         if cache_provider and dest.relative_to(work).parts[:2] == ("data", "jev_cache")
         else source.stat().st_size, dest)
        for source, dest in sources
    ]
    growth = sum(max(0, size - (dest.stat().st_size if dest.is_file() else 0)) for size, dest in sizes)
    largest = max((size for size, _ in sizes), default=0)
    extra = growth + 2 * largest + settings["scratch_headroom_gib"] * GIB
    project_bytes = tree_bytes(root)
    personal_bytes = tree_bytes(root.parent)
    print(json.dumps({"event": "scratch_capacity_estimate", "workspace_bytes": project_bytes,
                      "personal_bytes": personal_bytes, "growth_bytes": growth,
                      "largest_store_bytes": largest, "extra_bytes": extra,
                      "working_budget_gib": settings["scratch_working_set_gib"]}), flush=True)
    if project_bytes + extra > settings["scratch_working_set_gib"] * GIB:
        raise ScratchCapacityUnavailable(
            "JEVPFN GPFS byte budget exceeded; publish/release idle copies before staging"
        )
    if personal_bytes + extra > settings["scratch_allocation_gib"] * GIB:
        raise ScratchCapacityUnavailable(
            "Personal GPFS scratch budget/headroom exhausted; check myquota and other projects"
        )
    if shutil.disk_usage(work).free < extra:
        raise ScratchCapacityUnavailable("GPFS filesystem has insufficient free capacity for staging")


def _scheduler_read(command):
    try:
        return subprocess.check_output(command, text=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise QueueCheckUnavailable(f"Cannot verify scratch consumers: {command}: {exc}") from exc


def require_runtime_idle(runtime):
    """Protect queued consumers too: file leases alone only protect running workers."""
    if not paths.on_vsc():
        return
    from src.cluster.containers import RUNTIMES

    for cluster in ("wice", "mindwell"):
        listing = _scheduler_read(
            ["squeue", "--clusters=" + cluster, "--me", "--noheader", "--format=%i|%j|%Z"]
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
            detail = _scheduler_read(["scontrol", "--clusters=" + cluster, "show", "job", job, "-o"])
            tokens = shlex.split(detail)
            # These exact project entrypoints never consume an extraction workspace.
            # TAR uses fold-specific PROJECT bundles and its own temporary job cache;
            # CPU contracts use PROJECT directly. Both can overlap workspace release.
            independent = {
                "experiment_0_tokens.slurm",
                "experiment_0_cpu.slurm",
                "experiment_0_stage.slurm",
                "experiment_0_container_build.slurm",
                "experiment_0_embedding_stage.slurm",
                "stage_tabstar.slurm",
                "experiment_0_storage.slurm",
                "experiment_0_predictor_cpu.slurm",
                "prepare_tar.slurm",
                "prepare_tar_pca.slurm",
            }
            command = next(
                (t.removeprefix("Command=") for t in tokens if t.startswith("Command=")), ""
            )
            if Path(command).name in independent and "/JEVPFN/scripts/slurm/" in command:
                continue
            owners = {r for r in RUNTIMES if r in tokens or f"/jevpfn-work/{r}/" in detail}
            # Unknown JEVPFN consumers fail closed, including manually queued scripts
            # whose omitted arguments select a default runtime.
            if runtime in owners or not owners:
                raise RuntimeError(f"Scratch release refused: JEVPFN job {job} may use {runtime}")


def retire_publication_snapshots(work):
    """Retire exact redundant snapshots only, under an idle exclusive workspace lease.

    SQLite backup changes its header change/schema counters; every other byte must match
    the retained main store and both stores must have no live WAL/journal content.
    A mismatch/unknown file stays available for review instead of being cleaned.
    """
    def identity(path):
        if any(Path(str(path) + suffix).exists() and Path(str(path) + suffix).stat().st_size
               for suffix in ("-wal", "-journal")):
            return None
        before = path.stat()
        with path.open("rb") as stream:
            header = bytearray(stream.read(100))
            if len(header) != 100 or not header.startswith(b"SQLite format 3\0"):
                return None
            header[24:28] = header[40:44] = header[92:96] = b"\0" * 4
            sha = hashlib.sha256(header)
            for block in iter(lambda: stream.read(2**20), b""):
                sha.update(block)
        after = path.stat()
        return sha.hexdigest() if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns) else None

    retired, retained = [], []
    for folder in sorted(work.glob(".publish-*")):
        reject_links(folder)
        if not re.fullmatch(r"\.publish-[a-z0-9_]+", folder.name):
            retained.append(str(folder))
            continue
        files = [p for p in folder.rglob("*") if p.is_file() or p.is_symlink()]
        valid = bool(files)
        for copied in files:
            reject_links(copied)
            relative = copied.relative_to(folder)
            source = reject_links(work / relative)
            if (relative.parts[:2] != ("data", "jev_cache") or len(relative.parts) != 3
                    or copied.suffix != ".sqlite3" or not source.is_file()):
                valid = False
                break
            copied_identity = identity(copied)
            if copied_identity is None or copied_identity != identity(source):
                valid = False
                break
        if not valid:
            retained.append(str(folder))
            continue
        if folder.resolve().parent != work.resolve():
            raise ValueError("Snapshot cleanup escaped its owned workspace")
        size = tree_bytes(folder)
        shutil.rmtree(folder)
        retired.append({"path": str(folder), "bytes": size})
    return {"retired": retired, "retained_requires_review": retained}


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
        from src.cluster.control_runtime import host_view, verify

        control = durable / "containers/control-v1"
        control_view = host_view(control)
        if control_view != control:
            verify(control)  # Tykky masks bin in the executing CPU image; check its original host tree.
        # Require each asset tree's authoritative source. Failed staging is also
        # recoverable: GPU startup refuses a marker still marked 'staging'.
        for target in targets:
            source = reject_links(durable / target.relative_to(work))
            if not source.is_dir():
                raise ValueError(f"Missing durable input tree; preserve scratch: {source}")
            for copied in target.rglob("*"):
                relative = copied.relative_to(target)
                # Transformers creates package initializers and copies publisher Python
                # into HF_HOME/modules at runtime. These are not original checkpoint
                # assets; only this exact cache location and Python suffixes are exempt.
                if (
                    copied.relative_to(work).parts[:3] == ("checkpoints", "huggingface", "modules")
                    and copied.suffix in {".py", ".pyc"}
                    and not copied.is_symlink()
                ):
                    continue
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
                if original.is_relative_to(control):
                    original = control_view / original.relative_to(control)
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
        report["snapshot_retirement"] = retire_publication_snapshots(work)
        report["publication"] = _publish(durable, work, resume=True)
        # A conflict or failed publication above leaves every input copy intact.
        if marker.get("shared_immutable_root") == str(durable):
            if report["snapshot_retirement"]["retained_requires_review"]:
                raise ValueError("Unreviewed recovery snapshots prevent retiring mutable copies")
            report["published_mutable_retirement"] = retire_published_mutable(work)
            report["preserved"] = ["durable assets, unioned responses/features and recovery state",
                                   "DATA logs/manifests", "workspace identity and lease"]
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


def retire_published_mutable(work):
    """Called only after idle/exclusive lease and successful transactional publication.

    Shared-asset workspaces must not accumulate every finished provider's response
    stores. Retain their identity/lease, and refuse any file outside published data
    namespaces before removing anything. Legacy mirror release remains unchanged.
    """
    from src.cluster.scratch_stage import MARKER, PUBLICATION_PROGRESS

    candidates = []
    if work.is_mount():
        raise ValueError("Mounted working tree prevents retirement")
    for p in work.rglob("*"):
        if p.is_symlink() or p.is_mount():
            raise ValueError("Linked/mounted mutable work requires review before retirement")
        if not p.is_file():
            continue
        relative = p.relative_to(work)
        if relative.as_posix() in {MARKER, PUBLICATION_PROGRESS, ".workspace.lock"}:
            continue
        dataset = (len(relative.parts) >= 3 and relative.parts[:2] in {
            ("data", "jev_cache"), ("data", "text_embeddings")}
            and (p.suffix in {".sqlite3", ".lock"} or p.name.endswith(("-wal", "-shm", "-journal"))))
        result = (len(relative.parts) >= 4 and relative.parts[0] == "output_JEVPFN"
                  and relative.parts[2] == "results")
        if not dataset and not result:
            raise ValueError("Unpublished working file prevents retirement: " + str(relative))
        candidates.append(p)
    total = sum(p.stat().st_size for p in candidates)
    for p in candidates:
        p.unlink()
    for p in sorted((p for p in work.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if not any(p.iterdir()):
            p.rmdir()
    return {"files": [p.relative_to(work).as_posix() for p in candidates], "bytes": total}


def main():
    from src.cluster.containers import RUNTIMES
    from src.cluster.scheduling import require_cpu_staging
    from src.cluster.scratch_stage import PublicationLockUnavailable

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
        try:
            print(json.dumps(release(runtime, apply=args.apply)), flush=True)
        except QueueCheckUnavailable as exc:
            print(exc, flush=True)
            return QUEUE_CHECK_UNAVAILABLE
        except PublicationLockUnavailable as exc:
            print(exc, flush=True)
            return PUBLICATION_LOCK_UNAVAILABLE


if __name__ == "__main__":
    raise SystemExit(main())
