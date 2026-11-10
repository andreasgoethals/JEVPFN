"""Native B200 working sets with durable, conflict-checked publication to PROJECT.

CPU staging copies only a selected runtime and its required inputs. GPFS scratch is
never called a backup. Publication uses SQLite snapshots and existing cache unions;
interrupted jobs leave their working set intact for explicit recovery.
"""

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from contextlib import ExitStack, closing, contextmanager, suppress
from pathlib import Path

from src.cluster.maintenance import reject_links
from src.jev.merge import merge_cache, snapshot
from src.utils import paths
from src.utils.locking import file_lock
from src.utils.serialization import digest, file_sha256, write_json

MARKER = "_JEVPFN_WORKSPACE.json"
PUBLICATION_PROGRESS = "_JEVPFN_PUBLICATION_PROGRESS.json"


class PublicationLockUnavailable(TimeoutError):
    """A writer could not obtain the durable store's publication lock."""


def roots(runtime):
    from src.cluster.containers import RUNTIMES

    if runtime not in RUNTIMES:
        raise ValueError("Unknown runtime for B200 working set")
    durable = paths.durable_project_override()
    if durable is None:
        raise ValueError("An explicit durable PROJECT root is required")
    scratch = os.environ.get("VSC_SCRATCH_GPFS1")
    if not scratch or not Path(scratch).is_absolute():
        raise ValueError("VSC_SCRATCH_GPFS1 must identify your existing GPFS scratch allocation")
    scratch = reject_links(Path(scratch))
    durable = reject_links(durable)
    if not scratch.is_dir() or not durable.is_dir():
        raise ValueError("Both native scratch and durable project allocations must exist")
    if scratch == durable or scratch.is_relative_to(durable) or durable.is_relative_to(scratch):
        raise ValueError("Scratch and durable roots must be separate allocations")
    if paths.on_vsc() and (not str(scratch).startswith("/gpfs") or not any(
        str(durable).startswith(prefix) and "/project/" in str(durable)
        for prefix in ("/gpfs", "/lustre")
    )):
        raise ValueError("B200 needs native GPFS scratch and an allocated GPFS/Lustre PROJECT root")
    return (
        reject_links(durable / "JEVPFN"),
        reject_links(scratch / "jevpfn-work" / runtime / "JEVPFN"),
    )


@contextmanager
def staging_environment(root):
    previous = os.environ.get("JEVPFN_STAGING_ROOT")
    os.environ["JEVPFN_STAGING_ROOT"] = str(root.parent)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("JEVPFN_STAGING_ROOT", None)
        else:
            os.environ["JEVPFN_STAGING_ROOT"] = previous


@contextmanager
def workspace_lease(root, *, shared=False):
    """GPU jobs share a lease; restaging requires exclusivity. Kernel releases on crash."""
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".workspace.lock"
    with ExitStack() as stack:
        try:
            stack.enter_context(file_lock(lock, timeout=1, shared=shared))
        except TimeoutError as exc:
            raise RuntimeError("B200 working set is in use; do not restage an active runtime") from exc
        yield


def _regular_files(folder, *, link_boundary=None, immutable_assets=False):
    reject_links(folder)
    if not folder.exists():
        return
    for source in sorted(folder.rglob("*")):
        # HF download metadata and transient writers are not model assets.
        if any(part == ".cache" for part in source.relative_to(folder).parts):
            continue
        if (source.is_symlink() or getattr(source, "is_junction", lambda: False)()) and (
            link_boundary is None or not source.resolve().is_relative_to(link_boundary.resolve())
        ):
            raise ValueError(f"Linked input escapes the approved checkpoint tree: {source}")
        if source.is_file() and (immutable_assets or not source.name.endswith(
            (".lock", "-wal", "-shm", "-journal", ".partial")
        )):
            yield source


def _dataset_stores(folder):
    """Only active dataset DBs: recovery backups and control files are not inputs."""
    reject_links(folder)
    for source in sorted(folder.glob("*.sqlite3")):
        reject_links(source)
        if source.is_file():
            yield source


def _copy(source, destination, *, snapshot_sqlite=True):
    reject_links(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage-", dir=destination.parent) as temporary:
        copied = Path(temporary) / source.name
        if source.suffix == ".sqlite3" and snapshot_sqlite:
            snapshot(source, copied)
        else:
            if source.suffix == ".sqlite3" and any(
                Path(str(source) + suffix).exists() and Path(str(source) + suffix).stat().st_size
                for suffix in ("-wal", "-journal")
            ):
                raise ValueError("Prepared inputs must be closed/checkpointed before transfer")
            before = file_sha256(source)
            shutil.copyfile(source, copied)
            if before != file_sha256(source) or before != file_sha256(copied):
                raise ValueError("Input changed during staging")
        sha = file_sha256(copied)
        if destination.is_file() and file_sha256(destination) == sha:
            return sha
        os.replace(copied, destination)
        if file_sha256(destination) != sha:
            raise ValueError("Staging checksum mismatch")
        return sha


def _closed_db_identity(path):
    if not path.is_file() or any(
        sidecar.is_file() and sidecar.stat().st_size
        for sidecar in (path.with_name(path.name + suffix) for suffix in ("-wal", "-journal"))
    ):
        return None
    info = path.stat()
    return [info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _recover_rollback(path):
    """Let SQLite roll back an interrupted write under the existing writer lease."""
    journal = path.with_name(path.name + "-journal")
    if journal.is_file() and journal.stat().st_size:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True)) as db:
            db.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()


def _publish(durable, work, **options):
    """Publish, then remove this job's empty GPFS scratch fallback directory."""
    try:
        return _publish_stores(durable, work, **options)
    finally:
        job_root = _fallback_job_root(work)
        if job_root is not None and job_root.is_dir() and not job_root.is_symlink():
            with suppress(OSError):  # Kept only if a crash left a union in it.
                job_root.rmdir()


def _publish_stores(durable, work, *, periodic=False, dataset_id=None,
                    include_decision_stores=True, resume=False):
    """Never replace a durable response DB with a partial scratch copy."""
    from src.cluster.transfer import _publish_file
    from src.data.text_embeddings import merge as merge_embeddings
    from src.evaluation.store import merge as merge_evaluations

    counts = {"decision_stores": 0, "embedding_stores": 0, "outputs": 0, "busy_skipped": 0}
    if dataset_id is not None:
        dataset_id = paths.output_component(dataset_id)
    staged = (
        json.loads((work / MARKER).read_text()).get("files", {}) if (work / MARKER).exists() else {}
    )
    # Only staging and release own the exclusive workspace lease needed for
    # restart receipts. Dataset/periodic publishers retain their separate locks.
    progress_path = reject_links(work / PUBLICATION_PROGRESS)
    progress = {}
    if resume and progress_path.is_file():
        prior = json.loads(progress_path.read_text())
        if prior["durable_root"] != str(durable):
            raise ValueError("Publication checkpoint belongs to another durable root")
        progress = prior["stores"]
    durable.mkdir(parents=True, exist_ok=True)
    with staging_environment(durable):
        for kind, folder, merger in (
            ("decision_stores", "data/jev_cache", merge_cache),
            ("embedding_stores", "data/text_embeddings", merge_embeddings),
            ("outputs", "output_JEVPFN", None),
        ):
            if kind == "decision_stores" and not include_decision_stores:
                continue
            sources = (
                _dataset_stores(work / folder)
                if kind == "decision_stores"
                else _regular_files(work / folder)
            )
            for source in sources:
                relative = source.relative_to(work)
                if kind != "outputs" and source.suffix != ".sqlite3":
                    continue
                if kind != "outputs" and dataset_id is not None and source.stem != dataset_id:
                    continue
                if kind == "outputs" and "results" not in relative.parts:
                    continue
                if (
                    dataset_id is not None
                    and kind == "outputs"
                    and source.suffix == ".sqlite3"
                    and "evaluation" in relative.parts
                    and source.stem != dataset_id
                ):
                    continue
                if (
                    periodic
                    and kind == "outputs"
                    and not (source.suffix == ".sqlite3" and "evaluation" in relative.parts)
                ):
                    continue
                destination = durable / relative
                reject_links(destination)
                # Immutable CPU inputs already have an authoritative durable copy.
                if (
                    kind == "outputs"
                    and relative.as_posix() in staged
                    and file_sha256(source) == staged[relative.as_posix()]
                ):
                    continue
                # Unrelated datasets must publish concurrently. A periodic writer
                # can defer a busy destination; final publication must acquire it.
                lock = durable / ".publication-locks" / (digest(relative.as_posix()) + ".lock")
                with ExitStack() as stack:
                    try:
                        stack.enter_context(file_lock(lock, timeout=0 if periodic else 3600))
                    except TimeoutError as exc:
                        if not periodic:
                            raise PublicationLockUnavailable(
                                f"Timed out waiting for the publication lock for {relative.as_posix()}"
                            ) from exc
                        counts["busy_skipped"] += 1
                        continue
                    relative_key = relative.as_posix()
                    source_sha = None
                    if resume and kind == "decision_stores":
                        _recover_rollback(source)
                        if _closed_db_identity(source) is not None:
                            source_sha = file_sha256(source)
                            saved = progress.get(relative_key)
                            if saved and saved["source_sha256"] == source_sha:
                                with file_lock(destination.with_suffix(".extraction.lock"), timeout=3600,
                                               shared=True):
                                    destination_identity = _closed_db_identity(destination)
                                if destination_identity is not None and saved["destination"] == destination_identity:
                                    counts[kind] += 1
                                    print(json.dumps({"event": "publication_reused", "path": relative_key}),
                                          flush=True)
                                    continue
                    started = time.monotonic()
                    # The incoming snapshot and its union share one temporary root
                    # whenever that root can hold both.
                    needed = source.stat().st_size + (
                        _union_bytes(source.stat().st_size, destination)
                        if kind == "decision_stores" else 0
                    )
                    temporary_root = _publication_temporary_root(work, needed)
                    print(json.dumps({"event": "publication_started", "path": relative.as_posix(),
                                      "periodic": periodic,
                                      "temporary_storage": str(temporary_root)}), flush=True)
                    # Bound temporary bytes to ONE store, rather than retaining
                    # all eighteen snapshots until the release process exits.
                    with tempfile.TemporaryDirectory(prefix=".publish-", dir=temporary_root) as temporary:
                        copied = Path(temporary) / relative
                        copied.parent.mkdir(parents=True, exist_ok=True)
                        if source.suffix == ".sqlite3":
                            snapshot(source, copied)
                        else:
                            shutil.copyfile(source, copied)
                        if kind == "decision_stores":
                            identity = _merge_native_decision(copied, destination, work)
                            if source_sha is not None and identity is not None:
                                progress[relative_key] = {"source_sha256": source_sha, "destination": identity}
                                write_json(progress_path, {"durable_root": str(durable), "stores": progress})
                        elif merger is not None:
                            merger(copied, destination)
                        elif source.suffix == ".sqlite3" and "evaluation" in relative.parts:
                            merge_evaluations(copied, destination)
                        else:
                            _publish_file(copied, destination)
                    counts[kind] += 1
                    print(json.dumps({"event": "publication_completed", "path": relative.as_posix(),
                                      "seconds": time.monotonic() - started}), flush=True)
    return counts


def _union_bytes(incoming_bytes, destination):
    """Temporary bytes for one validated union: its two snapshots plus SQLite slack."""
    existing = destination.stat().st_size if destination.exists() else 0
    return 2 * (incoming_bytes + existing) + 512 * 2**20


FALLBACK_PARTS = ("jevpfn-transfers", "publication-temporary")
TERMINAL_STATES = ("BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
                   "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT")


def _fallback_job_root(work):
    """This job's own directory; only Mindwell jobs own a GPFS scratch fallback."""
    job = os.environ.get("SLURM_JOB_ID")
    if not job or not job.isdigit() or os.environ.get("SLURM_CLUSTER_NAME") == "wice":
        return None
    return work.parents[2].joinpath(*FALLBACK_PARTS, job)


def _retire_stale_fallbacks(root):
    """Remove unions left by killed jobs, only after Slurm reports them terminal.

    Normal exits remove their own directory. Unknown or unverifiable jobs are kept.
    """
    from src.cluster.scratch_capacity import QueueCheckUnavailable, _scheduler_read

    if not root.is_dir() or root.is_symlink():
        return []
    own = os.environ.get("SLURM_JOB_ID")
    stale = sorted((p for p in root.iterdir()
                    if p.name.isdigit() and p.name != own and p.is_dir() and not p.is_symlink()),
                   key=lambda p: int(p.name))
    if not stale:
        return []
    try:
        listing = _scheduler_read(["sacct", "--clusters=mindwell", "--noheader", "--parsable2",
                                   "--allocations", "--format=JobIDRaw,State",
                                   "--jobs=" + ",".join(p.name for p in stale)])
    except QueueCheckUnavailable:
        return []
    states = dict(line.split("|", 1) for line in listing.splitlines() if line.count("|") == 1)
    retired = []
    for p in stale:
        if states.get(p.name, "").split(" ")[0] in TERMINAL_STATES:
            if p.resolve().parent != root.resolve():
                raise ValueError("Fallback cleanup escaped its owned directory")
            shutil.rmtree(p)
            retired.append(p.name)
    return retired


def _native_scratch_fallback(work, required):
    """Node-local /tmp is shared and unreserved on Mindwell B200 nodes (TmpDisk=0).

    Durable stores grow with every provider, so a union can outgrow the space
    other jobs leave. Mindwell's native GPFS scratch then holds it in this job's
    own directory, outside every workspace and within the personal allocation.
    The directory is removed after publication; it never falls back to PROJECT.
    """
    from src.cluster.scratch_capacity import GIB, tree_bytes
    from src.utils.config import load_yaml

    job_root = _fallback_job_root(work)
    if job_root is None:
        return None  # WICE reaches Mindwell GPFS only through a cross-mount.
    scratch = work.parents[2]
    if not str(scratch).startswith("/gpfs") and paths.on_vsc():
        return None
    _retire_stale_fallbacks(job_root.parent)
    settings = load_yaml("experiment_0/scheduling")
    budget = (settings["scratch_allocation_gib"] - settings["scratch_headroom_gib"]) * GIB
    if tree_bytes(scratch) + required > budget or shutil.disk_usage(scratch).free < required:
        return None
    job_root.mkdir(parents=True, exist_ok=True)
    return reject_links(job_root)


def _publication_temporary_root(work, required=0):
    """Temporary unions use declared node-local disk; resumable stores stay on GPFS.

    When the node-local disk cannot hold ``required`` bytes, a native GPFS scratch
    directory outside the workspaces is used instead of failing a finished job.
    """
    cluster = os.environ.get("SLURM_CLUSTER_NAME")
    declared = os.environ.get("VSC_SCRATCH_NODE") or os.environ.get("SLURM_TMPDIR")
    if os.environ.get("SLURM_JOB_ID") and (declared or cluster == "wice"):
        local = Path(declared or "/tmp")
        if not local.is_absolute() or not local.is_dir() or local.is_relative_to(paths.data_root()):
            raise ValueError("Publication needs a verified node-local temporary directory")
        if not required or shutil.disk_usage(local).free >= required:
            return local
        fallback = _native_scratch_fallback(work, required)
        if fallback is None:
            raise OSError("Insufficient native space for a validated publication union")
        return fallback
    return work


def _merge_native_decision(incoming, destination, work):
    """Validate/union on native storage; transactionally back up the union to PROJECT.

    The destination extraction lock covers its snapshot, merge and publication,
    including other importers. SQLite backup commits the replacement as a single
    transaction without replacing an inode underneath a reader or losing WAL pages.
    A failed native merge never writes the durable destination.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = _union_bytes(incoming.stat().st_size, destination)
    temporary_root = _publication_temporary_root(work, required)
    if shutil.disk_usage(temporary_root).free < required:
        raise OSError("Insufficient native space for a validated publication union")
    with (
        file_lock(destination.with_suffix(".extraction.lock"), timeout=3600),
        tempfile.TemporaryDirectory(prefix=".union-", dir=temporary_root) as temporary,
    ):
        native_root = Path(temporary) / "JEVPFN"
        native = native_root / "data/jev_cache" / destination.name
        native.parent.mkdir(parents=True)
        existed = destination.exists()
        if existed:
            _recover_rollback(destination)
            snapshot(destination, native)
        with staging_environment(native_root):
            merged = merge_cache(incoming, native, backup_destination=False, deduplicate_ledger=True)
        if existed and not any(value for key, value in merged.items() if key.endswith("_added")):
            # The validated merge proved all incoming records already durable.
            # Avoid rewriting a multi-GB database and its rollback journal.
            return _closed_db_identity(destination)
        with (
            closing(sqlite3.connect(native.as_uri() + "?mode=ro", uri=True)) as src,
            closing(sqlite3.connect(destination)) as dst,
        ):
            src.backup(dst)
        # Read committed pages back with SQLite backup (including any WAL), then
        # perform random-read integrity checks on native temporary disk. Running
        # quick_check directly on PROJECT caused millions of GPFS page reads.
        verified = Path(temporary) / "published_readback.sqlite3"
        snapshot(destination, verified)
        if _backup_page_identity(native) != _backup_page_identity(verified):
            raise ValueError("Published SQLite readback differs from the validated union")
        return _closed_db_identity(destination)


def _backup_page_identity(path):
    """Compare backup content; only SQLite's storage/header counters may differ."""
    with path.open("rb") as stream:
        header = bytearray(stream.read(100))
        if len(header) != 100 or not header.startswith(b"SQLite format 3\0"):
            raise ValueError("Invalid SQLite backup header")
        if any(value not in (1, 2) for value in header[18:20]):
            raise ValueError("Unsupported SQLite journal format")
        # Backup preserves destination WAL/rollback mode and advances its change
        # and schema counters. No table/index/content byte is excluded.
        header[18:20] = b"\0" * 2
        header[24:28] = header[40:44] = header[92:96] = b"\0" * 4
        result = hashlib.sha256(header)
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def _provider_cache(source, destination, provider):
    """Warm only this open-weight provider; leave all durable history untouched.

    Imported ledgers are archival, never an executable scratch queue. Existing
    scratch queues are preserved by the caller's validated union. Predictor
    staging still receives every representation through its original path.
    """
    # Publication backs up the durable union under this same extraction lock.
    # Hold it before even reading identity, through the complete filtered copy;
    # SQLite's short busy timeout cannot span a multi-minute publication.
    with file_lock(source.with_suffix(".extraction.lock"), timeout=3600):
        _recover_rollback(source)
        return _provider_cache_locked(source, destination, provider)


def _provider_cache_locked(source, destination, provider):
    from src.jev.cache import ResponseCache
    from src.jev.merge import ARTIFACT_SQL, validate_cache

    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        dataset_id, = db.execute("SELECT dataset_id FROM dataset_identity").fetchone()
    cache = ResponseCache(destination, dataset_id)
    with cache._connect() as db:
        db.execute("ATTACH DATABASE ? AS upstream", (source.resolve().as_uri() + "?mode=ro",))
        db.execute("INSERT INTO responses SELECT * FROM upstream.responses "
                   "WHERE json_extract(settings_json,'$.provider')=?", (provider,))
        db.execute("INSERT INTO origins SELECT o.* FROM upstream.origins o "
                   "JOIN responses r ON r.request_key=o.request_key AND r.kind=o.kind")
        tables = {r[0] for r in db.execute("SELECT name FROM upstream.sqlite_master WHERE type='table'")}
        if "failures" in tables:
            db.execute("INSERT INTO failures SELECT * FROM upstream.failures "
                       "WHERE json_extract(settings_json,'$.provider')=?", (provider,))
        if "feature_artifacts" in tables:
            db.execute(ARTIFACT_SQL)
            db.execute("INSERT INTO feature_artifacts SELECT * FROM upstream.feature_artifacts "
                       "WHERE json_extract(manifest_json,'$.identity.provider')=?", (provider,))
    validate_cache(destination)
    return destination


def publish(runtime):
    durable, work = roots(runtime)
    if not (work / MARKER).is_file():
        raise ValueError("No staged B200 workspace; nothing can be published")
    with workspace_lease(work, shared=True):
        marker = json.loads((work / MARKER).read_text())
        if marker["durable_root"] != str(durable) or marker["runtime"] != runtime:
            raise ValueError("Workspace publication destination differs from its stage receipt")
        counts = _publish(durable, work)
    report = {"runtime": runtime, "status": "published", "durable_root": str(durable), **counts}
    write_json(
        paths.manifests_dir("experiment_0") / "scratch" / f"{runtime}_publication.json", report
    )
    return report


def stage(runtime):
    from src.cluster import containers
    from src.cluster.runtime_profiles import BLACKWELL, selected
    from src.cluster.scheduling import require_cpu_staging
    from src.cluster.scratch_capacity import capacity_lease, check_capacity

    if selected() != BLACKWELL:
        raise ValueError("B200 staging requires JEVPFN_RUNTIME_PROFILE=blackwell-cu130")

    durable, work = roots(runtime)
    require_cpu_staging()
    shared_assets = str(durable).startswith("/gpfs") or os.environ.get("JEVPFN_SHARED_IMMUTABLES") == "1"
    with capacity_lease(work), workspace_lease(work):
        previous = {}
        if (work / MARKER).exists():
            previous = json.loads((work / MARKER).read_text())
            if previous.get("runtime") != runtime or previous.get("durable_root") != str(durable):
                raise ValueError("Workspace publication destination differs from its stage receipt")
            if previous.get("status") != "staging":
                _publish(durable, work, resume=True)  # Protect earlier inference before restaging.
        warm = previous.get("warm_cache_progress", {}) if previous.get("status") == "staging" else {}
        staging_marker = {
            "runtime": runtime,
            "runtime_profile": selected(),
            "durable_root": str(durable),
            "status": "staging",
            "files": previous.get("files", {}),
            "warm_cache_progress": warm,
        }
        write_json(work / MARKER, staging_marker)
        inputs = ["data/raw", "data/decision_inputs", "data/jev_cache"]
        if runtime == "cloudflare" or runtime not in containers.SELF_HOSTED:
            inputs += ["data/text_embeddings"]
        if runtime in containers.SELF_HOSTED:
            inputs += [
                f"checkpoints/decision_models/{runtime}",
                f"checkpoints/tokenizers/{runtime}",
            ]
            if runtime in {"nimble", "strands"}:
                inputs += [f"checkpoints/decision_models/{runtime}_base"]
            if runtime == "strands":
                # The publisher loads its separate base by HF ID, even offline.
                from src.utils.config import load_yaml

                base = load_yaml("extractors")["candidates"][runtime]["base_model"]
                inputs += ["checkpoints/huggingface/hub/models--" + base.replace("/", "--")]
        else:
            # Predictor pins explicitly declare assets; never duplicate the ten
            # unrelated decision checkpoints in every predictor working set.
            from src.utils.config import load_yaml

            runtime_cfg = load_yaml("experiment_1/models")["models"][runtime]["runtime"]

            def pinned_paths(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key == "path" and item:
                            asset = Path(item)
                            if not asset.is_absolute():
                                asset = durable / "checkpoints" / asset
                            if not asset.resolve().is_relative_to(
                                (durable / "checkpoints").resolve()
                            ):
                                raise ValueError("Predictor pin is outside durable checkpoints")
                            yield asset.relative_to(durable)
                        else:
                            yield from pinned_paths(item)
                elif isinstance(value, list):
                    for item in value:
                        yield from pinned_paths(item)

            inputs += [str(p) for p in pinned_paths(runtime_cfg)]
            if runtime == "tabstar" and runtime_cfg.get("repositories"):
                from src.evaluation.tabstar_assets import MARKER as TABSTAR_MARKER
                from src.evaluation.tabstar_assets import asset_manifest
                assets = asset_manifest(runtime_cfg, root=durable / "checkpoints")
                inputs += ["checkpoints/" + p["path"] for p in assets["assets"]]
                inputs += ["checkpoints/predictors/tabstar/" + TABSTAR_MARKER]
        if runtime == "cloudflare":
            inputs += ["checkpoints/text_embeddings"]
        # Include prepared CPU artifacts (tokens and fold-specific PCA), never arbitrary outputs.
        inputs += [
            f"output_JEVPFN/{paths.phase_folder(p)}/results/{name}"
            for p, name in (
                *([("experiment_0", "token_audit")] if runtime in containers.SELF_HOSTED else []),
                ("experiment_2", "pca"),
                ("experiment_1", "pca"),
            )
        ]
        candidates = {}
        if runtime not in containers.SELF_HOSTED:
            # Completed fold records and their predictions are resume inputs too.
            # A fresh native workspace must see durable attempts before fitting;
            # otherwise it would refit them and conflict when publishing back.
            for folder in sorted(
                (durable / "output_JEVPFN").glob("*. experiment_*/results/evaluation")
            ):
                paths.phase_name(folder.parents[1].name)  # Require canonical experiment numbering.
                for source in _dataset_stores(folder):
                    candidates[source.relative_to(durable)] = source
        for folder in inputs:
            if shared_assets and (folder.startswith("checkpoints/") or folder in {"data/raw", "data/decision_inputs"}
                                  or "token_audit" in folder):
                continue
            location = durable / folder
            sources = (
                _dataset_stores(location)
                if folder == "data/jev_cache"
                else (
                    [location]
                    if location.is_file()
                    else _regular_files(location, link_boundary=durable / "checkpoints",
                                        immutable_assets=folder.startswith("checkpoints/"))
                )
            )
            for source in sources:
                relative = source.relative_to(durable)
                if "token_audit" in relative.parts and "shards" in relative.parts:
                    continue  # CPU restart partitions are not GPU inputs.
                if (
                    "token_audit" in relative.parts
                    and runtime in containers.SELF_HOSTED
                    and not source.name.startswith(runtime + "_")
                ):
                    continue
                candidates[relative] = source
        if not any(p.parts[:2] == ("data", "decision_inputs") for p in candidates) and not (
            shared_assets and (durable / "data/decision_inputs").is_dir()
        ):
            raise ValueError("Prepared questions are absent from durable storage")
        with staging_environment(durable):
            source_container = containers.container_dir(runtime)
        control = durable / "containers/control-v1"
        control_selected = (durable / "containers/control.active.json").is_file()
        if control_selected:
            from src.cluster.control_runtime import host_view
            from src.cluster.control_runtime import verify as verify_control

            verify_control(control)
            check_capacity(work, candidates, source_container, additional_containers=[control],
                           cache_provider=runtime if runtime in containers.SELF_HOSTED else None,
                           shared_assets=shared_assets)
        else:
            check_capacity(work, candidates, source_container,
                           cache_provider=runtime if runtime in containers.SELF_HOSTED else None,
                           shared_assets=shared_assets)
        files = {}
        for relative, source in candidates.items():
            destination = work / relative
            reject_links(destination)
            # Preserve operational queues in scratch; never replace them with the
            # durable union, whose imported queues intentionally remain archival.
            if relative.parts[:2] == ("data", "jev_cache") and runtime in containers.SELF_HOSTED:
                # A new runtime must not copy all previous models' multi-GB
                # response histories into every native workspace.
                key = relative.as_posix()
                before = _closed_db_identity(source)
                saved = warm.get(key)
                if (before is not None and saved and saved["source"] == before
                        and _closed_db_identity(destination) is not None
                        and file_sha256(destination) == saved["sha256"]):
                    files[key] = saved["sha256"]
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with file_lock(destination.with_suffix(".extraction.lock"), timeout=3600):
                    _recover_rollback(destination)
                with tempfile.TemporaryDirectory(prefix=".warm-", dir=destination.parent) as temp:
                    incoming = _provider_cache(source, Path(temp) / source.name, runtime)
                    with staging_environment(work):
                        merge_cache(incoming, destination, backup_destination=False, deduplicate_ledger=True)
                files[key] = file_sha256(destination)
                if before is not None and before == _closed_db_identity(source):
                    warm[key] = {"source": before, "sha256": files[key]}
                    write_json(work / MARKER, staging_marker)
            elif relative.parts[:2] == ("data", "jev_cache") and destination.is_file():
                with (
                    staging_environment(work),
                    tempfile.TemporaryDirectory(prefix=".merge-", dir=destination.parent) as temp,
                ):
                    incoming = snapshot(source, Path(temp) / source.name)
                    merge_cache(
                        incoming, destination, backup_destination=False, deduplicate_ledger=True
                    )
                files[relative.as_posix()] = file_sha256(destination)
            elif relative.parts[:2] == ("data", "text_embeddings") and destination.is_file():
                from src.data.text_embeddings import merge

                with tempfile.TemporaryDirectory(prefix=".merge-", dir=destination.parent) as temp:
                    merge(snapshot(source, Path(temp) / source.name), destination)
                files[relative.as_posix()] = file_sha256(destination)
            elif source.suffix == ".sqlite3" and "evaluation" in relative.parts:
                from src.evaluation.store import merge

                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix=".merge-", dir=destination.parent) as temp:
                    merge(snapshot(source, Path(temp) / source.name), destination)
                files[relative.as_posix()] = file_sha256(destination)
            else:
                # Token receipts bind the exact prepared-input file bytes. SQLite
                # backup changes its header, so immutable inputs must copy verbatim.
                files[relative.as_posix()] = _copy(
                    source,
                    destination,
                    snapshot_sqlite=relative.parts[:2] != ("data", "decision_inputs"),
                )
        if not shared_assets:
            with staging_environment(work):
                target_container = containers.container_dir(runtime)
            if not target_container.exists():
                shutil.copytree(source_container, target_container, symlinks=True)
            containers.relocate(source_container, target_container)
        if control_selected and not shared_assets:
            copied_control = work / "containers/control-v1"
            control_view = host_view(control)
            if not copied_control.exists():
                shutil.copytree(control_view, copied_control, symlinks=True)
            containers.relocate(control, copied_control,
                                **({"source_view": control_view} if control_view != control else {}))
        if runtime in containers.SELF_HOSTED:
            from src.jev.checkpoints import verify_checkpoint

            # Prime file identities on the native filesystem during the CPU transfer.
            with staging_environment(work):
                verify_checkpoint(runtime)
        report = {
            "schema": "jevpfn.scratch.v1",
            "runtime": runtime,
            "runtime_profile": selected(),
            "durable_root": str(durable),
            "working_root": str(work),
            "files": files,
            "input_identity": digest(files),
            "status": "staged_requires_b200_contracts",
            "quota_verified": False,
            "scratch_is_backup": False,
            "shared_immutable_root": str(durable) if shared_assets else None,
        }
        write_json(work / MARKER, report)
        write_json(
            paths.manifests_dir("experiment_0") / "scratch" / f"{runtime}_stage.json", report
        )
        return {k: v for k, v in report.items() if k != "files"}


def run(runtime, command):
    """Hold the working-set lease through the child, and publish even after failure."""
    import signal
    import subprocess
    import threading

    from src.cluster.runtime_profiles import selected

    durable, work = roots(runtime)
    if Path(os.environ["JEVPFN_STAGING_ROOT"]).resolve() != work.parent.resolve():
        raise ValueError("GPU job did not select its native working set")
    with workspace_lease(work, shared=True):
        # Read after acquiring the lease: a CPU restage can otherwise replace the
        # marker between validation and admission of this inference process.
        marker = json.loads((work / MARKER).read_text())
        if (
            marker["runtime"] != runtime
            or marker["durable_root"] != str(durable)
            or marker["runtime_profile"] != selected()
            or marker.get("status") != "staged_requires_b200_contracts"
        ):
            raise ValueError("Wrong scratch workspace identity")
        from src.cluster.control_runtime import child_command, child_environment

        process = subprocess.Popen(child_command(command), env=child_environment())
        stopped = threading.Event()
        publication_errors = []
        dataset_id = os.environ.get("JEVPFN_PUBLISH_DATASET")

        def checkpoint():
            # Only committed SQLite pages are copied. This also protects long
            # extraction jobs that never get a normal exit after node failure.
            while not stopped.wait(600):
                try:
                    _publish(durable, work, periodic=True, dataset_id=dataset_id)
                except Exception as exc:
                    publication_errors.append(exc)
                    process.terminate()
                    return

        thread = threading.Thread(target=checkpoint, daemon=True)
        thread.start()
        previous = signal.signal(signal.SIGTERM, lambda *_: process.terminate())
        try:
            code = process.wait()
        finally:
            stopped.set()
            thread.join()
            signal.signal(signal.SIGTERM, previous)
            _publish(durable, work, dataset_id=dataset_id)
        if publication_errors:
            raise RuntimeError(
                "Periodic durable publication failed; scratch is retained"
            ) from publication_errors[0]
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "publish", "run", "path"))
    parser.add_argument("--runtime", required=True)
    args, command = parser.parse_known_args()
    if args.action == "path":
        print(roots(args.runtime)[1].parent)
    elif args.action == "run":
        command = command[1:] if command[:1] == ["--"] else command
        if not command:
            parser.error("run needs -- COMMAND ...")
        raise SystemExit(run(args.runtime, command))
    else:
        if command:
            parser.error("Unknown arguments: " + " ".join(command))
        print(json.dumps((stage if args.action == "stage" else publish)(args.runtime), indent=2))


if __name__ == "__main__":
    main()
