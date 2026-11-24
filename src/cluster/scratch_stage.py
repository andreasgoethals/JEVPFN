"""Native B200 working sets with durable, conflict-checked publication to PROJECT.

CPU staging copies only a selected runtime and its required inputs. GPFS scratch is
never called a backup. Publication uses SQLite snapshots and existing cache unions;
interrupted jobs leave their working set intact for explicit recovery.
"""

import argparse
import errno
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
# (durable root, relative path) -> closed source identity published by THIS process.
_PUBLISHED_IDENTITIES = {}
# Workspace stores observed to differ from their staged bytes. Stores only grow, so a
# store never returns to its staged content; remembering this avoids rehashing it.
_DIVERGED_FROM_STAGE = set()
# Per-process rowids already unioned into durable; later passes start above them.
_PUBLISHED_ROWIDS = {}
# A shared dataset lock can be held by another job's validated union of a multi-GB
# store (Kickstarter took 73 min on 22-11-2026); one hour made staging fail spuriously.
LOCK_WAIT_SECONDS = 4 * 3600
PUBLICATION_PROGRESS = "_JEVPFN_PUBLICATION_PROGRESS.json"


class WorkspaceBusy(RuntimeError):
    """Another job holds this B200 working set's lease."""


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
            raise WorkspaceBusy("B200 working set is in use; do not restage an active runtime") from exc
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

    counts = {"decision_stores": 0, "embedding_stores": 0, "outputs": 0, "busy_skipped": 0,
              "unchanged_skipped": 0}
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
                # A closed SQLite store this process already published, unchanged
                # since (same inode, size, mtime and ctime, no journal), is durable:
                # durable stores only grow by validated unions.
                published_key = (str(durable), relative.as_posix())
                # Staging publishes any earlier workspace results first and then only
                # copies or unions durable records into the workspace, so a store whose
                # bytes still equal its staged hash holds nothing durable lacks. Without
                # this, every qualification job re-unioned all 18 multi-GB stores at
                # exit (5 h timeouts, 22-11-2026) while holding the shared dataset locks.
                if (
                    kind != "outputs"
                    and destination.is_file()
                    and _unchanged_since_stage(source, work, relative, staged)
                ):
                    counts[kind] += 1
                    counts["unchanged_skipped"] += 1
                    print(json.dumps({"event": "publication_unchanged_since_stage",
                                      "path": relative.as_posix(), "periodic": periodic}), flush=True)
                    continue
                if source.suffix == ".sqlite3":
                    unchanged = _closed_db_identity(source)
                    if (unchanged is not None and destination.is_file()
                            and _PUBLISHED_IDENTITIES.get(published_key) == unchanged):
                        counts[kind] += 1
                        counts["unchanged_skipped"] += 1
                        print(json.dumps({"event": "publication_unchanged", "path": relative.as_posix(),
                                          "periodic": periodic}), flush=True)
                        continue
                # Unrelated datasets must publish concurrently. A periodic writer
                # can defer a busy destination; final publication must acquire it.
                lock = durable / ".publication-locks" / (digest(relative.as_posix()) + ".lock")
                with ExitStack() as stack:
                    try:
                        stack.enter_context(file_lock(lock, timeout=0 if periodic else LOCK_WAIT_SECONDS))
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
                                with file_lock(destination.with_suffix(".extraction.lock"), timeout=LOCK_WAIT_SECONDS,
                                               shared=True):
                                    destination_identity = _closed_db_identity(destination)
                                if destination_identity is not None and saved["destination"] == destination_identity:
                                    counts[kind] += 1
                                    print(json.dumps({"event": "publication_reused", "path": relative_key}),
                                          flush=True)
                                    continue
                    started = time.monotonic()
                    before_snapshot = _closed_db_identity(source) if source.suffix == ".sqlite3" else None
                    if kind == "decision_stores":
                        # Incremental union: never re-copy and re-validate a multi-GB
                        # durable store to add a few rows (8 h contract timeouts, 23-11-2026).
                        print(json.dumps({"event": "publication_started", "path": relative.as_posix(),
                                          "periodic": periodic, "mode": "delta"}), flush=True)
                        identity = _merge_decision_delta(source, destination, work, published_key, relative)
                        if source_sha is not None and identity is not None:
                            progress[relative_key] = {"source_sha256": source_sha, "destination": identity}
                            write_json(progress_path, {"durable_root": str(durable), "stores": progress})
                        if before_snapshot is not None and _closed_db_identity(source) == before_snapshot:
                            _PUBLISHED_IDENTITIES[published_key] = before_snapshot
                        counts[kind] += 1
                        print(json.dumps({"event": "publication_completed", "path": relative.as_posix(),
                                          "seconds": time.monotonic() - started}), flush=True)
                        continue
                    temporary_root = _publication_temporary_root(work, source.stat().st_size)
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
                        if merger is not None:
                            merger(copied, destination)
                        elif source.suffix == ".sqlite3" and "evaluation" in relative.parts:
                            merge_evaluations(copied, destination)
                        else:
                            _publish_file(copied, destination)
                    # Remember only a source that stayed closed and unchanged across
                    # the whole snapshot and merge; any later write changes its identity.
                    if before_snapshot is not None and _closed_db_identity(source) == before_snapshot:
                        _PUBLISHED_IDENTITIES[published_key] = before_snapshot
                    counts[kind] += 1
                    print(json.dumps({"event": "publication_completed", "path": relative.as_posix(),
                                      "seconds": time.monotonic() - started}), flush=True)
    return counts


def _unchanged_since_stage(source, work, relative, staged):
    key = relative.as_posix()
    seen = (str(work), key)
    if key not in staged or seen in _DIVERGED_FROM_STAGE:
        return False
    if _closed_db_identity(source) is None:  # An open writer or a pending journal.
        return False
    if file_sha256(source) == staged[key]:
        return True
    _DIVERGED_FROM_STAGE.add(seen)
    return False


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
        file_lock(destination.with_suffix(".extraction.lock"), timeout=LOCK_WAIT_SECONDS),
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
        try:
            snapshot(destination, verified)
            verified_identity = _backup_page_identity(verified)
        except (OSError, sqlite3.OperationalError) as exc:
            # Free space was checked before the union. A co-tenant may have
            # filled node-local disk while it ran. Retry only the readback.
            if not (getattr(exc, "errno", None) == errno.ENOSPC
                    or getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_FULL):
                raise
            if temporary_root == _fallback_job_root(work):
                raise
            fallback = _native_scratch_fallback(work, native.stat().st_size + 512 * 2**20)
            if fallback is None:
                raise
            print(json.dumps({"event": "publication_readback_fallback", "path": destination.name,
                              "temporary_storage": str(fallback)}), flush=True)
            with tempfile.TemporaryDirectory(prefix=".readback-", dir=fallback) as retry:
                verified = snapshot(destination, Path(retry) / verified.name)
                verified_identity = _backup_page_identity(verified)
        if _backup_page_identity(native) != verified_identity:
            raise ValueError("Published SQLite readback differs from the validated union")
        return _closed_db_identity(destination)


DELTA_TABLES = {
    # table: (key columns, whether new rows carry a created_at column)
    "responses": (("request_key", "kind"), True),
    "origins": (("request_key", "kind", "provenance_hash"), False),
    "failures": (("event_id",), True),
    "feature_artifacts": (("artifact_id",), False),
}
LEDGER_TABLES = ("extraction_policy", "extraction_policy_history", "extraction_jobs",
                 "extraction_recoveries", "imported_extraction_records")
STAGE_CUTOFF_MARGIN_SECONDS = 3600


def _store_rowids(path):
    """Highest rowid per delta table; durable stores only grow, never delete or rewrite."""
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return {t: db.execute(f"SELECT COALESCE(MAX(rowid),0) FROM {t}").fetchone()[0]
                for t in DELTA_TABLES if t in tables}


def _stage_cutoff(work, relative):
    """Rows a GPU job may have added since staging: rowid watermark, else time cutoff.

    Workspaces staged before 24-11-2026 carry no rowid watermark. Their marker was
    last written when staging finished, before any inference could start (staging
    holds the exclusive workspace lease), so rows created since then minus a
    margin are a superset of every new or possibly conflicting row.
    """
    from datetime import UTC, datetime, timedelta

    marker = work / MARKER
    data = json.loads(marker.read_text()) if marker.is_file() else {}
    rowids = data.get("staged_rowids", {}).get(relative.as_posix())
    if rowids is not None:
        return {"rowids": rowids}
    if not marker.is_file():
        return {"created_at": ""}  # Unknown staging: compare every row.
    # An explicit staging time survives later marker edits; the file time does not.
    staged = (datetime.fromisoformat(data["staged_utc"]) if data.get("staged_utc")
              else datetime.fromtimestamp(marker.stat().st_mtime, UTC))
    return {"created_at": (staged - timedelta(seconds=STAGE_CUTOFF_MARGIN_SECONDS)).isoformat()}


def _merge_decision_delta(source, destination, work, key, relative):
    """Union only rows written since staging, in one durable transaction.

    Staging copies this runtime's rows FROM durable into scratch, and durable
    stores never delete or rewrite rows, so rows present at staging are durable
    by construction. Only rows written later (above the staged rowids, or for
    workspaces staged before 24-11-2026 created since staging) are copied, plus
    feature artifacts durable lacks with their referenced responses. ``merge_cache``
    validates every copied row and compares it with any durable row under the same
    key, so a conflicting paid response still aborts the whole union. Candidates
    are written in key order so durable B-tree pages are visited sequentially.
    Probing every scratch key against PROJECT instead costs ~5 ms per row
    (174 s for 34,633 rows, probe job 11870855, 24-11-2026). Rows up to an earlier
    successful publication by this process are skipped.
    """
    from src.jev.cache import ResponseCache
    from src.jev.features import load_block

    size = source.stat().st_size
    if not destination.exists():
        root = _publication_temporary_root(work, size + _union_bytes(size, destination))
        with tempfile.TemporaryDirectory(prefix=".publish-", dir=root) as t:
            return _merge_native_decision(snapshot(source, Path(t) / source.name), destination, work)
    published = _PUBLISHED_ROWIDS.get(key, {})
    cutoff = _stage_cutoff(work, relative)
    with (
        file_lock(destination.with_suffix(".extraction.lock"), timeout=LOCK_WAIT_SECONDS),
        tempfile.TemporaryDirectory(prefix=".delta-",
                                    dir=_publication_temporary_root(work, size + 512 * 2**20)) as temporary,
    ):
        _recover_rollback(destination)
        delta = Path(temporary) / source.name
        with closing(sqlite3.connect(delta.as_uri(), uri=True)) as db:
            db.execute("ATTACH DATABASE ? AS src", (source.resolve().as_uri() + "?mode=ro",))
            db.execute("ATTACH DATABASE ? AS dst", (destination.resolve().as_uri() + "?mode=ro",))
            db.execute("BEGIN")  # One consistent read of the scratch store.
            schema = dict(db.execute(
                "SELECT name, sql FROM src.sqlite_master WHERE type='table' AND sql IS NOT NULL "
                "AND name NOT LIKE 'sqlite_%'"))
            dst_tables = {r[0] for r in db.execute(
                "SELECT name FROM dst.sqlite_master WHERE type='table'")}
            for sql in schema.values():
                db.execute(sql)
            db.execute("INSERT INTO main.dataset_identity SELECT * FROM src.dataset_identity")
            high = {}
            for table, (columns, timed) in DELTA_TABLES.items():
                if table not in schema:
                    continue
                floor = published.get(table, 0)
                high[table] = db.execute(f"SELECT COALESCE(MAX(rowid),0) FROM src.{table}").fetchone()[0]
                order = ", ".join(columns)
                if "rowids" in cutoff:
                    new, args = "rowid > ?", (max(floor, cutoff["rowids"].get(table, 0)),)
                elif timed:
                    new, args = "rowid > ? AND created_at >= ?", (floor, cutoff["created_at"])
                elif table == "origins":
                    db.execute("INSERT OR IGNORE INTO main.origins SELECT o.* FROM main.responses AS r "
                               "JOIN src.origins AS o ON o.request_key=r.request_key AND o.kind=r.kind "
                               "WHERE o.rowid > ? ORDER BY o.request_key, o.kind, o.provenance_hash",
                               (floor,))
                    continue
                else:
                    new, args = "rowid > ?", (floor,)
                if table == "feature_artifacts" and table in dst_tables:
                    # Artifacts are few; durable ones are complete, so only absent IDs travel.
                    new += (" AND NOT EXISTS (SELECT 1 FROM dst.feature_artifacts AS x "
                            "WHERE x.artifact_id=s.artifact_id)")
                db.execute(f"INSERT OR IGNORE INTO main.{table} SELECT * FROM src.{table} AS s "
                           f"WHERE {new} ORDER BY {order}", args)
            for table in LEDGER_TABLES:
                if table in schema:
                    db.execute(f"INSERT INTO main.{table} SELECT * FROM src.{table}")
            db.execute("COMMIT")
            artifacts = ([r[0] for r in db.execute("SELECT artifact_id FROM main.feature_artifacts")]
                         if "feature_artifacts" in schema else [])
            if artifacts:
                db.execute("BEGIN")
                for artifact_id in artifacts:
                    keys = load_block(delta, artifact_id).requests.request_key.dropna().unique()
                    db.executemany(
                        "INSERT OR IGNORE INTO main.responses SELECT * FROM src.responses "
                        "WHERE request_key=? AND kind='verified_response'", ((k,) for k in keys))
                db.execute("COMMIT")
            db.execute("DETACH DATABASE src")
            db.execute("DETACH DATABASE dst")
        ResponseCache(delta)  # Same schema guarantees as every store.
        merged = merge_cache(delta, destination, backup_destination=False, deduplicate_ledger=True,
                             destination_locked=True, validate_destination=False,
                             commit_unchanged=False)
        _PUBLISHED_ROWIDS[key] = high
        print(json.dumps({"event": "publication_delta", "path": destination.name,
                          **{k: v for k, v in merged.items() if k.endswith(("_added", "_reused"))}}),
              flush=True)
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


def _provider_cache(source, destination, provider, *, after=None, rowids=None):
    """Warm only this open-weight provider; leave all durable history untouched.

    Imported ledgers are archival, never an executable scratch queue. Existing
    scratch queues are preserved by the caller's validated union. Predictor
    staging still receives every representation through its original path.
    With ``after`` (durable rowids of the previous warm), only later rows are read;
    ``rowids`` receives durable's rowids at this warm, captured under the lock.
    """
    # Publication backs up the durable union under this same extraction lock.
    # Hold it before even reading identity, through the complete filtered copy;
    # SQLite's short busy timeout cannot span a multi-minute publication.
    with file_lock(source.with_suffix(".extraction.lock"), timeout=LOCK_WAIT_SECONDS):
        _recover_rollback(source)
        if rowids is not None:
            rowids.update(_store_rowids(source))  # Durable cannot grow while the lock is held.
        return _provider_cache_locked(source, destination, provider, after)


def _provider_cache_locked(source, destination, provider, after=None):
    from src.jev.cache import ResponseCache
    from src.jev.merge import ARTIFACT_SQL, validate_cache

    after = after or {}
    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        dataset_id, = db.execute("SELECT dataset_id FROM dataset_identity").fetchone()
    cache = ResponseCache(destination, dataset_id)
    with cache._connect() as db:
        db.execute("ATTACH DATABASE ? AS upstream", (source.resolve().as_uri() + "?mode=ro",))
        db.execute("INSERT INTO responses SELECT * FROM upstream.responses "
                   "WHERE rowid > ? AND json_extract(settings_json,'$.provider')=?",
                   (after.get("responses", 0), provider))
        db.execute("INSERT INTO origins SELECT o.* FROM upstream.origins o "
                   "JOIN responses r ON r.request_key=o.request_key AND r.kind=o.kind")
        tables = {r[0] for r in db.execute("SELECT name FROM upstream.sqlite_master WHERE type='table'")}
        if "failures" in tables:
            db.execute("INSERT INTO failures SELECT * FROM upstream.failures "
                       "WHERE rowid > ? AND json_extract(settings_json,'$.provider')=?",
                       (after.get("failures", 0), provider))
        if "feature_artifacts" in tables:
            db.execute(ARTIFACT_SQL)
            db.execute("INSERT INTO feature_artifacts SELECT * FROM upstream.feature_artifacts "
                       "WHERE rowid > ? AND json_extract(manifest_json,'$.identity.provider')=?",
                       (after.get("feature_artifacts", 0), provider))
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


def stage(runtime, *, datasets=None):
    from src.cluster import containers
    from src.cluster.runtime_profiles import BLACKWELL, selected
    from src.cluster.scheduling import require_cpu_staging
    from src.cluster.scratch_capacity import capacity_lease, check_capacity

    if selected() != BLACKWELL:
        raise ValueError("B200 staging requires JEVPFN_RUNTIME_PROFILE=blackwell-cu130")

    durable, work = roots(runtime)
    require_cpu_staging()
    if datasets is not None:
        if runtime not in containers.SELF_HOSTED or not datasets:
            raise ValueError("Dataset-scoped staging requires a decision runtime and nonempty scope")
        datasets = [paths.output_component(value) for value in datasets]
        if len(set(datasets)) != len(datasets) or any(
            not (durable / "data/decision_inputs" / (value + ".sqlite3")).is_file()
            for value in datasets
        ):
            raise ValueError("Staging datasets must be distinct prepared dataset IDs")
        datasets = sorted(datasets)
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
        # Durable rowids already copied into each scratch store (24-11-2026); kept across restages.
        previous_warm = previous.get("warm_rowids", {})
        warm_rowids = dict(previous_warm)
        staging_marker = {
            "runtime": runtime,
            "runtime_profile": selected(),
            "durable_root": str(durable),
            "status": "staging",
            "files": previous.get("files", {}),
            "warm_cache_progress": warm,
            "datasets": datasets,
        }
        write_json(work / MARKER, staging_marker)
        inputs = ["data/raw", "data/decision_inputs", "data/jev_cache"]
        shared_predictor_inputs = runtime not in containers.SELF_HOSTED and paths.native_gpfs_project(durable)
        if (runtime == "cloudflare" and datasets is None) or runtime not in containers.SELF_HOSTED:
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
            f"output_JEVPFN/{paths.result_relative(p).as_posix()}/{name}"
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
            # B3 text-column refits (rq2), B4 (data_efficiency), C1 (representations) and
            # C2 (rq3) write their own fold stores and must be restored too (22-11-2026).
            evaluation_phases = ("experiment_0", "experiment_1", "experiment_2", "experiment_3",
                                 "experiment_4", "synthesis", "regression_readout", "rq2",
                                 "data_efficiency", "representations", "rq3")
            for folder in sorted(
                durable / "output_JEVPFN" / paths.result_relative(p, "evaluation") / "evaluation"
                for p in evaluation_phases
            ):
                if not folder.is_dir():
                    continue
                for source in _dataset_stores(folder):
                    candidates[source.relative_to(durable)] = source
        for folder in inputs:
            if runtime in containers.SELF_HOSTED and folder.endswith("/results/pca"):
                continue  # Decision extraction never consumes predictor PCA parameters.
            if shared_predictor_inputs and folder in {
                "data/jev_cache", "data/text_embeddings",
                f"output_JEVPFN/{paths.result_relative('experiment_2').as_posix()}/pca",
                f"output_JEVPFN/{paths.result_relative('experiment_1').as_posix()}/pca",
            }:
                continue  # Predictors open accepted, immutable features and PCA directly on PROJECT.
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
                if (datasets is not None and folder == "data/jev_cache"
                        and source.stem not in datasets):
                    continue  # Completed datasets remain durable; exclude before capacity estimation.
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
                with file_lock(destination.with_suffix(".extraction.lock"), timeout=LOCK_WAIT_SECONDS):
                    _recover_rollback(destination)
                # A scratch store that already holds this runtime's durable rows up to
                # recorded rowids needs only later rows; durable never deletes rows.
                after = previous_warm.get(key) if destination.is_file() else None
                rowids = {}
                with tempfile.TemporaryDirectory(prefix=".warm-", dir=destination.parent) as temp:
                    incoming = _provider_cache(source, Path(temp) / source.name, runtime,
                                               after=after, rowids=rowids)
                    with staging_environment(work):
                        merge_cache(incoming, destination, backup_destination=False, deduplicate_ledger=True,
                                    validate_destination=after is None)
                warm_rowids[key] = rowids
                staging_marker["warm_rowids"] = warm_rowids
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
        from datetime import UTC, datetime

        report = {
            "schema": "jevpfn.scratch.v1",
            "staged_utc": datetime.now(UTC).isoformat(),
            "warm_rowids": warm_rowids,
            # Rows above these rowids were written after staging; see _stage_cutoff.
            "staged_rowids": {
                p.relative_to(work).as_posix(): _store_rowids(p)
                for p in _dataset_stores(work / "data/jev_cache")
            },
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
            "datasets": datasets,
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

        dataset_id = os.environ.get("JEVPFN_PUBLISH_DATASET")
        if marker.get("datasets") is not None and dataset_id not in marker["datasets"]:
            raise ValueError("Requested dataset is outside the staged recovery scope")
        process = subprocess.Popen(child_command(command), env=child_environment())
        stopped = threading.Event()
        publication_errors = []

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
    parser.add_argument("--dataset", action="append", help="Stage only these unfinished decision datasets")
    args, command = parser.parse_known_args()
    if args.dataset and args.action != "stage":
        parser.error("--dataset applies only to stage")
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
        from src.cluster.scratch_capacity import (
            SCRATCH_CAPACITY_UNAVAILABLE,
            ScratchCapacityUnavailable,
        )

        try:
            report = stage(args.runtime, datasets=args.dataset) if args.action == "stage" else publish(args.runtime)
            print(json.dumps(report, indent=2))
        except ScratchCapacityUnavailable as exc:
            # A distinct status lets the dispatcher wait and retry instead of failing.
            print(exc, flush=True)
            raise SystemExit(SCRATCH_CAPACITY_UNAVAILABLE) from None


if __name__ == "__main__":
    main()
