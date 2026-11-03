"""Migration-only unions with stopped writers; normal live publication is separate.

Hold project leases/locks and exclude scheduler consumers before calling. SQLite
exclusive locks/checkpoints make each main-file copy consistent. Validate all unions
on node-local disk, then publish one complete file while no consumer uses its inode.
"""

import gc
import os
import shutil
import sqlite3
import stat
import tempfile
from contextlib import closing, contextmanager
from pathlib import Path
from uuid import uuid4

from src.cluster.maintenance import reject_links
from src.cluster.project_migration import sha
from src.cluster.scratch_stage import staging_environment
from src.jev.merge import merge_cache, validate_cache
from src.utils.locking import file_lock


@contextmanager
def exclusive_database(path):
    """Recover any hot journal, checkpoint WAL, and exclude SQLite writers/readers."""
    if not path.is_file():
        raise ValueError("A stopped source database is missing")
    with closing(sqlite3.connect(path, timeout=1)) as db:
        db.execute("PRAGMA trusted_schema=OFF")
        mode = db.execute("PRAGMA journal_mode").fetchone()[0]
        if mode == "wal":
            if db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]:
                raise RuntimeError("A WAL reader/writer prevents quiescent publication")
            # A rollback-mode exclusive lock also excludes old WAL readers before
            # migration replaces the inode. This changes no table/record content.
            if db.execute("PRAGMA journal_mode=DELETE").fetchone()[0] != "delete":
                raise RuntimeError("Database could not leave WAL mode while stopped")
        db.execute("BEGIN EXCLUSIVE")
        try:
            yield
        finally:
            db.rollback()


def closed_copy(source, destination):
    """Only while holding the source's exclusive SQLite lock, never a live copy."""
    with exclusive_database(source):
        if any(Path(str(source) + suffix).exists() and Path(str(source) + suffix).stat().st_size
               for suffix in ("-wal", "-journal")):
            raise ValueError("Uncheckpointed SQLite content prevents a main-file copy")
        before = sha(source)
        shutil.copyfile(source, destination)
        if before != sha(source) or before != sha(destination):
            raise ValueError("Quiescent SQLite copy changed")
    return before


def exact_subset(source, destination):
    """Read-only, byte-exact inclusion of every validated table/column/record.

    Both stores must already have passed validate_cache. Indexed primary keys
    locate candidates; type and BLOB comparisons prevent collation/coercion from
    treating different provenance, response bytes or feature blobs as identical.
    """
    def quoted(value):
        return '"' + value.replace('"', '""') + '"'

    with closing(sqlite3.connect(destination.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("ATTACH DATABASE ? AS incoming", (source.resolve().as_uri() + "?mode=ro",))
        tables = [r[0] for r in db.execute("SELECT name FROM incoming.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for table in tables:
            name = quoted(table)
            a = db.execute("PRAGMA incoming.table_info(" + name + ")").fetchall()
            b = db.execute("PRAGMA main.table_info(" + name + ")").fetchall()
            if not b or [(r[1], r[5]) for r in a] != [(r[1], r[5]) for r in b]:
                return False
            keys = [r[1] for r in a if r[5]]
            if not keys:
                return False
            predicates = [f"d.{quoted(k)} IS s.{quoted(k)}" for k in keys]
            for row in a:
                column = quoted(row[1])
                predicates += [f"typeof(d.{column})=typeof(s.{column})",
                               f"CAST(d.{column} AS BLOB) IS CAST(s.{column} AS BLOB)"]
            sql = (f"SELECT 1 FROM incoming.{name} AS s WHERE NOT EXISTS "
                   f"(SELECT 1 FROM main.{name} AS d WHERE {' AND '.join(predicates)}) LIMIT 1")
            if db.execute(sql).fetchone():
                return False
    return True


def guarded_dataset_union(sources, destination, temporary_root):
    """Picklable CPU worker; each dataset has its own locks and temporary files."""
    from src.cluster.maintenance import require_idle_vsc

    return union_closed_stores(sources, destination, temporary_root, writers_stopped=True,
                               consumer_check=lambda: require_idle_vsc(allow_current=True))


def single_thread_worker():
    import pyarrow
    from threadpoolctl import threadpool_limits

    pyarrow.set_cpu_count(1)
    threadpool_limits(limits=1)


def union_closed_stores(sources, destination, temporary_root, *, writers_stopped=False, consumer_check=None):
    """One validated durable publication per dataset, not per old working set.

    Atomic replacement is restricted to this explicitly quiescent migration API.
    Normal periodic publication continues to use SQLite's transactional backup API.
    """
    if not writers_stopped or not sources:
        raise ValueError("Quiescent publication requires stopped writers and source stores")
    destination = reject_links(Path(destination))
    sources = [reject_links(Path(p)) for p in sources]
    if destination in sources:
        raise ValueError("A migration source cannot be the durable publication target")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(temporary_root)
    required = sum(sorted((p.stat().st_size for p in sources), reverse=True)[:1])
    required += destination.stat().st_size if destination.exists() else 0
    if shutil.disk_usage(temporary_root).free < 3 * required + 512 * 2**20:
        raise OSError("Insufficient node-local space for a closed database union")
    with (file_lock(destination.with_suffix(".extraction.lock"), timeout=1),
          tempfile.TemporaryDirectory(prefix="jevpfn-migration-union-", dir=temporary_root) as temp):
        root = Path(temp) / "JEVPFN"
        native = root / "data/jev_cache" / destination.name
        native.parent.mkdir(parents=True)
        original_hash = None
        mode = 0o600
        if destination.exists():
            mode = stat.S_IMODE(destination.stat().st_mode)
            original_hash = closed_copy(destination, native)
        dataset = validate_cache(native) if native.exists() else None
        changed = False
        reports = []
        for number, source in enumerate(sources):
            incoming = Path(temp) / ("incoming_" + str(number) + ".sqlite3")
            source_hash = closed_copy(source, incoming)
            incoming_dataset = validate_cache(incoming)
            if dataset is not None and incoming_dataset != dataset:
                raise ValueError("Stopped stores belong to different datasets")
            if dataset is not None and exact_subset(incoming, native):
                result = {"responses_added": 0, "closed_subset_verified": True}
            else:
                with staging_environment(root):
                    result = merge_cache(incoming, native, backup_destination=False, deduplicate_ledger=True)
                dataset = validate_cache(native)
                changed = True
            reports.append({"source": str(source), "sha256": source_hash, "union": result})
            gc.collect()  # Close cyclic SQLite readers before retiring their temporary files.
            incoming.unlink()
        gc.collect()
        expected = sha(native)
        if not changed and original_hash is not None:
            if consumer_check is not None:
                consumer_check()
            with exclusive_database(destination):
                if sha(destination) != original_hash or expected != original_hash:
                    raise ValueError("A validated superset changed during migration")
            return {"dataset_id": dataset, "destination": str(destination), "sha256": original_hash,
                    "bytes": destination.stat().st_size, "sources": reports,
                    "consumer_guard": consumer_check is not None,
                    "protocol": "exclusive_sqlite_validated_exact_superset"}
        candidate = destination.parent / (".migration-union-" + uuid4().hex + ".partial")
        try:
            shutil.copyfile(native, candidate)
            candidate.chmod(mode)
            with candidate.open("rb+") as stream:
                os.fsync(stream.fileno())
            if sha(candidate) != expected or sha(native) != expected:
                raise ValueError("Complete migration candidate differs from its validated union")
            if consumer_check is not None:
                consumer_check()
            if destination.exists():
                if original_hash is None:
                    raise ValueError("A new destination appeared during stopped-writer publication")
                with exclusive_database(destination):
                    if sha(destination) != original_hash:
                        raise ValueError("Durable data changed during stopped-writer publication")
            elif original_hash is not None:
                raise ValueError("Durable data disappeared during stopped-writer publication")
            os.replace(candidate, destination)
            if os.name == "posix":
                fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            if sha(destination) != expected:
                raise ValueError("Durable migration publication failed readback")
        finally:
            candidate.unlink(missing_ok=True)
        return {"dataset_id": dataset, "destination": str(destination), "sha256": expected,
                "bytes": destination.stat().st_size, "sources": reports,
                "consumer_guard": consumer_check is not None,
                "protocol": "exclusive_sqlite_checkpoints_closed_union_atomic_replace"}
