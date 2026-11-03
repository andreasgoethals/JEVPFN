"""Explicit, source-preserving PROJECT copy/verification; host stdlib only.

Run bulk operations in a transfer allocation. This file deliberately works with
the host Python, outside Tykky's masked view of its own container wrapper tree.
No command follows symlinks, replaces existing science or deletes shared roots.
"""

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b""):
            h.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("File changed during verification: " + str(path))
    return h.hexdigest()


def roots(source, destination):
    source, destination = Path(source).absolute(), Path(destination).absolute()
    for p in (source, destination):
        if p.name != "JEVPFN" or p.is_symlink():
            raise ValueError("Migration requires exact real JEVPFN project directories")
        for parent in p.parents:
            if parent.is_symlink():
                raise ValueError("A migration ancestor is a symlink")
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Migration roots must be disjoint")
    if not source.is_dir() or not destination.parent.is_dir():
        raise ValueError("Existing source and allocated destination parent required")
    return source, destination


def inventory(root, workers=4):
    paths, links, groups = [], {}, {}
    for base, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            p = Path(base) / name
            key = p.relative_to(root).as_posix()
            if p.is_symlink():
                links[key] = os.readlink(p)
            elif p.is_file():
                s = p.stat()
                paths.append(p)
                groups.setdefault((s.st_dev, s.st_ino), []).append(key)
    def record(p):
        s = p.stat()
        return p.relative_to(root).as_posix(), {"sha256": sha(p), "bytes": s.st_size,
                                               "executable": bool(s.st_mode & 0o111)}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        files = dict(pool.map(record, paths))
    return {"files": files, "links": links,
            "hardlinks": [sorted(v) for v in groups.values() if len(v) > 1]}


def verify_sqlite(root, *, integrity=True, temporary_root=None, workers=1, expected_hashes=None):
    def check(p):
        if p.is_symlink():
            return None
        # A quiescent byte-for-byte transfer is consistent only when no WAL or
        # rollback journal contains committed pages outside the copied main DB.
        if any(Path(str(p) + s).exists() and Path(str(p) + s).stat().st_size
               for s in ("-wal", "-journal")):
            raise ValueError("SQLite has uncheckpointed pages; snapshot it first: " + str(p))
        if integrity:
            with tempfile.TemporaryDirectory(prefix="jevpfn-sqlite-check-", dir=temporary_root) as temp:
                checked = p
                if temporary_root:
                    if shutil.disk_usage(temporary_root).free < p.stat().st_size + 512 * 2**20:
                        raise OSError("Insufficient node-local space for a consistent SQLite check")
                    checked = Path(temp) / p.name
                    shutil.copyfile(p, checked)
                    expected = (expected_hashes or {}).get(p.relative_to(root).as_posix()) or sha(p)
                    if sha(checked) != expected:
                        raise ValueError("Node-local SQLite copy differs from verified PROJECT bytes")
                with closing(sqlite3.connect(checked.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                    if db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                        raise ValueError("SQLite integrity failure: " + str(p))
        return p.relative_to(root).as_posix()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [p for p in pool.map(check, sorted(root.rglob("*.sqlite3"))) if p is not None]


def rewrite_links(source, destination):
    rewritten = []
    for base, dirs, files in os.walk(destination, followlinks=False):
        for name in dirs + files:
            p = Path(base) / name
            if not p.is_symlink():
                continue
            target = Path(os.readlink(p).removeprefix("\\\\?\\"))
            if target.is_absolute() and (target == source or source in target.parents):
                p.unlink()
                p.symlink_to(destination / target.relative_to(source))
                rewritten.append(p.relative_to(destination).as_posix())
    return rewritten


def compare(source_inventory, destination_inventory, source, destination, *, relocated=False):
    a, b = source_inventory, destination_inventory
    if a["files"].keys() != b["files"].keys() or a["links"].keys() != b["links"].keys():
        raise ValueError("Source/destination file membership differs")
    for key, value in a["files"].items():
        if relocated and key.startswith("containers/") and not key.endswith((".sif", ".sqfs")):
            continue  # Relocation seals verify these small host wrappers separately.
        if value != b["files"][key]:
            raise ValueError("Copied bytes/executable mode differ: " + key)
    for key, target in a["links"].items():
        expected = target
        normalized_target = target.removeprefix("\\\\?\\")
        if relocated and Path(normalized_target).is_absolute() and (Path(normalized_target) == source or source in Path(normalized_target).parents):
            expected = str(destination / Path(normalized_target).relative_to(source))
        # Image-internal /LOCAL_TYKKY links intentionally remain unchanged.
        if b["links"][key].removeprefix("\\\\?\\") != expected.removeprefix("\\\\?\\"):
            raise ValueError("Copied link differs: " + key)
    for group in a["hardlinks"]:
        if not any(set(group) <= set(other) for other in b["hardlinks"]):
            raise ValueError("A required hardlink relationship was lost")


def copy(source, destination, receipt, *, writers_stopped=False, workers=4, defer_verification=False):
    source, destination = roots(source, destination)
    if not writers_stopped or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Consistent copy requires quiescent writers and a transfer allocation")
    print(json.dumps({"event": "migration_inventory_started", "source": str(source)}), flush=True)
    # Closed/checkpointed source bytes are copied and compared exactly. Run the
    # random-read integrity pass on native GPFS, not twice through slow Lustre.
    sqlite_files = verify_sqlite(source, integrity=False)
    original = inventory(source, workers)
    print(json.dumps({"event": "source_hashes_verified", "files": len(original["files"]),
                      "sqlite_files": len(sqlite_files)}), flush=True)
    if (destination / "_JEVPFN_PROJECT_MIGRATION.json").exists():
        raise ValueError("Destination has entered cutover; never overwrite it from the old authority")
    destination.mkdir(mode=0o2770, exist_ok=True)
    # The independent destination SHA-256 pass below is authoritative. A second
    # rsync checksum scan would redundantly read every large immutable asset.
    subprocess.run(["rsync", "-aH", "--no-owner", "--no-group", "--no-acls", "--no-xattrs",
                    "--chmod=Dg+rwx,Dg+s,Do-rwx,Fo-rwx", str(source) + "/", str(destination) + "/"], check=True)
    result = {"status": "copied_requires_native_verification",
              "source": str(source), "destination": str(destination), "source_inventory": original,
              "sqlite_files": sqlite_files, "created_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - host Python 3.9
              "job_id": os.environ["SLURM_JOB_ID"], "copied_logical_bytes": sum(v["bytes"] for v in original["files"].values())}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2))
    if not defer_verification:
        result = verify_copy(receipt, workers=workers)
    print(json.dumps({k: v for k, v in result.items() if k not in {"source_inventory", "sqlite_files"}}), flush=True)
    return result


def verify_copy(receipt, *, workers=4, temporary_root=None, sqlite_workers=1):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Destination verification requires a CPU allocation")
    result = json.loads(receipt.read_text())
    if result.get("status") != "copied_requires_native_verification":
        raise ValueError("Only a completed unaccepted copy can enter native verification")
    source, destination = roots(result["source"], result["destination"])
    compare(result["source_inventory"], inventory(destination, workers), source, destination)
    print(json.dumps({"event": "copy_bytes_verified"}), flush=True)
    if verify_sqlite(destination, temporary_root=temporary_root, workers=sqlite_workers,
                     expected_hashes={k:v["sha256"] for k,v in result["source_inventory"]["files"].items()}) != result["sqlite_files"]:
        raise ValueError("SQLite membership changed during migration")
    result.update(status="copied_and_verified_requires_relocation_and_runtime_acceptance",
                  verification_job_id=os.environ["SLURM_JOB_ID"],
                  sqlite_check_temporary_root=str(temporary_root) if temporary_root else None)
    receipt.write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("copy", "inventory", "rewrite-links", "verify-copy"))
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--destination", required=True, type=Path)
    p.add_argument("--receipt", required=True, type=Path)
    p.add_argument("--writers-stopped", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--defer-verification", action="store_true")
    p.add_argument("--temporary-root", type=Path)
    p.add_argument("--sqlite-workers", type=int, default=1)
    a = p.parse_args()
    if a.action == "copy":
        copy(a.source, a.destination, a.receipt, writers_stopped=a.writers_stopped, workers=a.workers,
             defer_verification=a.defer_verification)
    elif a.action == "verify-copy":
        verify_copy(a.receipt, workers=a.workers, temporary_root=a.temporary_root, sqlite_workers=a.sqlite_workers)
    elif a.action == "inventory":
        a.receipt.write_text(json.dumps(inventory(a.source, a.workers), indent=2))
    else:
        source, destination = roots(a.source, a.destination)
        print(json.dumps(rewrite_links(source, destination)))


if __name__ == "__main__":
    main()
