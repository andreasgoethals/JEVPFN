"""Reuse CPU-computed weight hashes while file identity/size/ctime/mtime stay unchanged."""

import os
import sqlite3
from contextlib import closing

from src.utils import paths
from src.utils.serialization import canonical_json, file_sha256


def signature(path):
    stat = path.stat()
    return canonical_json(
        [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    )


def hashes(folder):
    index = paths.checkpoints_dir("checkpoint_hashes.sqlite3")
    index.parent.mkdir(parents=True, exist_ok=True)
    result, changes = {}, []
    with closing(sqlite3.connect(index, timeout=60)) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS hashes (path TEXT PRIMARY KEY, signature TEXT NOT NULL, sha256 TEXT NOT NULL)"
        )
        for path in sorted(folder.rglob("*")):
            relative = path.relative_to(folder)
            if (
                not path.is_file()
                or path == index
                or ".cache" in relative.parts
                or path.name == "_JEVPFN_CHECKPOINT.json"
            ):
                continue
            before = signature(path)
            key = str(path.absolute())
            row = db.execute("SELECT signature,sha256 FROM hashes WHERE path=?", (key,)).fetchone()
            # On Windows/Python 3.12 st_ctime is creation time, not metadata-change
            # time. Rehash there: restoring mtime must not hide a replacement.
            if os.name != "nt" and row and row[0] == before:
                value = row[1]
            else:
                value = file_sha256(path)
                if signature(path) != before:
                    raise ValueError("Checkpoint changed while it was being hashed")
                changes.append((key, before, value))
            result[relative.as_posix()] = value
        with db:
            db.executemany("INSERT OR REPLACE INTO hashes VALUES (?,?,?)", changes)
    return result
