"""Ten independent CPU token partitions and a strict, transactional merge.

Each partition takes a disjoint contiguous fraction of every dataset. Publisher
rendering, input hashes and the final audit identity are unchanged. Existing
monolithic counts are read under a shared lock; no worker writes that database.
"""

import argparse
import json
import os
import sqlite3
import time
import zlib
from contextlib import closing
from itertools import islice
from pathlib import Path

from src.utils import paths
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import canonical_json, digest, file_sha256, write_json


def validate_shard(index, count):
    if (
        type(count) is not int
        or not 1 <= count <= 100
        or type(index) is not int
        or not 0 <= index < count
    ):
        raise ValueError("Shard count must be 1..100 and index must be 0..count-1")


def shard_path(original, index, count):
    validate_shard(index, count)
    return original.parent / "shards" / f"{original.stem}_part{index:02d}of{count:02d}.sqlite3"


def receipt_path(path):
    return (
        paths.manifests_dir("experiment_0")
        / "before_extraction"
        / "token_shards"
        / (path.stem + ".json")
    )


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=60)


def identity(db, recipe):
    row = db.execute("SELECT recipe_json FROM identity WHERE id=1").fetchone()
    if row != (canonical_json(recipe),):
        raise ValueError("Token partition input/tokenizer identity mismatch")


def has_prepared(db):
    return bool(db.execute("SELECT 1 FROM sqlite_master WHERE name='prepared_tokens'").fetchone())


def copy_counts(source, target, dataset, start, stop):
    """Compare existing counts in memory, then insert one dataset range in bulk."""
    rows = source.execute(
        "SELECT * FROM counts WHERE dataset_id=? AND input_no>=? AND input_no<? ORDER BY input_no",
        (dataset, start, stop),
    ).fetchall()
    existing = dict(
        (row[1], row)
        for row in target.execute(
            "SELECT * FROM counts WHERE dataset_id=? AND input_no>=? AND input_no<?",
            (dataset, start, stop),
        )
    )
    for row in rows:
        if row[1] in existing and existing[row[1]] != row:
            raise ValueError("Conflicting token counts; preserved the original database")
    target.executemany("INSERT OR IGNORE INTO counts VALUES (?,?,?,?)", rows)
    return rows


def copy_prepared(source, target, required, *, progress=print):
    """Scan compressed blobs in key order, avoiding random per-input file reads.

    A partition's count map is small; token arrays remain in bounded batches.
    Both incoming and already saved blobs retain checksum and token-count checks.
    """
    required = dict(required)
    total, copied, last = len(required), 0, time.monotonic()
    progress(f"prepared tokens: validating {total:,} inputs in key order")
    if not required:
        return
    records = iter(
        source.execute(
            "SELECT payload_sha256,prepared_sha256,content FROM prepared_tokens ORDER BY payload_sha256"
        )
    )
    while batch := list(islice(records, 256)):
        selected = [row for row in batch if row[0] in required]
        if not selected:
            continue
        placeholders = ",".join("?" for _ in selected)
        existing = {
            row[0]: row[1:]
            for row in target.execute(
                "SELECT payload_sha256,prepared_sha256,content FROM prepared_tokens "
                f"WHERE payload_sha256 IN ({placeholders})",
                [row[0] for row in selected],
            )
        }
        for key, checksum, content in selected:
            value = json.loads(zlib.decompress(content))
            if digest(value) != checksum or len(value["ids"]) != required[key]:
                raise ValueError("Prepared-token checksum or count mismatch")
            if key in existing:
                old_checksum, old_content = existing[key]
                if old_checksum != checksum:
                    raise ValueError("Conflicting prepared tokens")
                if digest(json.loads(zlib.decompress(old_content))) != old_checksum:
                    raise ValueError("Existing prepared-token checksum mismatch")
            del required[key]
        target.executemany("INSERT OR IGNORE INTO prepared_tokens VALUES (?,?,?)", selected)
        copied += len(selected)
        now = time.monotonic()
        if now - last >= 30 or copied == total:
            progress(f"prepared tokens: {copied:,}/{total:,} verified and copied")
            last = now
    if required:
        raise ValueError("Missing publisher-prepared tokens")


def copy_rows(source, target, dataset, start, stop, *, prepared):
    """Conflicts fail instead of replacing counts or publisher-prepared inputs."""
    for row in source.execute(
        "SELECT dataset_id,input_no,payload_sha256,input_tokens FROM counts "
        "WHERE dataset_id=? AND input_no>=? AND input_no<? ORDER BY input_no",
        (dataset, start, stop),
    ):
        prior = target.execute(
            "SELECT * FROM counts WHERE dataset_id=? AND input_no=?", row[:2]
        ).fetchone()
        if prior is not None and prior != row:
            raise ValueError("Conflicting token counts; preserved the original database")
        target.execute("INSERT OR IGNORE INTO counts VALUES (?,?,?,?)", row)
        if prepared:
            value = source.execute(
                "SELECT * FROM prepared_tokens WHERE payload_sha256=?", (row[2],)
            ).fetchone()
            if value is None:
                raise ValueError("Missing publisher-prepared tokens")
            decoded = json.loads(zlib.decompress(value[2]))
            if digest(decoded) != value[1] or len(decoded["ids"]) != row[3]:
                raise ValueError("Prepared-token checksum or count mismatch")
            old = target.execute(
                "SELECT prepared_sha256,content FROM prepared_tokens WHERE payload_sha256=?",
                (row[2],),
            ).fetchone()
            if old is not None and old[0] != value[1]:
                raise ValueError("Conflicting prepared tokens")
            if old is not None and digest(json.loads(zlib.decompress(old[1]))) != old[0]:
                raise ValueError("Existing prepared-token checksum mismatch")
            target.execute("INSERT OR IGNORE INTO prepared_tokens VALUES (?,?,?)", value)


def seed_existing(db, original, recipe, inputs, index, count):
    """Reuse only committed counts with the exact same recipe; never recount them."""
    db.execute(
        "CREATE TABLE IF NOT EXISTS partition (id INTEGER PRIMARY KEY CHECK(id=1), part INTEGER, total INTEGER)"
    )
    prior = db.execute("SELECT part,total FROM partition WHERE id=1").fetchone()
    if prior is not None and prior != (index, count):
        raise ValueError("Saved partition assignment differs")
    db.execute("INSERT OR IGNORE INTO partition VALUES (1,?,?)", (index, count))
    db.commit()
    destination = Path(db.execute("PRAGMA database_list").fetchone()[2])
    receipt = receipt_path(destination)
    if receipt.is_file():
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        if (
            saved.get("status") == "partition_complete_requires_merge"
            and saved.get("audit_id") == digest(recipe)
            and saved.get("index") == index
            and saved.get("shards") == count
        ):
            if saved.get("database_sha256") != file_sha256(destination):
                raise ValueError("Completed token partition changed; review before reuse")
            # The strict merge still checks every input and original-count conflict.
            # Avoid random blob rereads/copies from an older monolithic prefix here.
            return
    if not original.is_file():
        return
    with (
        file_lock(original.with_suffix(".lock"), shared=True, timeout=1),
        closing(readonly(original)) as old,
        db,
    ):
        identity(old, recipe)
        prepared = has_prepared(db)
        if prepared and not has_prepared(old):
            raise ValueError("Existing token counts lack prepared tokens; review before reuse")
        for item in inputs:
            n = item["unique_inputs"]
            copy_rows(
                old,
                db,
                item["dataset_id"],
                n * index // count,
                n * (index + 1) // count,
                prepared=prepared,
            )


def finish_shard(destination, recipe, shard, seconds):
    report = dict(
        status="partition_complete_requires_merge",
        audit_id=digest(recipe),
        index=shard[0],
        shards=shard[1],
        database=str(destination),
        database_sha256=file_sha256(destination),
        seconds=seconds,
    )
    write_json(receipt_path(destination), report)
    return report


def prepare(provider, inputs=None, *, counter=None):
    """Recover a cancelled writer's hot SQLite journal before parallel readers start."""
    from src.cluster.token_audit import selected_inputs
    from src.jev.catalog import ADDED, ExactCounter
    from src.jev.registry import require_prepared_coverage

    counter = ExactCounter(provider) if counter is None else counter
    inputs = selected_inputs() if inputs is None else inputs
    recipe = dict(
        schema="jevpfn.exact_tokens.v1",
        provider=provider,
        counter=counter.identity,
        inputs=[{k: v for k, v in i.items() if k != "path"} for i in inputs],
    )
    destination = (
        paths.require_project_output(paths.token_audit_dir())
        / f"{provider}_{digest(recipe)}.sqlite3"
    )
    reused = 0
    if destination.is_file():
        with (
            file_lock(destination.with_suffix(".lock"), timeout=1),
            closing(sqlite3.connect(destination)) as db,
        ):
            # A writable connection recovers a hot rollback journal automatically.
            # Do not remove journals manually or let parallel read-only jobs recover it.
            identity(db, recipe)
            if db.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ValueError("Existing token database failed integrity verification")
            for item in inputs:
                n, first, last = db.execute(
                    "SELECT COUNT(*),MIN(input_no),MAX(input_no) FROM counts WHERE dataset_id=?",
                    (item["dataset_id"],),
                ).fetchone()
                if (n and (first != 0 or last != n - 1)) or n > item["unique_inputs"]:
                    raise ValueError("Existing monolithic audit is not a resumable prefix")
                reused += n
            if db.execute("SELECT COUNT(*) FROM counts").fetchone()[0] != reused:
                raise ValueError("Existing token database contains an unknown dataset")
            if provider in ADDED:
                require_prepared_coverage(db)
    return dict(
        status="ready_for_parallel_counting",
        provider=provider,
        audit_id=digest(recipe),
        reusable_committed_inputs=reused,
    )


def merge(provider, inputs=None, *, counter=None, shards=10):
    from src.cluster.token_audit import audit_provider, selected_inputs
    from src.jev.catalog import ADDED, ExactCounter

    validate_shard(0, shards)
    counter = ExactCounter(provider) if counter is None else counter
    inputs = selected_inputs() if inputs is None else inputs
    recipe = dict(
        schema="jevpfn.exact_tokens.v1",
        provider=provider,
        counter=counter.identity,
        inputs=[{k: v for k, v in i.items() if k != "path"} for i in inputs],
    )
    audit_id = digest(recipe)
    destination = (
        paths.require_project_output(paths.token_audit_dir()) / f"{provider}_{audit_id}.sqlite3"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        file_lock(destination.with_suffix(".lock"), timeout=1),
        closing(sqlite3.connect(destination)) as db,
    ):
        db.executescript("""
            CREATE TABLE IF NOT EXISTS identity (id INTEGER PRIMARY KEY CHECK(id=1), recipe_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS counts (dataset_id TEXT NOT NULL, input_no INTEGER NOT NULL,
                payload_sha256 TEXT NOT NULL,input_tokens INTEGER NOT NULL CHECK(input_tokens>0),
                PRIMARY KEY(dataset_id,input_no)) WITHOUT ROWID;
        """)
        db.execute("INSERT OR IGNORE INTO identity VALUES (1,?)", (canonical_json(recipe),))
        db.commit()
        identity(db, recipe)
        prepared = provider in ADDED
        if prepared:
            db.execute(
                "CREATE TABLE IF NOT EXISTS prepared_tokens (payload_sha256 TEXT PRIMARY KEY, prepared_sha256 TEXT NOT NULL, content BLOB NOT NULL) WITHOUT ROWID"
            )
            # WITHOUT ROWID stores token blobs in the primary B-tree. A compact
            # covering index lets the final completeness join read keys alone.
            db.execute(
                "CREATE INDEX IF NOT EXISTS prepared_payload_keys ON prepared_tokens(payload_sha256)"
            )
            db.commit()
        db.execute("PRAGMA cache_size=-262144")  # Bounded 256 MiB, not the SQLite 2 MiB default.
        # One transaction: any missing partition, identity/hash conflict or incomplete
        # range rolls back every merge insert, preserving the old resumable prefix.
        with db:
            changes_before_merge = db.total_changes
            for index in range(shards):
                part = shard_path(destination, index, shards)
                part_started = time.monotonic()
                print(
                    f"{provider}: checking partition {index + 1}/{shards} checksum and coverage",
                    flush=True,
                )
                with (
                    file_lock(part.with_suffix(".lock"), timeout=1),
                    closing(readonly(part)) as source,
                ):
                    report = json.loads(receipt_path(part).read_text(encoding="utf-8"))
                    if (
                        report.get("status") != "partition_complete_requires_merge"
                        or report.get("audit_id") != audit_id
                        or report.get("index") != index
                        or report.get("shards") != shards
                        or report.get("database_sha256") != file_sha256(part)
                    ):
                        raise ValueError("Missing, stale or altered token partition receipt")
                    identity(source, recipe)
                    if source.execute("SELECT part,total FROM partition WHERE id=1").fetchone() != (
                        index,
                        shards,
                    ):
                        raise ValueError("Token partition assignment mismatch")
                    expected_total = 0
                    required = {}
                    for item in inputs:
                        n = item["unique_inputs"]
                        start, stop = n * index // shards, n * (index + 1) // shards
                        expected_total += stop - start
                        with closing(readonly(item["path"])) as bank:
                            expected = bank.execute(
                                "SELECT input_no,payload_sha256 FROM unique_inputs WHERE input_set_id=? AND input_no>=? AND input_no<? ORDER BY input_no",
                                (item["input_set_id"], start, stop),
                            )
                            actual = source.execute(
                                "SELECT input_no,payload_sha256 FROM counts WHERE dataset_id=? AND input_no>=? AND input_no<? ORDER BY input_no",
                                (item["dataset_id"], start, stop),
                            )
                            for offset in range(stop - start):
                                row = next(actual, None)
                                if (
                                    row is None
                                    or row != next(expected, None)
                                    or row[0] != start + offset
                                ):
                                    raise ValueError(
                                        "Incomplete or mismatched token partition coverage"
                                    )
                        rows = copy_counts(source, db, item["dataset_id"], start, stop)
                        if prepared:
                            for _, _, key, count in rows:
                                if key in required and required[key] != count:
                                    raise ValueError(
                                        "Conflicting counts for the same prepared input"
                                    )
                                required[key] = count
                    if (
                        source.execute("SELECT COUNT(*) FROM counts").fetchone()[0]
                        != expected_total
                    ):
                        raise ValueError("Token partition contains out-of-range inputs")
                    if prepared:
                        copy_prepared(
                            source,
                            db,
                            required,
                            progress=lambda message, part_no=index + 1: print(
                                f"{provider} partition {part_no}/{shards}: {message}", flush=True
                            ),
                        )
                print(
                    f"{provider}: verified partition {index + 1}/{shards}; "
                    f"{expected_total:,} inputs; {time.monotonic() - part_started:.1f}s",
                    flush=True,
                )
            if db.execute("SELECT COUNT(*) FROM counts").fetchone()[0] != sum(
                i["unique_inputs"] for i in inputs
            ):
                raise ValueError("Merged token audit has unexpected inputs")
            if prepared:
                statistics = (
                    db.execute("SELECT 1 FROM sqlite_master WHERE name='sqlite_stat1'").fetchone()
                    and db.execute(
                        "SELECT 1 FROM sqlite_stat1 WHERE tbl='prepared_tokens' AND idx='prepared_payload_keys'"
                    ).fetchone()
                )
                if db.total_changes != changes_before_merge or not statistics:
                    print(f"{provider}: analyzing the compact coverage index", flush=True)
                    db.execute("ANALYZE prepared_tokens")
    # Existing summary/overflow logic and GPU consumers retain precisely the same schema.
    # All counts already exist; this does not call the tokenizer again.
    print(f"{provider}: merge committed; producing final dataset/context summaries", flush=True)
    return audit_provider(provider, inputs, counter=counter)


def plan(provider, root, *, dependencies=()):
    from src.cluster.token_preparation import resources
    from src.jev.catalog import ADDED

    count = load_yaml("experiment_0/scheduling").get("token_audit_shards", 10)
    validate_shard(0, count)
    preparation = dict(
        name=provider + "_tokens_prepare",
        dependencies=list(dependencies),
        script="experiment_0_tokens.slurm",
        args=[root, provider, "prepare", str(count)],
        flags=["--cpus-per-task=1", "--mem=6800M", "--time=02:00:00"],
        cluster="wice",
    )
    parts = [
        dict(
            name=f"{provider}_tokens_part{i:02d}",
            dependencies=[preparation["name"]],
            script="experiment_0_tokens.slurm",
            args=[root, provider, str(i), str(count)],
            flags=resources(provider, sharded=True),
            cluster="wice",
            pool="tokens",
        )
        for i in range(count)
    ]
    return [
        preparation,
        *parts,
        dict(
            name=provider + "_tokens",
            dependencies=[p["name"] for p in parts],
            script="experiment_0_tokens.slurm",
            args=[root, provider, "merge", str(count)],
            flags=[
                "--cpus-per-task=1",
                "--mem=6800M",
                "--time=24:00:00" if provider in ADDED else "--time=04:00:00",
            ],
            cluster="wice",
        ),
    ]


def main():
    from src.cluster.preflight import storage_policy_failures
    from src.cluster.token_audit import audit_provider, save_reports
    from src.jev.catalog import ADDED, SELF_HOSTED

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=SELF_HOSTED, required=True)
    parser.add_argument("--shards", type=int, default=10)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--index", type=int)
    action.add_argument("--merge", action="store_true")
    action.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if paths.on_vsc() and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Token partitions and merge require a CPU allocation")
    if failures := storage_policy_failures(os.environ.get("SLURM_CLUSTER_NAME")):
        raise ValueError("; ".join(failures))
    if args.prepare:
        report = prepare(args.provider)
    elif args.merge:
        report = merge(args.provider, shards=args.shards)
        save_reports([report])
    else:
        workers = (
            min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))) if args.provider in ADDED else 1
        )
        report = audit_provider(
            args.provider, preparation_workers=workers, shard=(args.index, args.shards)
        )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
