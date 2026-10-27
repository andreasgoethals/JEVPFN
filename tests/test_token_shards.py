import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_token_audit import inputs as _inputs

from src.cluster import token_audit as audit
from src.cluster import token_shards as shards
from src.utils import paths
from src.utils.serialization import file_sha256, write_json

inputs = _inputs


@pytest.fixture
def counter():
    calls = []

    def count(payloads):
        calls.extend(payloads)
        return [len(p["state"]["text"]["text"]) for p in payloads]

    return SimpleNamespace(
        identity={"counter_id": "fixture"}, limit=4, count_many=count, calls=calls
    )


def parts(inputs, counter, count=10):
    return [
        audit.audit_provider("liquid", inputs, counter=counter, shard=(i, count), batch_size=1)
        for i in range(count)
    ]


def test_ten_parts_cover_even_tiny_dataset_and_merge_matches_serial(inputs, counter):
    reports = parts(inputs, counter)
    assert len(counter.calls) == 3
    result = shards.merge("liquid", inputs, counter=counter)
    assert result["unique_inputs"] == 3 and result["total_input_tokens"] == 15
    assert result["overflow_unique_inputs"] == 2
    assert len(counter.calls) == 3  # Merge must never invoke inference/tokenization.
    before = file_sha256(Path(result["counts_database"]))
    shards.merge("liquid", inputs, counter=counter)
    assert len(counter.calls) == 3
    assert file_sha256(Path(result["counts_database"])) == before
    assert len(reports) == 10


def test_cancelled_monolithic_batches_reused_without_recounting(inputs, counter):
    real = counter.count_many

    def fail(payloads):
        if len(counter.calls) == 2:
            raise RuntimeError("interruption")
        return real(payloads)

    counter.count_many = fail
    with pytest.raises(RuntimeError):
        audit.audit_provider("liquid", inputs, counter=counter, batch_size=2)
    counter.count_many = real
    parts(inputs, counter)
    assert len(counter.calls) == 3
    shards.merge("liquid", inputs, counter=counter)
    assert len(counter.calls) == 3


def test_interrupted_partition_resumes_nonzero_range(inputs, counter):
    real = counter.count_many

    def fail(payloads):
        if len(counter.calls) == 1:
            raise RuntimeError("interruption")
        return real(payloads)

    counter.count_many = fail
    with pytest.raises(RuntimeError):
        audit.audit_provider("liquid", inputs, counter=counter, shard=(1, 2), batch_size=1)
    counter.count_many = real
    audit.audit_provider("liquid", inputs, counter=counter, shard=(1, 2), batch_size=1)
    assert len(counter.calls) == 2
    assert [p["state"]["text"]["text"] for p in counter.calls] == ["longer", "third"]


@pytest.mark.parametrize("damage", ["missing", "hash", "identity", "coverage", "assignment"])
def test_merge_fails_closed_and_rolls_back_all_inserts(inputs, counter, damage):
    reports = parts(inputs, counter)
    path = Path(reports[-1]["database"])
    if damage == "missing":
        shards.receipt_path(path).unlink()
    else:
        with sqlite3.connect(path) as db:
            if damage in {"hash", "identity"}:
                db.execute("UPDATE identity SET recipe_json='{}'")
            elif damage == "coverage":
                db.execute("UPDATE counts SET payload_sha256='wrong'")
            else:
                db.execute("UPDATE partition SET part=0")
        if damage != "hash":
            report = json.loads(shards.receipt_path(path).read_text())
            report["database_sha256"] = file_sha256(path)
            write_json(shards.receipt_path(path), report)
    with pytest.raises((ValueError, FileNotFoundError)):
        shards.merge("liquid", inputs, counter=counter)
    destination = next(paths.token_audit_dir().glob("liquid_*.sqlite3"))
    with sqlite3.connect(destination) as db:
        assert db.execute("SELECT COUNT(*) FROM counts").fetchone() == (0,)


def test_conflicting_existing_count_is_never_overwritten(inputs, counter):
    parts(inputs, counter)
    result = audit.audit_provider("liquid", inputs, counter=counter)
    with sqlite3.connect(result["counts_database"]) as db:
        db.execute("UPDATE counts SET input_tokens=999 WHERE input_no=2")
    with pytest.raises(ValueError, match="Conflicting"):
        shards.merge("liquid", inputs, counter=counter)
    with sqlite3.connect(result["counts_database"]) as db:
        assert db.execute("SELECT input_tokens FROM counts WHERE input_no=2").fetchone() == (999,)


@pytest.mark.parametrize("index,count", [(-1, 10), (10, 10), (0, 0), (True, 10), (0, 101)])
def test_invalid_partitions_refused(index, count):
    with pytest.raises(ValueError):
        shards.validate_shard(index, count)


def test_all_datasets_partitioned_once_without_gaps():
    for n in (0, 1, 9, 10, 11, 103, 1001):
        numbers = [j for i in range(10) for j in range(n * i // 10, n * (i + 1) // 10)]
        assert numbers == list(range(n))


def test_parallel_plan_keeps_merge_as_gpu_prerequisite():
    from src.cluster.setup_extractors import plan

    steps = {s["name"]: s for s in plan(["liquid"], reuse=["liquid"])}
    assert len(steps["liquid_tokens"]["dependencies"]) == 10
    assert "liquid_tokens" in steps["liquid_scratch"]["dependencies"]
    assert all(s["pool"] == "tokens" for s in steps.values() if "_tokens_part" in s["name"])


def test_cancelled_sqlite_transaction_is_recovered_before_readers(inputs, counter):
    import subprocess
    import sys

    result = audit.audit_provider("liquid", inputs, counter=counter)
    destination = result["counts_database"]
    code = """import sqlite3,sys,os
db=sqlite3.connect(sys.argv[1])
db.execute('PRAGMA cache_size=1')
db.execute('BEGIN IMMEDIATE')
db.execute('UPDATE counts SET input_tokens=999')
db.execute('CREATE TABLE crash_pages(value BLOB)')
db.executemany('INSERT INTO crash_pages VALUES (?)',[(b'x'*4096,)]*100)
os._exit(0)
"""
    subprocess.run([sys.executable, "-c", code, destination], check=True)
    assert Path(destination + "-journal").exists()
    report = shards.prepare("liquid", inputs, counter=counter)
    assert report["reusable_committed_inputs"] == 3
    with sqlite3.connect(destination) as db:
        assert db.execute("SELECT SUM(input_tokens) FROM counts").fetchone() == (15,)
        assert (
            db.execute("SELECT name FROM sqlite_master WHERE name='crash_pages'").fetchone() is None
        )


@pytest.fixture
def prepared_counter():
    import zlib

    from src.utils.serialization import digest

    class Counter:
        identity = {"counter_id": "prepared-fixture"}
        limit = 100

        def attach_cache(self, db, *, writable):
            self.db = db
            db.execute(
                "CREATE TABLE IF NOT EXISTS prepared_tokens (payload_sha256 TEXT PRIMARY KEY, prepared_sha256 TEXT, content BLOB)"
            )
            db.commit()

        def count_many(self, payloads):
            lengths = []
            for p in payloads:
                value = {"ids": list(range(len(p["state"]["text"]["text"])))}
                self.db.execute(
                    "INSERT OR IGNORE INTO prepared_tokens VALUES (?,?,?)",
                    (digest(p), digest(value), zlib.compress(json.dumps(value).encode())),
                )
                lengths.append(len(value["ids"]))
            return lengths

    return Counter()


def test_prepared_tokens_survive_partition_reuse_and_merge(inputs, prepared_counter):
    c = prepared_counter
    for i in range(3):
        audit.audit_provider("h2o", inputs, counter=c, shard=(i, 3))
    result = shards.merge("h2o", inputs, counter=c, shards=3)
    with sqlite3.connect(result["counts_database"]) as db:
        assert db.execute("SELECT COUNT(*) FROM prepared_tokens").fetchone() == (3,)
        assert db.execute(
            "SELECT name FROM sqlite_master WHERE name='prepared_payload_keys'"
        ).fetchone() == ("prepared_payload_keys",)


@pytest.mark.parametrize("damage", ["missing", "checksum", "length"])
def test_damaged_prepared_source_rolls_back_every_partition(inputs, prepared_counter, damage):
    import zlib

    from src.utils.serialization import digest

    reports = [
        audit.audit_provider("h2o", inputs, counter=prepared_counter, shard=(i, 3))
        for i in range(3)
    ]
    damaged = Path(reports[-1]["database"])
    with sqlite3.connect(damaged) as db:
        if damage == "missing":
            db.execute("DELETE FROM prepared_tokens")
        elif damage == "checksum":
            db.execute("UPDATE prepared_tokens SET prepared_sha256='wrong'")
        else:
            value = {"ids": [1]}
            db.execute(
                "UPDATE prepared_tokens SET content=?,prepared_sha256=?",
                (zlib.compress(json.dumps(value).encode()), digest(value)),
            )
    report = reports[-1]
    report["database_sha256"] = file_sha256(damaged)
    write_json(shards.receipt_path(damaged), report)
    with pytest.raises(ValueError, match="prepared|Prepared"):
        shards.merge("h2o", inputs, counter=prepared_counter, shards=3)
    destination = next(paths.token_audit_dir().glob("h2o_*.sqlite3"))
    with sqlite3.connect(destination) as db:
        assert db.execute("SELECT COUNT(*) FROM counts").fetchone() == (0,)
        assert db.execute("SELECT COUNT(*) FROM prepared_tokens").fetchone() == (0,)


@pytest.mark.parametrize("damage", ["conflict", "corruption"])
def test_existing_prepared_content_is_checked_and_preserved(inputs, prepared_counter, damage):
    reports = [
        audit.audit_provider("h2o", inputs, counter=prepared_counter, shard=(i, 3))
        for i in range(3)
    ]
    result = shards.merge("h2o", inputs, counter=prepared_counter, shards=3)
    destination = Path(result["counts_database"])
    with sqlite3.connect(destination) as db:
        if damage == "conflict":
            db.execute("UPDATE prepared_tokens SET prepared_sha256='different'")
        else:
            import zlib

            db.execute("UPDATE prepared_tokens SET content=?", (zlib.compress(b"{}"),))
    before = file_sha256(destination)
    with pytest.raises(ValueError, match="prepared|Prepared"):
        shards.merge("h2o", inputs, counter=prepared_counter, shards=3)
    assert file_sha256(destination) == before
    assert len(reports) == 3


def test_finished_partition_resume_does_not_recopy_old_blob_cache(
    inputs, prepared_counter, monkeypatch
):
    for i in range(3):
        audit.audit_provider("h2o", inputs, counter=prepared_counter, shard=(i, 3))
    shards.merge("h2o", inputs, counter=prepared_counter, shards=3)

    def recopy(*a, **k):
        raise AssertionError("A finished partition should not recopy the monolithic cache")

    monkeypatch.setattr(shards, "copy_rows", recopy)
    report = audit.audit_provider("h2o", inputs, counter=prepared_counter, shard=(1, 3))
    assert report["status"] == "partition_complete_requires_merge"


def test_finished_partition_cannot_be_reused_after_its_database_changes(inputs, counter):
    report = audit.audit_provider("liquid", inputs, counter=counter, shard=(0, 1))
    with sqlite3.connect(report["database"]) as db:
        db.execute("UPDATE counts SET input_tokens=999 WHERE input_no=0")
    with pytest.raises(ValueError, match="Completed token partition changed"):
        audit.audit_provider("liquid", inputs, counter=counter, shard=(0, 1))


def test_blob_merge_uses_bounded_batches_and_sublinear_database_queries():
    import zlib

    from src.utils.serialization import digest

    with sqlite3.connect(":memory:") as source, sqlite3.connect(":memory:") as target:
        for db in (source, target):
            db.execute(
                "CREATE TABLE prepared_tokens (payload_sha256 TEXT PRIMARY KEY, prepared_sha256 TEXT, content BLOB) WITHOUT ROWID"
            )
        required, values = {}, []
        for i in range(1025):
            key = digest({"row": i})
            value = {"ids": [i] * 20, "extra": "Unicode: éλ"}
            required[key] = 20
            values.append((key, digest(value), zlib.compress(json.dumps(value).encode())))
        source.executemany("INSERT INTO prepared_tokens VALUES (?,?,?)", values)
        queries = []
        target.set_trace_callback(queries.append)
        shards.copy_prepared(source, target, required, progress=lambda text: None)
        assert target.execute("SELECT COUNT(*) FROM prepared_tokens").fetchone() == (1025,)
        assert len([q for q in queries if q.startswith("SELECT")]) < len(values) // 10
        assert list(
            source.execute("SELECT * FROM prepared_tokens ORDER BY payload_sha256")
        ) == list(target.execute("SELECT * FROM prepared_tokens ORDER BY payload_sha256"))


def test_added_merge_has_time_for_large_saved_token_arrays():
    for provider in ("h2o", "sii", "kev", "decision3", "nimble"):
        step = shards.plan(provider, "/project")[-1]
        assert "--time=24:00:00" in step["flags"]
        assert len(step["dependencies"]) == 10
