import json
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import pytest
from test_cache_merge import cache, feature, request, response

from src.cluster.closed_publication import union_closed_stores
from src.utils.serialization import file_sha256


def test_closed_union_preserves_providers_artifacts_recovery_and_reuses_rows(tmp_path):
    node = tmp_path / "node"
    node.mkdir()
    destination = tmp_path / "project/fixture.sqlite3"
    cache(destination)
    first = cache(tmp_path / "first.sqlite3", request("intern"))
    second = cache(tmp_path / "second.sqlite3", request("cloudflare"))
    feature(first)
    with first._connect() as db:
        db.execute("CREATE TABLE extraction_jobs (request_key TEXT PRIMARY KEY,status TEXT)")
        db.execute("INSERT INTO extraction_jobs VALUES ('uncertain','attempting')")
    result = union_closed_stores([first.path, second.path], destination, node, writers_stopped=True)
    for provider in ("jev", "intern", "cloudflare"):
        assert cache(destination).get(request(provider), kind="verified_response") == response()
    with closing(sqlite3.connect(destination)) as db:
        assert db.execute("SELECT COUNT(*) FROM feature_artifacts").fetchone() == (1,)
        assert any(json.loads(r[0]).get("status") == "attempting"
                   for r in db.execute("SELECT record_json FROM imported_extraction_records"))
    again = union_closed_stores([first.path, second.path], destination, node, writers_stopped=True)
    assert all(s["union"]["responses_added"] == 0 for s in again["sources"])
    assert result["protocol"] == "exclusive_sqlite_checkpoints_closed_union_atomic_replace"
    assert first.path.is_file() and second.path.is_file() and not list(node.iterdir())


def test_closed_union_conflict_keeps_durable_and_original_sources(tmp_path):
    destination = cache(tmp_path / "destination.sqlite3", value=0.7)
    incoming = cache(tmp_path / "incoming.sqlite3", value=0.9)
    before = file_sha256(destination.path)
    with pytest.raises(ValueError, match="Conflicting"):
        union_closed_stores([incoming.path], destination.path, tmp_path, writers_stopped=True)
    assert file_sha256(destination.path) == before
    assert incoming.path.is_file() and not list(tmp_path.glob(".migration-union-*"))


def test_closed_union_refuses_an_actual_sqlite_writer_even_if_flag_is_set(tmp_path):
    destination = cache(tmp_path / "destination.sqlite3")
    incoming = cache(tmp_path / "incoming.sqlite3", request("intern"))
    before = file_sha256(destination.path)
    with closing(sqlite3.connect(incoming.path)) as writer:
        writer.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            union_closed_stores([incoming.path], destination.path, tmp_path, writers_stopped=True)
        writer.rollback()
    assert file_sha256(destination.path) == before
    with pytest.raises(ValueError, match="stopped"):
        union_closed_stores([incoming.path], destination.path, tmp_path)


def test_new_consumer_blocks_replacement_without_losing_either_store(tmp_path):
    destination = cache(tmp_path / "destination.sqlite3")
    incoming = cache(tmp_path / "incoming.sqlite3", request("intern"))
    before = file_sha256(destination.path)

    def queued_consumer():
        raise RuntimeError("Queued consumer appeared")

    with pytest.raises(RuntimeError, match="Queued"):
        union_closed_stores([incoming.path], destination.path, tmp_path,
                            writers_stopped=True, consumer_check=queued_consumer)
    assert file_sha256(destination.path) == before and incoming.path.is_file()


def test_native_allocated_project_closed_union_and_resume():
    context = os.environ.get("JEVPFN_MIGRATION_QA_CONTEXT")
    if not context:
        pytest.skip("Explicit native GPFS PROJECT/Slurm integration only")
    context = json.loads(context)
    assert context["cluster"] == "mindwell"
    target = Path(context["retained_root"]) / ("closed_publication_" + context["job_id"])
    target.mkdir(parents=True, exist_ok=False)
    with (tempfile.TemporaryDirectory(prefix="jevpfn-publication-qa-", dir=context["gpfs_scratch"]) as source,
          tempfile.TemporaryDirectory(prefix="jevpfn-publication-qa-", dir=context["node_scratch"]) as node):
        sources = [cache(Path(source) / (p + ".sqlite3"), request(p)).path for p in ("intern", "cloudflare")]
        destination = target / "fixture.sqlite3"
        cache(destination)
        first = union_closed_stores(sources, destination, Path(node), writers_stopped=True)
        second = union_closed_stores(sources, destination, Path(node), writers_stopped=True)
        assert all(s["union"]["responses_added"] == 0 for s in second["sources"])
        for p in ("jev", "intern", "cloudflare"):
            assert cache(destination).get(request(p), kind="verified_response") == response()
    (target / "passed.json").write_text(json.dumps({"status": "native_closed_union_and_resume_passed",
        "job_id": context["job_id"], "first": first, "resume": second}, indent=2))
