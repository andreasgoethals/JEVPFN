"""Two-tier export/local import, corruption rejection and repeated downloads."""

import json
import zipfile

import pytest
from test_cache_merge import cache, request, response

from src.cluster.transfer import export_bundle, import_bundle
from src.utils import paths
from src.utils.serialization import file_sha256


def test_project_and_data_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("VSC_DATA", str(tmp_path / "DATA"))
    project = tmp_path / "PROJECT"
    project.mkdir()
    monkeypatch.setenv("JEVPFN_STAGING_ROOT", str(project))
    remote = cache(paths.dataset_cache_path("fixture"), request("liquid"))
    log = paths.logs_dir("feature_creation") / "run.log"
    log.parent.mkdir(parents=True)
    log.write_text("finished")
    prediction = paths.results_dir("experiment_1") / "predictions.parquet"
    prediction.parent.mkdir(parents=True)
    prediction.write_bytes(b"synthetic result")
    bundle = export_bundle()
    assert bundle.is_relative_to(project)
    assert file_sha256(remote.path)
    monkeypatch.delenv("VSC_DATA")
    monkeypatch.delenv("JEVPFN_STAGING_ROOT")
    localroot = tmp_path / "LOCAL"
    monkeypatch.setattr(paths, "REPO_ROOT", localroot)
    local = cache(paths.dataset_cache_path("fixture"))
    local_log = paths.logs_dir("feature_creation") / "run.log"
    local_log.parent.mkdir(parents=True)
    local_log.write_text("local log must survive")
    report = import_bundle(bundle)
    assert report["caches"][0]["responses_added"] == 1
    assert local.get(request(), kind="verified_response") == response()
    assert local.get(request("liquid"), kind="verified_response") == response()
    assert local_log.read_text() == "local log must survive"
    assert len(list(local_log.parent.glob("run__vsc_*.log"))) == 1
    assert (
        paths.results_dir("experiment_1") / prediction.name
    ).read_bytes() == prediction.read_bytes()
    again = import_bundle(bundle)
    assert again["caches"][0]["responses_added"] == 0
    assert all(item["status"] == "reused" for item in again["outputs"])


@pytest.mark.parametrize(
    "path,body,sha",
    [
        ("../outside.txt", b"x", "bad"),
        ("output_JEVPFN/3. experiment_0/logs/a.log", b"tampered", "bad"),
        ("output_JEVPFN/C:/secret", b"x", "bad"),
    ],
)
def test_bad_bundle_never_writes_research_files(tmp_path, monkeypatch, path, body, sha):
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    bundle = tmp_path / "bad.zip"
    record = {
        "path": path,
        "member": "files/file",
        "kind": "output",
        "sha256": sha,
        "bytes": len(body),
    }
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps({"schema": "jevpfn.transfer.v1", "bundle_id": "test", "files": [record]}),
        )
        archive.writestr("files/file", body)
    with pytest.raises(ValueError):
        import_bundle(bundle)
    assert not paths.logs_dir("experiment_0").exists()


def test_import_not_allowed_on_vsc(isolated_output, tmp_path):
    with pytest.raises(RuntimeError, match="local"):
        import_bundle(tmp_path / "none.zip")


def test_token_summary_rebases_and_retains_other_provider(tmp_path, monkeypatch):
    from src.cluster.transfer import _combine_token_reports
    from src.utils.serialization import write_json

    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    local, incoming = tmp_path / "local.json", tmp_path / "incoming.json"
    record = {"provider": "liquid", "audit_id": "fixed", "counts_database": "/remote/file.sqlite3"}
    write_json(local, {"self_hosted": [record], "hosted_unverified": {}})
    write_json(
        incoming,
        {
            "self_hosted": [{**record, "provider": "intern", "audit_id": "new"}],
            "hosted_unverified": {},
        },
    )
    _combine_token_reports(incoming, local)
    combined = json.loads(local.read_text())
    assert {r["provider"] for r in combined["self_hosted"]} == {"liquid", "intern"}
    assert all(str(tmp_path) in r["counts_database"] for r in combined["self_hosted"])


def test_interrupted_import_resumes_completed_datasets(tmp_path, monkeypatch):
    from src.cluster import transfer
    from src.jev.cache import ResponseCache

    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path / "remote")
    cache(paths.dataset_cache_path("fixture"))
    other = ResponseCache(paths.dataset_cache_path("second"), "second")
    other.put(request(dataset="second"), response(), kind="verified_response")
    bundle = export_bundle()
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path / "local")
    real_merge = transfer.merge_cache
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return real_merge(*args, **kwargs)

    monkeypatch.setattr(transfer, "merge_cache", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        import_bundle(bundle)
    assert paths.dataset_cache_path("fixture").exists()
    monkeypatch.setattr(transfer, "merge_cache", real_merge)
    report = import_bundle(bundle)
    assert [r["responses_added"] for r in report["caches"]] == [0, 1]
