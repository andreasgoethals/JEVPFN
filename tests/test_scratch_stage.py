"""B200 staging preserves durable caches, boundaries and SQLite WAL commits."""

import json
import sqlite3
from pathlib import Path

import pytest
from test_cache_merge import cache, request, response

from src.cluster import scratch_stage as stage
from src.utils import paths
from src.utils.serialization import file_sha256


@pytest.fixture
def roots(tmp_path, monkeypatch):
    durable, scratch = tmp_path / "lustre", tmp_path / "gpfs"
    durable.mkdir()
    scratch.mkdir()
    monkeypatch.delenv("VSC_DATA", raising=False)
    monkeypatch.setenv("JEVPFN_DURABLE_ROOT", str(durable))
    monkeypatch.setenv("VSC_SCRATCH_GPFS1", str(scratch))
    monkeypatch.setenv("JEVPFN_RUNTIME_PROFILE", "blackwell-cu130")
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path / "repo")
    return stage.roots("intern")


def marker(durable, work):
    work.mkdir(parents=True, exist_ok=True)
    (work / stage.MARKER).write_text(
        json.dumps(
            {
                "runtime": "intern",
                "durable_root": str(durable),
                "runtime_profile": "blackwell-cu130",
                "status": "staged_requires_b200_contracts",
                "files": {},
            }
        )
    )


def test_publication_unions_providers_without_overwriting_paid_local_response(roots):
    durable, work = roots
    marker(durable, work)
    destination = durable / "data/jev_cache/fixture.sqlite3"
    original = cache(destination)
    remote = cache(work / "data/jev_cache/fixture.sqlite3", request("intern"))
    assert stage.publish("intern")["decision_stores"] == 1
    assert original.get(request(), kind="verified_response") == response()
    assert original.get(request("intern"), kind="verified_response") == response()
    first = file_sha256(remote.path)
    stage.publish("intern")
    assert file_sha256(remote.path) == first
    assert remote.path.is_file()  # Recovery copy never deleted.


def test_conflict_keeps_both_originals(roots):
    durable, work = roots
    marker(durable, work)
    destination = cache(durable / "data/jev_cache/fixture.sqlite3", value=0.7)
    source = cache(work / "data/jev_cache/fixture.sqlite3", value=0.9)
    with pytest.raises(ValueError):
        stage.publish("intern")
    assert destination.get(request(), kind="verified_response") == response(0.7)
    assert source.get(request(), kind="verified_response") == response(0.9)


def test_periodic_publication_ignores_historical_backups_and_does_not_multiply_files(roots):
    durable, work = roots
    marker(durable, work)
    cache(durable / "data/jev_cache/fixture.sqlite3")
    remote = cache(work / "data/jev_cache/fixture.sqlite3", request("intern"))
    with remote._connect() as db:
        db.execute("CREATE TABLE extraction_jobs (request_key TEXT PRIMARY KEY,status TEXT)")
        db.execute("INSERT INTO extraction_jobs VALUES ('job','completed')")
    for root in (durable, work):
        historical = root / "data/jev_cache/merge_backups/historical.sqlite3"
        historical.parent.mkdir(parents=True)
        historical.write_bytes(b"Historical backup must not be treated as an active dataset")
    before_files = sorted(str(p) for p in durable.rglob("*.sqlite3"))
    for i in range(3):
        remote.put(request("intern", text=str(i)), response(), kind="verified_response")
        assert stage._publish(durable, work, periodic=True)["decision_stores"] == 1
    assert sorted(str(p) for p in durable.rglob("*.sqlite3")) == before_files
    assert (work / "data/jev_cache/merge_backups/historical.sqlite3").exists()
    with sqlite3.connect(durable / "data/jev_cache/fixture.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM imported_extraction_records").fetchone() == (1,)


def test_copy_snapshots_committed_wal_without_copying_open_journal(tmp_path):
    source = tmp_path / "source.sqlite3"
    with sqlite3.connect(source) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE x (value)")
        db.execute("INSERT INTO x VALUES ('committed')")
        db.commit()
        target = tmp_path / "native/copy.sqlite3"
        stage._copy(source, target)
        with sqlite3.connect(target) as copied:
            assert copied.execute("SELECT * FROM x").fetchall() == [("committed",)]


def test_immutable_prepared_inputs_preserve_exact_audited_bytes(tmp_path):
    source = tmp_path / "source.sqlite3"
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE x (value)")
        db.execute("INSERT INTO x VALUES ('committed')")
        db.commit()
    db.close()
    destination = tmp_path / "native/input.sqlite3"
    stage._copy(source, destination, snapshot_sqlite=False)
    assert file_sha256(source) == file_sha256(destination)


def test_runtime_paths_cannot_escape_or_select_other_projects(roots):
    with pytest.raises(ValueError, match="Unknown runtime"):
        stage.roots("../CreditPFN")
    durable, work = roots
    assert work.parent.name == "intern"
    assert work.name == durable.name == "JEVPFN"


def test_wrong_publication_destination_refused(roots):
    durable, work = roots
    marker(durable / "wrong", work)
    with pytest.raises(ValueError, match="destination"):
        stage.publish("intern")


def test_restage_cannot_publish_to_a_different_durable_root(roots):
    durable, work = roots
    marker(durable / "wrong", work)
    source = cache(work / "data/jev_cache/fixture.sqlite3", request("intern"))
    before = file_sha256(source.path)
    with pytest.raises(ValueError, match="publication destination"):
        stage.stage("intern")
    assert file_sha256(source.path) == before
    assert not (durable / "data/jev_cache/fixture.sqlite3").exists()


@pytest.mark.parametrize("action", ["run", "publish"])
def test_runtime_marker_is_checked_after_admitting_workspace_lease(roots, monkeypatch, action):
    from contextlib import contextmanager

    durable, work = roots
    marker(durable, work)
    monkeypatch.setenv("JEVPFN_STAGING_ROOT", str(work.parent))

    @contextmanager
    def changed_before_lock(*args, **kwargs):
        marker(durable / "changed", work)
        yield

    monkeypatch.setattr(stage, "workspace_lease", changed_before_lock)
    with pytest.raises(ValueError, match="identity|destination"):
        if action == "run":
            stage.run("intern", ["must-not-execute"])
        else:
            stage.publish("intern")


def test_stage_requires_blackwell_and_retains_original_queue(roots, monkeypatch):
    durable, work = roots
    monkeypatch.setenv("JEVPFN_RUNTIME_PROFILE", "legacy-cu126")
    with pytest.raises(ValueError, match="blackwell"):
        stage.stage("intern")
    assert not work.exists()


def test_gpu_failure_still_publishes_and_leaves_working_set(roots, monkeypatch):
    import sys

    durable, work = roots
    marker(durable, work)
    source = cache(work / "data/jev_cache/fixture.sqlite3", request("intern"))
    monkeypatch.setenv("JEVPFN_STAGING_ROOT", str(work.parent))
    assert stage.run("intern", [sys.executable, "-c", "raise SystemExit(7)"]) == 7
    assert source.path.is_file()
    assert (durable / "data/jev_cache/fixture.sqlite3").is_file()


def test_symlink_root_rejected(tmp_path):
    outside, linked = tmp_path / "other", tmp_path / "linked"
    outside.mkdir()
    (outside / "sensitive.txt").write_text("private")
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Local account cannot create directory links")
    with pytest.raises(ValueError, match="Linked"):
        list(stage._regular_files(linked))


def test_derived_workspace_link_is_rejected_before_marker_or_lock_writes(roots):
    _, work = roots
    outside = work.parents[3] / "outside"
    outside.mkdir()
    work.parent.mkdir(parents=True)
    try:
        work.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Local account cannot create directory links")
    with pytest.raises(ValueError, match="Linked"):
        stage.roots("intern")
    assert list(outside.iterdir()) == []


def test_cpu_staging_includes_tokenizers_and_adapter_base(roots, monkeypatch):
    from src.cluster import containers
    from src.jev import checkpoints

    durable, _ = roots
    source = durable / "data/decision_inputs/fixture.sqlite3"
    source.parent.mkdir(parents=True)
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE inputs (id)")
    for name in ("nimble", "nimble_base"):
        model = durable / "checkpoints/decision_models" / name
        model.mkdir(parents=True)
        (model / "model.safetensors").write_bytes(b"fixture")
    tokenizer = durable / "checkpoints/tokenizers/nimble"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer.json").write_text("{}")
    image = durable / "containers/blackwell-cu130/nimble"
    image.mkdir(parents=True)
    (image / "_JEVPFN_CONTAINER.json").write_text("{}")
    monkeypatch.setattr(containers, "relocate", lambda *args: None)
    observed = []
    monkeypatch.setattr(
        checkpoints, "verify_checkpoint", lambda p: observed.append(paths.tokenizer_dir(p))
    )
    result = stage.stage("nimble")
    work = Path(result["working_root"])
    assert (work / "checkpoints/decision_models/nimble_base/model.safetensors").exists()
    assert observed == [work / "checkpoints/tokenizers/nimble"]
    assert result["scratch_is_backup"] is False


def test_restage_unions_active_dataset_db_on_native_root_and_preserves_its_queue(
    roots, monkeypatch
):
    from src.cluster import containers
    from src.jev import checkpoints

    durable, work = roots
    source = durable / "data/decision_inputs/fixture.sqlite3"
    source.parent.mkdir(parents=True)
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE inputs (id)")
    image = durable / "containers/blackwell-cu130/intern"
    image.mkdir(parents=True)
    (image / "_JEVPFN_CONTAINER.json").write_text("{}")
    monkeypatch.setattr(containers, "relocate", lambda *args: None)
    monkeypatch.setattr(checkpoints, "verify_checkpoint", lambda *args: None)
    monkeypatch.setenv("JEVPFN_STAGING_ROOT", str(durable.parent))
    original = cache(durable / "data/jev_cache/fixture.sqlite3")
    backup = durable / "data/jev_cache/merge_backups/historical.sqlite3"
    backup.parent.mkdir()
    backup.write_bytes(b"Never stage this backup")
    stage.stage("intern")
    active = cache(work / "data/jev_cache/fixture.sqlite3", request("intern"))
    with active._connect() as db:
        db.execute("CREATE TABLE extraction_policy (id INTEGER PRIMARY KEY, settings_json TEXT)")
        db.execute("INSERT INTO extraction_policy VALUES (1, ?)", ('{"budget": 17}',))
    original.put(request("liquid"), response(), kind="verified_response")
    stage.stage("intern")
    for provider in ("jev", "intern", "liquid"):
        assert active.get(request(provider), kind="verified_response") == response()
    with active._connect() as db:
        assert json.loads(
            db.execute("SELECT settings_json FROM extraction_policy").fetchone()[0]
        ) == {"budget": 17}
    assert not (work / "data/jev_cache/merge_backups").exists()
    assert sorted(p.name for p in backup.parent.iterdir()) == ["historical.sqlite3"]


@pytest.mark.parametrize(
    "phase", ["experiment_0", "experiment_1", "experiment_2", "experiment_3", "experiment_5"]
)
def test_fresh_predictor_workspace_reuses_durable_folds_and_unions_later_results(
    roots, monkeypatch, tmp_path, phase
):
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from src.cluster import containers
    from src.data.loaders import LoadedDataset
    from src.data.metadata import extract_metadata
    from src.evaluation.protocol import evaluate_cv, shared_splits
    from src.evaluation.store import read_predictions, recipes

    durable, _ = roots
    # Real source/config paths, isolated DATA/PROJECT output paths.
    monkeypatch.setattr(paths, "REPO_ROOT", Path(stage.__file__).resolve().parents[2])
    monkeypatch.setenv("VSC_DATA", str(tmp_path / "data"))
    source = durable / "data/decision_inputs/fixture.sqlite3"
    source.parent.mkdir(parents=True)
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE inputs (id)")
    image = durable / "containers/blackwell-cu130/linear"
    image.mkdir(parents=True)
    (image / containers.MARKER).write_text("{}")
    monkeypatch.setattr(containers, "relocate", lambda *args: None)

    frame = pd.DataFrame({"text": ["sample"] * 20, "x": np.arange(20.0), "y": np.arange(20.0)})
    official = {"target": "y", "task_type": "reg"}
    metadata = extract_metadata(
        frame, official, dataset_id="fixture", dataset_name="Fixture", text_columns=["text"]
    )
    dataset = LoadedDataset(frame, metadata, official, {"data_identity": "fixed"}, [])
    matrix = frame[["x"]].set_axis([f"csv:{i}" for i in range(20)])
    options = dict(
        specification=shared_splits(dataset),
        model_id="linear",
        checkpoint_id="synthetic",
        feature_set="non_text",
        allow_model_runs=True,
        estimator_settings={},
        feature_artifact_ids=[],
        phase=phase,
    )

    def factory():
        return SimpleNamespace(fit=lambda x, y: None, predict=lambda x: x.x.to_numpy() + 0.1)

    def no_refit():
        pytest.fail("A durable completed fold must be reused, not fitted again")

    with stage.staging_environment(durable):
        original = evaluate_cv(dataset, matrix, factory, fold_numbers=[0], **options)
    work = Path(stage.stage("linear")["working_root"])
    with stage.staging_environment(work):
        reused = evaluate_cv(dataset, matrix, no_refit, fold_numbers=[0], **options)
        assert [r.key for r in reused] == [r.key for r in original]
        pd.testing.assert_frame_equal(read_predictions(original[0]), read_predictions(reused[0]))
        evaluate_cv(dataset, matrix, factory, fold_numbers=[1], **options)
    # Another completed durable fold must be unioned, not replace scratch progress.
    with stage.staging_environment(durable):
        evaluate_cv(dataset, matrix, factory, fold_numbers=[2], **options)
    stage.stage("linear")
    with stage.staging_environment(work):
        evaluate_cv(dataset, matrix, no_refit, fold_numbers=[0, 1, 2], **options)
    stage.publish("linear")
    with stage.staging_environment(durable):
        assert len(list(recipes(phase))) == 3
    assert len(list((work / "output_JEVPFN").glob("*/results/evaluation/*.sqlite3"))) == 1
