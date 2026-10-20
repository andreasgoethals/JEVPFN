import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from src.evaluation import store
from src.evaluation.metrics import regression_metrics
from src.evaluation.records import PhaseTimer, complete_fold, completed_fold, save_evaluation
from src.utils import paths
from src.utils.serialization import digest


def save(index):
    recipe = {"dataset_id": "fixture", "task_type": "regression", "outer_fold": index}
    values = pd.DataFrame(
        {"row_id": ["csv:1", "csv:4"], "y_true": [1.0, 2.0], "prediction": [1.1, 2.1]}
    )
    ref = save_evaluation(
        run_id=f"fold-{index}",
        predictions=values,
        timer=PhaseTimer(),
        scores=regression_metrics(values.y_true, values.prediction),
        metadata={
            "dataset_id": "fixture",
            "data_identity": "data",
            "model_id": "fixture",
            "feature_set": "non_text",
            "outer_fold": index,
            "split_id": "fixed",
            "n_train": 8,
            "n_validation": 0,
            "checkpoint_id": "fixed",
            "target_scale": "original",
            "fold_recipe_id": digest(recipe),
            "prediction_role": "outer_test",
        },
    )
    complete_fold(recipe, [ref], "experiment_1")
    return recipe, ref, values


def test_concurrent_folds_one_database_roundtrip_and_corruption(isolated_output):
    with ThreadPoolExecutor(max_workers=3) as pool:
        saved = list(pool.map(save, range(8)))
    root = paths.results_dir(phase="experiment_1")
    assert len(list(root.rglob("*.sqlite3"))) == 1
    assert not list(root.rglob("*.parquet")) and not list(root.rglob("record.json"))
    assert len(list(paths.manifests_dir("experiment_1").rglob("*.json"))) == 1
    for recipe, ref, values in saved:
        assert completed_fold(recipe, "experiment_1") == [ref]
        pd.testing.assert_frame_equal(store.read_predictions(ref), values)
    recipe, ref, _ = saved[0]
    with store.connect(ref.database) as db:
        db.execute("UPDATE records SET predictions=? WHERE id=?", (b"corrupt", ref.key))
    with pytest.raises(ValueError, match="Corrupt"):
        completed_fold(recipe, "experiment_1")


def test_merge_is_idempotent_portable_and_atomic_on_conflict(isolated_output, tmp_path):
    recipe, ref, _ = save(0)
    destination = tmp_path / "fixture.sqlite3"
    assert store.merge(ref.database, destination) == "merged"
    assert store.merge(ref.database, destination) == "merged"
    before = destination.read_bytes()
    with store.connect(ref.database) as db:
        row = json.loads(ref.read_text())
        row["metadata"]["n_train"] = 9
        db.execute("UPDATE records SET body=? WHERE id=?", (json.dumps(row), ref.key))
    with pytest.raises(ValueError, match="Conflicting"):
        store.merge(ref.database, destination)
    assert destination.read_bytes() == before


def test_transfer_import_merges_evaluation_records(isolated_output, tmp_path, monkeypatch):
    from src.cluster.transfer import export_bundle, import_bundle

    recipe, _, values = save(0)
    first = export_bundle()
    save(1)
    second = export_bundle()
    monkeypatch.delenv("VSC_DATA")
    monkeypatch.delenv("JEVPFN_STAGING_ROOT")
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path / "local")
    import_bundle(first)
    import_bundle(second)
    import_bundle(second)
    refs = completed_fold(recipe, "experiment_1")
    pd.testing.assert_frame_equal(store.read_predictions(refs[0]), values)
    assert len(list(store.recipes("experiment_1"))) == 2
    assert len(list(paths.results_dir("evaluation", phase="experiment_1").glob("*.sqlite3"))) == 1


def test_completion_accepts_equivalent_path_alias_but_not_another_store(
    isolated_output, monkeypatch, tmp_path
):
    original = store.database("fixture", "experiment_1")
    detour = original.parent / "detour"
    detour.mkdir(parents=True)
    monkeypatch.setattr(store, "database", lambda *a: detour / ".." / original.name)
    recipe, ref, values = save(3)
    pd.testing.assert_frame_equal(
        store.read_predictions(store.completed(recipe, "experiment_1")[0]), values
    )
    with pytest.raises(ValueError, match="own dataset"):
        store.complete(recipe, [store.RecordRef(tmp_path / original.name, ref.key)], "experiment_1")
