"""Cross-module regressions found during the November qualification review."""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.cluster.preflight import validate_decision_group
from src.evaluation.qualification import verify_arm_recipe
from src.jev import feature_retirement


def test_regression_probability_block_preserves_missing_groups():
    group = pd.DataFrame([[1 / 9] * 9, [np.nan] * 9])
    missing = pd.Series([False, True])
    validate_decision_group(group, missing, 9)
    for invalid in (group.fillna(0), group * 0.5, group.iloc[:, :1]):
        with pytest.raises(ValueError):
            validate_decision_group(invalid, missing, 9)


def test_same_width_pca_profiles_cannot_change_recipe_artifacts_or_runtime():
    arm = dict(dataset_id="wine", model="linear", feature_set="embedding", encoder="e5",
               pca_scope="joint", pca_components=64)
    runtime = {"container_id": "sealed"}
    metadata = dict(dataset_id="wine", model_id="linear", feature_set="embedding",
                    representation_recipe={"embedding": {"encoder_id": "e5", "pca_scope": "joint",
                                                         "pca_components_per_column": 64}},
                    representation_feature_count=69, feature_artifact_ids=["frozen"],
                    estimator_settings={"execution_runtime": runtime})
    verify_arm_recipe(metadata, arm, ["frozen"], runtime)
    for recipe, assets, image in (
        (arm | {"pca_scope": "per_column", "pca_components": 32}, ["frozen"], runtime),
        (arm, ["other"], runtime), (arm, ["frozen"], {"container_id": "other"}),
    ):
        with pytest.raises(ValueError, match="planned fold"):
            verify_arm_recipe(metadata, recipe, assets, image)


def test_same_published_features_survive_retry_and_both_merge_orders(monkeypatch):
    import hashlib

    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE feature_artifacts (artifact_id TEXT PRIMARY KEY, manifest_json TEXT, features_parquet BLOB, requests_parquet BLOB)")
    monkeypatch.setattr(feature_retirement, "REPLACEMENTS", {"old": "new"})

    def row(name, data=b"features"):
        blobs = (data, b"requests")
        sums = {key: hashlib.sha256(blob).hexdigest() for key, blob in zip(("features.parquet", "requests.parquet"), blobs)}
        return (name, json.dumps({"artifact_id": name, "sha256": sums, "identity": {"model": "nimble", "code_sha256": name}}), *blobs)

    def insert(item):
        with db:
            if not feature_retirement.canonical_equivalent(db, item):
                db.execute("INSERT OR IGNORE INTO feature_artifacts VALUES (?,?,?,?)", item)

    try:
        for first, second in (("old", "new"), ("new", "old")):
            db.execute("DELETE FROM feature_artifacts")
            insert(row(first))
            insert(row(second))
            insert(row("old"))
            assert db.execute("SELECT artifact_id FROM feature_artifacts").fetchall() == [("new",)]
            with pytest.raises(ValueError):
                insert(row("old", b"different"))
            with pytest.raises(ValueError, match="Equivalent feature publication"):
                insert(row("unreviewed"))
            assert db.execute("SELECT artifact_id FROM feature_artifacts").fetchall() == [("new",)]
    finally:
        db.close()


def test_saved_fold_modes_reach_paired_tables_without_colliding(isolated_output, monkeypatch, tmp_path):
    from sklearn.dummy import DummyRegressor

    from src.cluster.predictor_contracts import feature_cases, fixtures
    from src.evaluation import results
    from src.evaluation.notebook_study import headline_summary
    from src.evaluation.protocol import evaluate_cv, shared_splits
    from src.evaluation.representation_views import MODES, decision_vs_embedding

    dataset = next(d for d in fixtures() if d.metadata.task_type == "regression")
    spec = shared_splits(dataset, seed=42)
    features = next(feature_cases(dataset, "linear"))[1]
    for mode in MODES:
        evaluate_cv(dataset, features, DummyRegressor, specification=spec,
                    model_id="linear", checkpoint_id="fixed", feature_set=mode,
                    phase="B1", allow_model_runs=True, estimator_settings={"fixture": True},
                    feature_artifact_ids=[mode], representation_recipe={"provider": "liquid"})
    for scope, width in (("joint", 64), ("per_column", 32)):
        evaluate_cv(dataset, features, DummyRegressor, specification=spec,
                    model_id="linear", checkpoint_id="fixed", feature_set="embedding",
                    phase="B2", allow_model_runs=True, estimator_settings={"fixture": True},
                    feature_artifact_ids=["encoder"], representation_recipe={"embedding": {
                        "encoder_id": "e5_small_v2", "pca_scope": scope, "pca_components_per_column": width}})
    # A baseline on the same rows from a different estimator runtime is ineligible.
    evaluate_cv(dataset, features, DummyRegressor, specification=spec,
                model_id="linear", checkpoint_id="fixed", feature_set="non_text",
                phase="B1", allow_model_runs=True, estimator_settings={"fixture": "different-image"},
                feature_artifact_ids=[], representation_recipe={"provider": None})
    monkeypatch.setenv("JEVPFN_NOTEBOOK_SCRATCH", str(tmp_path / "notebooks"))
    status, folds, _, gains = results.collect_for_notebook()
    assert status["complete_arms"] == 6
    assert gains.difference_from_non_text.isna().all()
    paired = decision_vs_embedding(folds)
    assert len(paired) == 5 * 3 * 2
    assert set(paired.feature_set) == set(MODES)
    assert not headline_summary(folds).empty
    assert headline_summary(folds[folds.outer_fold != 4]).empty
    monkeypatch.setattr(results, "collect", lambda *_: pytest.fail("Unchanged notebook snapshot was read twice"))
    cached = results.collect_for_notebook()[1]
    pd.testing.assert_frame_equal(folds, cached, check_dtype=False)
