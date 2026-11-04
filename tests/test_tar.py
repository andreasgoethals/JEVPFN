"""Local TAR boundaries; synthetic vectors do not certify upstream GPU training."""

import json
import runpy
import sys
from types import ModuleType

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from src.cluster.predictor_contracts import fixtures
from src.evaluation import tar
from src.evaluation.features import assemble_feature_set
from src.evaluation.protocol import evaluate_cv, shared_splits
from src.utils import paths
from src.utils.serialization import digest, file_sha256, write_json


@pytest.fixture
def bundles(isolated_output, monkeypatch):
    monkeypatch.setattr(paths, "tar_results_dir", lambda *p: isolated_output.joinpath("tar", *p))
    dataset = next(fixtures())
    split = shared_splits(dataset, seed=tar.load_yaml("experiment_3/tar")["random_seed"])
    rng = np.random.default_rng(42)
    bodies = []
    for fold, partition in enumerate(split["folds"]):
        folder = tar.directory(dataset.metadata.dataset_id, fold)
        (folder / "adapter").mkdir(parents=True)
        (folder / "adapter" / "unit_fixture.json").write_text("{}")
        for i in range(2):
            vectors = rng.normal(size=(len(dataset.frame), 384)).astype("float32")
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            np.save(folder / f"text_{i}.npy", vectors)
        body = {
            "recipe": tar.recipe(dataset, split, fold), "status": "complete",
            "adapter_roundtrip_checked": True,
            "runtime": {"container_id": "unit_fixture_not_real_evidence"}, "job_id": "unit_fixture",
            "optimizer_rows": partition["train"][20:], "early_stopping_rows": partition["train"][:20],
            "files": {p.relative_to(folder).as_posix(): file_sha256(p) for p in folder.rglob("*") if p.is_file()},
        }
        body["artifact_id"] = digest(body)
        write_json(folder / "manifest.json", body)
        bodies.append(body)
    return dataset, split, bodies


def test_copied_source_is_complete_and_byte_sealed():
    receipt = tar.upstream()
    assert len(receipt["files"]) == 12
    assert receipt["commit"] == "bc9e87130df9884ccd49d04055599c0eda9f3238"


def test_upstream_validation_split_keeps_columns_of_each_row_together(monkeypatch):
    tar.upstream()
    constants = ModuleType("multabench.constants")
    constants.SEED = 42
    monkeypatch.setitem(sys.modules, "multabench.constants", constants)
    # Execute the exact upstream splitting implementation without its GPU dependencies.
    split_to_val = runpy.run_path(str(tar.VENDOR / "multabench/preprocessing/splits.py"))["split_to_val"]
    dataset = next(fixtures())
    x, y = dataset.frame[["text", "notes"]], dataset.frame.target
    a, b, ay, by = split_to_val(x.iloc[:140], y.iloc[:140], is_cls=True)
    assert not set(a.index) & set(b.index)
    assert set(a.index) | set(b.index) == set(range(140))
    assert a.index.equals(ay.index) and b.index.equals(by.index)
    assert len(b) == 16  # Upstream binary split adds one validation example per class.
    assert len(a.columns) == len(b.columns) == 2


def test_tar_cache_training_identity_and_file_integrity(bundles):
    dataset, split, bodies = bundles
    assert tar.read_bundle(dataset, split, 0) == bodies[0]
    dataset.frame.loc[split["folds"][0]["test"], "target"] = 999
    assert tar.read_bundle(dataset, split, 0) == bodies[0]  # Test targets never enter encoder identity.
    dataset.frame.loc[split["folds"][0]["train"][0], "target"] = 999
    with pytest.raises(ValueError, match="training partition"):
        tar.read_bundle(dataset, split, 0)


def test_tar_rejects_corrupt_vectors(bundles):
    dataset, split, _ = bundles
    file = tar.directory(dataset.metadata.dataset_id, 0) / "text_0.npy"
    values = np.load(file)
    values[0, 0] += 1
    np.save(file, values)
    with pytest.raises(ValueError, match="missing, changed"):
        tar.read_bundle(dataset, split, 0)


@pytest.mark.parametrize("scope,components,columns", [("per_column", 32, 68), ("joint", 64, 68)])
def test_tar_pca_and_predictor_reuse_exact_outer_fold(bundles, scope, components, columns):
    dataset, split, bodies = bundles
    tar.prepare_pca(dataset, 0)
    base = assemble_feature_set(dataset, [], "non_text")
    train, test = split["folds"][0]["train"], split["folds"][0]["test"]
    arm = {"pca_scope": scope, "pca_components": components}
    controls = [tar.TarControl(DummyClassifier(), dataset, split, bodies, arm) for _ in range(2)]
    for control in controls:
        control.fit(base.iloc[train], dataset.frame.target.iloc[train])
        assert control.output_feature_count_ == columns
        assert control.predict_proba(base.iloc[test]).shape == (len(test), 2)
        transformed = control.control.transform(control._features(base.iloc[test])[0])
        assert transformed.shape[1] == columns
        # Upstream encodes its missing placeholder; the original text-presence flags survive.
        assert transformed.iloc[:, :base.shape[1]].equals(base.iloc[test])
    np.testing.assert_array_equal(controls[0].control.pcas[0].components_, controls[1].control.pcas[0].components_)
    before = controls[0].control.pcas[0].components_.copy()
    train_vectors, groups = tar.vector_frame(dataset, 0, train)
    columns_to_scale = groups[0] if scope == "per_column" else sum(groups, [])
    np.testing.assert_allclose(controls[0].control.pcas[0].input_scaler_mean_, train_vectors[columns_to_scale].mean().to_numpy(), atol=1e-7)
    changed = base.iloc[test].copy()
    changed["number"] = 1e30
    controls[0].predict_proba(changed)
    np.testing.assert_array_equal(before, controls[0].control.pcas[0].components_)
    with pytest.raises(ValueError, match="exact outer training"):
        controls[0].fit(base.iloc[train[:50]], dataset.frame.target.iloc[train[:50]])


def test_tar_0b_requires_every_fold_and_cpu_pca(bundles):
    dataset, _, _ = bundles
    with pytest.raises(ValueError, match="PCA"):
        tar.validate_dataset(dataset)
    for fold in range(5):
        tar.prepare_pca(dataset, fold)
    report = tar.validate_dataset(dataset)
    assert len(set(report["artifact_ids"])) == 5
    assert len(report["prepared_pca_keys"]) == 15
    path = tar.directory(dataset.metadata.dataset_id, 4) / "manifest.json"
    body = json.loads(path.read_text())
    body["early_stopping_rows"][0] = body["optimizer_rows"][0]
    body["artifact_id"] = digest({k: v for k, v in body.items() if k != "artifact_id"})
    write_json(path, body)
    with pytest.raises(ValueError, match="overlap"):
        tar.validate_dataset(dataset)


def test_tar_fixed_threshold_no_inner_refits_and_resumes(isolated_output):
    dataset = next(fixtures())
    base = assemble_feature_set(dataset, [], "non_text")
    fitted = []

    class Count(DummyClassifier):
        def fit(self, x, y):
            fitted.append(x.index.tolist())
            return super().fit(x, y)

    kwargs = dict(model_id="fixture", checkpoint_id="fixture", feature_set="non_text_plus_tar",
                  specification=shared_splits(dataset, seed=42), estimator_settings={"fixture": True},
                  phase="experiment_3", binary_threshold_policy="fixed_0_5", allow_model_runs=True)
    records = evaluate_cv(dataset, base, Count, **kwargs)
    assert len(records) == len(fitted) == 5
    for record in records:
        body = json.loads(record.read_text())
        assert body["metadata"]["prediction_role"] == "outer_test"
        assert body["metadata"]["selected_threshold"] == .5
        assert "validation_selected" not in body["scores"]["decisions"]
        assert "default_0_5" in body["scores"]["decisions"]
    assert evaluate_cv(dataset, base, Count, **kwargs) == records
    assert len(fitted) == 5


def test_tar_shares_durable_gpfs_without_copying_into_predictor_scratch(monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(paths, "on_vsc", lambda: True)
    monkeypatch.setattr(paths, "durable_project_override", lambda: Path("/gpfs1/project/prj_00013"))
    monkeypatch.setattr(paths, "staging_override", lambda: Path("/gpfs1/scratch/user/working"))
    assert "project/prj_00013/JEVPFN" in paths.tar_results_dir("tar").as_posix()
    monkeypatch.setattr(paths, "durable_project_override", lambda: Path("/lustre1/project/stg_00211"))
    with pytest.raises(ValueError, match="native GPFS"):
        paths.tar_results_dir("tar")
