"""RQ2-E dataset moderators and label-free decision-input row cohorts (synthetic only)."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.evaluation import moderators as m
from src.evaluation import row_benefit as rb
from src.evaluation.decision_comparison import TIED


def _gains(values, moderator):
    datasets = [f"d{i}" for i in range(len(values))]
    gains = pd.DataFrame({
        "model_id": "tabpfn", "checkpoint_id": "c", "family": "decision",
        "aggregation": "per_decision_model", "representation": "decision/joint/jev",
        "feature_set": "non_text_plus_joint", "provider": "jev", "encoder": None,
        "pca_scope": None, "pca_components": np.nan, "dataset_id": datasets,
        "task_type": "binary_classification", "baseline": 0.7, "augmented": 0.8,
        "gain": values, "decision_models": 1,
    })
    covariates = pd.DataFrame({"dataset_id": datasets, "task_type": "binary_classification",
                               "n_rows": moderator, "share_long_text": 0.5})
    return gains, covariates


def test_spearman_with_dataset_bootstrap_and_reported_gaps():
    gains, covariates = _gains(np.arange(8.0), np.arange(8) ** 2)
    table = m.moderator_associations(gains, covariates, n_boot=200, seed=3)
    rows = table.set_index("moderator")
    assert rows.loc["n_rows", "spearman_rho"] == pytest.approx(1.0)
    assert rows.loc["n_rows", ["ci_low", "ci_high"]].tolist() == pytest.approx([1.0, 1.0])
    assert rows.loc["n_rows", "n_datasets"] == 8 and rows.loc["n_rows", "status"] == "descriptive"
    assert rows.loc["share_long_text", "status"] == "constant_input"
    assert rows.loc["baseline_headroom", "status"] == "constant_input"
    assert m.moderator_associations(gains, covariates, n_boot=200, seed=3).equals(table)
    reversed_gains, cov = _gains(np.arange(8.0)[::-1].copy(), np.arange(8))
    assert m.moderator_associations(reversed_gains, cov, n_boot=0).set_index(
        "moderator").loc["n_rows", "spearman_rho"] == pytest.approx(-1.0)
    few, cov = _gains(np.arange(3.0), np.arange(3))
    assert m.moderator_associations(few, cov, n_boot=10).set_index(
        "moderator").loc["n_rows", "status"] == "too_few_datasets"


def test_spearman_rows_handles_ties_like_scipy():
    from scipy.stats import spearmanr

    x, y = np.array([1, 2, 2, 3, 5.0]), np.array([2, 1, 4, 4, 3.0])
    assert m.spearman_rows(x, y)[0] == pytest.approx(spearmanr(x, y).statistic)


def test_headroom_and_provider_specific_decision_properties(tmp_path):
    gains, covariates = _gains(np.arange(6.0), np.arange(6))
    gains["baseline"] = np.linspace(0.5, 0.9, 6)[::-1]
    pd.DataFrame([
        {"dataset_id": f"d{i}", "provider": "jev", "representation": "joint",
         "weighting": "rows", "variable": "max_probability", "mean": 0.5 + i / 20}
        for i in range(6)
    ]).to_csv(tmp_path / "distributions.csv", index=False)
    pd.DataFrame([
        {"dataset_id": f"d{i}", "model_a": "jev", "model_b": "kev", "representation": "joint",
         "weighting": "rows", "agreement": 0.9 - i / 20}
        for i in range(6)
    ]).to_csv(tmp_path / "pairs.csv", index=False)
    properties, status = m.decision_properties(tmp_path)
    assert status.startswith("available") and set(properties.provider) == {"jev", "kev"}
    covariates.attrs["decision_properties"] = properties
    for column in m.DECISION_PROPERTIES:
        covariates[column] = np.nan
    rows = m.moderator_associations(gains, covariates, n_boot=0).set_index("moderator")
    assert rows.loc["baseline_headroom", "spearman_rho"] == pytest.approx(1.0)
    assert rows.loc["decision_mean_max_probability", "spearman_rho"] == pytest.approx(1.0)
    assert rows.loc["decision_mean_pairwise_agreement", "spearman_rho"] == pytest.approx(-1.0)
    empty, pending = m.decision_properties(tmp_path / "absent")
    assert empty.empty and pending.startswith("pending")


def test_dataset_covariates_are_label_free(tmp_path):
    frame = pd.DataFrame({
        "number": [1.0, 2, 3, 4], "body": ["a b", None, "a b", "long text here"],
        "code": ["X1", "X2", "X3", "X4"], "target": [0, 1, 0, 1],
    })
    official = {"target": "target", "task_type": "cls"}
    metadata = extract_metadata(frame, official, dataset_id="cov", dataset_name="Cov",
                                text_columns=["body", "code"])
    dataset = LoadedDataset(frame, metadata, official, {"data_identity": "x"}, [])
    row = m.dataset_covariates(dataset)
    assert (row["n_rows"], row["n_text_fields"], row["n_classes"]) == (4, 2, 2)
    assert row["missing_text_share"] == pytest.approx(1 / 8)
    assert row["repeated_text_share"] == pytest.approx(2 / 7)
    assert row["median_text_chars"] == 2.0
    assert sum(row[f"share_{k}"] for k in (*m.FIELD_TYPES, "empty")) == pytest.approx(1)
    flipped = LoadedDataset(frame.assign(target=[1, 0, 1, 0]), metadata, official,
                            {"data_identity": "x"}, [])
    assert m.dataset_covariates(flipped) == row
    table = m.dataset_moderators([dataset], decision_tables=tmp_path / "absent")
    assert table.attrs["headroom_status"] == "pending"


def test_decision_choices_ties_abstain_and_missing_rows_are_unavailable():
    p = np.array([[0.7, 0.3], [0.5, 0.5], [np.nan, np.nan], [0.2, 0.8]])
    top, choice = rb.decision_summaries(p)
    np.testing.assert_allclose(top[[0, 1, 3]], [0.7, 0.5, 0.8])
    assert np.isnan(top[2])
    assert choice.tolist() == [0, TIED, rb.UNAVAILABLE, 1]


def test_agreement_uses_plurality_share_with_abstentions():
    u, t = rb.UNAVAILABLE, TIED
    choices = np.array([
        # rows: unanimous, 2 of 3, tie abstains, one model only, all abstain
        [1, 1, 1, 1, t],
        [1, 1, t, u, t],
        [1, 0, 0, u, u],
    ])
    agreement = rb.group_agreement(choices)
    np.testing.assert_allclose(agreement[:3], [1.0, 2 / 3, 1 / 3])
    assert np.isnan(agreement[3]) and agreement[4] == 0
    assert rb.agreement_bands([1.0, 0.75, 0.6, 0.2, np.nan]).tolist() == [
        "unanimous", "0.75_to_1", "0.5_to_0.75", "lt_0.5", "missing"]
    assert rb.confidence_bands([0.95, 0.9, 0.69, 0.4, np.nan]).tolist() == [
        "ge_0.9", "ge_0.9", "0.5_to_0.7", "lt_0.5", "missing"]


def _block(provider, mode, rows, values, task="binary_classification"):
    groups = ["all_text"] if mode == "joint" else ["text_000", "text_001"]
    frame = {"row_id": [f"csv:{i}" for i in range(rows)]}
    columns = []
    for g, group in enumerate(groups):
        name = f"{provider}__{mode}__text_only__{group}__p_positive"
        frame[name] = values[g] if mode == "per_column" else values
        columns.append(name)
    return SimpleNamespace(
        values=pd.DataFrame(frame),
        manifest={"feature_columns": columns, "text_columns": ["a", "b"],
                  "identity": {"representation": mode}},
    )


def test_decision_inputs_average_blocks_and_mark_missing_rows():
    rows = 4
    stores = {
        ("jev", "per_column"): _block("jev", "per_column", rows,
                                      [[0.9, 0.6, np.nan, 0.5], [0.7, np.nan, np.nan, 0.5]]),
        ("jev", "joint"): _block("jev", "joint", rows, [0.8, 0.4, np.nan, 0.5]),
        ("kev", "joint"): _block("kev", "joint", rows, [0.7, 0.6, 0.9, 0.5]),
    }

    def loader(dataset_id, provider, mode, cfg):
        block = stores.get((provider, mode))
        return (block, "fixture") if block is not None else (None, "artifact_not_found")

    dataset = SimpleNamespace(frame=pd.DataFrame(index=range(rows)),
                              metadata=SimpleNamespace(task_type="binary_classification",
                                                       dataset_id="x"))
    inputs = rb.DecisionInputs(dataset, {"candidate_extractors": ["jev", "kev", "nim"]},
                               loader=loader)
    confidence = inputs.confidence("jev", ("per_column",))
    np.testing.assert_allclose(confidence[[0, 1, 3]], [0.8, 0.6, 0.5])
    assert np.isnan(confidence[2])
    both = inputs.confidence("jev", ("per_column", "joint"))
    assert both[0] == pytest.approx((0.9 + 0.7 + 0.8) / 3)
    agreement = inputs.agreement(("joint",))
    # Row 1: jev says 0, kev says 1; row 2: only kev available; row 3: both tie.
    assert agreement[0] == 1.0 and agreement[1] == 0.5
    assert np.isnan(agreement[2]) and agreement[3] == 0
    assert inputs.status["nim/joint"] == "missing:artifact_not_found"
    assert np.isnan(inputs.confidence("nim", ("joint",))).all()
    assert rb.decision_modes({"provider": "jev"}, "non_text_plus_both", "experiment_1") == (
        "per_column", "joint")
    assert rb.decision_modes({"provider": None}, "non_text", "experiment_1") is None


def test_paired_cohorts_add_decision_axes_only_when_supplied():
    ref = pd.DataFrame(dict(row_id=["csv:0", "csv:1"], y_true=[1.0, 2.0], prediction=[0.0, 3.0],
                            text_present=[True, False], joint_text_seen_in_train=[False, False]))
    aug = ref.assign(prediction=[1.0, 2.0])
    plain, _ = rb.paired_cohorts(ref, aug, [], 2, [2.0, 4.0, 8.0], [100.0, np.nan])
    assert not {r["axis"] for r in plain} & {"decision_confidence", "decision_agreement"}
    rows, _ = rb.paired_cohorts(ref, aug, [], 2, [2.0, 4.0, 8.0], [100.0, np.nan],
                                decision={"confidence": np.array([0.95, np.nan]),
                                          "agreement": np.array([0.6, np.nan])})
    cohorts = {(r["axis"], r["cohort"]) for r in rows}
    assert {("decision_confidence", "ge_0.9"), ("decision_confidence", "missing"),
            ("decision_agreement", "0.5_to_0.75"), ("decision_agreement", "missing")} <= cohorts
    with pytest.raises(ValueError, match="misaligned"):
        rb.paired_cohorts(ref, aug, [], 2, [2.0], [100.0, np.nan],
                          decision={"confidence": np.array([0.9]), "agreement": np.array([1.0])})


def test_saved_decision_arm_joins_decision_inputs_by_row(isolated_output, monkeypatch):
    from sklearn.dummy import DummyClassifier

    from src.cluster.predictor_contracts import feature_cases, fixtures
    from src.evaluation.protocol import evaluate_cv, shared_splits

    ds = next(d for d in fixtures() if d.metadata.task_type == "binary_classification")
    _, features, _ = next(feature_cases(ds, "linear"))
    name = ds.metadata.dataset_id
    monkeypatch.setattr(rb, "load_catalog", lambda: {"datasets": [{"id": name}]})
    monkeypatch.setattr(rb, "load_dataset", lambda *a, **kw: ds)
    monkeypatch.setattr(rb, "load_yaml", lambda name: {"random_seed": 20260929,
                                                       "candidate_extractors": ["jev", "kev"]})
    spec = shared_splits(ds, seed=20260929, folds=5)
    for feature_set, provider in [("non_text", None), ("non_text_plus_joint", "jev")]:
        evaluate_cv(ds, features, DummyClassifier, specification=spec, model_id="linear",
                    checkpoint_id="fixture", feature_set=feature_set, phase="experiment_1",
                    allow_model_runs=True, estimator_settings={"seed": 1},
                    representation_recipe={"provider": provider, "feature_set": feature_set},
                    feature_artifact_ids=[])
    rows = len(ds.frame)
    values = np.where(np.arange(rows) % 3 == 0, 0.95, 0.6)
    values[ds.frame.text.isna().to_numpy() & ds.frame.notes.isna().to_numpy()] = np.nan

    def loader(dataset_id, provider, mode, cfg):
        if mode != "joint" or provider not in ("jev", "kev"):
            return None, "artifact_not_found"
        return _block(provider, "joint", rows, values if provider == "jev" else 1 - values), "x"

    monkeypatch.setattr(rb.paths, "durable_project_override", rb.paths.staging_override)
    body = json.loads(rb.run(name, decision_loader=loader).read_text())
    assert body["decision_inputs"] == {"jev/joint": "x", "kev/joint": "x"}
    by, _ = rb.summaries()
    confidence = by[by.axis.eq("decision_confidence")].set_index("cohort").rows
    assert confidence["ge_0.9"] == int((values > 0.9).sum())
    assert confidence.sum() == rows
    agreement = by[by.axis.eq("decision_agreement")].set_index("cohort").rows
    # jev and kev always disagree, so each available row has a 1-of-2 plurality.
    assert set(agreement.index) <= {"0.5_to_0.75", "missing"} and agreement.sum() == rows
