"""0B behaviour expected once the open-weight campaign publishes its stores.

Synthetic stores and score tables exercise the real 0B gate and analysis code paths; they do
not establish checkpoint, GPU or real-provider behaviour.
"""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.cluster import preflight
from src.cluster.predictor_contracts import required_fixture_cases
from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.evaluation import group_b, inference, robustness_export, study_reports
from src.evaluation import qualification as q
from src.jev.cache import ResponseCache
from src.jev.features import materialize_cached_block, save_block
from src.jev.requests import build_requests
from src.utils import paths
from src.utils.config import load_config
from tests.test_benchmark_qualification import reports, write_approval  # noqa: F401 (fixture)

# ------------------------------------------------------------- decision stores in 0B preflight


def _sparse_dataset(rows=40):
    """Two text fields; `notes` is sparse (sparse-text dataset) and row 0 has no text at all."""
    frame = pd.DataFrame({
        "text": [None if i % 7 == 0 else f"Review number {i}." for i in range(rows)],
        "notes": [f"Note {i}." if i % 5 == 1 else None for i in range(rows)],
        "n": np.arange(rows, dtype=float),
        "y": [i % 2 for i in range(rows)],
    })
    meta = extract_metadata(frame, {"target": "y", "task_type": "cls"}, dataset_id="sparse",
                            dataset_name="Sparse", text_columns=["text", "notes"])
    return LoadedDataset(frame, meta, {}, {"data_identity": "sparse-raw"}, [])


def _publish_store(dataset):
    """Both decision modes through the real request builder, cache and store writer."""
    path = paths.dataset_cache_path(dataset.metadata.dataset_id)
    cache = ResponseCache(path, dataset_id=dataset.metadata.dataset_id)
    for mode in ("per_column", "joint"):
        cfg = load_config("feature_creation")
        cfg["jev"]["representation"] = mode
        for position in range(len(dataset.frame)):
            for request in build_requests(dataset.features, dataset.metadata, position, cfg):
                cache.put(request, {"model": "jev-1.14.0",
                                    "answers": {"semantic": {"type": "noul", "noul": 0.25}}},
                          kind="verified_response")
        save_block(materialize_cached_block(dataset, cfg, cache), path)
    return path


@pytest.fixture
def decision_store(isolated_output, monkeypatch):
    dataset = _sparse_dataset()
    _publish_store(dataset)
    entry = {"id": dataset.metadata.dataset_id}
    monkeypatch.setattr(preflight, "selected_entries", lambda catalog, cfg: [entry])
    monkeypatch.setattr(preflight, "load_catalog", lambda: {"datasets": [entry]})
    monkeypatch.setattr(preflight, "load_dataset", lambda e, allow_download=False: dataset)
    monkeypatch.setattr(q, "decision_selector", lambda cfg, provider: {"provider": provider})
    return dataset


def test_sparse_text_store_passes_and_missing_provider_store_fails_closed(decision_store):
    cfg = {"candidate_extractors": ["jev", "glide"], "feature_selectors": {}}
    results = {r["provider"]: r for r in preflight.decision_feature_checks(cfg)}
    assert results["jev"]["status"] == "passed", results["jev"]
    assert set(results["jev"]["artifact_ids"]) == {"joint", "per_column"}
    assert results["jev"]["rows"] == len(decision_store.frame)
    # A provider whose store has not been published yet (partial campaign) is a failure.
    assert results["glide"]["status"] == "failed" and "exactly one" in results["glide"]["error"]
    with pytest.raises(ValueError, match="complete passing decision feature"):
        q._complete(list(results.values()), ("dataset_id", "provider"),
                    {("sparse", "jev"), ("sparse", "glide")}, "decision feature")


def test_store_of_another_dataset_version_is_rejected(decision_store, monkeypatch):
    changed = deepcopy(decision_store)
    changed.source["data_identity"] = "republished-raw"
    monkeypatch.setattr(preflight, "load_dataset", lambda e, allow_download=False: changed)
    (result,) = preflight.decision_feature_checks({"candidate_extractors": ["jev"]})
    assert result["status"] == "failed" and "different dataset version" in result["error"]


@pytest.mark.parametrize(("task", "classes"), [("binary_classification", ()),
                                               ("multiclass_classification", ("a", "b", "c")),
                                               ("regression", ())])
def test_probability_columns_must_follow_the_ordered_class_or_level_schema(task, classes):
    labels = preflight.decision_probability_labels(task, classes)
    width = len(labels)
    assert width == {"binary_classification": 1, "multiclass_classification": 3, "regression": 9}[task]
    values = np.full((3, width), 1 / width) if width > 1 else np.array([[0.2], [0.7], [np.nan]])
    if width > 1:
        values[2] = np.nan
    frame = pd.DataFrame(values, columns=[f"m__joint__text_only__all_text__{c}" for c in labels])
    missing = pd.Series([False, False, True])
    preflight.validate_decision_group(frame, missing, width, labels)
    if width > 1:
        # Same values, ordinal levels or classes permuted: the width check alone would pass.
        swapped = frame[list(frame.columns[::-1])]
        preflight.validate_decision_group(swapped, missing, width)
        with pytest.raises(ValueError, match="ordered class/level schema"):
            preflight.validate_decision_group(swapped, missing, width, labels)
    with pytest.raises(ValueError, match="ordered class/level schema"):
        preflight.validate_decision_group(frame.iloc[:, :-1], missing, width, labels[:-1] + ["x"])
    infinite = frame.copy()
    infinite.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="benchmark input contract"):
        preflight.validate_decision_group(infinite, missing, width, labels)


# ------------------------------------------------------------------- 0B acceptance evidence


@pytest.mark.parametrize("fault", ["subsampled_profile", "partial_training_rows", "no_fixture_records",
                                   "repeated_fixture_records", "absent_split_identity", "four_folds"])
def test_vacuous_or_row_capped_0b_evidence_is_rejected(tmp_path, reports, fault):  # noqa: F811
    arm, bodies = deepcopy(reports)
    if fault == "subsampled_profile":
        bodies[-1]["roles"][0]["rows"] = 1
    elif fault == "partial_training_rows":
        bodies[-1]["training_rows"] = 5
    elif fault == "no_fixture_records":
        for fixture in bodies[1]["fixtures"]:
            fixture.pop("records")
    elif fault == "repeated_fixture_records":
        fixture = bodies[1]["fixtures"][0]
        fixture["records"] = [fixture["records"][0]] * 10
    elif fault == "absent_split_identity":
        # Missing on both sides used to compare equal (None == None).
        bodies[0]["datasets"][0].pop("split_id")
        bodies[-1].pop("split_id")
    else:
        bodies[0]["datasets"][0]["outer_folds"] = 4
    with pytest.raises(ValueError, match="0B"):
        q.benchmark_evidence(write_approval(tmp_path, bodies), arm)


def test_complete_0b_evidence_still_passes(tmp_path, reports):  # noqa: F811
    arm, bodies = reports
    features, runtime = q.benchmark_evidence(write_approval(tmp_path, bodies), arm)
    assert runtime == {"container_id": "seal"} and features[0]["data",]["rows"] == 10


def test_fixture_tasks_include_the_largest_vocabulary_and_nine_level_regression():
    for model in ("linear", "tabstar"):
        cases = required_fixture_cases(model)
        assert {(t, c) for t, c, _ in cases} == {("binary_classification", 2),
                                                 ("multiclass_classification", 3),
                                                 ("multiclass_classification", 30),
                                                 ("regression", 0)}


# --------------------------------------------------------- A2 averaged-presentation contrasts


def _measured(p):
    return {"status": "measured", "probabilities": list(p)}


@pytest.mark.parametrize("task", ["binary_classification", "multiclass_classification", "regression"])
def test_a2_metrics_all_have_an_orientation_and_convex_scores_never_lose_to_mean_single(task):
    """Every exported metric needs a declared orientation, or its improvement sign is a guess."""
    rng = np.random.default_rng(11)
    width = {"binary_classification": 2, "multiclass_classification": 4, "regression": 9}[task]
    rows = []
    for dataset in ("D1", "D2", "D3"):
        trials = {n: {v: _measured(rng.dirichlet(np.ones(width))) for v in robustness_export.WORDINGS}
                  for n in range(50)}
        found = robustness_export.averaged_presentations(trials)
        links = pd.DataFrame({"input_no": range(50), "row_position": range(50), "mode": ["joint"] * 50})
        targets = (rng.normal(size=50) if task == "regression"
                   else np.resize(np.arange(width), 50))  # every class observed
        rows += [{"dataset": dataset, "provider": "glide", **r} for r in
                 robustness_export.quality_records(found, links, targets, task, 10)]
    table = pd.DataFrame(rows)
    assert set(table.metric) <= set(robustness_export.METRIC_ROLES)
    if task == "regression":
        assert "ranked_probability_score" in set(table.metric)
    summary = study_reports.averaging_quality_summary(table)
    jensen = summary[summary.contrast.eq("average_vs_mean_single") & summary.guaranteed]
    assert not jensen.empty and set(jensen.datasets) == {3}
    assert (jensen.improvement >= -1e-12).all() and jensen.datasets_improved.eq(3).all()
    assert not summary[summary.contrast.eq("average_vs_original")].guaranteed.any()


def test_a2_outcome_levels_follow_the_target_order_of_the_nine_evidence_levels():
    targets = np.linspace(-1, 1, 900)
    levels = robustness_export.outcome_levels(targets, targets)
    assert levels.min() == 0 and levels.max() == 8 and (np.diff(levels) >= 0).all()
    # A forecast on the observed level beats one that reverses the evidence scale.
    right = np.eye(9)[levels]
    reversed_scale = np.eye(9)[8 - levels]
    assert robustness_export.ranked_probability_score(levels, right) == 0
    assert robustness_export.ranked_probability_score(levels, reversed_scale) > 0.3


# ------------------------------------------------------------------- B3 pilot and LODO sets


def test_b3_pilot_with_too_few_answered_rows_is_undefined_and_leaves_the_common_cohort():
    rng = np.random.default_rng(5)
    non_text = pd.DataFrame({"x": rng.normal(size=4000)})
    decision = (1 / (1 + np.exp(-non_text.x.to_numpy())))[:, None]
    pilot = group_b.pilot_rows(len(non_text), dataset_id="sparse", size=200, seed=20261123)
    sparse = np.full_like(decision, np.nan)
    sparse[pilot[:9]] = decision[pilot[:9]]  # nine answered pilot rows: below the ten-row floor
    dropped = group_b.pilot_redundancy(non_text, sparse, dataset_id="sparse",
                                       task_type="binary_classification", size=200, seed=20261123)
    assert np.isnan(dropped["pilot_r2"])
    full = group_b.pilot_redundancy(non_text, decision, dataset_id="sparse",
                                    task_type="binary_classification", size=200, seed=20261123)
    assert full["pilot_r2"] > 0.8
    # The rows are chosen before any answer exists: sparsity does not move the pilot.
    assert np.array_equal(pilot, group_b.pilot_rows(len(non_text), dataset_id="sparse", size=200,
                                                    seed=20261123))
    ids = [f"d{i:02d}" for i in range(18)]
    properties = pd.DataFrame({"dataset_id": ids, "log10_median_text_chars": rng.normal(size=18),
                               "pilot_redundancy_r2": rng.uniform(size=18),
                               "decision_redundancy_r2": rng.uniform(size=18)})
    properties.loc[3, "pilot_redundancy_r2"] = dropped["pilot_r2"]
    outcome = pd.DataFrame({"dataset_id": ids, "raw_gain": rng.normal(0, 0.01, 18)})
    from src.utils.config import load_yaml

    sets = load_yaml("rq2/when_text_helps")["predict_in_advance"]["input_sets"]
    predictions, summary = group_b.lodo_input_sets(outcome, properties, input_sets=sets, n_shuffles=5)
    assert list(summary.input_set) == ["pre_extraction", "pilot", "post_extraction"]
    assert summary.datasets.eq(17).all() and summary.excluded_datasets.eq("d03").all()
    assert "d03" not in set(predictions.dataset_id)


# -------------------------------------------------------- learner comparison, end to end


def _fold_records(learners, datasets=18, drop=None):
    """Five outer-fold headline records per (dataset, predictor) of one arm, as exported."""
    rng = np.random.default_rng(9)
    rows = []
    for d in range(datasets):
        task = ("binary_classification", "multiclass_classification", "regression")[d % 3]
        metric = {"binary_classification": "roc_auc", "multiclass_classification": "roc_auc_ovr_macro",
                  "regression": "r2"}[task]
        for j, learner in enumerate(learners):
            for fold in range(5):
                if drop == (f"d{d:02d}", learner, fold):
                    continue
                rows.append(dict(dataset_id=f"d{d:02d}", task_type=task, model_id=learner,
                                 checkpoint_id="c", split_id="s", outer_fold=fold, feature_set="non_text",
                                 provider=None, metric=metric,
                                 value=0.6 + 0.02 * j + rng.normal(0, 0.002)))
    return pd.DataFrame(rows)


def _dataset_scores(folds):
    head = group_b.headline(folds)
    complete = head.groupby(["dataset_id", "model_id"]).outer_fold.transform("nunique").eq(5)
    return (head[complete].groupby(["dataset_id", "model_id"], as_index=False).value.mean()
            .rename(columns={"model_id": "learner", "value": "score"}))


def test_learner_comparison_runs_from_fold_records_and_stops_on_an_incomplete_matrix():
    learners = ["linear", "catboost", "tabpfn_3_5"]
    scores = _dataset_scores(_fold_records(learners))
    assert len(scores) == 18 * 3
    omnibus, table = inference.compare_learners(scores, higher_is_better=True, control="linear", rope=0.01)
    best = table.set_index("learner").loc["tabpfn_3_5"]
    assert best.wins == 18 and best.median_oriented_difference > 0 and best.p_holm < 0.01
    assert min(omnibus["mean_ranks"], key=omnibus["mean_ranks"].get) == "tabpfn_3_5"
    # A loss (lower is better) is oriented so that positive still means better.
    losses = scores.assign(score=1 - scores.score)
    _, oriented = inference.compare_learners(losses, higher_is_better=False, control="linear", rope=0.01)
    assert oriented.set_index("learner").loc["tabpfn_3_5"].wins == 18
    # One missing outer fold makes that dataset incomplete; inference must stop, not shrink N.
    partial = _dataset_scores(_fold_records(learners, drop=("d04", "catboost", 2)))
    with pytest.raises(ValueError, match="inference stops"):
        inference.compare_learners(partial, higher_is_better=True, control="linear", rope=0.01)
    with pytest.raises(ValueError, match="Nonfinite"):
        inference.compare_learners(scores.assign(score=scores.score.where(scores.index != 0, np.inf)),
                                   higher_is_better=True, rope=0.01)
