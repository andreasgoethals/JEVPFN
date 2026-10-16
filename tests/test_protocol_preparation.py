"""Exercise orchestration with fake estimators/transport, never paid calls or ML fitting."""

import copy
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.evaluation.features import assemble_feature_set, shuffle_feature_block, text_statistics
from src.evaluation.protocol import evaluate_cv, select_f1_threshold, shared_splits
from src.jev.alternatives import preview
from src.jev.cache import ResponseCache
from src.jev.extraction import (
    DatasetExtractor,
    ExtractionLimits,
    RejectedRetryable,
    TypeSafeTransport,
    campaign_limits,
)
from src.jev.features import FeatureBlock, feature_view
from src.jev.requests import build_requests
from src.utils import paths
from src.utils.config import load_config, load_yaml


@pytest.fixture(autouse=True)
def _outputs(isolated_output):
    pass


@pytest.fixture
def dataset():
    frame = pd.DataFrame(
        {"text": ["one two", "", None, "one two"] * 25, "n": np.arange(100), "y": [0, 1] * 50}
    )
    official = {"target": "y", "task_type": "cls", "num_classes": 2}
    meta = extract_metadata(
        frame, official, dataset_id="fake", dataset_name="Fake", text_columns=["text"]
    )
    return LoadedDataset(frame, meta, official, {"data_identity": "fixed"}, [])


def test_text_statistics_are_label_free_and_missing_aware(dataset):
    stats = text_statistics(dataset)
    assert stats.iloc[0].tolist() == [7, 2, 0, 7, 2, 1]
    assert stats.iloc[1].tolist() == [0, 0, 1, 0, 0, 0]
    changed = copy.deepcopy(dataset)
    changed.frame["y"] = 999
    changed.frame["n"] = -888
    pd.testing.assert_frame_equal(stats, text_statistics(changed))
    baseline = assemble_feature_set(dataset, [], "non_text_plus_text_statistics")
    assert "n" in baseline and not {"text", "y"} & set(baseline.columns)
    assert baseline.attrs["feature_artifact_ids"] == []


def test_shuffle_preserves_vectors_without_crossing_partitions():
    train = pd.DataFrame({"a": range(12), "b": range(100, 112)}, index=range(12))
    test = train + 500
    a, b = shuffle_feature_block(train, test, seed=42)
    assert list(a.index) == list(train.index)
    assert sorted(map(tuple, a.to_numpy())) == sorted(map(tuple, train.to_numpy()))
    assert b.a.min() >= 500 and not a.equals(train)
    pd.testing.assert_frame_equal(a, shuffle_feature_block(train, test, seed=42)[0])


def test_regression_default_view_is_one_direction_not_eleven_features():
    columns = [f"jev__all_text__p_{i}" for i in range(9)] + [
        "jev__all_text__confidence",
        "jev__all_text__expected_direction",
    ]
    block = FeatureBlock(
        pd.DataFrame([["csv:0", *([1 / 9] * 9), 0.2, 0]], columns=["row_id", *columns]),
        pd.DataFrame(),
        {"feature_columns": columns},
    )
    assert list(feature_view(block, "regression").columns) == ["row_id", columns[-1]]
    assert feature_view(block, "regression", "probabilities").shape[1] == 10
    assert feature_view(block, "regression", "retained").shape[1] == 12


def test_fast_threshold_search_matches_exhaustive_search():
    rng = np.random.default_rng(7)
    y, p = rng.integers(0, 2, 100), rng.uniform(0, 1, 100)
    candidates = np.unique(np.r_[0, 0.5, 1, p])
    best = max(f1_score(y, p >= t) for t in candidates)
    expected = min(
        [t for t in candidates if f1_score(y, p >= t) == best], key=lambda t: (abs(t - 0.5), -t)
    )
    assert select_f1_threshold(y, p) == expected


def test_shared_cv_and_binary_oof_refit_save_all_predictions(dataset):
    calls = []

    class FakeEstimator:
        classes_ = [1, 0]  # Deliberately reversed to test probability alignment.

        def fit(self, x, y):
            calls.append(set(x.index))
            assert len(x) == len(y)

        def predict_proba(self, x):
            p = 0.1 + 0.8 * (x.n.to_numpy() % 2)
            return np.column_stack([p, 1 - p])

    features = dataset.features[["n"]].copy()
    features.index = [f"csv:{i}" for i in range(100)]
    specification = shared_splits(dataset)
    assert specification == shared_splits(dataset)
    with pytest.raises(RuntimeError, match="disabled"):
        evaluate_cv(
            dataset,
            features,
            FakeEstimator,
            specification=specification,
            model_id="fake",
            checkpoint_id="fake",
            feature_set="n",
        )
    records = evaluate_cv(
        dataset,
        features,
        FakeEstimator,
        specification=specification,
        model_id="fake",
        checkpoint_id="fake",
        feature_set="n",
        allow_model_runs=True,
        estimator_settings={"fixture": "fake"},
        feature_artifact_ids=[],
    )
    assert len(records) == 10 and len(calls) == 20
    for i, fold in enumerate(specification["folds"]):
        outer_train = {f"csv:{p}" for p in fold["train"]}
        outer_test = {f"csv:{p}" for p in fold["test"]}
        for fitted in calls[i * 4 : i * 4 + 3]:
            assert fitted < outer_train and not fitted & outer_test
        assert calls[i * 4 + 3] == outer_train
    outer_ids = []
    for path in records:
        record = json.loads(path.read_text())
        assert path.is_relative_to(paths.large_outputs_dir())
        table = pd.read_parquet(path.parent / "predictions.parquet")
        if record["metadata"]["prediction_role"] == "outer_test":
            outer_ids += table.row_id.tolist()
            assert {
                "text_present",
                "joint_text_seen_in_train",
                "any_text_column_seen_in_train",
            } <= set(table)
            assert "text_overlap" in record["scores"]
        assert record["metadata"]["selected_threshold"] >= 0
    assert sorted(outer_ids) == sorted(features.index)


@pytest.mark.parametrize("task", ["regression", "multiclass_classification"])
def test_other_tasks_use_no_binary_inner_fitting(dataset, task):
    dataset.metadata = replace(
        dataset.metadata, task_type=task, classes=() if task == "regression" else (0, 1, 2)
    )
    dataset.frame["y"] = np.arange(100, dtype=float) if task == "regression" else np.arange(100) % 3
    calls = []

    class FakeEstimator:
        classes_ = [0, 1, 2]

        def fit(self, x, y):
            calls.append(len(x))

        def predict(self, x):
            return np.zeros(len(x))

        def predict_proba(self, x):
            return np.full((len(x), 3), 1 / 3)

    features = dataset.features[["n"]].copy()
    features.index = [f"csv:{i}" for i in range(100)]
    records = evaluate_cv(
        dataset,
        features,
        FakeEstimator,
        specification=shared_splits(dataset),
        model_id="fake",
        checkpoint_id="fake",
        feature_set="n",
        allow_model_runs=True,
        estimator_settings={"fixture": "fake"},
        feature_artifact_ids=[],
    )
    assert len(records) == 5 and calls == [80] * 5


def test_rare_classes_require_split_review(dataset):
    dataset.frame.loc[1:, "y"] = 0
    with pytest.raises(ValueError, match="Every class"):
        shared_splits(dataset)


class NoWait:
    def acquire(self):
        pass


@pytest.fixture
def intended_request(dataset):
    return build_requests(dataset.features, dataset.metadata, 0, load_config("feature_creation"))[0]


@pytest.fixture
def limits():
    return ExtractionLimits(0.1, 1, 100, 10, 1000, workers=2)


class FakeClient:
    def __init__(self, failure=None):
        self.calls = 0
        self.failure = failure

    def evaluate(self, intended_request):
        self.calls += 1
        if self.failure == "retry" and self.calls == 1:
            raise RejectedRetryable()
        if self.failure == "timeout":
            raise TimeoutError()
        return {
            "model": "jev-1.13.0",
            "answers": {"semantic": {"type": "noul", "noul": 0.4}},
            "usage": {"input_tokens": 10},
        }


def make_extractor(tmp_path, client, limits):
    return DatasetExtractor(
        ResponseCache(tmp_path / "fake.sqlite3", "fake"),
        client,
        limits,
        gate=NoWait(),
        validate_context=lambda r: None,
    )


def test_network_guard_precedes_key_or_sdk_access():
    with pytest.raises(RuntimeError, match="disabled"):
        TypeSafeTransport(load_config("feature_creation"))


def test_extraction_deduplicates_and_charges_measured_usage(tmp_path, intended_request, limits):
    client = FakeClient()
    extractor = make_extractor(tmp_path, client, limits)
    assert extractor.evaluate(intended_request) == "completed"
    assert extractor.evaluate(intended_request) == "cache_hit" and client.calls == 1
    with extractor.cache._connect() as db:
        spent, reserved, attempts = db.execute(
            "SELECT spent_usd,reserved_usd,attempts FROM extraction_jobs"
        ).fetchone()
    assert spent == pytest.approx(0.00001) and reserved == 0 and attempts == 1


def test_known_rejection_retries_but_timeout_does_not(tmp_path, intended_request, limits):
    client = FakeClient("retry")
    extractor = make_extractor(tmp_path, client, limits)
    assert extractor.evaluate(intended_request) == "completed" and client.calls == 2
    second = tmp_path / "second"
    client = FakeClient("timeout")
    extractor = make_extractor(second, client, limits)
    assert extractor.evaluate(intended_request) == "ambiguous"
    restarted = make_extractor(second, client, limits)
    assert restarted.evaluate(intended_request) == "ambiguous" and client.calls == 1


def test_budget_stops_before_sending_and_cannot_reset(tmp_path, intended_request, limits):
    tiny = replace(limits, dataset_budget_usd=0.00001)
    client = FakeClient()
    extractor = make_extractor(tmp_path, client, tiny)
    assert extractor.evaluate(intended_request) == "budget_stop" and client.calls == 0
    with pytest.raises(ValueError, match="permanent allocation"):
        make_extractor(tmp_path, client, limits)


def test_parallel_dispatch_reuses_identical_input(tmp_path, intended_request, limits):
    client = FakeClient()
    extractor = make_extractor(tmp_path, client, limits)
    counts = extractor.run(iter([intended_request] * 40))
    assert client.calls == 1 and sum(counts.values()) == 40


def test_missing_usage_keeps_spending_reserved(tmp_path, intended_request, limits):
    class MissingUsage(FakeClient):
        def evaluate(self, intended_request):
            value = super().evaluate(intended_request)
            value.pop("usage")
            return value

    extractor = make_extractor(tmp_path, MissingUsage(), limits)
    assert extractor.evaluate(intended_request) == "ambiguous_usage"
    with extractor.cache._connect() as db:
        assert db.execute("SELECT reserved_usd FROM extraction_jobs").fetchone()[0] > 0
    assert extractor.cache.get(intended_request, kind="verified_response") is None


def test_campaign_cannot_exceed_owner_spending_allocation():
    cfg = load_config("feature_creation")
    review = cfg["live_extraction"]
    review.update(
        dataset_budget_allocations_usd={"a": 0.7, "b": 0.7}, campaign_spending_limit_usd=1
    )
    with pytest.raises(ValueError, match="exceed"):
        campaign_limits(cfg)


def test_recorded_raw_response_recovers_without_another_call(tmp_path, intended_request, limits):
    client = FakeClient()
    extractor = make_extractor(tmp_path, client, limits)
    assert extractor._reserve(intended_request) == "claimed"
    raw = client.evaluate(intended_request)
    assert extractor._update(intended_request, "received", response=raw) == "received"
    restarted = make_extractor(tmp_path, client, limits)
    assert restarted.evaluate(intended_request) == "recovered_response" and client.calls == 1
    assert restarted.cache.get(intended_request, kind="verified_response") == raw


def test_six_offline_mappings_share_identical_questions(intended_request):
    catalog = load_yaml("extractors")
    assert len(catalog["candidates"]) == 6 and catalog["enabled"] is False
    original = preview(intended_request, "jev")["body"]["questions"]
    for provider, candidate in catalog["candidates"].items():
        assert candidate["enabled"] is False
        mapped = preview(intended_request, provider)
        if provider == "perplexity":
            assert mapped["arguments"]["rows"][0]["question"] == original["semantic"]
        else:
            body = mapped.get("body", mapped.get("arguments"))
            assert body["questions"] == original
    with pytest.raises(ValueError, match="Unknown extractor"):
        preview(intended_request, "microsoft")


def test_batch_cache_lookup_does_not_mutate_origins(tmp_path, dataset):
    cfg = load_config("feature_creation")
    cfg["jev"]["representation"] = "per_column"
    requests = [build_requests(dataset.features, dataset.metadata, p, cfg)[0] for p in (0, 3)]
    cache = ResponseCache(tmp_path / "cache.sqlite3", "fake")
    supplied = FakeClient().evaluate(requests[0])
    cache.put(requests[0], supplied, kind="verified_response")
    assert cache.get_many(requests) == {requests[0].cache_key: supplied}
    with cache._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM origins").fetchone()[0] == 1
