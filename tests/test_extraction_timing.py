"""Timing evidence must distinguish new work, reuse and failures without changing features."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from src.utils import paths
from src.utils.timing import TimingJournal


@pytest.mark.parametrize("phase", ["experiment_0", "feature_creation"])
def test_journal_counts_batches_reuse_failures_and_routes_tiers(isolated_output, monkeypatch, phase):
    journal = TimingJournal("fixture", phase=phase)
    clock = iter([10.0, 12.0, 15.0, 18.0])
    monkeypatch.setattr("src.utils.timing.time.perf_counter", lambda: next(clock))
    sync = []
    with journal.measure(
        "inference", items=3, dataset_id="data", synchronize=lambda: sync.append(1)
    ):
        pass
    journal.record("cache_reuse", items=4, dataset_id="data")
    with (
        pytest.raises(ValueError, match="failure"),
        journal.measure("inference", dataset_id="data"),
    ):
        raise ValueError("failure")
    report = journal.write_summary()
    stages = {r["stage"]: r for r in report["stages"]}
    inference, reuse = stages["inference"], stages["cache_reuse"]
    assert inference["observations"] == 2 and inference["items"] == 4
    assert inference["wall_seconds"] == 5 and inference["failures"] == 1
    assert inference["cuda_synchronized_observations"] == 1 and len(sync) == 2
    assert reuse["stage"] == "cache_reuse" and reuse["wall_seconds"] is None
    assert journal.path.is_relative_to(paths.large_outputs_dir())
    manifest = next(paths.manifests_dir(phase).glob("timings/*.json"))
    assert json.loads(manifest.read_text())["database"] == str(journal.path)


def test_concurrent_journal_and_transfer_preserve_every_observation(
    isolated_output, tmp_path, monkeypatch
):
    from src.cluster.transfer import export_bundle, import_bundle

    journal = TimingJournal("fixture", phase="experiment_0")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda i: journal.record("inference", seconds=i / 100, item_keys=[str(i)]),
                range(130),
            )
        )
    assert journal.write_summary()["stages"][0]["observations"] == 130
    bundle = export_bundle()
    relative = journal.path.relative_to(paths.large_outputs_dir())
    monkeypatch.delenv("VSC_DATA")
    monkeypatch.delenv("JEVPFN_STAGING_ROOT")
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path / "imported")
    import_bundle(bundle)
    with sqlite3.connect(paths.outputs_dir() / relative) as db:
        assert db.execute("SELECT COUNT(*) FROM measurements").fetchone()[0] == 130


def test_api_timing_failure_keeps_paid_response_recoverable(isolated_output, tmp_path, monkeypatch):
    from src.jev.cache import ResponseCache
    from src.jev.extraction import DatasetExtractor, ExtractionLimits
    from tests.test_spending_policy import request

    response = {
        "model": "jev-1.13.0",
        "answers": {"semantic": {"type": "noul", "noul": 0.4}},
        "usage": {"input_tokens": 10},
    }
    calls = []
    client = SimpleNamespace(evaluate=lambda r: calls.append(r.cache_key) or response)
    journal = TimingJournal("jev", phase="experiment_0")
    record = journal.record

    def broken_record(stage, **kw):
        if stage == "api_request":
            raise OSError("timing storage unavailable")
        return record(stage, **kw)

    monkeypatch.setattr(journal, "record", broken_record)
    extractor = DatasetExtractor(
        ResponseCache(tmp_path / "cache.sqlite3", "fixture"),
        client,
        ExtractionLimits(1, 0.042, 100, 10, 1000),
        gate=SimpleNamespace(acquire=lambda: None),
        validate_context=lambda _: None,
        timings=journal,
    )
    with pytest.raises(OSError, match="timing storage"):
        extractor.evaluate(request())
    monkeypatch.setattr(journal, "record", record)
    assert extractor.evaluate(request()) == "recovered_response"
    assert len(calls) == 1
    with extractor.cache._connect() as db:
        assert db.execute("SELECT spent_usd FROM extraction_jobs").fetchone()[0] > 0


def test_embedding_timing_reuse_does_not_change_artifact(isolated_output, tmp_path):
    from src.data import text_embeddings as embeddings
    from src.utils.config import load_yaml
    from tests.test_frozen_text import Encoder, fixture_data

    encoder = Encoder()
    encoder.timings = TimingJournal("embeddings", phase="experiment_0")
    cfg = load_yaml("feature_creation/text_embeddings") | {"dimensions": 3, "batch_size": 2}
    path = tmp_path / "features.sqlite3"
    first = embeddings.build(fixture_data(), cfg, encoder, destination=path)
    assert embeddings.build(fixture_data(), cfg, encoder, destination=path) == first
    stages = {r["stage"]: r for r in encoder.timings.summary()["stages"]}
    assert stages["embedding_batch"]["items"] == 3
    assert stages["cache_reuse"]["items"] == 3
    assert len(encoder.calls) == 3


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_three_task_metric_sets_are_retained_in_prediction_store(isolated_output, task):
    import numpy as np
    import pandas as pd

    from src.evaluation.metrics import classification_metrics, regression_metrics
    from src.evaluation.records import PhaseTimer, save_evaluation
    from src.evaluation.store import read_predictions

    y = [0, 1, 2] if task != "binary" else [0, 1, 0]
    if task == "regression":
        prediction = np.array([0.1, 0.8, 2.1])
        scores = regression_metrics(y, prediction)
        assert {"mae", "rmse", "r2", "spearman_rho"} <= scores["metrics"].keys()
    else:
        k = 2 if task == "binary" else 3
        prediction = np.full((3, k), 0.1 / (k - 1))
        prediction[np.arange(3), y] = 0.9
        scores = classification_metrics(
            y,
            prediction,
            classes=list(range(k)),
            positive_class=1 if k == 2 else None,
            selected_threshold=0.4 if k == 2 else None,
        )
        assert {"log_loss", "brier_multiclass_sum", "ece"} <= scores["metrics"].keys()
        assert all("f1_macro" in r["metrics"] for r in scores["decisions"].values())
    table = pd.DataFrame({"row_id": ["csv:0", "csv:1", "csv:2"], "y_true": y})
    if task == "regression":
        table["prediction"] = prediction
    else:
        for i in range(prediction.shape[1]):
            table[f"p_c{i:03d}"] = prediction[:, i]
    ref = save_evaluation(
        run_id=task,
        scores=scores,
        predictions=table,
        timer=PhaseTimer(),
        metadata=dict(
            dataset_id="fixture",
            data_identity="fixed",
            model_id="fixture",
            feature_set="non_text",
            outer_fold=0,
            split_id="fixed",
            n_train=9,
            n_validation=0,
            checkpoint_id="fixed",
            target_scale="original",
        ),
    )
    assert json.loads(ref.read_text())["scores"] == scores
    pd.testing.assert_frame_equal(read_predictions(ref), table)
