"""Known predictions and failure cases, without fitting any estimator."""

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import classification_metrics, regression_metrics
from src.evaluation.records import PhaseTimer, save_evaluation


def test_binary_probability_metrics_are_independent_of_threshold_and_column_order():
    y = ["yes", "no", "yes", "no"]
    # Positive class is FIRST, intentionally unlike a default sorted/sklearn convention.
    p = np.array([[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.1, 0.9]])
    result = classification_metrics(
        y, p, classes=["yes", "no"], positive_class="yes", selected_threshold=0.3
    )
    assert result["metrics"]["roc_auc"] == pytest.approx(0.75)
    assert result["metrics"]["brier"] == pytest.approx(0.175)
    assert result["metrics"]["brier_multiclass_sum"] == pytest.approx(0.35)
    assert result["decisions"]["default_0_5"]["metrics"]["f1"] == pytest.approx(2 / 3)
    assert result["decisions"]["validation_selected"]["metrics"]["f1"] == pytest.approx(0.8)
    reversed_result = classification_metrics(
        y, p[:, ::-1], classes=["no", "yes"], positive_class="yes", selected_threshold=0.3
    )
    assert result["metrics"] == reversed_result["metrics"]
    assert sum(b["count"] for b in result["reliability_bins"]) == len(y)


def test_multiclass_metrics_keep_declared_vocabulary_and_brier_scale():
    p = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]
    result = classification_metrics(["a", "b", "c"], p, classes=["a", "b", "c"])
    assert result["metrics"]["brier_multiclass_sum"] == pytest.approx((0.14 + 0.06 + 0.24) / 3)
    assert result["metrics"]["roc_auc_ovr_macro"] == 1
    assert result["metrics"]["roc_auc_ovo_weighted"] == 1
    assert result["decisions"]["argmax"]["metrics"]["f1_macro"] == 1
    assert result["metrics"]["top_3_accuracy"] is None
    assert "top_3_accuracy" in result["undefined"]


def test_absent_classes_and_constant_targets_are_explicitly_undefined():
    r = classification_metrics(["a", "a"], [[0.7, 0.2, 0.1]] * 2, classes=["a", "b", "c"])
    assert r["metrics"]["roc_auc_ovr_macro"] is None
    assert r["metrics"]["log_loss"] == pytest.approx(-np.log(0.7))
    assert r["decisions"]["argmax"]["per_class"][1]["support"] == 0
    r = regression_metrics([1, 1], [2, 3])
    assert r["metrics"]["r2"] is None and r["metrics"]["spearman_rho"] is None
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize("p", [[[0.4, 0.4]], [[np.nan, 0.4]], [[1.1, -0.1]]])
def test_invalid_probabilities_are_not_silently_repaired(p):
    with pytest.raises(ValueError):
        classification_metrics([0], p, classes=[0, 1], positive_class=1)


def test_binary_positive_class_is_mandatory_and_multiclass_threshold_is_rejected():
    with pytest.raises(ValueError, match="positive_class"):
        classification_metrics([0], [[0.4, 0.6]], classes=[0, 1])
    with pytest.raises(ValueError, match="multiclass"):
        classification_metrics([0], [[0.2, 0.3, 0.5]], classes=[0, 1, 2], selected_threshold=0.5)


def test_regression_known_errors_and_domain_restrictions():
    r = regression_metrics([1, 2, 3], [1, 3, 2])
    assert r["metrics"]["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert r["metrics"]["mae"] == pytest.approx(2 / 3)
    assert r["metrics"]["r2"] == 0
    assert r["metrics"]["pearson_r"] == pytest.approx(0.5)
    assert r["metrics"]["mape_pct"] is None  # Positive data alone do not establish units.
    explicit = regression_metrics([1, 2, 3], [1, 3, 2], positive_ratio_scale=True)
    assert explicit["metrics"]["mape_pct"] == pytest.approx((0 + 0.5 + 1 / 3) / 3 * 100)
    assert (
        regression_metrics([0, 1], [0, -1], nonnegative_ratio_scale=True)["metrics"]["rmsle"]
        is None
    )


def test_quantile_metrics_are_conditional_and_crossing_is_rejected():
    r = regression_metrics([1, 2], [1, 2], quantiles={0.1: [0, 1], 0.9: [2, 3]})
    assert r["interval_metrics"]["0.8"]["coverage"] == 1
    assert r["interval_metrics"]["0.8"]["mean_width"] == 2
    assert r["quantile_metrics"]["0.1"]["pinball_loss"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="cross"):
        regression_metrics([1], [1], quantiles={0.1: [2], 0.9: [0]})


def test_timing_synchronizes_cuda_and_records_failed_attempts(monkeypatch):
    from src.evaluation import records

    wall = iter([1.0, 3.0, 4.0, 7.0])
    cpu = iter([1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(records.time, "perf_counter", lambda: next(wall))
    monkeypatch.setattr(records.time, "process_time", lambda: next(cpu))
    sync = []
    timer = PhaseTimer(device="cuda:0", synchronize=lambda: sync.append(True))
    with timer.measure("fit"):
        pass
    with pytest.raises(RuntimeError), timer.measure("fit"):
        raise RuntimeError("failed attempt")
    assert timer.to_dict()["stages"]["fit"] == {
        "wall_seconds": 5.0,
        "cpu_seconds": 2.0,
        "attempts": 2,
        "failures": 1,
    }
    assert len(sync) == 3  # Before both attempts, after successful device work.
    with pytest.raises(ValueError, match="synchronization"):
        PhaseTimer(device="cuda")


def test_result_record_preserves_predictions_timings_and_undefined_values(tmp_path):
    predictions = pd.DataFrame(
        {"row_id": ["csv:0", "csv:1"], "y_true": [1.0, 1.0], "prediction": [1.0, 2.0]}
    )
    scores = regression_metrics(predictions.y_true, predictions.prediction)
    timer = PhaseTimer()
    with timer.measure("metric_evaluation"):
        pass
    metadata = dict(
        dataset_id="test",
        model_id="no_model_fixture",
        feature_set="non_text",
        outer_fold=0,
        split_id="fixture",
        n_train=0,
        n_validation=0,
        checkpoint_id="none",
        target_scale="original",
    )
    path = save_evaluation(
        tmp_path, scores=scores, predictions=predictions, timer=timer, metadata=metadata
    )
    data = json.loads(path.read_text())
    assert data["scores"]["metrics"]["r2"] is None
    assert "fit" not in data["timing"]["stages"]  # No fabricated zero fit time.
    assert data["resources"]["peak_rss_bytes"] is None
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "predictions.parquet"), predictions)
    with pytest.raises(FileExistsError):
        save_evaluation(
            tmp_path, scores=scores, predictions=predictions, timer=timer, metadata=metadata
        )
