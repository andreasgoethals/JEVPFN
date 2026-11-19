"""Exp 4 appendix candidate: decision models scored directly on outer-test rows (synthetic)."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import log_loss, roc_auc_score

from src.evaluation import zero_shot as z


def _dataset(task, target, classes=(), text_columns=("text",)):
    frame = pd.DataFrame({"y": target, **{c: "x" for c in text_columns}})
    metadata = SimpleNamespace(dataset_id="D", task_type=task, target="y", classes=list(classes),
                               text_columns=list(text_columns))
    return SimpleNamespace(metadata=metadata, frame=frame)


def _block(columns, values, mode="joint", text_columns=("text",)):
    n = len(next(iter(values.values())))
    table = pd.DataFrame({"row_id": [f"csv:{i}" for i in range(n)], **values})
    manifest = {"feature_columns": list(columns), "text_columns": list(text_columns),
                "identity": {"representation": mode}}
    return SimpleNamespace(values=table, manifest=manifest)


SPLITS = {"folds": [{"test": list(range(0, 10))}, {"test": list(range(10, 20))}]}


def test_binary_scores_match_sklearn_and_fill_missing_text_with_uniform():
    rng = np.random.default_rng(0)
    y = np.tile([0, 1], 10)
    p = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.15, 20), 0.01, 0.99)
    p[3] = np.nan  # A row without usable text.
    block = _block(["m__joint__text_only__all_text__p_positive"],
                   {"m__joint__text_only__all_text__p_positive": p})
    table = z.dataset_scores(_dataset("binary_classification", y, (0, 1)), SPLITS, {"m": (block, "joint")})
    rows = table.set_index("metric")
    filled = np.where(np.isnan(p), 0.5, p)
    expected = np.mean([roc_auc_score(y[f["test"]], filled[f["test"]]) for f in SPLITS["folds"]])
    assert rows.loc["roc_auc", "value"] == pytest.approx(expected)
    probs = np.column_stack([1 - filled, filled])
    expected_loss = np.mean([log_loss(y[f["test"]], probs[f["test"]], labels=[0, 1]) for f in SPLITS["folds"]])
    assert rows.loc["log_loss", "value"] == pytest.approx(expected_loss)
    assert rows.loc["roc_auc", "coverage"] == pytest.approx(0.95)
    assert set(table.label) == {z.REFERENCE_LABEL}


def test_multiclass_macro_auc_and_regression_rank_only():
    rng = np.random.default_rng(1)
    y = np.tile(["a", "b", "c"], 7)[:20]
    names = [f"m__joint__text_only__all_text__p_c{i:03d}" for i in range(3)]
    raw = rng.dirichlet(np.ones(3), 20)
    block = _block(names, dict(zip(names, raw.T)))
    table = z.dataset_scores(_dataset("multiclass_classification", y, ("a", "b", "c")), SPLITS,
                             {"m": (block, "joint")}).set_index("metric")
    index = np.array([{"a": 0, "b": 1, "c": 2}[v] for v in y])
    expected = np.mean([roc_auc_score(index[f["test"]], raw[f["test"]], multi_class="ovr", average="macro",
                                      labels=[0, 1, 2]) for f in SPLITS["folds"]])
    assert table.loc["roc_auc_ovr_macro", "value"] == pytest.approx(expected)

    target = rng.normal(size=20)
    levels = [f"m__joint__text_only__all_text__p_{i}" for i in range(9)]
    probs = rng.dirichlet(np.ones(9), 20)
    block = _block(levels, dict(zip(levels, probs.T)))
    table = z.dataset_scores(_dataset("regression", target), SPLITS, {"m": (block, "joint")})
    assert list(table.metric) == [z.REGRESSION_METRIC]  # Rank only: no R², no log loss.
    score = probs @ z.LEVELS
    expected = np.mean([spearmanr(score[f["test"]], target[f["test"]]).statistic for f in SPLITS["folds"]])
    assert table.value.iloc[0] == pytest.approx(expected)


def test_per_column_only_with_a_single_text_column_and_pending_providers():
    y = np.tile([0, 1], 10)
    columns = [f"m__per_column__text_only__text_{i:03d}__p_positive" for i in range(2)]
    two = _block(columns, {c: np.full(20, 0.6) for c in columns}, mode="per_column",
                 text_columns=("t1", "t2"))
    dataset = _dataset("binary_classification", y, (0, 1), text_columns=("t1", "t2"))
    table = z.dataset_scores(dataset, SPLITS, {"m": (two, "per_column"), "absent": (None, "artifact_not_found")})
    status = table.set_index("provider").status
    assert status["m"].startswith("not_scored") and status["absent"] == "pending: artifact_not_found"
    loaded = []

    def loader(dataset_id, provider, mode, cfg):
        loaded.append(mode)
        return None, "artifact_not_found"

    z._blocks(_dataset("binary_classification", y, (0, 1)), ["m"], {}, loader)
    assert loaded == ["joint", "per_column"]  # Fallback only with one text column.
    loaded.clear()
    z._blocks(dataset, ["m"], {}, loader)
    assert loaded == ["joint"]


def test_references_pending_without_results_and_figure_labels():
    references, status = z.structured_references(pd.DataFrame())
    assert references.empty and status.startswith("pending")
    from src.visualize import style
    from src.visualize import zero_shot as plots

    style.apply()
    scores = pd.DataFrame([
        {"dataset_id": "BIN_TEXT_FAKE_JOB_POSTING", "task_type": "binary_classification", "provider": "kev",
         "metric": "roc_auc", "value": 0.7, "status": "descriptive"},
        {"dataset_id": "BIN_TEXT_FAKE_JOB_POSTING", "task_type": "binary_classification", "provider": "kev",
         "metric": "log_loss", "value": 0.3, "status": "descriptive"},
        {"dataset_id": "REG_TEXT_BOOK_PRICE", "task_type": "regression", "provider": "kev",
         "metric": z.REGRESSION_METRIC, "value": 0.4, "status": "descriptive"},
    ])
    figures = list(plots.direct_score_figures(scores, references))
    assert [name for _, name, _ in figures] == ["direct_decision_scores_classification",
                                                 "direct_decision_scores_regression"]
    for fig, _, caption in figures:
        assert z.REFERENCE_LABEL in fig._suptitle.get_text()
        assert caption and "Reference only" not in caption  # Captions describe, titles label.
        plt.close(fig)
    assert "pending" in figures[0][2]
