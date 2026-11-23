"""Diagnostic semantics, row alignment, exact case provenance and rendering."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.evaluation.decision_comparison import Comparison, GroupFeatures, analyze_group
from src.evaluation.decision_diagnostics import (
    case_browser,
    correlation_vectors,
    inspect_cases,
    matching_correlations,
    peer_majority_agreement,
)
from src.utils.config import load_config
from src.visualize import style


def compare(values, *, task="binary_classification", hashes=None, mode="joint"):
    names = list(style.DECISION_COLORS)[:len(values)]
    groups = []
    for name, p in zip(names, values):
        p = np.asarray(p, dtype=float)
        if p.ndim == 1:
            p = np.column_stack([1-p, p])
        ids = np.array([f"csv:{i}" for i in range(len(p))])
        groups.append(GroupFeatures(name, ids, p, np.array(hashes or [f"h{i}" for i in range(len(p))], dtype=object),
                                    () if task == "regression" else tuple(range(p.shape[1])), task, "same"))
    cfg = {**load_config("feature_comparison")["comparison"], "providers": names}
    result = Comparison()
    analyze_group(groups, {"dataset": "Fixture", "dataset_id": "fixture", "representation": mode,
                           "group": "all_text", "text_column": "text"}, cfg, result)
    return result


def test_agreement_is_not_same_probabilities_and_missing_rows_stay_missing():
    result = compare([[.99, .5, np.nan, .1], [.51, .5, np.nan, .9]])
    rows = result.table("row_diagnostics")
    assert rows.loc[0, "unanimous"]
    assert rows.loc[0, "mean_pair_tv"] == pytest.approx(.48)
    assert rows.loc[1, "ambiguous"] and not rows.loc[1, "unanimous"]
    assert rows.loc[2, "status"] == "missing_text"
    assert pd.isna(rows.loc[2, "unanimous"])
    assert pd.isna(rows.loc[2, "choice_jev"])
    assert rows.loc[3, "vote_pattern"] == "1:1"
    assert result.table("row_summary").n.eq(3).all()
    assert result.table("row_summary").unanimous.eq(1/3).all()


def test_binary_strongest_split_and_multiclass_all_different():
    result = compare([[.9], [.8], [.7], [.1], [.2]])
    assert result.table("row_diagnostics").vote_pattern.item() == "3:2"
    values = [np.eye(5)[[i]] for i in range(5)]
    result = compare(values, task="multiclass_classification")
    assert result.table("row_diagnostics").vote_pattern.item() == "1:1:1:1:1"
    assert result.table("row_diagnostics").mean_pair_tv.item() == 1


def test_fully_missing_joint_group_is_exported_but_not_compared():
    result = compare([[np.nan, np.nan], [np.nan, np.nan]])
    assert len(result.table("row_diagnostics")) == 2
    assert result.table("row_diagnostics").status.eq("missing_text").all()
    assert result.table("row_summary").empty
    assert result.table("pairs").empty


def test_regression_same_direction_can_have_large_score_range():
    result = compare([np.eye(9)[[5]], np.eye(9)[[8]]], task="regression")
    row = result.table("row_diagnostics").iloc[0]
    assert row.unanimous and row.score_range == 3
    assert result.table("row_summary").within_tolerance.eq(0).all()


def test_duplicates_do_not_crowd_out_case_selection_or_reweight_unique_histograms():
    result = compare([[.9, .9, .1], [.9, .9, .9]], hashes=["repeat", "repeat", "other"])
    rows = result.table("row_diagnostics")
    assert rows.input_multiplicity.tolist() == [2, 2, 1]
    summary = result.table("row_summary").set_index("weighting")
    assert summary.loc["rows", "unanimous"] == pytest.approx(2/3)
    assert summary.loc["unique_inputs", "unanimous"] == .5
    density = result.table("confidence_density")
    assert np.allclose(density.groupby("weighting").fraction.sum(), 1)
    assert density.groupby("weighting")["count"].sum().to_dict() == {"rows": 3, "unique_inputs": 2}
    assert result.table("cases").query("category == 'disputed'").row_id.item() == "csv:2"


def test_matching_class_correlations_differ_from_identical_max_confidence():
    p = np.array([[.9, .1, 0], [.7, .3, 0], [.6, .4, 0]])
    q = p[:, [1, 0, 2]]
    result = matching_correlations(correlation_vectors(np.stack([p, q]), "multiclass_classification"), 0, 1)
    assert result["representation_spearman"] == pytest.approx(-1)
    assert result["representation_pearson"] == pytest.approx(-1)
    assert result["correlation_dimensions"] == 2
    assert result["correlation_total_dimensions"] == 3
    assert np.array_equal(p.max(axis=1), q.max(axis=1))


def test_constant_answer_agreement_does_not_imply_nontrivial_kappa():
    result = compare([[.9, .9, .9], [.8, .8, .8]])
    pairs = result.table("pairs")
    assert pairs.agreement.eq(1).all()
    assert pairs.marginal_expected_agreement.eq(1).all()
    assert pairs.cohen_kappa.isna().all()
    result = compare([[.9, .1], [.1, .9]])
    assert result.table("pairs").cohen_kappa.eq(-1).all()


def test_matching_correlations_match_scipy_with_ties_and_constants():
    p = np.array([[.8, .2], [.8, .2], [.3, .7], [.1, .9]])
    q = np.array([[.2, .8], [.7, .3], [.1, .9], [.9, .1]])
    result = matching_correlations(correlation_vectors(np.stack([p, q]), "binary_classification"), 0, 1)
    assert result["representation_spearman"] == pytest.approx(spearmanr(p[:, 1], q[:, 1]).statistic)
    constant = np.full((2, 3, 2), .5)
    result = matching_correlations(correlation_vectors(constant, "binary_classification"), 0, 1)
    assert np.isnan(result["representation_pearson"])
    assert result["correlation_dimensions"] == 0


def test_case_text_is_hash_matched(tmp_path, monkeypatch):
    from src.jev import inputs
    from src.utils import paths
    result = compare([[.9], [.1]])
    path = tmp_path / "questions.sqlite3"
    path.touch()
    monkeypatch.setattr(paths, "dataset_question_path", lambda _: path)
    monkeypatch.setattr(inputs, "prepared_row", lambda *a, **kw: [{"payload_sha256": "h0", "payload": {"text": "<script>bad</script>"}}])
    cases = inspect_cases(result)
    assert cases.payload_status.eq("Exact payload hash verified").all()
    monkeypatch.setattr(inputs, "prepared_row", lambda *a, **kw: [{"payload_sha256": "different", "payload": {"text": "wrong"}}])
    with pytest.raises(ValueError, match="does not match"):
        inspect_cases(result)


@pytest.mark.notebook
def test_case_browser_html_is_escaped():
    pytest.importorskip("IPython", reason="Rich display is a local notebook dependency")
    cases = compare([[.9], [.1]]).table("cases").copy()
    assert not cases.empty
    cases["payload_json"] = json.dumps({"text": "<script>bad</script>"})
    cases["payload_status"] = "Synthetic escaping test"
    html = case_browser(cases).data
    assert "<script>bad</script>" not in html
    assert "&lt;script&gt;bad&lt;/script&gt;" in html


def test_peer_majority_excludes_tied_peers():
    # Rows: all agree; model 0 alone dissents; the other two tie (one each).
    result = compare([[.9, .9, .9], [.9, .1, .1], [.9, .1, .9]])
    table = peer_majority_agreement(result).set_index("provider")
    first = list(style.DECISION_COLORS)[0]
    assert table.loc[first, "n"] == 2 and table.loc[first, "tied_peers"] == 1
    assert table.loc[first, "agreement"] == pytest.approx(.5)
