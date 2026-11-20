"""Group A (A1 outputs, A2 stability, A3 geometry): synthetic checks of the new analyses."""

import copy
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score

from src.evaluation import comparison_report, geometry_export, robustness_export, study_export
from src.evaluation import zero_shot as z
from src.evaluation.decision_comparison import (
    TIED,
    Comparison,
    GroupFeatures,
    analyze_group,
    consensus_distance,
    earth_movers,
    leave_one_out_consensus,
    output_shape,
    top_answers,
)
from src.evaluation.model_properties import compute_per_thousand, profile_table
from src.utils.config import load_config

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def options(providers):
    result = copy.deepcopy(load_config("feature_comparison")["comparison"])
    result["providers"] = list(providers)
    result["scatter_rows_per_group"] = 2
    return result


def group(provider, p, task):
    p = np.asarray(p, dtype=float)
    labels = () if task == "regression" else tuple(range(p.shape[1]))
    return GroupFeatures(provider, np.array([f"csv:{i}" for i in range(len(p))]), p,
                         np.array([f"input{i}" for i in range(len(p))], dtype=object), labels, task, "recipe")


CONTEXT = {"dataset_id": "fixture", "dataset": "Fixture", "representation": "joint", "group": "all_text",
           "text_column": "All text"}


# ---------------------------------------------------------------- A1: distances and ties

def test_earth_movers_distance_matches_wasserstein_on_ordered_levels():
    rng = np.random.default_rng(0)
    p, q = rng.dirichlet(np.ones(9), 20), rng.dirichlet(np.ones(9), 20)
    levels = np.arange(-4, 5)
    expected = [wasserstein_distance(levels, levels, a, b) for a, b in zip(p, q)]
    assert earth_movers(p, q) == pytest.approx(expected)
    point = np.eye(9)
    assert earth_movers(point[4], point[6]) == pytest.approx(2)  # Level 0 to level +2.
    assert earth_movers(point[0], point[8]) == pytest.approx(8)  # The whole scale.
    assert earth_movers(point[3], point[3]) == 0
    # Unlike total variation, a near miss is closer than a far miss.
    assert earth_movers(point[4], point[5]) < earth_movers(point[4], point[8])


def test_top_answers_abstain_on_exact_and_roundoff_ties():
    assert top_answers(np.array([[0.5, 0.5], [0.3, 0.7], [0.4, 0.6]])).tolist() == [TIED, 1, 1]
    assert top_answers(np.array([[0.15, 0.15, 0.7], [0.2, 0.4, 0.4]])).tolist() == [2, TIED]
    # An averaged consensus can miss an exact tie by float roundoff only: still a tie.
    assert 0.1 + 0.2 != 0.3
    assert top_answers(np.array([[0.1 + 0.2, 0.3, 0.4 - 0.1 - 0.3 + 0.1]])).tolist() == [TIED]


def test_leave_one_out_consensus_excludes_own_model_and_requires_two_others():
    ps = np.array([
        [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]],
        [[0.7, 0.3], [0.4, 0.6], [np.nan, np.nan]],
        [[0.5, 0.5], [0.6, 0.4], [0.1, 0.9]],
    ])
    present = np.isfinite(ps).all(axis=2)
    consensus, eligible = leave_one_out_consensus(ps, present, "binary_classification", 2)
    assert consensus[0, 0] == pytest.approx([0.6, 0.4])  # Mean of models 1 and 2 only.
    assert consensus[2, 1] == pytest.approx([0.3, 0.7])
    # Row 2 has only two available models: each has one other, below the minimum.
    assert eligible[:, 2].tolist() == [False, False, False]
    assert np.isnan(consensus[:, 2]).all()
    _, relaxed = leave_one_out_consensus(ps, present, "binary_classification", 1)
    assert relaxed[:, 2].tolist() == [True, False, True]
    with pytest.raises(ValueError):
        leave_one_out_consensus(ps, present, "binary_classification", 0)


def test_consensus_distance_per_task():
    p, c = np.array([[0.2, 0.8]]), np.array([[0.5, 0.5]])
    assert consensus_distance(p, c, "binary_classification")["distance"] == pytest.approx([0.3])
    p3, c3 = np.array([[0.6, 0.3, 0.1]]), np.array([[0.2, 0.3, 0.5]])
    assert consensus_distance(p3, c3, "multiclass_classification")["distance"] == pytest.approx([0.4])
    levels = np.eye(9)
    regression = consensus_distance(levels[[2]], levels[[5]], "regression")
    assert regression["distance"] == pytest.approx([3])
    assert regression["expected_level_difference"] == pytest.approx([3])


def test_output_shape_and_regression_spread():
    p = np.array([[0.97, 0.03], [0.6, 0.4]])
    shape = output_shape(p, "binary_classification", 0.95)
    assert shape["mean_top_probability"] == pytest.approx(0.785)
    assert shape["confident_share"] == 0.5
    levels = np.zeros((2, 9))
    levels[0, 4] = 1  # All mass on level 0: spread 0.
    levels[1, [3, 5]] = 0.5  # Levels -1 and +1: SD 1, expected level 0.
    shape = output_shape(levels, "regression", 0.95)
    assert shape["mean_level_std"] == pytest.approx(0.5)
    assert shape["expected_level_median"] == 0


def test_analyze_group_records_pairs_loo_consensus_and_level_histogram():
    a = [[0.9, 0.1], [0.2, 0.8], [0.5, 0.5], [0.6, 0.4]]
    b = [[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.3, 0.7]]
    c = [[0.6, 0.4], [0.6, 0.4], [0.2, 0.8], [0.4, 0.6]]
    groups = [group(n, p, "binary_classification") for n, p in zip(("jev", "liquid", "strands"), (a, b, c))]
    result = Comparison()
    analyze_group(groups, CONTEXT, options(("jev", "liquid", "strands")), result)
    pairs = result.table("pairs").query("weighting == 'rows'").set_index(["model_a", "model_b"])
    # jev ties on row 2 and abstains; on the other three rows jev/liquid agree on two.
    assert pairs.loc[("jev", "liquid"), "top_tied_either_count"] == 1
    assert pairs.loc[("jev", "liquid"), "top_agreement"] == pytest.approx(2 / 3)
    loo = result.table("loo_consensus").query("weighting == 'rows'").set_index("provider")
    assert loo.loc["jev", "coverage"] == 1 and loo.loc["jev", "eligible"] == 4
    # jev vs mean(liquid, strands): rows 0 (0, 0), 1 (1, tie 0.45/0.55 -> 1), 3 (0 vs 1); row 2 jev tied.
    assert loo.loc["jev", "tied_either"] == 1
    assert loo.loc["jev", "top_agreement"] == pytest.approx(2 / 3)
    consensus = (np.array(b) + np.array(c)) / 2
    assert loo.loc["jev", "distance"] == pytest.approx(np.abs(np.array(a)[:, 1] - consensus[:, 1]).mean())
    shape = result.table("output_shape")
    assert set(shape.provider) == {"jev", "liquid", "strands"}

    levels = np.random.default_rng(3).dirichlet(np.ones(9), 6)
    reg = [group(n, np.roll(levels, k, axis=1), "regression") for k, n in enumerate(("jev", "liquid", "strands"))]
    result = Comparison()
    analyze_group(reg, CONTEXT, options(("jev", "liquid", "strands")), result)
    pair = result.table("pairs").query("weighting == 'rows' and model_a == 'jev' and model_b == 'liquid'").iloc[0]
    assert pair.earth_movers == pytest.approx(earth_movers(levels, np.roll(levels, 1, axis=1)).mean())
    histogram = result.table("level_histogram")
    assert histogram.groupby("provider")["count"].sum().eq(6).all()
    assert "expected_level_difference" in result.table("loo_consensus")


def test_output_figures_render_at_print_width():
    from src.visualize import outputs as plots
    from src.visualize import style

    style.apply()
    rng = np.random.default_rng(1)
    result = Comparison()
    for dataset, task, k in (("Bin", "binary_classification", 2), ("Mul", "multiclass_classification", 3),
                             ("Reg", "regression", 9)):
        groups = [group(n, rng.dirichlet(np.ones(k), 12), task) for n in ("jev", "liquid", "strands")]
        analyze_group(groups, {**CONTEXT, "dataset_id": dataset, "dataset": dataset},
                      options(("jev", "liquid", "strands")), result)
    shape, pairs, loo = result.table("output_shape"), result.table("pairs"), result.table("loo_consensus")
    figures = [*plots.output_shape_figure(shape), *plots.regression_levels_figure(shape),
               *plots.pairwise_figures(pairs), *plots.consensus_figure(loo)]
    assert [name for _, name, _ in figures] == [
        "output_shape_joint", "regression_levels_joint", "pairwise_agreement_binary_classification_joint",
        "pairwise_agreement_multiclass_classification_joint", "pairwise_agreement_regression_joint",
        "consensus_agreement_joint"]
    for fig, _, caption in figures:
        assert fig.get_size_inches()[0] <= 6.31 and caption
        plt.close(fig)
    assert "Qwen3.8-27B" in figures[-1][2] and "not truth" in figures[-1][2]
    table = comparison_report.pairwise_table(pairs)
    assert {"top_agreement", "total_variation", "earth_movers"} <= set(table)


# ---------------------------------------------------------------- A1: stand-alone quality (moved from Exp 4)

def _dataset(task, target, classes=()):
    frame = pd.DataFrame({"y": target, "text": "x"})
    metadata = SimpleNamespace(dataset_id="D", task_type=task, target="y", classes=list(classes),
                               text_columns=["text"])
    return SimpleNamespace(metadata=metadata, frame=frame)


def _block(columns, values, mode="joint"):
    n = len(next(iter(values.values())))
    table = pd.DataFrame({"row_id": [f"csv:{i}" for i in range(n)], **values})
    return SimpleNamespace(values=table, manifest={"feature_columns": list(columns), "text_columns": ["text"],
                                                   "identity": {"representation": mode}})


def test_standalone_quality_scores_joint_rows_only_without_imputation():
    from sklearn.metrics import brier_score_loss, log_loss

    rng = np.random.default_rng(0)
    y = np.tile([0, 1], 20)
    p = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.15, 40), 0.01, 0.99)
    p[[3, 8]] = np.nan  # Rows without a joint request are not scored.
    name = "m__joint__text_only__all_text__p_positive"
    scores, curves = z.dataset_scores(_dataset("binary_classification", y, (0, 1)), {"m": (_block([name], {name: p}), "s")})
    rows = scores.set_index("metric")
    keep = np.isfinite(p)
    assert rows.loc["roc_auc", "value"] == pytest.approx(roc_auc_score(y[keep], p[keep]))
    assert rows.loc["log_loss", "value"] == pytest.approx(log_loss(y[keep], np.column_stack([1 - p[keep], p[keep]])))
    assert rows.loc["brier", "value"] == pytest.approx(brier_score_loss(y[keep], p[keep]))
    assert rows.loc["roc_auc", "coverage"] == pytest.approx(38 / 40) and rows.loc["roc_auc", "scored_rows"] == 38
    assert curves["count"].sum() == 38 and set(curves.provider) == {"m"}
    used = curves[curves["count"] > 0]
    ece = (used["count"] * (used.accuracy - used.mean_confidence).abs()).sum() / 38
    assert rows.loc["ece", "value"] == pytest.approx(ece)
    per_column = _block(["m__per_column__text_only__text_000__p_positive"],
                        {"m__per_column__text_only__text_000__p_positive": p}, mode="per_column")
    with pytest.raises(ValueError, match="joint"):
        z.dataset_scores(_dataset("binary_classification", y, (0, 1)), {"m": (per_column, "s")})


def test_standalone_multiclass_ties_and_regression_rank_only():
    y = np.array([0, 1, 2, 0])
    p = np.array([[0.5, 0.5, 0.0], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.4, 0.3, 0.3]])
    assert z.top_label_correct(y, p).tolist() == [0.5, 1, 1, 1]
    names = [f"m__joint__text_only__all_text__p_c{i:03d}" for i in range(3)]
    scores, _ = z.dataset_scores(_dataset("multiclass_classification", np.array(list("abca")), "abc"),
                                 {"m": (_block(names, dict(zip(names, p.T))), "s")})
    assert {"roc_auc_ovr_macro", "log_loss", "brier", "ece"} == set(scores.metric)
    levels = [f"m__joint__text_only__all_text__p_{i}" for i in range(9)]
    probs = np.random.default_rng(2).dirichlet(np.ones(9), 10)
    scores, curves = z.dataset_scores(_dataset("regression", np.arange(10.0)),
                                      {"m": (_block(levels, dict(zip(levels, probs.T))), "s"), "x": (None, "absent")})
    assert scores.dropna(subset=["metric"]).metric.tolist() == [z.REGRESSION_METRIC] and curves.empty
    assert scores.set_index("provider").loc["x", "status"] == "pending: absent"


def test_standalone_analysis_moved_from_exp04_to_exp01():
    exp04 = (NOTEBOOKS / "exp04_decision_features.ipynb").read_text(encoding="utf-8")
    exp01 = (NOTEBOOKS / "exp01_outputs.ipynb").read_text(encoding="utf-8")
    assert "zero_shot" not in exp04 and "direct_score" not in exp04
    assert "direct_score_figures" in exp01 and "joint" in exp01
    assert not hasattr(z, "structured_references") and not hasattr(z, "prediction_matrix")
    from src.visualize import style
    from src.visualize import zero_shot as plots

    style.apply()
    scores = pd.DataFrame([
        {"dataset_id": "BIN_TEXT_FAKE_JOB_POSTING", "task_type": "binary_classification", "provider": "kev",
         "metric": m, "value": v, "status": "descriptive"} for m, v in (("roc_auc", .7), ("log_loss", .3), ("ece", .1))])
    reliability = pd.DataFrame([{"dataset_id": "BIN_TEXT_FAKE_JOB_POSTING", "provider": "kev", "bin": 9,
                                 "count": 3, "mean_confidence": .95, "accuracy": .8}])
    figures = list(plots.direct_score_figures(scores, reliability))
    assert [n for _, n, _ in figures] == ["standalone_quality_classification", "standalone_reliability"]
    for fig, _, caption in figures:
        assert z.REFERENCE_LABEL in fig._suptitle.get_text() and "Reference only" not in caption
        plt.close(fig)


# ---------------------------------------------------------------- A1: model profile

def test_compute_per_thousand_is_compute_only_and_never_invented():
    ledger = {"summary": [
        {"stage": "decision_feature_creation", "decision_model": "kev", "successful_jobs": 2, "successful_gpu_hours": 10.0},
        {"stage": "decision_feature_creation", "decision_model": "strands", "successful_jobs": 0, "successful_gpu_hours": 0.0},
        {"stage": "benchmark", "decision_model": None, "successful_jobs": 3, "successful_gpu_hours": 99.0}]}
    audit = {"self_hosted": [{"provider": "kev", "unique_inputs": 20000}, {"provider": "strands", "unique_inputs": 5}]}
    frame = compute_per_thousand(ledger, audit, ["kev", "strands", "jev"]).set_index("provider")
    assert frame.loc["kev", "gpu_hours_per_1000_requests"] == pytest.approx(0.5)
    assert np.isnan(frame.loc["strands", "gpu_hours_per_1000_requests"])
    assert frame.loc["jev", "compute_status"] == "no successful GPU job in the ledger"
    assert compute_per_thousand(None, audit, ["kev"]).compute_status.tolist() == ["no compute ledger"]
    assert not any("price" in c or "token" in c for c in frame.columns)
    profile = profile_table(frame.reset_index())
    assert {"parameters_billion", "readout_type", "architecture_family", "gpu_hours_per_1000_requests"} <= set(profile)


# ---------------------------------------------------------------- exports

def test_study_export_waiting_roundtrip_tamper_and_superseded(tmp_path):
    target = tmp_path / "x.zip"
    frames, receipt = study_export.read_export(target, schema="s.v1", tables=("a",))
    assert frames == {} and receipt["status"] == study_export.AWAITING
    assert "Waiting for export" in study_export.describe(receipt)
    study_export.write_export(target, {"a": pd.DataFrame({"v": [1, 2]})}, schema="s.v1", identity="id", origin="test")
    frames, receipt = study_export.read_export(target, schema="s.v1", tables=("a",), identity="id")
    assert frames["a"].v.tolist() == [1, 2] and receipt["status"] == "verified_export"
    assert receipt["sha256"] in study_export.describe(receipt) and receipt["created_utc"]
    with pytest.raises(ValueError, match="configuration"):
        study_export.read_export(target, schema="s.v1", tables=("a",), identity="other")
    assert study_export.read_export(target, schema="s.v2", tables=("a",))[1]["status"] == "superseded_schema"
    with ZipFile(target) as archive:
        members = {n: archive.read(n) for n in archive.namelist()}
    members["a.parquet"] += b"x"
    with ZipFile(target, "w") as archive:
        for n, data in members.items():
            archive.writestr(n, data)
    with pytest.raises(ValueError, match="checksum"):
        study_export.read_export(target, schema="s.v1", tables=("a",))
    with pytest.raises(ValueError, match="limit"):
        study_export.write_export(tmp_path / "big.zip", {"a": pd.DataFrame({"v": range(10)})}, schema="s",
                                  identity="i", origin="o", max_bytes=10)
    with pytest.raises(RuntimeError, match="Mindwell"):
        study_export.require_mindwell_cpu("test")


def test_previous_a1_schema_is_identified_not_read(tmp_path):
    target = tmp_path / "comparison.zip"
    study_export.write_export(target, {"inventory": pd.DataFrame({"provider": ["kev"]})},
                              schema="jevpfn.rq1a.report.v1", identity="old", origin="old")
    result, receipt = comparison_report.read_report(comparison_report.report_config(), target)
    assert receipt["status"] == "superseded_schema" and result.table("pairs").empty
    assert "waiting for a new export" in comparison_report.describe_receipt(receipt)
    _, receipt = comparison_report.read_report(comparison_report.report_config(), tmp_path / "none.zip")
    assert "Waiting for export" in comparison_report.describe_receipt(receipt)


def test_geometry_export_skips_pilots(tmp_path):
    common = {"data_identity": "d", "config": "c", "samples": [{"field": "f", "text_keys": ["a", "b"]}],
              "missing_providers": [], "encoder_references": []}
    (tmp_path / "A.json").write_text(json.dumps({**common, "dataset": "A", "pilot": False, "comparisons": [
        {"dataset": "A", "field": "f", "encoder": "e", "provider": "kev", "spearman": 0.4,
         "text_block_spearman_range": [0.3, 0.5]}]}))
    (tmp_path / "B.json").write_text(json.dumps({**common, "dataset": "B", "pilot": True, "comparisons": []}))
    tables = geometry_export.collect(tmp_path)
    assert tables["sources"].dataset.tolist() == ["A"] and tables["coverage"].texts.tolist() == [2]
    assert tables["comparisons"].text_block_spearman_max.tolist() == [0.5]


# ---------------------------------------------------------------- A2: averaging over presentations

def _measured(p):
    return {"status": "measured", "probabilities": list(p)}


def test_wording_averaging_compares_averaged_pairs_with_single_versions():
    trials = {1: {"original": _measured([0.8, 0.2]), "paraphrase_1": _measured([0.6, 0.4]),
                  "paraphrase_2": _measured([0.4, 0.6]), "paraphrase_3": _measured([0.6, 0.4]),
                  "repeat_1": _measured([0.8, 0.2]), "repeat_2": _measured([0.7, 0.3])},
              2: {"original": _measured([0.5, 0.5]), "paraphrase_1": _measured([0.5, 0.5])}}
    table = robustness_export.wording_records(trials, regression=False)
    assert table.input_no.tolist() == [1]  # Incomplete wording sets are excluded.
    row = table.iloc[0]
    # Cross pairs: |.8-.4|, |.8-.6|, |.6-.4|, |.6-.6| -> mean 0.2; averages .7 vs .5 -> 0.2 (TV of binary = |dp|).
    assert row.single_tv == pytest.approx(0.2) and row.averaged_tv == pytest.approx(0.2)
    assert row.replay_tv == pytest.approx(0.05)
    trials[1]["paraphrase_3"] = _measured([0.8, 0.2])  # Averaging now cancels part of the wording effect.
    row = robustness_export.wording_records(trials, regression=False).iloc[0]
    assert row.averaged_tv < row.single_tv
    levels = np.eye(9)
    reg = {1: {v: _measured(levels[i]) for v, i in zip(robustness_export.WORDINGS, (4, 5, 4, 5))}}
    row = robustness_export.wording_records(reg, regression=True).iloc[0]
    assert row.averaged_emd == 0 and row.single_emd == pytest.approx(0.5)


def test_averaged_presentation_quality_uses_joint_rows_and_labels_only_for_scoring():
    trials = {}
    rng = np.random.default_rng(4)
    for n in range(40):
        base = rng.uniform(0.2, 0.8)
        trials[n] = {v: _measured([1 - p, p]) for v, p in zip(
            (*robustness_export.WORDINGS, "field_order"), np.clip(base + rng.normal(0, 0.1, 5), 0.01, 0.99))}
        trials[n]["option_order"] = _measured([0.5, 0.5])
    trials[0]["field_order"] = {"status": "canonicalized_by_runtime"}
    found = robustness_export.averaged_presentations(trials)
    assert 0 not in found["field_order"] and len(found["wording"]) == 40
    original, average = found["wording"][5]
    assert average == pytest.approx(np.mean([trials[5][v]["probabilities"] for v in robustness_export.WORDINGS], axis=0))
    links = pd.DataFrame({"input_no": [*range(40), 0], "row_position": [*range(40), 40],
                          "mode": ["joint"] * 40 + ["per_column"]})
    targets = (np.array([found["wording"][n][0][1] for n in range(40)] + [0.5]) > 0.5).astype(int)
    rows = pd.DataFrame(robustness_export.quality_records(found, links, targets, "binary_classification", 10))
    original_auc = rows.query("factor == 'wording' and presentation == 'original' and metric == 'roc_auc'")
    assert original_auc.value.iloc[0] == pytest.approx(1.0) and original_auc.rows.iloc[0] == 40
    assert set(rows.factor) == {"wording", "option_order", "field_order"}
    assert rows.query("factor == 'field_order'").requests.iloc[0] == 39


def test_robustness_export_collect_reads_completed_stores_only(tmp_path, monkeypatch):
    def store(dataset, provider, pilot):
        folder = tmp_path / dataset / provider
        folder.mkdir(parents=True)
        trials = {n: {v: _measured([0.3 + 0.01 * n, 0.7 - 0.01 * n]) for v in
                      ("original", "repeat_1", "repeat_2", *robustness_export.WORDINGS[1:])} for n in range(6)}
        for n in trials:
            for v in trials[n]:
                if v != "original":
                    trials[n][v]["effect"] = {"tv": 0.01 * (v.startswith("para") + 1), "flip": False}
        with sqlite3.connect(folder / "run.sqlite3") as db:
            db.execute("CREATE TABLE results (input_no INTEGER, variant TEXT, data TEXT)")
            db.executemany("INSERT INTO results VALUES (?,?,?)",
                           [(n, v, json.dumps(d)) for n, item in trials.items() for v, d in item.items()])
        body = {"design": {"dataset": dataset, "provider": provider, "pilot": pilot, "source": "set"},
                "completed": True, "sample": {str(n): {} for n in trials},
                "counts": [{"variant": "original", "status": "measured", "count": 6}],
                "effects": {"paraphrase_1": {"measured": 6, "mean_tv": 0.02}},
                "replay_contrasts": {"paraphrase_1": {"matched_requests": 6, "excess_tv": 0.01}}}
        (folder / "run.json").write_text(json.dumps(body))

    store("D1", "kev", pilot=False)
    store("D2", "kev", pilot=True)
    links = pd.DataFrame({"input_no": range(6), "row_position": range(6), "mode": ["joint"] * 3 + ["per_column"] * 3})
    monkeypatch.setattr(robustness_export, "request_links", lambda *a: links)
    tables = robustness_export.collect(
        tmp_path, {"averaging": {"calibration_bins": 10}},
        load_targets=lambda dataset, _: ("binary_classification", np.array([0, 1, 0, 1, 0, 1])),
        question_path=lambda dataset: tmp_path / "unused")
    assert tables["sources"].dataset.tolist() == ["D1"]
    assert set(tables["wording_averaging"]["mode"]) == {"joint", "per_column"}
    modes = tables["mode_effects"].set_index(["mode", "variant"])
    assert modes.loc[("joint", "paraphrase_1"), "excess_tv"] == pytest.approx(0.01)
    assert set(tables["averaging_quality"].factor) == {"wording"}
    assert tables["averaging_quality"].rows.max() == 3  # Joint source rows only.
    target = tmp_path / "robustness.zip"
    tables.update(robustness_export.historical_tables((pd.DataFrame({"probe": ["p"]}), None, [])))
    study_export.write_export(target, tables, schema=robustness_export.SCHEMA, identity="i", origin="test")
    frames, receipt = study_export.read_export(target, schema=robustness_export.SCHEMA,
                                               tables=robustness_export.TABLES)
    assert receipt["status"] == "verified_export" and len(frames) == len(robustness_export.TABLES)


def test_behavior_config_declares_averaging_without_new_calls():
    from src.utils.config import load_yaml

    cfg = load_yaml("rq1/behavior")
    assert cfg["averaging"]["additional_model_calls"] is False and cfg["lexical_rewordings"] is False
    assert cfg["averaging"]["wording"]["averaged_pairs"] == [["original", "paraphrase_1"],
                                                             ["paraphrase_2", "paraphrase_3"]]
    assert cfg["requests_per_dataset"] == 500


def test_robustness_export_carries_historical_0a_probes_without_local_paths(tmp_path):
    """exp02 section 2 reads the dated 0A probes from the ZIP, never the local receipts."""
    changes = pd.DataFrame({"provider": ["jev"], "source_sha256": ["a" * 64],
                            "source": [r"C:\local\receipt.json"], "probe": ["real_X__base"],
                            "total_variation_from_base": [0.0], "same_top_option": [True]})
    sources = [{"path": r"C:\local\receipt.json", "sha256": "a" * 64, "created_utc": "t",
                "status": "s", "scope": "x", "input_sets": ["i"]}]
    tables = robustness_export.historical_tables((changes, None, sources))
    assert set(tables) == set(robustness_export.HISTORICAL)
    assert "source" not in tables["historical_0a"]
    assert tables["historical_0a_sources"].path.tolist() == ["receipt.json"]
    empty = {name: pd.DataFrame({"x": [1]}) for name in robustness_export.COLLECTED}
    target = study_export.write_export(tmp_path / "robustness.zip", {**empty, **tables},
                                       schema=robustness_export.SCHEMA, identity="i", origin="test")
    frames, _ = study_export.read_export(
        target, schema=robustness_export.SCHEMA, tables=robustness_export.TABLES)
    assert frames["historical_0a"].probe.tolist() == ["real_X__base"]
