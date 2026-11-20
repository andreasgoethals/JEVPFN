"""Exp 8 additions (19 Nov 2026): model axis, statistics, complementarity, data map, figure map.

Synthetic inputs only; no benchmark result exists yet.
"""

import copy
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.evaluation import aggregate
from src.evaluation import complementarity as comp
from src.evaluation import moderator_statistics as ms
from src.evaluation import moderators as m
from src.evaluation import row_benefit as rb
from src.utils import paper_figures
from src.visualize import style
from src.visualize import when_text_helps as views

PROVIDERS = ["a", "b", "c", "d", "e", "f"]


def _paired(datasets=8, *, providers=PROVIDERS):
    """Paired fold rows: decision arms per provider and one embedding recipe per dataset."""
    rows = []
    rng = np.random.default_rng(4)
    for d in range(datasets):
        base = 0.55 + 0.04 * d
        arms = [("experiment_1", "non_text_plus_joint", p, None, i) for i, p in enumerate(providers)]
        arms.append(("experiment_2", "non_text_plus_frozen_e5", None, "e5_small_v2", len(providers)))
        for phase, feature_set, provider, encoder, rank in arms:
            for fold in range(5):
                value = base + 0.01 * rank + 0.002 * d + 0.004 * rng.standard_normal()
                rows.append(dict(
                    arm_id=f"D{d}/{phase}/{provider}/{encoder}", phase=phase, dataset_id=f"D{d}",
                    task_type="binary_classification", model_id="catboost", checkpoint_id="cb",
                    feature_set=feature_set, provider=provider, encoder=encoder,
                    pca_scope="per_column" if encoder else None,
                    pca_components=32 if encoder else None, split_id=f"D{d}", outer_fold=fold,
                    metric="roc_auc", value=value, non_text_value=base + 0.003 * (fold - 2),
                    difference=value - base, reference="non_text", inference_status="prespecified",
                    view="main"))
    return pd.DataFrame(rows)


def _moderators(gains):
    datasets = sorted(gains.dataset_id.unique())
    n = len(datasets)
    table = pd.DataFrame({
        "dataset_id": datasets, "task_type": "binary_classification",
        "n_rows": np.arange(n) * 100 + 50, "free_text_share": np.linspace(0, 1, n),
        "median_text_chars": np.arange(n)[::-1] * 3.0, "unseen_text_share": (np.arange(n) % 3) / 3,
        "share_long_text": np.linspace(0, 0.5, n),
    })
    return table


# ------------------------------------------------------------------ gains and model axis


def test_dataset_gains_report_raw_gain_and_fold_variance_for_median_rows():
    gains = aggregate.dataset_gains(_paired())
    one = gains[gains.representation.eq("decision/joint/a") & gains.dataset_id.eq("D0")].iloc[0]
    assert one.raw_gain == pytest.approx(one.augmented - one.baseline)
    assert one.gain == pytest.approx(one.raw_gain / (1 - one.baseline))
    assert one.gain_variance > 0
    median = gains[gains.aggregation.eq(aggregate.MEDIAN)]
    assert len(median) == 8 and median.gain_variance.gt(0).all() and median.raw_gain.notna().all()


def test_model_gain_table_ranks_models_on_shared_datasets_and_robustness_joins():
    gains = aggregate.dataset_gains(_paired())
    table = m.model_gain_table(gains)
    assert list(table.provider) == PROVIDERS[::-1] and table["rank"].tolist() == [1, 2, 3, 4, 5, 6]
    assert table.n_datasets.eq(8).all() and table.mean_raw_gain.gt(0).all()
    effects = pd.DataFrame([
        {"dataset": f"D{d}", "provider": p, "variant": v, "mean_tv": 0.01 * i + 0.001 * d,
         "excess_tv": None if v.startswith("repeat") else 0.02 * i}
        for i, p in enumerate(PROVIDERS) for d in range(2)
        for v in ("repeat_1", "repeat_2", "option_order")])
    robustness, status = m.model_robustness(effects)
    assert status.startswith("available") and len(robustness) == 6
    row = robustness.set_index("provider").loc["c"]
    assert row.replay_noise_tv == pytest.approx(0.02 + 0.0005)
    assert row.sensitivity_excess_tv == pytest.approx(0.04) and row.robustness_datasets == 2
    covariates = pd.DataFrame({"provider": PROVIDERS, "log10_parameters": np.arange(6.0),
                               "architecture_family": ["decoder_lm_head"] * 6,
                               "readout_type": ["api"] * 6})
    assoc = m.model_level_associations(gains, covariates, robustness=robustness, n_boot=50)
    rho = assoc[assoc.statistic.eq("spearman_rho")].set_index("covariate").value
    assert rho["replay_noise_tv"] == pytest.approx(1.0) and rho["log10_parameters"] == pytest.approx(1.0)
    empty, pending = m.model_robustness(pd.DataFrame())
    assert empty.empty and pending.startswith("pending")


def test_moderator_associations_report_raw_gain_against_headroom():
    gains = aggregate.dataset_gains(_paired())
    table = m.moderator_associations(gains, _moderators(gains), n_boot=0)
    assert set(table.outcome) == {"gain", "raw_gain"}
    raw = table[table.outcome.eq("raw_gain") & table.moderator.eq("baseline_headroom")]
    assert raw.status.eq("descriptive").all()


def test_unseen_and_free_text_shares_are_label_free():
    from src.data.loaders import LoadedDataset
    from src.data.metadata import extract_metadata

    frame = pd.DataFrame({
        "number": [1.0, 2, 3, 4, 5],
        "body": ["a b", "a b", None, "unique words here", "other"],
        "note": ["x", "x", None, "y", None],
        "target": [0, 1, 0, 1, 0],
    })
    official = {"target": "target", "task_type": "cls"}
    metadata = extract_metadata(frame, official, dataset_id="u", dataset_name="U",
                                text_columns=["body", "note"])
    row = m.dataset_covariates(LoadedDataset(frame, metadata, official, {"data_identity": "x"}, []))
    # Rows 0 and 1 repeat their joint text; rows 3 and 4 are unique; row 2 has no text.
    assert row["unseen_text_share"] == pytest.approx(2 / 4)
    assert row["free_text_share"] == pytest.approx(row["share_long_text"] + row["share_short_text"])


# ---------------------------------------------------------------------------- statistics


def test_plan_config_singles_out_no_moderator_in_advance():
    plan = ms.load_plan()
    assert plan["status"] == "proposed_before_results"
    assert ms.primary_columns(plan) == {}  # Owner decision 19-11-2026.
    broken = copy.deepcopy(plan)
    broken["exploratory_correction"] = "none"
    with pytest.raises(ValueError, match="Benjamini"):
        ms.load_plan(broken)
    with pytest.raises(ValueError, match="proposed"):
        ms.load_plan({**plan, "status": "after_results"})


def test_benjamini_hochberg_matches_hand_computation():
    p = [0.01, 0.04, 0.03, np.nan, 0.20]
    q = ms.benjamini_hochberg(p)
    # m = 4 finite: sorted 0.01, 0.03, 0.04, 0.20 -> 0.04, 0.0533, 0.0533, 0.20.
    np.testing.assert_allclose(q[[0, 2, 1, 4]], [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.20])
    assert np.isnan(q[3])


def test_permutation_p_is_small_for_perfect_and_large_for_null():
    x = np.arange(12.0)
    assert ms.permutation_p(x, x, n_permutations=999, seed=1) == pytest.approx(1 / 1000)
    rng = np.random.default_rng(0)
    assert ms.permutation_p(x, rng.permutation(x), n_permutations=999, seed=1) > 0.05
    assert np.isnan(ms.permutation_p(x, np.ones(12), n_permutations=10, seed=1))


def test_without_primaries_every_moderator_is_fdr_adjusted():
    plan = {**ms.load_plan(), "n_permutations": 199, "n_boot": 50}
    gains = aggregate.dataset_gains(_paired())
    table = ms.moderator_tests(gains, _moderators(gains), plan)
    assert set(table.role) == {"exploratory"}
    tested = table[table.p_value.notna()]
    assert len(tested) and tested.q_value.ge(tested.p_value - 1e-12).all()


def test_moderator_tests_flag_primary_and_adjust_only_exploratory():
    plan = {**ms.load_plan(), "n_permutations": 199, "n_boot": 50,
            "primary_moderators": [{"name": n, "column": c} for n, c in (("free_text_share", "free_text_share"), ("median_text_length", "median_text_chars"),
                                                         ("unseen_text_share", "unseen_text_share"), ("baseline_headroom", "baseline_headroom"))]}
    gains = aggregate.dataset_gains(_paired())
    table = ms.moderator_tests(gains, _moderators(gains), plan)
    assert set(table.role) == {"primary", "exploratory"}
    primary = table[table.role.eq("primary")]
    assert set(primary.moderator) == set(ms.primary_columns(plan))
    assert primary.q_value.isna().all() and not primary.significant_fdr.any()
    explore = table[table.role.eq("exploratory") & table.p_value.notna()]
    assert len(explore) and explore.q_value.ge(explore.p_value - 1e-12).all()
    # Only headline rows (median over decision models, recipes) enter.
    assert set(table.representation) == {"decision/joint/median", "embedding/e5_small_v2/per_column32"}


def test_partial_spearman_removes_a_shared_driver():
    rng = np.random.default_rng(3)
    z = rng.standard_normal(200)
    x = z + 0.1 * rng.standard_normal(200)
    y = z + 0.1 * rng.standard_normal(200)
    assert m.spearman_rows(x, y)[0] > 0.9
    assert abs(ms.partial_spearman(x, y, z[:, None])) < 0.2
    w = rng.standard_normal(200)
    assert ms.partial_spearman(w + z, w, z[:, None]) > 0.9
    assert np.isnan(ms.partial_spearman(x[:3], y[:3], z[:3, None]))
    plan = {**ms.load_plan(), "n_boot": 20, "primary_moderators": [{"name": n, "column": c} for n, c in (("free_text_share", "free_text_share"), ("median_text_length", "median_text_chars"),
                                                         ("unseen_text_share", "unseen_text_share"), ("baseline_headroom", "baseline_headroom"))]}
    gains = aggregate.dataset_gains(_paired())
    table = ms.partial_primary(gains, _moderators(gains), plan)
    assert set(table.moderator) == set(ms.primary_columns(plan))
    assert table.controls.str.count(",").eq(2).all()


def test_dersimonian_laird_matches_hand_computation():
    y, v = np.array([0.1, 0.3, 0.5]), np.array([0.01, 0.02, 0.01])
    out = ms.dersimonian_laird(y, v)
    w = 1 / v
    fixed = (w * y).sum() / w.sum()
    q = (w * (y - fixed) ** 2).sum()
    tau2 = max(0, (q - 2) / (w.sum() - (w**2).sum() / w.sum()))
    ws = 1 / (v + tau2)
    assert out["q"] == pytest.approx(q) and out["tau2"] == pytest.approx(tau2)
    assert out["pooled"] == pytest.approx((ws * y).sum() / ws.sum())
    assert out["se"] == pytest.approx(math.sqrt(1 / ws.sum()))
    assert out["i2"] == pytest.approx((q - 2) / q)
    homogeneous = ms.dersimonian_laird([0.2, 0.2, 0.2], [0.01, 0.02, 0.03])
    assert homogeneous["tau2"] == 0 and homogeneous["i2"] == 0
    assert ms.dersimonian_laird([0.2, 0.3], [0.0, np.nan])["status"] == "too_few_datasets"
    table = ms.random_effects(aggregate.dataset_gains(_paired()))
    assert table.status.eq("estimated").all() and table.variance_note.str.contains("approximate").all()
    assert (table.ci_low <= table.pooled_gain).all() and table.k.eq(8).all()


# ------------------------------------------------------------------------- row cohorts


def test_field_novelty_is_stratified_by_field_type_and_post_treatment_is_labelled():
    from types import SimpleNamespace

    frame = pd.DataFrame({
        "code": [f"C{i % 3}" for i in range(120)],
        "review": [f"a long review number {i} with many distinct words" if i % 4 else None
                   for i in range(120)],
    })
    ds = SimpleNamespace(frame=frame, metadata=SimpleNamespace(text_columns=["code", "review"]))
    novelty = rb.field_novelty(ds, list(range(116)), list(range(116, 120)))
    assert set(novelty) == {"categorical", "short_text"}
    assert novelty["categorical"].tolist() == ["seen"] * 4
    assert novelty["short_text"].tolist() == ["missing", "unseen", "unseen", "unseen"]
    ref = pd.DataFrame(dict(row_id=[f"csv:{i}" for i in range(4)], y_true=[1.0, 2, 3, 4],
                            prediction=[0.0, 2, 3, 5], text_present=[True] * 4,
                            joint_text_seen_in_train=[False] * 4))
    # Group B (20-11-2026): novelty and prediction-shift cohorts are no longer reported.
    rows, _ = rb.paired_cohorts(ref, ref.assign(prediction=[1.0, 2, 3, 4]), [], 2.0,
                                [1.0, 2.0], [1.0, 2.0, 3.0, 4.0])
    axes = {r["axis"] for r in rows}
    assert not {a for a in axes if a.startswith(("novelty", "prediction_shift"))}
    per_axis = pd.DataFrame(rows).groupby("axis").rows.sum()
    assert per_axis.eq(4).all()  # Each axis partitions the held-out rows.
    assert rb.POST_TREATMENT_AXES == ()
    assert rb.axis_interpretation("length").startswith("pre-treatment")


# ----------------------------------------------------------------------- complementarity


def test_pair_statistics_rank_correlation_and_top_overlap():
    a = np.arange(100.0)
    same = comp.pair_statistics(a, a * 2)
    assert same["spearman_rho"] == pytest.approx(1.0) and same["jaccard_top"] == 1.0
    assert same["top_k"] == 10
    assert same["jaccard_chance"] == pytest.approx(1 / (20 - 1))
    opposite = comp.pair_statistics(a, -a)
    assert opposite["spearman_rho"] == pytest.approx(-1.0) and opposite["jaccard_top"] == 0.0
    with_nan = comp.pair_statistics(np.r_[a, np.nan], np.r_[a, 1.0])
    assert with_nan["n_rows"] == 100
    with pytest.raises(ValueError, match="same rows"):
        comp.pair_statistics(a, a[:5])


def _benefits(rng, n=200):
    shared = rng.standard_normal(n)
    return {("catboost", "cb"): {
        "decision/joint/a": {"family": "decision", "values": shared + 0.1 * rng.standard_normal(n)},
        "decision/joint/b": {"family": "decision", "values": rng.standard_normal(n)},
        "embedding/e5/per_column32": {"family": "embedding", "values": shared},
        "tar/e5/joint64": {"family": "tar", "values": rng.standard_normal(n)},
    }}


def test_complementarity_pairs_and_equal_dataset_summary():
    rng = np.random.default_rng(0)
    pairs = pd.concat([comp.complementarity_pairs(_benefits(rng), dataset_id=f"D{i}",
                                                  task_type="binary_classification")
                       for i in range(4)], ignore_index=True)
    assert len(pairs) == 4 * 2 * 2
    close = pairs[pairs.decision_representation.eq("decision/joint/a")
                  & pairs.other_family.eq("embedding")]
    assert close.spearman_rho.gt(0.9).all() and close.jaccard_top.gt(close.jaccard_chance).all()
    summary = comp.complementarity_summary(pairs, n_boot=100, seed=1)
    assert set(summary.aggregation) == {"per_pair", "mode_by_family"}
    family = summary[summary.aggregation.eq("mode_by_family")].set_index("other")
    assert family.n_datasets.eq(4).all()
    assert family.loc["embedding", "mean_spearman_rho"] > family.loc["tar", "mean_spearman_rho"]
    assert comp.complementarity_summary(pairs.iloc[0:0]).empty


def test_data_map_grid_counts_and_sums():
    x = np.array([0.05, 0.05, 0.95, np.nan])
    y = np.array([0.05, 0.05, 0.95, 0.5])
    grid = comp.data_map_grid(x, y, [1.0, -3.0, 2.0, 5.0], bins=2)
    assert grid["counts"] == [[2, 0], [0, 1]] and grid["rows"] == 3
    assert grid["benefit_sum"] == [[-2.0, 0.0], [0.0, 2.0]]
    merged = comp.merge_grids([grid, grid])
    assert merged["counts"] == [[4, 0], [0, 2]] and merged["rows"] == 6
    reference = pd.DataFrame({"p_c000": [0.5, 0.9], "p_c001": [0.5, 0.1]})
    confidence = comp.baseline_confidence(reference, 2)
    np.testing.assert_allclose(confidence["baseline_max_probability"], [0.5, 0.9])
    assert confidence["baseline_certainty"][0] == pytest.approx(0.0)


def test_saved_predictions_to_complementarity_export(isolated_output, monkeypatch):
    from sklearn.dummy import DummyClassifier

    from src.cluster.predictor_contracts import feature_cases, fixtures
    from src.evaluation.protocol import evaluate_cv, shared_splits

    ds = next(d for d in fixtures() if d.metadata.task_type == "binary_classification")
    _, features, _ = next(feature_cases(ds, "linear"))
    name = ds.metadata.dataset_id
    monkeypatch.setattr(rb, "load_catalog", lambda: {"datasets": [{"id": name}]})
    monkeypatch.setattr(rb, "load_dataset", lambda *a, **kw: ds)
    monkeypatch.setattr(rb, "load_yaml", lambda name: {"random_seed": 20260929,
                                                       "candidate_extractors": ["jev"]})
    spec = shared_splits(ds, seed=20260929, folds=5)
    embedding = {"encoder_id": "e5_small_v2", "pca_scope": "per_column",
                 "pca_components_per_column": 32}
    for phase, feature_set, recipe in [
        ("experiment_1", "non_text", {"provider": None}),
        ("experiment_1", "non_text_plus_joint", {"provider": "jev"}),
        ("experiment_2", "non_text_plus_frozen_e5", {"provider": None, "embedding": embedding}),
    ]:
        evaluate_cv(ds, features, DummyClassifier, specification=spec, model_id="linear",
                    checkpoint_id="fixture", feature_set=feature_set, phase=phase,
                    allow_model_runs=True, estimator_settings={"seed": 1},
                    representation_recipe={**recipe, "feature_set": feature_set},
                    feature_artifact_ids=[])
    rows = len(ds.frame)

    def loader(dataset_id, provider, mode, cfg):
        from tests.test_moderators_and_decision_cohorts import _block

        if mode != "joint":
            return None, "artifact_not_found"
        return _block(provider, "joint", rows, np.linspace(0.05, 0.95, rows)), "fixture"

    monkeypatch.setattr(rb.paths, "durable_project_override", rb.paths.staging_override)
    body = json.loads(comp.run(name, decision_loader=loader).read_text())
    assert body["schema"] == comp.SCHEMA and not body["incomplete_or_unmatched"]
    (pair,) = body["pairs"]
    assert pair["decision_representation"] == "decision/joint/jev"
    assert pair["other_representation"] == "embedding/e5_small_v2/per_column32"
    assert pair["n_rows"] == rows
    assert {g["x_axis"] for g in body["data_map"]} == set(comp.X_AXES)
    assert all(g["rows"] == rows and g["providers"] == ["jev"] for g in body["data_map"])
    pairs, grids = comp.summaries()
    assert len(pairs) == 1 and len(grids) == 2 and grids.dataset_id.eq(name).all()


# --------------------------------------------------------------------------------- figures


def _check(figures):
    figures = list(figures)
    assert figures
    for fig, name, caption in figures:
        fig.canvas.draw()
        assert fig.get_size_inches()[0] == pytest.approx(style.WIDTH_FULL)
        assert fig.get_size_inches()[1] <= style.MAX_HEIGHT + 1e-9
        assert name and caption and not caption.startswith(name)
        plt.close(fig)
    return figures


def test_exp08_figures_render_at_print_width_and_stay_empty_when_pending():
    style.apply()
    gains = aggregate.dataset_gains(_paired())
    covariates = pd.DataFrame({"provider": PROVIDERS, "parameters_billion": [1, 2, 4, 9, 27, 31],
                               "architecture_family": ["decoder_lm_head"] * 6,
                               "readout_type": ["api", None, "api", "api", "api", "api"]})
    _check(views.model_gain_figures(m.model_gain_table(gains), covariates))
    _check(views.raw_gain_headroom_figures(gains))
    balanced = pd.DataFrame([
        {"task": "binary_classification", "phase": phase, "model": "catboost", "axis": axis,
         "cohort": cohort, "mean_benefit": 0.01 * i, "datasets": 3}
        for phase in ("experiment_1", "experiment_2")
        for axis, cohorts in {"availability": ["present", "absent"],
                              "prediction_shift": ["shift_0", "shift_1"],
                              "novelty_by_field_type:long_text": ["seen", "unseen", "missing"]}.items()
        for i, cohort in enumerate(cohorts)])
    (fig, _, caption), = _check(views.cohort_figures(balanced))
    assert "post-treatment" in caption
    rng = np.random.default_rng(0)
    pairs = pd.concat([comp.complementarity_pairs(_benefits(rng), dataset_id=f"D{i}",
                                                  task_type="binary_classification")
                       for i in range(3)])
    _check(views.complementarity_figures(comp.complementarity_summary(pairs, n_boot=0)))
    grid = comp.data_map_grid(rng.uniform(size=300), rng.uniform(size=300),
                              rng.standard_normal(300), bins=10)
    grids = pd.DataFrame([{"dataset_id": f"D{i}", "task_type": "binary_classification",
                           "model_id": "catboost", "checkpoint_id": "cb",
                           "feature_set": "non_text_plus_joint", "x_axis": axis, **grid}
                          for i in range(11) for axis in comp.X_AXES])
    maps = _check(views.data_map_figures(grids))
    assert [name for _, name, _ in maps] == ["data_map_joint_baseline_max_probability_1",
                                             "data_map_joint_baseline_max_probability_2"]
    for function in (views.model_gain_figures, views.raw_gain_headroom_figures,
                     views.cohort_figures, views.complementarity_figures, views.data_map_figures):
        assert list(function(pd.DataFrame())) == []


# ------------------------------------------------------------------------- paper figures


def test_paper_figure_map_validates_and_resolves_saved_files(tmp_path):
    raw = paper_figures.load()
    ids = [f["id"] for f in raw["figures"]]
    assert ids[:4] == ["fig1_headline_gain", "fig2_per_predictor_gains_and_compute",
                       "fig3_dataset_and_model_explanations", "fig4_fields_rows_complementarity"]
    assert all(f["placement"] == "undecided" for f in raw["figures"])
    folder = tmp_path / "exp08_when_text_helps"
    folder.mkdir()
    (folder / "03_raw_gain_vs_headroom.pdf").write_bytes(b"%PDF")
    (folder / "04_decision_models_ranked_joint.pdf").write_bytes(b"%PDF")
    table = paper_figures.resolve(raw, figures_root=tmp_path).set_index("id")
    assert table.loc["fig3_dataset_and_model_explanations", "saved_figures"] == (
        "decision_models_ranked_joint, raw_gain_vs_headroom")
    assert table.loc["fig4_fields_rows_complementarity", "saved_figures"] == "pending"
    for mutate, message in [
        (lambda r: r["figures"][0].update(notebook="exp99_missing"), "unknown notebook"),
        (lambda r: r["figures"][0].update(placement="front"), "placement"),
        (lambda r: r["figures"].append(dict(r["figures"][0])), "unique"),
        (lambda r: r["figures"][0].update(figures=["Bad Name"]), "stems"),
    ]:
        broken = copy.deepcopy(raw)
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            paper_figures.load(broken)
    # A reserved (planned) folder admits only planned figures. No number is reserved since
    # Group C activated Exp 9 (20-11-2026), so a synthetic reservation is used.
    from src.utils import structure

    reg = structure.registry()
    reserved = structure.Registry(reg.entries, reg.paper, (structure.Planned(
        "exp12", "exp12_future", "Future", "planned_not_implemented"),))
    broken = copy.deepcopy(raw)
    broken["figures"][0].update(notebook="exp12_future", status="available_when_results_exist")
    with pytest.raises(ValueError, match="reserved"):
        paper_figures.load(broken, registry=reserved)
