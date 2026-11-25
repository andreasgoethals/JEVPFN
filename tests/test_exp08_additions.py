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
    gains = aggregate.dataset_gains(_paired(), roster=PROVIDERS)
    one = gains[gains.representation.eq("decision/joint/a") & gains.dataset_id.eq("D0")].iloc[0]
    assert one.raw_gain == pytest.approx(one.augmented - one.baseline)
    assert one.gain == pytest.approx(one.raw_gain / (1 - one.baseline))
    assert one.gain_variance > 0
    median = gains[gains.aggregation.eq(aggregate.MEDIAN)]
    assert len(median) == 8 and median.gain_variance.gt(0).all() and median.raw_gain.notna().all()


def test_model_gain_table_ranks_models_on_shared_datasets_and_robustness_joins():
    gains = aggregate.dataset_gains(_paired(), roster=PROVIDERS)
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
    gains = aggregate.dataset_gains(_paired(), roster=PROVIDERS)
    table = m.moderator_associations(gains, _moderators(gains), n_boot=0)
    assert set(table.outcome) == {"gain", "raw_gain"}
    raw = table[table.outcome.eq("raw_gain") & table.moderator.eq("baseline_headroom")]
    assert raw.status.eq("descriptive").all()


def test_moderator_outcomes_keep_raw_auc_and_r2_separate():
    paired = _paired()
    regression = paired.dataset_id.isin(["D4", "D5", "D6", "D7"])
    paired.loc[regression, ["task_type", "metric"]] = ["regression", "r2"]
    gains = aggregate.dataset_gains(paired, roster=PROVIDERS)
    covariates = _moderators(gains)
    covariates["task_type"] = covariates.dataset_id.map(gains.set_index("dataset_id").task_type.to_dict())
    plan = {**ms.load_plan(), "min_datasets": 3, "n_boot": 0, "n_permutations": 0,
            "primary_moderators": [{"name": "rows", "column": "n_rows"}]}
    tables = [m.moderator_associations(gains, covariates, n_boot=0),
              ms.moderator_tests(gains, covariates, plan), ms.partial_primary(gains, covariates, plan)]
    for table in tables:
        counts = set(table[["outcome", "task_family", "n_datasets"]].itertuples(index=False, name=None))
        assert counts == {("gain", "all", 8), ("raw_gain", "classification", 4), ("raw_gain", "regression", 4)}
    ranking = m.model_gain_table(gains)
    assert ranking.mean_raw_gain.isna().all() and ranking.mean_gain.notna().all()
    covariates.loc[covariates.task_type.eq("regression"), "n_rows"] = np.nan
    partial = ms.partial_primary(gains, covariates, plan)
    missing = partial[partial.task_family.eq("regression")]
    assert len(missing) and missing.n_datasets.eq(0).all() and missing.status.eq("too_few_datasets").all()


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
    gains = aggregate.dataset_gains(_paired(), roster=PROVIDERS)
    table = ms.moderator_tests(gains, _moderators(gains), plan)
    assert set(table.role) == {"exploratory"}
    tested = table[table.p_value.notna()]
    assert len(tested) and tested.q_value.ge(tested.p_value - 1e-12).all()


def test_moderator_tests_flag_primary_and_adjust_only_exploratory():
    plan = {**ms.load_plan(), "n_permutations": 199, "n_boot": 50,
            "primary_moderators": [{"name": n, "column": c} for n, c in (("free_text_share", "free_text_share"), ("median_text_length", "median_text_chars"),
                                                         ("unseen_text_share", "unseen_text_share"), ("baseline_headroom", "baseline_headroom"))]}
    gains = aggregate.dataset_gains(_paired(), roster=PROVIDERS)
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
    gains = aggregate.dataset_gains(_paired(), roster=PROVIDERS)
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
    table = ms.random_effects(aggregate.dataset_gains(_paired(), roster=PROVIDERS))
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
    expected = sum(math.comb(10, i) * math.comb(90, 10 - i) / math.comb(100, 10) * i / (20 - i)
                   for i in range(11))
    assert same["jaccard_chance"] == pytest.approx(expected)
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
    summary = comp.complementarity_summary(pairs, roster=("a", "b"), n_boot=100, seed=1)
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
    from src.evaluation import benchmark_export, group_b
    from src.evaluation.design import decision_versions
    from src.evaluation.protocol import evaluate_cv, shared_splits
    from src.utils.config import load_yaml

    ds = next(d for d in fixtures() if d.metadata.task_type == "binary_classification")
    _, features, _ = next(feature_cases(ds, "linear"))
    name = ds.metadata.dataset_id
    monkeypatch.setattr(rb, "load_catalog", lambda: {"datasets": [{"id": name}]})
    monkeypatch.setattr(rb, "load_dataset", lambda *a, **kw: ds)
    spec = shared_splits(ds, seed=load_yaml("experiment_1/main")["random_seed"], folds=5)
    embedding = {"encoder_id": "e5_small_v2", "pca_scope": "per_column",
                 "pca_components_per_column": 32}
    version, = decision_versions(["jev"])
    selector = {k: version[k] for k in ("provider", "model", "version")}
    selector.update(question_bank_id="bank", question_variant="fixture", settings_sha256="settings")
    for phase, feature_set, recipe in [
        ("experiment_1", "non_text", {"provider": None}),
        ("experiment_1", "non_text_plus_joint", {"provider": "jev", "selector": selector}),
        ("experiment_1", "non_text_plus_joint", {"provider": "jev", "selector": {**selector, "version": "older"}}),
        ("experiment_1", "non_text_plus_joint", {"provider": "jev"}),
        ("experiment_2", "non_text_plus_frozen_e5", {"provider": None, "embedding": embedding}),
    ]:
        evaluate_cv(ds, features, DummyClassifier, specification=spec, model_id="linear",
                    checkpoint_id="fixture", feature_set=feature_set, phase=phase,
                    allow_model_runs=True, estimator_settings={"seed": 1},
                    representation_recipe={**recipe, "feature_set": feature_set},
                    feature_artifact_ids=[f"joint-{(recipe.get('selector') or {}).get('version', 'legacy')}"]
                    if recipe.get("provider") else [])
    rows = len(ds.frame)

    def loader(dataset_id, provider, mode, cfg):
        from tests.test_moderators_and_decision_cohorts import _block

        if mode != "joint":
            return None, "artifact_not_found"
        return _block(provider, "joint", rows, np.linspace(0.05, 0.95, rows)), "fixture"

    monkeypatch.setattr(rb.paths, "durable_project_override", rb.paths.staging_override)
    report_path = comp.run(name, decision_loader=loader, cohort="all")
    body = json.loads(report_path.read_text())
    assert body["schema"] == comp.SCHEMA and not body["incomplete_or_unmatched"]
    (pair,) = body["pairs"]
    assert pair["decision_representation"] == "decision/joint/jev"
    assert pair["other_representation"] == "embedding/e5_small_v2/per_column32"
    assert pair["n_rows"] == rows
    assert {g["x_axis"] for g in body["data_map"]} == set(comp.X_AXES)
    assert all(g["rows"] == rows and g["providers"] == ["jev"] for g in body["data_map"])
    pairs, grids = comp.summaries(cohort="all")
    assert len(pairs) == 1 and len(grids) == 2 and grids.dataset_id.eq(name).all()
    monkeypatch.setenv("JEVPFN_DECISION_COHORT", "all")
    monkeypatch.setattr(benchmark_export, "coverage", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(benchmark_export.study_export, "require_mindwell_cpu", lambda study: None)
    monkeypatch.setattr(benchmark_export.study_export, "job_origin", lambda: "synthetic")
    monkeypatch.setattr(benchmark_export, "export_path", lambda *args: isolated_output / "b2.zip")
    benchmark_export.build("B2")
    tables, receipt = benchmark_export.read("B2")
    assert receipt["decision_cohort"] == "all"
    folds = tables["folds"]
    assert folds.metric.eq("roc_auc").sum() == 3 * 5
    assert rb.Pairing(name, cohort="open_weight").arms.keys() == {
        ("experiment_2", folds.loc[folds.phase.eq("experiment_2"), "recipe_id"].iloc[0])}
    result = tables["complementarity_pairs"]
    assert result.iloc[0].decision_recipe_id == pair["decision_recipe_id"]
    head = group_b.headline(folds)
    missing, _ = comp.summaries(folder=isolated_output / "no-reports", cohort="all", headline_folds=head)
    assert missing.empty
    with pytest.raises(ValueError, match="selected decision-model cohort"):
        comp.summaries(folder=report_path.parent, cohort="open_weight", headline_folds=head)
    for column in ("split_id", "checkpoint_id", "data_identity"):
        with pytest.raises(ValueError, match="sources differ from the benchmark folds"):
            comp.summaries(cohort="all", headline_folds=head.assign(**{column: "other"}))
    for label in ("baseline", "decision/joint/jev", "frozen/e5_small_v2/per_column32"):
        different = head.copy()
        different.loc[different.representation.eq(label), "recipe_id"] = "another-recipe"
        with pytest.raises(ValueError, match="sources differ from the benchmark folds"):
            comp.summaries(cohort="all", headline_folds=different)
    del body["pairs"][0]["other_recipe_id"]
    report_path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks source recipe identities"):
        comp.summaries(cohort="all", headline_folds=head)


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


def test_b3_figures_wait_on_empty_tables():
    """B3 figures now read the Group B export schema (tests/test_group_b_views.py renders them
    from realistic fixtures); empty tables give no figure."""
    for function in (views.cohort_benefits, views.column_refits):
        assert list(function(pd.DataFrame(), name="x", what="x")) == []
    assert list(views.headroom_and_associations(pd.DataFrame(), None, name="x", what="x")) == []
    assert list(views.diagnostics_vs_usefulness(pd.DataFrame(), None, name="x", what="x")) == []
    assert list(views.lodo(pd.DataFrame(), None, name="x", what="x")) == []


# ------------------------------------------------------------------------- paper figures


def test_paper_figure_map_validates_and_resolves_saved_files(tmp_path):
    raw = paper_figures.load()
    ids = [f["id"] for f in raw["figures"]]
    assert ids[:4] == ["fig1_headline_gain", "fig2_representation_ladder_and_compute",
                       "fig3_dataset_and_model_explanations", "fig4_fields_and_prediction_in_advance"]
    assert all(f["placement"] == "undecided" for f in raw["figures"])
    folder = tmp_path / "B" / "exp08_when_text_helps"
    folder.mkdir(parents=True)
    (folder / "03_b3_headroom_and_properties.pdf").write_bytes(b"%PDF")
    (folder / "04_b3_model_diagnostics_vs_usefulness.pdf").write_bytes(b"%PDF")
    table = paper_figures.resolve(raw, figures_root=tmp_path).set_index("id")
    assert table.loc["fig3_dataset_and_model_explanations", "saved_figures"] == (
        "b3_headroom_and_properties, b3_model_diagnostics_vs_usefulness")
    assert table.loc["fig4_fields_and_prediction_in_advance", "saved_figures"] == "pending"
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
