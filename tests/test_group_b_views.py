"""Group B notebook tables and figures on realistic fixtures built through the real export code.

18 datasets (3 binary, 6 multiclass, 9 regression), 17 decision models, 7 predictors and 5 folds
(`tests/group_b_fixture.py`); B1, B2 and B5 round-trip through `study_export.write_export` and
`benchmark_export.read`. Values are synthetic and only exercise shapes, units and layout.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import group_b_fixture as fx  # noqa: E402

from src.evaluation import benchmark_export, data_efficiency, group_b  # noqa: E402
from src.evaluation import group_b_views as views  # noqa: E402
from src.visualize import group_b as figures  # noqa: E402
from src.visualize import group_b_synthesis as synthesis  # noqa: E402
from src.visualize import representation_views, style  # noqa: E402
from src.visualize import when_text_helps as b3  # noqa: E402

ROSTER = fx.roster()


@pytest.fixture(scope="module")
def exports(tmp_path_factory):
    folder = tmp_path_factory.mktemp("group_b_exports")
    return {study: fx.export(study, folder)[0] for study in ("B1", "B2", "B5")}


def _check(generator, *, width=style.WIDTH_FULL, count=1):
    produced = list(generator)
    assert len(produced) == count
    for fig, name, caption in produced:
        fig.canvas.draw()
        assert fig.get_figwidth() == pytest.approx(width)
        assert fig.get_figheight() <= style.MAX_HEIGHT + 1e-9
        assert name and len(caption) > 80 and not caption.startswith(name)
        plt.close(fig)
    return produced


def test_fixture_has_the_real_shape_and_reads_back_verified(exports):
    b1 = exports["B1"]
    gains = b1["dataset_gains"]
    assert gains.dataset_id.nunique() == 18 and gains.model_id.nunique() == 7
    combinations = 7 * 3 * len(ROSTER)
    assert gains.task_type.value_counts().to_dict() == {
        "regression": 9 * combinations, "multiclass_classification": 6 * combinations,
        "binary_classification": 3 * combinations}
    assert set(b1["median_gains"].decision_models) == {len(ROSTER)}


def test_b1_model_tables_rank_every_model_and_never_mix_units(exports):
    gains = exports["B1"]["dataset_gains"]
    per, summary = views.model_ranks(gains, roster=ROSTER, n_boot=100)
    assert len(summary) == len(ROSTER) and per.dataset_id.nunique() == 18
    assert per["rank"].between(1, len(ROSTER)).all()
    assert summary.mean_rank.mean() == pytest.approx((len(ROSTER) + 1) / 2)
    family = views.family_summaries(exports["B1"]["median_gains"], ["decision/both/median"], n_boot=50)
    assert set(family.task_family) == {"classification", "regression", "all"}
    assert set(family[family.task_family.ne("all")].datasets) == {9}
    assert set(family[family.task_family.eq("all")].datasets) == {18}
    agreement = views.predictor_rank_agreement(gains, roster=ROSTER)
    assert len(agreement) == 49 and agreement.tau_b.between(-1, 1).all()
    contrasts = views.view_contrasts(gains, roster=ROSTER, text_columns=views.text_column_counts())
    assert contrasts.dataset_id.nunique() == 16  # one-text datasets are identical by construction


def test_fold_win_rates_pool_predictors_as_cells_of_the_same_dataset(exports):
    head = views.fold_long(exports["B1"]["folds"])
    head = pd.concat([head, views.median_fold_scores(head, roster=ROSTER)], ignore_index=True)
    rates = views.fold_win_rates(head, ["decision/both/median"], resamples=100)
    pooled = rates[rates.model_id.eq("all")].iloc[0]
    assert pooled.fold_cells == 18 * 7 * 5 and pooled.datasets == 18
    assert set(rates.model_id) == {*style.MAIN_PREDICTORS, "all"}
    assert (rates[rates.model_id.ne("all")].fold_cells == 90).all()


def test_b1_figures_render(exports):
    b1 = exports["B1"]
    gains, medians = b1["dataset_gains"], b1["median_gains"]
    _check(figures.gain_landscape(views.model_gains(gains, roster=ROSTER), views.median_model_gains(medians),
                                  name="a", what="Test"))
    per, summary = views.model_ranks(gains, roster=ROSTER, n_boot=50)
    head = views.fold_long(b1["folds"])
    wins = views.fold_win_rates(head, [f"decision/both/{p}" for p in ROSTER], by_predictor=False, resamples=50)
    _check(figures.model_rank_heatmap(per, summary, win_rates=wins, name="b", what="Test"))
    texts = views.text_column_counts()
    _check(figures.view_contrast(views.view_contrasts(gains, roster=ROSTER, text_columns=texts), name="c",
                                 what="Test", single_text=["x"]))
    modes = ["decision/per_column/median", "decision/joint/median", "decision/both/median"]
    family = views.family_summaries(medians, modes, n_boot=50)
    _check(figures.gain_forest(family, modes, name="d", what="Test"))
    _check(figures.gain_forest(family, modes, name="e", what="Test", share=True))
    head = pd.concat([head, views.median_fold_scores(head, roster=ROSTER)], ignore_index=True)
    _check(figures.fold_win_rate_forest(views.fold_win_rates(head, modes, resamples=50), modes, name="f", what="Test"))
    _check(figures.model_by_predictor(gains, order=list(summary.provider), name="g", what="Test"))
    _check(figures.predictor_rank_agreement(views.predictor_rank_agreement(gains, roster=ROSTER), name="h",
                                            what="Test"), width=style.WIDTH_HALF)


def test_b2_and_b5_figures_render(exports):
    b2 = exports["B2"]
    gains = views.combined_gains(b2)
    pair = views.paired_gains(gains, x="frozen/e5_small_v2/per_column32", y="decision/both/median")
    assert len(pair) == 18 and (pair.x_low <= pair.x).all() and (pair.x <= pair.x_high).all()
    _check(figures.paired_scatter({"E5": pair}, name="a", what="Test"))
    _check(figures.paired_scatter({"TAR": views.paired_gains(gains, x="tar/e5_small_v2_tar/per_column32",
                                                             y="decision/both/median"), "Probe": pair},
                                  name="b", what="Test"))
    widths = views.arm_widths(b2["folds"], roster=ROSTER)
    arms = ["decision/both/median", "decision+frozen/both/median", "frozen/e5_small_v2/per_column32"]
    _check(figures.compactness(views.compactness(gains, widths, arms), name="c", what="Test"))
    _check(representation_views.complementarity(b2["complementarity_summary"], name="d", what="Test"))
    scores = views.native_scores(views.fold_long(exports["B5"]["folds"]), roster=ROSTER)
    assert {"native/tabstar", "baseline", "decision/both/median"} <= set(scores.representation)
    _check(figures.native_references(scores, name="e", what="Test"))


def test_synthesis_ladder_win_matrix_and_compute(exports):
    b1, b2, b5 = exports["B1"], exports["B2"], exports["B5"]
    ladder = views.ladder(b1, b2, b5, n_boot=50)
    assert set(ladder[ladder.representation.str.startswith("native/")].model_id) == {"native"}
    _check(synthesis.representation_ladder(ladder, name="a", what="Test"))
    _check(synthesis.representation_ladder(ladder, name="b", what="Test", share=True))
    scores = views.main_arm_fold_scores(b1, b2, roster=ROSTER)
    pairwise = views.pairwise_win_rates(scores, list(views.MAIN_ARMS), resamples=50)
    assert len(pairwise) == len(views.MAIN_ARMS) * (len(views.MAIN_ARMS) - 1)
    # Ties aside, a wins against b exactly when b loses against a.
    one = pairwise.set_index(["learner", "versus"])
    a, b = views.MAIN_ARMS[:2]
    assert one.loc[(a, b), "wins"] == one.loc[(b, a), "losses"]
    _check(synthesis.win_rate_matrix(pairwise, name="c", what="Test"), width=style.WIDTH_HALF)
    extraction = pd.DataFrame({"provider": ROSTER, "gpu_hours_per_1000_requests": np.linspace(0.01, 0.5, len(ROSTER))})
    costs = views.compute_costs(b1["folds"], rows=views.dataset_rows(), extraction=extraction, roster=ROSTER)
    single = b1["summary"][b1["summary"].representation.str.startswith("decision/both/")]
    _check(synthesis.quality_vs_compute(views.pareto_points([ladder, single], costs), name="d", what="Test"))
    no_cost = views.compute_costs(b1["folds"], rows=views.dataset_rows(), roster=ROSTER)
    assert list(synthesis.quality_vs_compute(views.pareto_points([ladder], no_cost), name="x", what="x")) == []


def test_extraction_and_fit_costs_preserve_their_units_and_roster():
    folds = fx.folds(("experiment_1",))
    fit = folds[folds.metric.eq("seconds.fit") & folds.provider.eq(ROSTER[0])].copy()
    fit["n_train"] = 200
    fit["value"] = 360
    extraction = pd.DataFrame({"provider": [ROSTER[0]], "gpu_hours_per_1000_requests": [0.02]})
    costs = views.compute_costs(fit, extraction=extraction, roster=ROSTER).set_index("representation")
    both = costs.loc[f"decision/both/{ROSTER[0]}"]
    assert both.fit_hours_per_1000_rows == pytest.approx(0.5)
    assert both.gpu_hours_per_1000_requests == pytest.approx(0.02)
    assert pd.isna(costs.loc[f"decision/joint/{ROSTER[0]}", "gpu_hours_per_1000_requests"])
    assert pd.isna(costs.loc["decision/both/median", "gpu_hours_per_1000_requests"])
    assert "total_hours_per_1000_rows" not in costs


def test_b3_tables_and_figures_render():
    tables = fx.b3_tables(fx.folds(("experiment_1", "rq2")), n_permutations=50, n_shuffles=10)
    medians = group_b.median_over_decision_models(tables["dataset_gains"], roster=ROSTER)
    headroom = views.headroom_table(medians)
    assert len(headroom) == 18 and (headroom.headroom > 0).all()
    _check(b3.headroom_and_associations(headroom, tables["associations"], name="a", what="Test"))
    usefulness, relations = views.model_usefulness_relations(tables["dataset_gains"], tables["model_diagnostics"],
                                                            roster=ROSTER, n_permutations=50)
    assert len(usefulness) == len(ROSTER) and relations.q_value_bh.between(0, 1).all()
    _check(b3.diagnostics_vs_usefulness(usefulness, relations, name="b", what="Test"))
    types = pd.DataFrame([{"dataset_id": d, "field": f"c{i}", "field_type": "short_text"}
                          for d, n in views.text_column_counts().items() for i in range(n)])
    columns = views.column_refits(tables["column_importance"], field_types=types, roster=ROSTER)
    assert len(columns) == 78 and columns.dataset_id.nunique() == 16
    _check(b3.column_refits(columns, name="c", what="Test"))
    cohorts = views.cohort_benefits(tables["row_cohorts"])
    assert set(cohorts.axis) == set(views.COHORT_AXES)
    _check(b3.cohort_benefits(cohorts, name="d", what="Test"))
    _check(b3.lodo(tables["lodo_predictions"], tables["lodo"], name="e", what="Test"))


def test_b4_curves_pool_per_task_family_with_fixed_cohorts():
    tables = benchmark_export.build_tables("B4", fx.size_folds(), cohort="all")
    head = views.fold_long(tables["folds"])
    curves = views.size_curves(head, roster=ROSTER, curves=data_efficiency.config()["curves"],
                               representations=figures.SIZE_ARMS)
    main = curves[curves.curve.eq("main") & curves.model_id.eq("all")]
    assert set(main.task_family) == {"classification", "regression"}
    assert set(main.datasets) == {9}
    extended = curves[curves.curve.eq("extended") & curves.model_id.eq("all")]
    assert extended.groupby("task_family").datasets.max().to_dict() == {"classification": 4, "regression": 3}
    assert (main.ci_low <= main.pooled).all() and (main.pooled <= main.ci_high).all()
    _check(figures.size_curves(curves, name="a", what="Test"))
    _check(figures.size_curves(curves, name="b", what="Test", curve="extended"))
    _check(figures.size_curves_by_predictor(curves, name="c", what="Test", family="regression"))
    slopes = views.size_slopes(head, roster=ROSTER, representations=figures.SIZE_ARMS)
    assert slopes.dataset_id.nunique() == 18
    _check(figures.size_slopes(slopes, name="d", what="Test"))


def test_pca_variance_figure_and_empty_states():
    rows = [{"dataset_id": d, "task_type": t, "encoder": e, "outer_fold": f, "pca_scope": s, "text_group": 0,
             "training_explained_variance_fraction": 0.3 + 0.1 * f}
            for d, t in (("BIN_TEXT_FAKE_JOB_POSTING", "binary_classification"),
                         ("REG_TEXT_BOOK_PRICE", "regression"))
            for e in ("e5_small_v2", "e5_large_v2", "qwen3_embedding_0_6b") for f in range(5)
            for s in ("per_column", "joint")]
    _check(representation_views.pca_variance(pd.DataFrame(rows)))
    empty = pd.DataFrame()
    assert list(representation_views.pca_variance(empty)) == []
    assert list(figures.gain_landscape(empty, empty, name="x", what="x")) == []
    assert list(figures.size_curves(empty, name="x", what="x")) == []
    assert list(synthesis.representation_ladder(empty, name="x", what="x")) == []
    assert views.ladder({}, {}, {}).empty


def test_normalized_score_views_and_figures(exports):
    b1, b2, b5 = exports["B1"], exports["B2"], exports["B5"]
    arms = ["baseline", "decision/per_column/median", "decision/joint/median", "decision/both/median"]
    per, summary = views.normalized_view(views.fold_long(b1["folds"]), arms, roster=ROSTER, n_boot=50)
    assert per.normalized_score.between(0, 1).all() and per.dataset_id.nunique() == 18
    assert set(per.representation) == set(arms) and per.model_id.nunique() == 7
    pooled = summary[summary.task_family.eq("all")]
    assert pooled.datasets.eq(18).all() and pooled.configurations.eq(28).all()
    assert set(summary[summary.task_family.ne("all")].datasets) == {9}
    # Per dataset and fold the best configuration scores 1 and the worst 0, so every dataset's
    # fold-mean scores span at most [0, 1] and someone is near each end.
    spread = per.groupby("dataset_id").normalized_score.agg(["min", "max"])
    assert (spread["min"] < 0.5).all() and (spread["max"] > 0.5).all()
    produced = _check(figures.normalized_score_forest(summary, arms, name="n", what="Test"))
    assert "28 compared configurations" in produced[0][2] and "clipped at 0" in produced[0][2]
    _, ladder = views.normalized_ladder(b1, b2, b5, roster=ROSTER, n_boot=50)
    assert {"baseline", "native/tabstar", "tar/e5_small_v2_tar/per_column32"} <= set(ladder.representation)
    assert ladder.configurations.max() == (1 + len(views.LADDER)) * 7 + 3
    _check(synthesis.normalized_ladder(ladder, name="l", what="Test"))
    assert list(figures.normalized_score_forest(pd.DataFrame(), arms, name="x", what="x")) == []
    assert all(t.empty for t in views.normalized_view(pd.DataFrame(), arms, roster=ROSTER))


def test_glance_adds_the_normalized_scope_when_fold_tables_exist(exports):
    from src.evaluation.synthesis_overview import NO_TEXT, glance_table
    from src.visualize.overview import study_at_a_glance

    table = glance_table({"B1": exports["B1"], "B2": exports["B2"], "C2": {}}, n_boot=50)
    normalized = table[table.scope.eq("normalized")]
    assert NO_TEXT in set(normalized.arm) and normalized.datasets.eq(18).all()
    assert normalized.estimate.between(0, 1).all()
    fig, name, caption = study_at_a_glance(table)
    assert name == "study_at_a_glance" and len(fig.axes) == 4 and "MulTaBench" in caption
    assert NO_TEXT not in [t.get_text() for t in fig.axes[0].get_yticklabels()]
    plt.close(fig)


def test_glance_rejects_incompatible_fold_sources(exports):
    from src.evaluation.synthesis_overview import glance_table

    b2 = dict(exports["B2"])
    b2["folds"] = b2["folds"].assign(split_id="other-split")
    with pytest.raises(ValueError, match="disagree|different split"):
        glance_table({"B1": exports["B1"], "B2": b2}, n_boot=5)


def test_confirmatory_family_runs_on_the_declared_contrasts(exports):
    from src.evaluation import learner_scores
    from src.visualize.overview import confirmatory_forest

    frames = {"B1_decision_features": exports["B1"], "B2_representations": exports["B2"]}
    result = learner_scores.run_confirmatory_family(frames, n_boot=100)
    assert result["status"] == "compared", result.get("reason")
    results, forest = result["results"], result["forest"]
    assert len(results) == len(forest) == 21 and results.holm_family_size.eq(21).all()
    assert results.datasets.eq(18).all() and forest.datasets.eq(18).all()
    assert ((forest.low <= forest.estimate) & (forest.estimate <= forest.high)).all()
    fig, name, _ = confirmatory_forest(forest)
    assert name == "confirmatory_family"
    plt.close(fig)
