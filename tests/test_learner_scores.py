"""Framework application: synthetic export -> learner scores -> compare_learners -> CD diagram."""

import copy

import matplotlib
import numpy as np
import pandas as pd
import pytest

from src.evaluation import group_b, inference, learner_scores
from src.utils.config import load_yaml
from src.visualize import critical_difference, style

matplotlib.use("Agg")

TASKS = (["binary_classification"] * 5 + ["multiclass_classification"] * 4 + ["regression"] * 9)
DATASETS = {f"D{i:02d}": task for i, task in enumerate(TASKS)}
ROSTER = ("m1", "m2", "m3")
MODELS = ("catboost", "linear")
EFFECT = {"per_column": 0.02, "joint": 0.04, "both": 0.06}


def design(**changes):
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            rows.append(dict(phase="experiment_1", dataset_id=dataset, model_id=model,
                             representation="baseline", width_status="within_known_width_limit"))
            for mode in EFFECT:
                for provider in ROSTER:
                    rows.append(dict(phase="experiment_1", dataset_id=dataset, model_id=model,
                                     representation=f"decision/{mode}/{provider}",
                                     width_status="within_known_width_limit"))
            for scope in ("per_column32", "joint64"):
                rows.append(dict(phase="experiment_2", dataset_id=dataset, model_id=model,
                                 representation=f"frozen/e5_small_v2/{scope}",
                                 width_status="within_known_width_limit"))
    multiclass = [d for d, t in DATASETS.items() if t == "multiclass_classification"]
    spec = dict(datasets=dict(DATASETS), roster=ROSTER, plan=pd.DataFrame(rows),
                lodo_input_sets=("pre_extraction", "pilot", "post_extraction"), lodo_threshold=0.0,
                a2_datasets={"wording": list(DATASETS), "option_order": multiclass, "field_order": None})
    return learner_scores.Design(**(spec | changes))


def fold_rows(dataset, task, model, feature_set, value, *, provider=None, folds=5, **extra):
    metric = {"binary_classification": "roc_auc", "multiclass_classification": "roc_auc_ovr_macro",
              "regression": "r2"}[task]
    return [dict(phase="experiment_1", dataset_id=dataset, task_type=task, model_id=model,
                 checkpoint_id="ck", feature_set=feature_set, provider=provider, view="main",
                 encoder=extra.get("encoder"), pca_scope=extra.get("pca_scope"),
                 pca_components=extra.get("pca_components"), probe_view=None, dropped_text_column=None,
                 training_size=None, draw=None, hidden_view=None, hidden_position=None, hidden_layer=None,
                 adaptation_cell=None, split_id=f"split-{dataset}", outer_fold=fold, recipe_id="r",
                 metric=metric, value=value + 0.001 * (fold - 2)) for fold in range(folds)]


def synthetic_folds(baselines=None, skip=()):
    """B1-style fold table: decision modes help by EFFECT x headroom, per model jitter."""
    rng = np.random.default_rng(7)
    rows = []
    for i, (dataset, task) in enumerate(DATASETS.items()):
        for model in MODELS:
            base = (baselines or {}).get(dataset, 0.6 + 0.01 * i)
            rows += fold_rows(dataset, task, model, "non_text", base)
            for mode, effect in EFFECT.items():
                for provider in ROSTER:
                    if (dataset, model, mode, provider) in skip:
                        continue
                    gain = (effect + rng.normal(0, 0.004)) * (1 - min(base, 0.99))
                    rows += fold_rows(dataset, task, model, f"non_text_plus_{mode}", base + gain,
                                      provider=provider)
    return pd.DataFrame(rows)


def approved_analysis():
    analysis = copy.deepcopy(load_yaml("synthesis/analysis"))
    analysis["framework"]["status"] = "approved"
    return analysis


def test_normalised_gain_is_undefined_without_headroom():
    gain = learner_scores.normalised_gain([0.9, 1.0, 0.5, 1.2], [0.8, 1.0, 0.6, 1.0])
    assert gain[0] == pytest.approx(0.5)
    assert np.isnan(gain[1]) and np.isnan(gain[3])  # baseline >= 1: undefined, no floor
    assert gain[2] == pytest.approx(-0.25)


def test_application_names_match_the_framework_entries():
    entries = set(load_yaml("synthesis/analysis")["framework"]["application"])
    assert set(learner_scores.APPLICATIONS) == entries - {"A3_geometry"}


def test_every_application_awaits_approval_and_reads_nothing(monkeypatch):
    def refuse(_source):
        raise AssertionError("No export may be read before approval")

    monkeypatch.setattr(learner_scores, "read_frames", refuse)
    analysis = load_yaml("synthesis/analysis")
    assert analysis["framework"]["status"] != "approved"
    for name in analysis["framework"]["application"]:
        result = learner_scores.run_application(name)
        assert result["status"] == learner_scores.AWAITING
        assert "groups" not in result
        assert list(critical_difference.cd_figures(result)) == []
    state = learner_scores.confirmatory_family_state()
    assert state["status"] == learner_scores.AWAITING and state["pooled_contrasts"] == 0


def test_b1_end_to_end_synthetic_export_to_cd_diagram(tmp_path):
    frames = {"folds": synthetic_folds()}
    result = learner_scores.run_application("B1_decision_features", frames, analysis=approved_analysis(),
                                            design=design())
    assert result["status"] == "compared" and result["rope"] == 0.01
    assert [g["group"] for g in result["groups"]] == list(MODELS)
    group = result["groups"][0]
    assert group["status"] == "compared" and group["control"] == "baseline"
    omnibus, table = group["omnibus"], group["table"]
    assert omnibus["datasets"] == 18 and omnibus["learners"] == 4
    ranks = omnibus["mean_ranks"]
    assert ranks["decision/both/median"] == 1 and ranks["baseline"] == 4
    assert set(table.versus) == {"baseline"} and "p_finner" in table
    assert (table.p_holm < 0.05).all()
    # The median over the roster equals the per-dataset normalised median gain.
    scores = learner_scores.application_scores("B1_decision_features", frames, design())
    both = scores[scores.learner.eq("decision/both/median") & scores.group.eq("catboost")]
    assert both.status.eq("ok").all() and both.score.between(0.04, 0.08).all()
    figures = list(critical_difference.cd_figures(result))
    assert len(figures) == 2
    fig, name, caption = figures[0]
    assert name == "cd_B1_decision_features_catboost" and "Nemenyi" in caption
    assert fig.get_size_inches()[0] == pytest.approx(style.WIDTH_FULL)
    fig.savefig(tmp_path / "cd.pdf")


def test_zero_headroom_missing_dataset_and_incomplete_folds_stop_inference():
    analysis = approved_analysis()
    baselines = {"D10": 1.0}  # R2 baseline of 1: zero headroom
    result = learner_scores.run_application("B1_decision_features", {"folds": synthetic_folds(baselines)},
                                            analysis=analysis, design=design())
    group = result["groups"][0]
    assert group["status"] == "stopped" and "omnibus" not in group
    assert {c["status"] for c in group["cells"]} == {learner_scores.ZERO_HEADROOM}
    assert {c["dataset_id"] for c in group["cells"]} == {"D10"}

    folds = synthetic_folds()
    folds = folds[~folds.dataset_id.eq("D03")]
    group = learner_scores.run_application("B1_decision_features", {"folds": folds}, analysis=analysis,
                                           design=design())["groups"][0]
    assert group["status"] == "stopped"
    assert {c["dataset_id"] for c in group["cells"]} == {"D03"}
    assert {c["status"] for c in group["cells"]} == {learner_scores.MISSING}

    folds = synthetic_folds()
    folds = folds[~(folds.dataset_id.eq("D04") & folds.outer_fold.eq(4) & folds.feature_set.eq("non_text"))]
    group = learner_scores.run_application("B1_decision_features", {"folds": folds}, analysis=analysis,
                                           design=design())["groups"][0]
    assert group["status"] == "stopped"
    assert {c["status"] for c in group["cells"]} == {learner_scores.INCOMPLETE}


def test_median_requires_the_whole_configured_roster():
    folds = synthetic_folds(skip={("D05", "catboost", "joint", "m2")})
    result = learner_scores.run_application("B1_decision_features", {"folds": folds},
                                            analysis=approved_analysis(), design=design())
    catboost, linear = result["groups"]
    assert catboost["status"] == "stopped" and linear["status"] == "compared"
    assert catboost["cells"] == [{"dataset_id": "D05", "learner": "decision/joint/median",
                                  "status": learner_scores.MISSING}]


def test_unsupported_width_arms_are_excluded_by_design_and_listed():
    plan = design().plan.copy()
    mask = (plan.model_id.eq("linear") & plan.dataset_id.eq("D01")
            & plan.representation.eq("frozen/e5_small_v2/joint64"))
    plan.loc[mask, "width_status"] = "unsupported_width"
    folds = synthetic_folds()
    extra = []
    for dataset, task in DATASETS.items():
        for model in MODELS:
            for scope, components in (("per_column", 32), ("joint", 64)):
                if model == "linear" and dataset == "D01" and scope == "joint":
                    continue
                extra += fold_rows(dataset, task, model, "non_text_plus_frozen_e5", 0.62, encoder="e5_small_v2",
                                   pca_scope=scope, pca_components=components)
    folds = pd.concat([folds, pd.DataFrame(extra)], ignore_index=True)
    groups = learner_scores.application_groups("B2_representations", {"folds": folds}, design(plan=plan))
    linear = next(g for g in groups if g.group == "linear")
    assert linear.excluded_by_design == ["frozen/e5_small_v2/joint64"]
    assert "frozen/e5_small_v2/joint64" not in linear.learners
    assert linear.control == "decision/both/median"
    # decision+frozen medians are planned in the real design only; here they are missing cells.
    assert set(linear.scores.loc[linear.scores.status.ne("ok"), "learner"]) == {"decision+frozen/both/median"}


def averaging_quality(zero_reference=None):
    rows = []
    rng = np.random.default_rng(3)
    for dataset, task in DATASETS.items():
        metric = "ranked_probability_score" if task == "regression" else "log_loss"
        for provider in ROSTER:
            original = 0.5 + rng.uniform(0, 0.2)
            if (dataset, provider) == zero_reference:
                original = 0.0
            for presentation, value in (("original", original), ("average", original * 0.9),
                                        ("mean_single", original * 0.95)):
                rows.append(dict(dataset=dataset, provider=provider, task=task, factor="wording",
                                 presentation=presentation, metric=metric, value=value))
                rows.append(dict(dataset=dataset, provider=provider, task=task, factor="wording",
                                 presentation=presentation, metric="brier", value=value / 2))
    return pd.DataFrame(rows)


def test_a2_proper_score_skill_against_a_zero_skill_reference():
    analysis = approved_analysis()
    result = learner_scores.run_application("A2_stability", {"averaging_quality": averaging_quality()},
                                            analysis=analysis, design=design())
    groups = {g["group"]: g for g in result["groups"]}
    primary = groups["wording/average_vs_original"]
    assert primary["status"] == "compared" and primary["control"] == "reference"
    assert set(primary["table"].learner) == set(ROSTER) and primary["table"].p_holm.lt(0.05).all()
    assert primary["table"].median_oriented_difference.tolist() == pytest.approx([0.1] * 3)
    assert groups["field_order/average_vs_original"]["status"] == "stopped"
    assert "not declared" in groups["field_order/average_vs_original"]["reason"]
    assert groups["option_order/average_vs_original"]["status"] == "stopped"  # no rows: missing
    broken = learner_scores.run_application(
        "A2_stability", {"averaging_quality": averaging_quality(zero_reference=("D02", "m1"))},
        analysis=analysis, design=design())
    primary = next(g for g in broken["groups"] if g["group"] == "wording/average_vs_original")
    assert primary["status"] == "stopped"
    assert primary["cells"] == [{"dataset_id": "D02", "learner": "m1", "status": learner_scores.UNDEFINED}]


def test_b3_regret_matches_the_lodo_evaluation_policy_regret():
    rng = np.random.default_rng(5)
    rows = []
    for model in MODELS:
        actual = rng.normal(0.01, 0.02, len(DATASETS))
        for name in ("pre_extraction", "pilot", "post_extraction"):
            predicted = actual + rng.normal(0, 0.01, len(DATASETS))
            rows += [dict(dataset_id=d, actual=a, predicted=p, input_set=name, model_id=model)
                     for d, a, p in zip(DATASETS, actual, predicted)]
    table = pd.DataFrame(rows)
    groups = learner_scores.application_groups("B3_when_text_helps", {"lodo_predictions": table}, design())
    scores = groups[0].scores.set_index("learner")
    part = table[table.model_id.eq("catboost") & table.input_set.eq("pilot")]
    expected = group_b.lodo_evaluation(part[["dataset_id", "actual", "predicted"]], n_shuffles=0)
    assert scores.loc["rule/pilot", "score"].mean() == pytest.approx(expected["regret"])
    assert scores.loc["always", "score"].mean() == pytest.approx(expected["regret_always"])
    assert scores.loc["never", "score"].mean() == pytest.approx(expected["regret_never"])
    result = learner_scores.run_application("B3_when_text_helps", {"lodo_predictions": table},
                                            analysis=approved_analysis(), design=design())
    assert result["groups"][0]["status"] == "compared" and result["rope"] == 0.01
    assert not result["higher_is_better"] and "p_mean_rank_shaffer" in result["groups"][0]["table"]


def test_a1_without_a_prespecified_rope_stops():
    quality = pd.DataFrame([dict(dataset_id=d, provider=p, metric="log_loss", value=0.5 + 0.1 * j)
                            for d, t in DATASETS.items() if t != "regression"
                            for j, p in enumerate(ROSTER)])
    missing = approved_analysis()
    missing["framework"]["rope"].pop("log_loss", None)
    result = learner_scores.run_application("A1_outputs", {"quality": quality}, analysis=missing,
                                            design=design())
    assert result["groups"][0]["status"] == "stopped" and "ROPE" in result["groups"][0]["reason"]
    analysis = approved_analysis()
    assert analysis["framework"]["rope"]["log_loss"] == 0.01  # prespecified for A1
    group = learner_scores.run_application("A1_outputs", {"quality": quality}, analysis=analysis,
                                           design=design())["groups"][0]
    assert group["status"] == "compared" and group["omnibus"]["datasets"] == 9
    assert group["omnibus"]["mean_ranks"]["m1"] == 1  # lower log loss is better


# ------------------------------------------------------------------ pooled contrasts


def pooled_plan(contrasts, *, framework="approved"):
    analysis = copy.deepcopy(load_yaml("synthesis/analysis"))
    analysis.update(approved_before_results=True, reviewed_by="owner", contrasts=contrasts)
    analysis["framework"]["status"] = framework
    return analysis


def pooled_contrast(learner, versus="baseline", group="catboost"):
    return dict(type="pooled_normalised_gain", application="B1_decision_features", group=group,
                learner=learner, versus=versus, datasets=list(DATASETS), direction=1)


def test_pooled_contrasts_are_unreachable_until_approved_and_complete():
    pooled = learner_scores.pooled_scores({"B1_decision_features": {"folds": synthetic_folds()}}, design())
    contrasts = [pooled_contrast("decision/both/median"), pooled_contrast("decision/joint/median")]
    with pytest.raises(ValueError, match="framework.status"):
        inference.pooled_confirmatory_tests(pooled, pooled_plan(contrasts, framework="awaiting_owner_approval"))
    with pytest.raises(ValueError, match="approved before results"):
        inference.pooled_confirmatory_tests(pooled, pooled_plan(contrasts) | {"approved_before_results": False})
    with pytest.raises(ValueError, match="Duplicate"):
        inference.pooled_confirmatory_tests(pooled, pooled_plan([contrasts[0], contrasts[0]]))
    gap = pooled_scores_with_gap = pooled.copy()
    gap.loc[gap.dataset_id.eq("D07") & gap.learner.eq("baseline") & gap.group.eq("catboost"),
            ["score", "status"]] = [np.nan, learner_scores.ZERO_HEADROOM]
    with pytest.raises(ValueError, match="inference stops"):
        inference.pooled_confirmatory_tests(pooled_scores_with_gap, pooled_plan(contrasts))
    fewer = pooled_contrast("decision/both/median") | {"datasets": list(DATASETS)[:17]}
    with pytest.raises(ValueError, match="inference stops"):
        inference.pooled_confirmatory_tests(pooled, pooled_plan([fewer]))
    with pytest.raises(ValueError, match="declared score type"):
        bad = pooled_contrast("decision/both/median") | {"type": "pooled_proper_score_skill"}
        inference.pooled_confirmatory_tests(pooled, pooled_plan([bad]))


def test_pooled_family_holm_covers_every_declared_contrast():
    pooled = learner_scores.pooled_scores({"B1_decision_features": {"folds": synthetic_folds()}}, design())
    contrasts = [pooled_contrast("decision/both/median"), pooled_contrast("decision/joint/median"),
                 pooled_contrast("decision/both/median", "decision/per_column/median", group="linear")]
    result = inference.pooled_confirmatory_tests(pooled, pooled_plan(contrasts))
    assert len(result) == 3 and result.holm_family_size.eq(3).all()
    assert result.datasets.eq(18).all() and result.contrast_type.eq("pooled_normalised_gain").all()
    assert result.p_holm.tolist() == pytest.approx(inference.holm(result.p_raw).tolist())
    assert {"bayes_p_better", "bayes_rope"} <= set(result.columns)
    first = pooled[pooled.group.eq("catboost")].pivot(index="dataset_id", columns="learner", values="score")
    assert result.p_raw.iloc[0] == pytest.approx(
        inference.wilcoxon_p((first["decision/both/median"] - first["baseline"]).to_numpy()))
    # Per-task contrasts in the same plan join the same Holm family.
    datasets = pd.DataFrame([dict(task_type="binary_classification", model_id="catboost", checkpoint_id="ck",
                                  feature_set="non_text_plus_both", comparison_id="c", reference="non_text",
                                  metric="roc_auc", dataset_id=d, mean_difference=0.01 * (i + 1),
                                  complete=True) for i, d in enumerate(list(DATASETS)[:5])])
    match = {k: datasets[k].iloc[0] for k in inference.GROUP}
    mixed = contrasts + [dict(match=match, datasets=list(DATASETS)[:5], direction=1)]
    result = inference.pooled_confirmatory_tests(pooled, pooled_plan(mixed), datasets=datasets)
    assert len(result) == 4 and result.holm_family_size.eq(4).all()
    assert set(result.contrast_type) == {"pooled_normalised_gain", "per_task"}
    assert result.p_holm.tolist() == pytest.approx(inference.holm(result.p_raw).tolist())


# ------------------------------------------------------------------ CD diagram


def test_no_difference_groups_join_only_pairwise_non_significant_runs():
    order = ["a", "b", "c", "d"]
    table = pd.DataFrame([("a", "b", 0.5), ("a", "c", 0.01), ("a", "d", 0.01),
                          ("b", "c", 0.3), ("b", "d", 0.02), ("c", "d", 0.4)],
                         columns=["learner", "versus", "p_holm"])
    assert critical_difference.no_difference_groups(order, table) == [(0, 1), (1, 2), (2, 3)]
    table.loc[table.learner.eq("b") & table.versus.eq("d"), "p_holm"] = 0.9
    assert critical_difference.no_difference_groups(order, table) == [(0, 1), (1, 3)]


def test_cd_diagram_draws_all_pairwise_and_control_modes_at_full_width():
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(18, 9)) + np.linspace(0, 1.5, 9)
    rows = [{"dataset_id": f"d{i}", "learner": f"decision/both/m{j}", "score": v}
            for i, row in enumerate(matrix) for j, v in enumerate(row)]
    frame = pd.DataFrame(rows)
    omnibus, table = inference.compare_learners(frame, higher_is_better=True, rope=0.01)
    fig = critical_difference.cd_diagram(omnibus, table, title="Synthetic")
    width, height = fig.get_size_inches()
    assert width == pytest.approx(style.WIDTH_FULL) and height <= style.MAX_HEIGHT
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("Nemenyi CD" in t for t in texts) and any("Bars join" in t for t in texts)
    fig.canvas.draw()
    omnibus, table = inference.compare_learners(frame, higher_is_better=True, rope=0.01,
                                                control="decision/both/m0")
    assert critical_difference.control_of(table) == "decision/both/m0"
    fig = critical_difference.cd_diagram(omnibus, table)
    assert any("Filled circle" in t.get_text() for t in fig.axes[0].texts)
    fig.canvas.draw()


def test_planned_design_reads_the_configuration_only():
    real = learner_scores.planned_design()
    assert len(real.datasets) == 18 and len(real.roster) == 17
    assert sum(t == "regression" for t in real.datasets.values()) == 9
    assert set(real.plan.phase) >= {"experiment_1", "experiment_2", "experiment_3", "experiment_4",
                                    "representations", "rq3", "data_efficiency"}
    assert real.a2_datasets["field_order"] is None
