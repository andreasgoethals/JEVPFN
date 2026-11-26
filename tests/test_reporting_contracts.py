"""Scientific reporting contracts at the production entry points, with small in-memory inputs."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    benchmark_export,
    complementarity,
    data_efficiency,
    group_a,
    group_b,
    group_b_views,
    group_c_views,
    inference,
    learner_scores,
)
from src.evaluation.attribution import _representation


def folds(dataset="d", providers=("strands",), *, size=128, draws=1):
    rows = []
    for draw in range(draws):
        for outer_fold in range(5):
            for provider in (None, *providers):
                rows.append({"dataset_id": dataset, "task_type": "binary_classification", "model_id": "linear",
                             "phase": "B5", "metric": "roc_auc", "training_size": size,
                             "draw": draw, "checkpoint_id": "checkpoint", "split_id": f"{dataset}/{draw}",
                             "outer_fold": outer_fold, "provider": provider, "pairing_identity": "same-data",
                             "feature_set": "non_text_plus_both" if provider else "non_text",
                             "value": 0.99975 if provider else 0.9995})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("identity", ["split_id", "checkpoint_id", "pairing_identity"])
def test_all_reducers_reject_different_measurement_identities(identity):
    mixed = folds()
    mixed.loc[mixed.provider.notna(), identity] = "other"
    with pytest.raises(ValueError, match=identity):
        learner_scores.arm_means(mixed)
    with pytest.raises(ValueError, match=identity):
        learner_scores.fold_values(mixed)
    with pytest.raises(ValueError, match=identity):
        group_b_views.size_draw_gains(group_b.headline(mixed), roster=("strands",))
    with pytest.raises(ValueError):
        group_b_views.merged_heads(folds(), mixed)


def test_b4_requires_finite_folds_planned_draws_and_the_declared_dataset_cohort():
    clean = folds(draws=5)
    head = group_b.headline(clean)
    result = data_efficiency.size_gains(head)
    assert result.headroom_share.item() == pytest.approx(0.5)
    assert result.draws.item() == 5
    bad = head.copy()
    bad.loc[bad.draw.eq(1) & bad.outer_fold.eq(4) & bad.provider.notna(), "value"] = np.nan
    assert data_efficiency.size_gains(bad).empty
    assert data_efficiency.size_gains(head[head.draw.ne(2)]).empty
    assert data_efficiency.size_gains(head.assign(draw=head.draw + 1)).empty
    curves = {n: {"sizes": [128], "datasets": ["d", "missing"]} for n in ("main", "extended")}
    assert data_efficiency.cohort_gains(result, curves).empty


def test_empty_b3_exports_lodo_schema_and_exposes_missing_cells(monkeypatch):
    from src.evaluation import row_benefit

    records = folds().loc[lambda r: r.provider.isna()].assign(phase="B1")
    monkeypatch.setattr(benchmark_export, "coverage", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(benchmark_export, "dataset_properties_and_redundancy",
                        lambda **k: (pd.DataFrame({"dataset_id": ["d"]}), pd.DataFrame()))
    monkeypatch.setattr(benchmark_export, "_group_a_frames", lambda **k: {})
    monkeypatch.setattr(row_benefit, "summaries", lambda **k: (pd.DataFrame(), pd.DataFrame()))
    tables = benchmark_export.build_tables("B4", records)
    assert tables["dataset_gains"].empty and "family" in tables["dataset_gains"]
    assert tables["lodo_predictions"].empty and {"model_id", "dataset_id", "input_set"} <= set(tables["lodo_predictions"])
    assert tables["lodo"].empty and "status" in tables["lodo"]


def test_b3_export_predicts_and_shuffles_headroom_share_on_the_common_dataset_cohort(monkeypatch):
    from copy import deepcopy

    from src.evaluation import row_benefit
    from src.evaluation.moderator_statistics import benjamini_hochberg
    from src.utils.config import load_yaml

    cfg = deepcopy(load_yaml("B4"))
    cfg["n_permutations"] = cfg["predict_in_advance"]["n_shuffles"] = 7
    rule = cfg["predict_in_advance"]
    ids = [f"d{i}" for i in range(12)] + ["zero", "unmeasured"]
    tasks = ["binary_classification", "regression"] * 7
    shares = np.array([0.2, -0.1, 0.1, -0.3, 0.0, 0.05, -0.05, 0.4, -0.2, 0.15, 0.3, -0.15])
    x = np.random.default_rng(42).normal(size=(len(ids), 3))
    properties = pd.DataFrame(dict(zip(["log10_median_text_chars", "pilot_redundancy_r2", "decision_redundancy_r2"],
                                       x.T))).assign(dataset_id=ids, task_type=tasks)
    records = []
    for i, (dataset, task) in enumerate(zip(ids[:-1], tasks)):
        base = 1.0 if dataset == "zero" else 0.9 if task == "binary_classification" else -0.8
        gain = 0 if dataset == "zero" else (1 - base) * shares[i]
        for fold in range(5):
            for provider in (None, "jev"):
                records.append({"dataset_id": dataset, "task_type": task, "model_id": "linear", "outer_fold": fold,
                                "phase": "B1", "metric": "roc_auc" if task == "binary_classification" else "r2",
                                "checkpoint_id": "fixed", "split_id": dataset, "provider": provider,
                                "feature_set": "non_text_plus_both" if provider else "non_text",
                                "value": base + gain if provider else base})
    monkeypatch.setattr(benchmark_export, "load_yaml", lambda name: cfg)
    monkeypatch.setattr(benchmark_export, "coverage", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(benchmark_export, "dataset_properties_and_redundancy",
                        lambda **kwargs: (properties, pd.DataFrame()))
    monkeypatch.setattr(benchmark_export, "_group_a_frames", lambda **kwargs: {})
    monkeypatch.setattr(row_benefit, "summaries", lambda **kwargs: (pd.DataFrame(), pd.DataFrame()))
    tables = benchmark_export.build_tables("B4", pd.DataFrame(records), cohort="paid")
    got = tables["lodo_predictions"]
    summary = tables["lodo"].set_index("input_set")
    assert set(got.outcome) == set(summary.outcome) == {"headroom_share"}
    assert set(summary.excluded_datasets) == {"unmeasured,zero"}
    assert summary.datasets.eq(12).all()
    # The numerical oracle receives only the twelve manually specified dimensionless gains.
    reference = pd.DataFrame({"dataset_id": ids[:12], "value": shares})
    for name, spec in rule["input_sets"].items():
        args = {"features": spec["properties"], "value": "value"}
        expected = group_b.lodo_predictions(reference, properties, **args)
        actual = got[got.input_set.eq(name)].set_index("dataset_id").loc[ids[:12]]
        assert actual.actual.to_numpy() == pytest.approx(shares)
        assert actual.predicted.to_numpy() == pytest.approx(expected.predicted)
        score = group_b.lodo_evaluation(expected, outcome=reference, properties=properties, n_shuffles=7, **args)
        for metric in ("regret", "regret_always", "regret_never", "shuffle_p_rho", "shuffle_p_accuracy"):
            assert summary.loc[name, metric] == pytest.approx(score[metric])
    associations = tables["associations"]
    assert set(zip(associations.task_family, associations.outcome)) == {
        ("all", "headroom_share"), ("classification", "raw_gain"), ("regression", "raw_gain")}
    assert set(associations.loc[associations.task_family.eq("all"), "units"]) == {12}
    assert associations.q_value_bh.to_numpy() == pytest.approx(benjamini_hochberg(associations.p_value))
    mixed = reference.assign(raw_gain=shares, task_type=tasks[:12])
    with pytest.raises(ValueError, match="cannot be pooled"):
        group_b.lodo_predictions(mixed, properties, features=rule["input_sets"]["pre_extraction"]["properties"])

def test_undefined_omnibus_keeps_valid_pairwise_results_and_invalid_p_values_stop():
    scores = pd.DataFrame([{"dataset_id": d, "learner": learner, "score": d}
                           for d in range(5) for learner in ("x", "y", "z")])
    omnibus, report = inference.compare_learners(scores, higher_is_better=True, rope=0.01)
    assert omnibus["status"] == "all_tied" and omnibus["friedman_p"] is None
    assert set(omnibus["mean_ranks"].values()) == {2}
    assert report.p_holm.eq(1).all() and report.bayes_p_equivalent.eq(1).all()
    for bad_p in (np.nan, np.inf, -0.01, 1.1):
        with pytest.raises(ValueError, match="finite p-values"):
            inference.holm([0.01, bad_p])


def test_complementarity_differentiates_real_arms_and_does_not_break_ties_by_row_order():
    emb = {"embedding": {"encoder_id": "e5", "pca_scope": "per_column", "pca_components_per_column": 32}}
    labels = [_representation({**emb, "provider": provider}, "non_text_plus_both_plus_frozen_e5", "B2")
              for provider in ("strands", "liquid")]
    labels += [_representation(emb, "non_text_plus_frozen_e5", "B2"),
               _representation({**emb, "probe": {"view": "joint"}}, "non_text_plus_supervised_probe", "B3")]
    assert len(set(labels)) == 4
    result = complementarity.pair_statistics(np.zeros(100), np.zeros(100))
    assert result["top_overlap_status"] == "tied_cutoff" and np.isnan(result["jaccard_top"])
    assert complementarity.chance_jaccard(4, 2) == pytest.approx((1 + 4 / 3) / 6)


def test_reductions_distinguish_units_denominators_and_partial_c_exports():
    g = pd.DataFrame([{"dataset_id": "one", "task_type": "binary_classification", "representation": "x",
                       "model_id": "linear", "raw_gain": 0.1, "headroom_share": 0.2}])
    other = g.assign(dataset_id="two", task_type="regression", raw_gain=0.8, headroom_share=0.4)
    summary = group_b.summarize(pd.concat([g, other]), n_boot=10).iloc[0]
    assert np.isnan(summary.mean_raw_gain) and np.isnan(summary.raw_ci_low)
    assert summary.mean_headroom_share == pytest.approx(0.3)
    scope = group_c_views.combined_gains({"dataset_gains": pd.concat([g, other])}, {"dataset_gains": g.copy()})
    assert set(scope.dataset_id) == {"one", "two"}
    assert group_c_views.predictor_average(scope).empty
    with pytest.raises(ValueError, match="disagree"):
        group_c_views.combined_gains({"dataset_gains": g}, {"dataset_gains": g.assign(raw_gain=1.0)})
    effects = pd.DataFrame([{"dataset": "d", "provider": "p", "mode": "joint", "variant": "paraphrase_" + str(i),
                             "mean_tv": 0.1, "excess_tv": 0.05, "flip_rate": 0.0, "matched_requests": 4}
                            for i in (1, 2, 3)])
    counts = group_a.excess_by_dataset(effects).iloc[0]
    assert counts.matched_presentations == 12
    assert counts.min_matched_requests_per_variant == counts.max_matched_requests_per_variant == 4


def test_fixed_paid_extension_has_its_own_archive_and_never_changes_by_measured_results(monkeypatch, tmp_path):
    from src.evaluation import comparison_report, robustness_export, study_export

    monkeypatch.delenv("JEVPFN_DECISION_COHORT", raising=False)
    open_models = benchmark_export.decision_roster()
    assert len(open_models) == 16 and benchmark_export.decision_roster("with_jev") == [*open_models, "jev"]
    assert benchmark_export.export_path("B1") != benchmark_export.export_path("B1", "all")
    assert comparison_report.report_path() != comparison_report.report_path("all")
    assert robustness_export.export_path() != robustness_export.export_path("all")
    assert comparison_report.report_config()["comparison"]["providers"] == open_models
    for cohort in ("open_weight", "all"):
        record = benchmark_export.cohort_record(cohort)
        result = {"status": "compared", **record, "results": pd.DataFrame({"test": ["a"]}),
                  "forest": pd.DataFrame({"n": [18]})}
        target = tmp_path / (benchmark_export.cohort_key(cohort) + ".zip")
        learner_scores.publish_confirmatory(result, target=target,
                                           analysis={"contrasts": [{"group": "a"}], "framework": {}})
        _, receipt = study_export.read_export(target, schema="jevpfn.confirmatory_family.v1", tables=("results", "forest"))
        assert receipt["decision_roster"] == record["decision_roster"]
    assert len(list(tmp_path.glob("*.zip"))) == 2
    with pytest.raises(ValueError, match="expanded primary release"):
        learner_scores.publish_confirmatory({"status": "compared", **benchmark_export.cohort_record("paid")},
                                            target=tmp_path / "paid.zip")
    assert learner_scores.run_confirmatory_family(cohort="with_jev_added")["status"] == "not_applicable"


def test_primary_archive_preserves_analysis_time_versions_after_a_registered_pin_changes(tmp_path, monkeypatch):
    from copy import deepcopy

    from src.evaluation import design, study_export
    from src.utils import paths

    config = design.load_yaml
    candidates = deepcopy(config("extractors"))
    monkeypatch.setattr(design, "load_yaml", lambda name: candidates if name == "extractors" else config(name))
    monkeypatch.setattr(paths, "notebook_artifacts", lambda *args, **kwargs: tmp_path)
    old = benchmark_export.cohort_record("with_jev")
    candidates["candidates"]["jev"]["version"] = "99.0"
    new = benchmark_export.cohort_record("with_jev")
    plan = {"contrasts": [{"group": "fixture"}], "framework": {}}
    result = {"status": "compared", "results": pd.DataFrame({"stat": [0.0]}), "forest": pd.DataFrame({"effect": [0.0]})}
    archives = []
    for record in (old, new):
        archive = learner_scores.publish_confirmatory({**result, **record}, analysis=plan)
        _, meta = study_export.read_export(archive, schema="jevpfn.confirmatory_family.v1", tables=("results", "forest"))
        assert meta["decision_model_versions"] == record["decision_model_versions"]
        assert meta["roster_sha256"] == record["roster_sha256"]
        assert archive.name == f"with_jev-{record['roster_sha256'][:16]}.zip"
        archives.append(archive)
    assert archives[0] != archives[1] and all(a.is_file() for a in archives)
    with pytest.raises(ValueError, match="recorded ordered cohort"):
        learner_scores.publish_confirmatory({**result, **new, "decision_model_versions": old["decision_model_versions"]},
                                            analysis=plan)
    with pytest.raises(ValueError, match="Primary table roster_sha256"):
        learner_scores.publish_confirmatory({**result, **old, "results": result["results"].assign(roster_sha256=new["roster_sha256"])},
                                            analysis=plan)
    monkeypatch.setattr(learner_scores, "read_frames", lambda *args, **kwargs: ({"folds": pd.DataFrame()}, old))
    stopped = learner_scores.run_confirmatory_family(cohort="with_jev")
    assert stopped["status"] == "stopped" and "different decision-model versions" in stopped["reason"]

def test_row_benefits_use_the_same_models_and_equal_dataset_weights():
    from src.evaluation.models import MAIN_MODELS

    records = [{"dataset": dataset, "task": task, "model": model, "features": "non_text_plus_both",
                "axis": "length", "cohort": "bin_1", "benefit": value, "phase": "B1",
                "representation": '{"provider": "' + p + '"}'}
               for dataset, task, value in (("bin", "binary_classification", 0.6),
                                            ("multi1", "multiclass_classification", 0.3),
                                            ("multi2", "multiclass_classification", 0.0))
               for model in MAIN_MODELS for p in ("p1", "p2")]
    data = pd.DataFrame(records)
    roster = ("p1", "p2")
    cells = group_b_views.row_cohort_cells(data, roster=roster)
    result = group_b_views.cohort_benefits(cells).iloc[0]
    assert result.mean_benefit == pytest.approx(0.3) and result.datasets == 3
    incomplete = data.drop(index=[0])
    result = group_b_views.cohort_benefits(group_b_views.row_cohort_cells(incomplete, roster=roster))
    assert result.datasets.item() == 2 and result.mean_benefit.item() == pytest.approx(0.15)
    assert group_b_views.model_pama(group_b.headline(folds()), roster=("strands", "unrun")).empty


def test_a2_canonicalized_field_order_and_descriptive_jensen_are_distinguished():
    import test_learner_scores as fx
    from matplotlib import pyplot as plt

    from src.visualize import critical_difference

    values = fx.averaging_quality()
    rows = pd.DataFrame([{"dataset": d, "provider": p, "variant": "field_order",
                          "status": "canonicalized_by_runtime", "count": 3}
                         for d in fx.DATASETS for p in fx.ROSTER])
    design = fx.design(a2_datasets={"wording": list(fx.DATASETS), "field_order": list(fx.DATASETS)})
    result = learner_scores.run_application("A2_stability", {"averaging_quality": values, "coverage": rows},
                                            analysis=fx.approved_analysis(), design=design)
    groups = {g["group"]: g for g in result["groups"]}
    canonical = groups["field_order/average_vs_original"]
    assert canonical["status"] == "stopped" and {c["status"] for c in canonical["cells"]} == {"canonicalized_by_runtime"}
    assert result["procedure"]["inference"] == "confirmatory"
    assert result["multiplicity_scope"] == "selected_models_vs_original_within_each_factor"
    primary = groups["wording/average_vs_original"]
    assert primary["inference"] == "confirmatory" and len(primary["table"]) == len(fx.ROSTER)
    assert primary["table"].holm_family_size.eq(len(fx.ROSTER)).all()
    assert primary["table"].p_holm.tolist() == pytest.approx(inference.holm(primary["table"].p_wilcoxon))
    assert primary["table"].holm_family.unique().tolist() == ["A2_stability/wording/average_vs_original"]
    jensen = groups["wording/average_vs_mean_single"]
    assert jensen["inference"] == "descriptive" and jensen["table"].empty
    counts = next(g for g in learner_scores.summary_record(result)["groups"] if g["group"] == jensen["group"])
    assert counts["holm_tests"] == 0 and counts["holm_significant"] is None
    (fig, _, caption), = critical_difference.cd_figures({**result, "groups": [jensen]})
    try:
        texts = [label.get_text() for label in fig.axes[0].texts]
        assert "descriptive" in caption and any("Mean single-presentation score" in t for t in texts)
        assert all("Wilcoxon" not in t and "not significant" not in t for t in texts)
        assert all(line.get_marker() != "o" for line in fig.axes[0].lines)
    finally:
        plt.close(fig)
    framework = fx.approved_analysis()["framework"]
    assert framework["confirmatory_scores"]["A2"] == "proper_score_skill"
    for name in ("B1_decision_features", "B2_frozen_embeddings"):
        entry = framework["application"][name]
        assert entry["inference"] == "confirmatory" and entry["holm_family"].endswith("within_each_predictor")


def test_moderator_properties_read_verified_current_and_legacy_a1_archives(tmp_path, monkeypatch):
    from copy import deepcopy
    from zipfile import ZipFile

    from src.evaluation import comparison_report, moderators, study_export
    from src.evaluation.decision_comparison import Comparison

    report = Comparison()
    for site, field in (("strands", 0.6), ("kev", 0.7)):
        report.add("distributions", {"dataset_id": "d", "provider": site, "weighting": "rows",
                                     "representation": "joint", "variable": "max_probability", "mean": field})
    report.add("pairs", {"dataset_id": "d", "model_a": "strands", "model_b": "kev", "representation": "joint",
                         "weighting": "rows", "agreement": 0.8})
    target = tmp_path / "comparison.open_weight.zip"
    legacy = tmp_path / "comparison.zip"
    cfg = comparison_report.report_config()
    monkeypatch.setattr(comparison_report, "report_path", lambda cohort=None: target)
    for path in (None, target):
        props, status = moderators.decision_properties(path)
        assert props.empty and "awaiting_cpu_report" in status

    study_export.write_export(legacy, {"archived": pd.DataFrame()}, schema="jevpfn.rq1a.report.v1",
                              identity="old", origin="fixture")
    props, status = moderators.decision_properties()
    assert props.empty and "superseded_schema" in status and str(legacy) in status
    comparison_report.write_report(report, cfg, legacy, origin="fixture", joint_rows={})
    for path in (None, legacy):
        props, status = moderators.decision_properties(path)
        assert "available" in status and props.decision_mean_pairwise_agreement.tolist() == [0.8, 0.8]
    assert learner_scores.read_frames("A1")[1]["path"] == str(legacy)

    stale = deepcopy(cfg)
    stale["comparison"]["random_seed"] += 1
    comparison_report.write_report(report, stale, target, origin="fixture", joint_rows={})
    for path in (None, target):
        props, status = moderators.decision_properties(path)
        assert props.empty and "superseded_configuration" in status
    assert learner_scores.read_frames("A1")[0] == {}

    comparison_report.write_report(report, cfg, target, origin="fixture", joint_rows={})
    assert moderators.decision_properties()[0].decision_mean_pairwise_agreement.tolist() == [0.8, 0.8]
    with ZipFile(target) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["pairs.parquet"] += b"damaged"
    with ZipFile(target, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    for path in (None, target):
        with pytest.raises(ValueError, match="checksum"):
            moderators.decision_properties(path)


def test_paid_comparisons_include_saved_controls_without_planning_new_control_fits(monkeypatch):
    from src.evaluation.campaign import benchmark_plan

    pending_fits = benchmark_plan(cohort="paid")
    assert len(pending_fits) == 3843 and {arm["provider"] for arm in pending_fits} == {"jev"}
    design = learner_scores.planned_design("paid")
    plan = design.plan
    assert design.roster == ("jev",) and len(design.datasets) == 18
    assert plan.provider.notna().sum() == len(pending_fits)
    assert set(plan.loc[plan.phase.eq("B5") & plan.provider.isna(), "representation"]) >= {"baseline"}
    assert plan.loc[plan.phase.eq("B6"), "provider"].isna().all()
    dataset = next(d for d, task in design.datasets.items() if task == "binary_classification")
    phases = ("B1", "B2", "B3")  # The B2 and B3 exports share the B1 controls.
    arms = plan[plan.dataset_id.eq(dataset) & plan.model_id.eq("linear") & plan.phase.isin(phases)].copy()
    arms["dataset_id"] = "fixture"
    arms = arms.to_dict("records")
    rows = []
    for arm in arms:
        row = {key: arm.get(key) for key in benchmark_export.RECIPE_COLUMNS}
        rows.extend({**row, "outer_fold": fold, "task_type": "binary_classification", "metric": "roc_auc",
                     "checkpoint_id": "linear", "split_id": "same-split", "pairing_identity": "same-data",
                     "value": 0.5 if arm["representation"] == "baseline" else 0.6} for fold in range(5))
    folds = pd.DataFrame(rows)
    monkeypatch.setattr(benchmark_export, "comparison_plan", lambda cohort: arms)
    monkeypatch.setattr(complementarity, "summaries", lambda **kwargs: (pd.DataFrame(), pd.DataFrame()))
    unrelated = folds[folds.provider.eq("jev")].assign(provider="strands")
    mixed = pd.concat([folds, unrelated])
    local = learner_scores.Design({"fixture": "binary_classification"}, design.roster, pd.DataFrame(arms))
    expected = {"B2": ({"frozen"}, {"frozen/e5_small_v2/per_column32"}),
                "B3": ({"tar", "probe"}, {"tar/e5_small_v2_tar/per_column32", "probe/e5_small_v2/joint"})}
    for study, (families, learners) in expected.items():
        own = mixed[mixed.phase.isin(benchmark_export.STUDIES[study]["phases"])]
        tables = benchmark_export.build_tables(study, own, cohort="paid")
        assert set(tables["folds"].provider.dropna()) == {"jev"}
        coverage = tables["coverage"]
        planned = [a for a in arms if a["phase"] in benchmark_export.STUDIES[study]["phases"]]
        assert coverage.planned_arms.sum() == len(planned) and coverage.complete_arms.eq(coverage.planned_arms).all()
        assert families <= set(tables["contrasts"].family)
        group, = getattr(learner_scores, f"{study.lower()}_groups")(tables, local)
        assert learners <= set(group.learners)
        assert group.scores.status.eq(learner_scores.OK).all()
    assert learner_scores.run_confirmatory_family(cohort="paid")["status"] == "not_applicable"


def test_pooled_inference_cli_reads_verified_exports_without_the_raw_fold_store(monkeypatch, capsys):
    import sys

    from src.evaluation import results

    calls = []
    def family(*, analysis, cohort):
        calls.append((len(analysis["contrasts"]), cohort))
        return {"status": "compared", "results": pd.DataFrame({"family_tests": [21]})}

    monkeypatch.setattr(learner_scores, "run_confirmatory_family", family)
    monkeypatch.setattr(results, "collect", lambda: pytest.fail("Pooled inference should read compact exports"))
    monkeypatch.setattr(sys, "argv", ["results", "--infer", "--cohort", "all"])
    results.main()
    assert calls == [(21, "all")] and "family_tests" in capsys.readouterr().out


def test_all_21_contrasts_run_with_identical_procedure_before_and_after_paid_extension(tmp_path):
    from src.evaluation import study_export
    from src.evaluation.endpoints import HEADLINES
    from src.evaluation.models import MAIN_MODELS

    planned = learner_scores.planned_design("all")
    all_providers = benchmark_export.decision_roster("all")
    records = []
    for number, (dataset, task) in enumerate(planned.datasets.items()):
        for model in MAIN_MODELS:
            for fold in range(5):
                base = {"dataset_id": dataset, "task_type": task, "model_id": model, "outer_fold": fold,
                        "checkpoint_id": model + "-checkpoint", "split_id": dataset + "-split", "metric": HEADLINES[task]}
                baseline = 0.5 + number / 1000
                records.append({**base, "phase": "B1", "feature_set": "non_text", "value": baseline})
                for index, provider in enumerate(all_providers):
                    records.append({**base, "phase": "B1", "feature_set": "non_text_plus_both",
                                    "provider": provider, "value": baseline + 0.01 + index / 1000})
                for phase, fs, encoder, gain in (("B2", "non_text_plus_frozen_e5", "e5_small_v2", 0.005),
                                                 ("B3", "non_text_plus_tar", "e5_small_v2_tar", 0.007)):
                    records.append({**base, "phase": phase, "feature_set": fs, "encoder": encoder,
                                    "pca_scope": "per_column", "pca_components": 32, "value": baseline + gain})
    complete = pd.DataFrame(records)
    archives = []
    for cohort in ("open_weight", "all"):
        available = complete[complete.provider.ne("jev")] if cohort == "open_weight" else complete
        inputs = {"B1_decision_features": {"folds": available[available.phase.eq("B1")]},
                  "B2_frozen_embeddings": {"folds": available[available.phase.isin(["B1", "B2"])]},
                  "B3_label_trained": {"folds": available[available.phase.isin(["B1", "B3"])]}}
        if cohort == "open_weight":
            stopped = learner_scores.run_confirmatory_family(inputs, cohort="all", n_boot=5)
            assert stopped["status"] == "stopped"
        result = learner_scores.run_confirmatory_family(inputs, cohort=cohort, n_boot=10)
        assert result["status"] == "compared"
        tests = result["results"]
        assert len(tests) == 21 and tests.datasets.eq(18).all()
        assert tests.p_holm.tolist() == pytest.approx(inference.holm(tests.p_raw))
        assert tests.decision_models.eq(len(benchmark_export.decision_roster(cohort))).all()
        path = tmp_path / (benchmark_export.cohort_key(cohort) + ".zip")
        learner_scores.publish_confirmatory(result, target=path)
        tables, receipt = study_export.read_export(path, schema="jevpfn.confirmatory_family.v1", tables=("results", "forest"))
        assert len(tables["results"]) == len(tables["forest"]) == receipt["contrasts"] == 21
        archives.append(path)
    assert len(set(archives)) == 2 and all(p.is_file() for p in archives)
