"""General notebooks (data, feature creation, qualification, study audit, synthesis): readers and
figures on deterministic fixtures with the study's real shapes (18 datasets = 3 binary +
6 multiclass + 9 regression, 17 decision models, 7 predictors, 5 folds)."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.evaluation import study_audit  # noqa: E402
from src.evaluation.endpoints import HEADLINES  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
from src.visualize import style  # noqa: E402

PREDICTORS = list(style.MAIN_PREDICTORS)


def roster():
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries

    main = load_yaml("experiment_1/main")
    datasets = [e["id"] for e in selected_entries(load_catalog(), main)]
    tasks = {d: load_yaml("dataset_semantics")["datasets"][d]["task_type"] for d in datasets}
    return datasets, tasks, list(main["candidate_extractors"])


def fixture_store_status():
    datasets, _, models = roster()
    local = pd.DataFrame([{"dataset_id": d, "provider": m, "representation": v, "rows": 10}
                          for d in datasets for m in models[1:12] for v in ("per_column", "joint")])
    manifests = pd.DataFrame([{"dataset_id": d, "provider": "glide", "representation": "joint", "rows": 10}
                              for d in datasets[:5]])
    return study_audit.feature_store_status(local, manifests, datasets=datasets, providers=models,
                                            partial={("jev", d) for d in datasets[:15]})


def fixture_timings():
    datasets, _, models = roster()
    rng = np.random.default_rng(0)
    rows = []
    for k, model in enumerate(models[1:9]):
        for d in datasets:
            n = int(rng.integers(5_000, 300_000))
            rows.append({"model_id": model, "kind": "decision", "dataset_id": d, "requests": n,
                         "gpu_seconds": n * 0.01 * (1 + k) * rng.uniform(0.8, 1.2), "job_id": "1", "source": "x"})
    for k, encoder in enumerate(("e5_small_v2", "e5_large_v2", "qwen3_embedding_0_6b")):
        for d in datasets:
            rows.append({"model_id": encoder, "kind": "encoder", "dataset_id": d, "requests": 60_000,
                         "gpu_seconds": 60_000 * 0.0005 * (k + 1), "job_id": "2", "source": "y"})
    return pd.DataFrame(rows)


def fixture_ledger():
    names = {"jevpfn-features": 120, "jevpfn-embeddings": 54, "jevpfn-benchmark": 400, "jevpfn-tar": 90,
             "jevpfn-prepare-pca": 90, "jevpfn-c1-hidden": 18, "jevpfn-c2-adaptation": 270,
             "jevpfn-prepare-probe": 54, "jevpfn-exp0-contracts": 30, "jevpfn-group-b-export": 5}
    states = ["COMPLETED"] * 8 + ["FAILED", "TIMEOUT", "CANCELLED"]
    jobs, k = [], 0
    for name, count in names.items():
        for i in range(count):
            gpus = 0 if name in ("jevpfn-prepare-pca", "jevpfn-prepare-probe", "jevpfn-group-b-export") else 1
            seconds = 600 + 37 * (i % 13)
            jobs.append({"job_id": str(k), "job_name": name, "state": states[(i * 7 + k) % len(states)],
                         "gpus": gpus, "cpus": 8, "elapsed_seconds": seconds,
                         "gpu_hours": seconds * gpus / 3600, "cpu_core_hours": seconds * 8 / 3600})
            k += 1
    return {"schema": "jevpfn.compute-ledger.v1", "jobs": jobs, "summary": []}


def fixture_folds(plan, *, drop_models=("sieve",), drop_dataset_family=None):
    """Five complete folds of every planned arm, minus a roster gap and one missing family."""
    rows = []
    for arm in plan.to_dict("records"):
        if arm.get("provider") in drop_models:
            continue
        if drop_dataset_family and (arm["dataset_id"], study_audit.FEATURE_FAMILY[arm["feature_set"]]) == drop_dataset_family:
            continue
        task = arm["task_type"]
        for fold in range(5):
            base = {k: arm.get(k) for k in study_audit.ARM_KEYS if k != "model_id"}
            base.update(model_id=arm["model"], task_type=task, outer_fold=fold)
            rows.append({**base, "metric": HEADLINES[task], "value": 0.7 + fold / 100})
            rows.append({**base, "metric": "seconds.fit", "value": 1 + arm["predictor_columns"] / 50})
            rows.append({**base, "metric": "columns.representation", "value": arm["predictor_columns"]})
    return pd.DataFrame(rows)


def fixture_plan(phases=("experiment_1", "experiment_2", "experiment_3")):
    from src.evaluation.campaign import benchmark_plan

    _, tasks, _ = roster()
    plan = pd.DataFrame(benchmark_plan())
    plan = plan[plan.phase.isin(phases)].copy()
    plan["task_type"] = plan.dataset_id.map(tasks)
    return plan


def fixture_gains():
    """Group B/C export tables with dataset x predictor gains for the glance arms."""
    datasets, tasks, models = roster()
    rng = np.random.default_rng(1)
    frames = {"B1": {}, "B2": {}, "C2": {}}
    from src.evaluation.synthesis_overview import GLANCE_ARMS

    for k, (_, study, table, representation) in enumerate(GLANCE_ARMS):
        rows = []
        for d in datasets:
            for m in PREDICTORS:
                baseline = rng.uniform(0.55, 0.9)
                gain = rng.normal(0.01 * (k + 1), 0.01)
                rows.append({"dataset_id": d, "task_type": tasks[d], "model_id": m,
                             "representation": representation, "raw_gain": gain, "baseline": baseline,
                             "headroom_share": gain / (1 - baseline)})
        frame = pd.DataFrame(rows)
        frames[study][table] = pd.concat([frames[study].get(table, pd.DataFrame()), frame], ignore_index=True)
    return frames


def fixture_profiles():
    from src.evaluation.qualification import profile_plan

    predictors = pd.DataFrame([{"model": m, "contract_passed": i % 3 != 0, "source": None}
                               for i, m in enumerate([*PREDICTORS, "tabstar", "contexttab", "ag_mm"])])
    profiles = {}
    plan = profile_plan()
    from src.evaluation.notebook_study import profile_evidence

    table = profile_evidence(predictors, profiles, plan)
    table["passed"] = (table.required * np.linspace(0, 1, len(table))).round().astype(int)
    return predictors, table


# ------------------------------------------------------------------------------ tests


def test_feature_store_status_marks_complete_partial_and_unknown_views():
    status = fixture_store_status()
    model_count = len(roster()[2])
    assert len(status) == 18 * model_count * 2
    jev = status[status.provider.eq("jev")]
    assert set(jev.status) == {"partial", "unknown"}
    glide = status[status.provider.eq("glide")]
    assert glide.status.eq("observed").sum() == 5 and set(glide[glide.status.eq("observed")].view) == {"joint"}
    from src.visualize.audit import feature_store_figure

    fig, name, caption = feature_store_figure(status)
    assert name == "feature_store_completeness" and f"{model_count} decision" in caption
    assert "Local stores contain 396" in caption and "records alone account for 5" in caption
    plt.close(fig)


def test_extraction_workload_pools_timed_requests_and_never_estimates_untimed_models():
    timings = fixture_timings()
    audit = {"self_hosted": [{"provider": m, "unique_inputs": 1_698_127} for m in roster()[2][1:]]}
    work = study_audit.extraction_workload(timings, token_audit=audit, distinct_texts=1_181_154)
    assert len(work) == len(roster()[2]) + 3
    timed = work.dropna(subset=["inference_gpu_hours_per_1000"])
    part = timings[timings.model_id.eq(timed.model_id.iloc[0])]
    assert timed.inference_gpu_hours_per_1000.iloc[0] == pytest.approx(
        part.gpu_seconds.sum() / part.requests.sum() * 1000 / 3600)
    assert work.set_index("model_id").loc["sieve", "inference_gpu_hours"] != work.set_index("model_id").loc["sieve", "inference_gpu_hours"]
    from src.visualize.build_review import extraction_figure

    # Near-complete coverage must remain distinguishable from an exact workload match.
    work.loc[work.model_id.eq(timed.model_id.iloc[0]), "distinct_requests"] = part.requests.sum() + 1
    fig, name, caption = extraction_figure(work, study_audit.dataset_rates(timings), jev_charges=4.2)
    assert name == "extraction_compute" and "audited number of distinct inputs" in caption
    assert "<100%" in [label.get_text() for label in fig.axes[1].texts]
    assert len(fig.axes[1].patches) == len(timed)
    plt.close(fig)


def test_reuse_separates_cross_view_workload_from_within_view_repetition(monkeypatch):
    from matplotlib.colors import to_rgba

    from src.data.text_reuse import text_reuse_profile
    from src.visualize import exploration

    fields = pd.DataFrame({"a": ["x", "x", "y", None], "b": ["z", "z", None, None]})
    summary = pd.DataFrame([
        {"dataset_id": name, "dataset": name, "task_type": "regression",
         **text_reuse_profile(frame, dataset_id=name)["summary"]}
        for name, frame in (("included", fields), ("excluded", fields[["a"]]))
    ])
    lengths = pd.DataFrame({"dataset_id": summary.dataset_id, "dataset": summary.dataset,
                           "text_columns": [2, 1], **{f"joint_tokens_{q}": [1, 1] for q in
                           ("p01", "p25", "median", "p75", "p99", "max")}})
    monkeypatch.setattr(exploration, "_removed", lambda: {"excluded"})
    fig, _, _ = exploration.reuse_figure(summary, lengths)
    per_column, extra_joint = fig.axes[1].containers
    assert [bar.get_width() for bar in per_column] == [3, 2]
    assert [bar.get_width() for bar in extra_joint] == [1, 0]
    per_column, joint = fig.axes[2].collections
    np.testing.assert_allclose(per_column.get_offsets()[:, 0], [40, 100 / 3])
    np.testing.assert_allclose(joint.get_offsets()[:, 0], [100 / 3, 100 / 3])
    assert per_column.get_offsets()[1, 1] != joint.get_offsets()[1, 1]
    for points in (per_column, joint):
        np.testing.assert_allclose(points.get_facecolors()[1], to_rgba(style.REMOVED_COLOR))
    plt.close(fig)


def test_batched_gpu_rate_uses_requests_inside_batches(tmp_path):
    stage = {"stage": "decision_inference", "dataset_id": "d", "items": 8,
             "observations": 4, "wall_seconds": 36}
    (tmp_path / "batch.json").write_text(json.dumps({"name": "cloudflare", "stages": [stage]}))
    timings = study_audit.inference_timings(tmp_path)
    assert timings.requests.tolist() == [8]
    assert study_audit.dataset_rates(timings).gpu_hours_per_1000.iloc[0] == pytest.approx(1.25)


def test_jev_spend_counts_each_request_once_across_snapshots(tmp_path):
    path = tmp_path / "BIN_TEXT_FAKE_JOB_POSTING.sqlite3"
    request = json.dumps({"settings": {"provider": "jev"}})
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE imported_extraction_records (snapshot_sha256 TEXT, table_name TEXT, "
                   "record_key TEXT, record_json TEXT)")
        for snapshot in ("a", "b"):
            for key, spent in (("k1", 0.1), ("k2", 0.2)):
                db.execute("INSERT INTO imported_extraction_records VALUES (?,?,?,?)",
                           (snapshot, "extraction_jobs", key, json.dumps(
                               {"request_json": request, "status": "completed", "spent_usd": spent,
                                "reserved_usd": 0.0})))
        db.execute("INSERT INTO imported_extraction_records VALUES (?,?,?,?)",
                   ("a", "extraction_policy", "1", json.dumps({"settings_json": json.dumps({"budget": 0.3})})))
    spend = study_audit.jev_spend(tmp_path)
    assert spend.jobs.iloc[0] == 2 and spend.spent_usd.iloc[0] == pytest.approx(0.3)
    assert spend.allocation_usd.iloc[0] == 0.3
    budget = study_audit.jev_budget(spend)
    assert budget["ceiling_usd"] == 5.50 and budget["remaining_usd"] == pytest.approx(5.2)


def test_arm_completeness_separates_roster_gaps_missing_and_excluded():
    plan = fixture_plan(("experiment_1",))
    datasets = plan.dataset_id.unique()
    folds = fixture_folds(plan, drop_dataset_family=(datasets[0], "no text"))
    table = study_audit.arm_completeness(folds, plan, roster=roster()[2])
    decision = table[table.family.eq("decision")]
    assert set(decision.status) == {"roster_gap"}
    baseline = table[table.family.eq("no text")]
    assert set(baseline[baseline.dataset_id.eq(datasets[0])].status) == {"missing"}
    assert set(baseline[~baseline.dataset_id.eq(datasets[0])].status) == {"complete"}
    checks = study_audit.fold_checks(folds, "B1")
    assert checks["duplicate_records"] == 0 and checks["arms_without_five_folds"] == 0
    duplicated = study_audit.fold_checks(pd.concat([folds, folds.head(3)]), "B1")
    assert duplicated["duplicate_records"] == 3
    from src.visualize.audit import completeness_figure

    fig, name, _ = completeness_figure(table)
    assert name == "arm_completeness"
    plt.close(fig)


def test_ledger_views_classify_group_b_and_c_jobs_and_draw():
    from src.cluster.compute_ledger import stage_of

    assert stage_of("jevpfn-c1-hidden") == "strands_hidden_states"
    assert stage_of("jevpfn-c2-adaptation") == "strands_adaptation"
    assert stage_of("jevpfn-prepare-probe") == "label_trained_classifier"
    assert stage_of("jevpfn-group-c-export") == "result_export"
    jobs, by_study = study_audit.ledger_tables(fixture_ledger())
    assert "Other" not in set(by_study.study)
    assert (by_study.successful_gpu_hours <= by_study.gpu_hours + 1e-12).all()
    register = study_audit.failure_register(jobs)
    assert register.jobs.sum() == len(jobs)
    from src.visualize.audit import compute_figure, failure_register_figure

    for fig, name, _ in (failure_register_figure(register), compute_figure(by_study)):
        assert name in {"failure_register", "compute_and_spending"}
        plt.close(fig)


def test_runtime_table_and_figure_from_export_folds():
    plan = fixture_plan(("experiment_1", "experiment_2"))
    plan = plan[plan.provider.isna() | plan.provider.isin(["jev", "strands"])]
    fits = study_audit.fit_seconds(fixture_folds(plan, drop_models=()))
    assert set(fits.family) >= {"no text", "decision", "frozen", "decision+frozen"}
    from src.visualize.audit import runtime_figure

    fig, name, _ = runtime_figure(fits)
    assert name == "fit_runtime"
    plt.close(fig)


def test_glance_table_keeps_task_families_apart_and_pools_only_the_share():
    from src.evaluation.synthesis_overview import glance_table
    from src.visualize.overview import study_at_a_glance

    table = glance_table(fixture_gains(), n_boot=200)
    assert set(table.scope) == {"pooled", "classification", "regression"}
    assert table[table.scope.eq("pooled")].datasets.eq(18).all()
    assert table[table.scope.ne("pooled")].datasets.eq(9).all()
    assert table.predictors.eq(7).all()
    fig, name, _ = study_at_a_glance(table)
    assert name == "study_at_a_glance"
    plt.close(fig)


def test_confirmatory_forest_and_qualification_matrix_draw():
    from src.visualize.overview import confirmatory_forest
    from src.visualize.study import qualification_figure

    results = pd.DataFrame({"contrast": [f"contrast {i}" for i in range(8)],
                            "estimate": np.linspace(-0.02, 0.05, 8), "low": np.linspace(-0.04, 0.02, 8),
                            "high": np.linspace(0.0, 0.08, 8), "p_holm": np.linspace(0.001, 0.9, 8),
                            "p_left": 0.1, "p_rope": 0.3, "p_right": 0.6})
    fig, name, _ = confirmatory_forest(results)
    assert name == "confirmatory_family"
    plt.close(fig)
    predictors, profiles = fixture_profiles()
    assert "Kumo override" in set(profiles.family)
    fig, name, caption = qualification_figure(predictors, profiles)
    assert name == "predictor_qualification" and "500" in caption
    assert {"column refit", "training size", "Kumo override"} <= {tick.get_text() for tick in fig.axes[0].get_xticklabels()}
    plt.close(fig)


def test_feature_design_wording_is_current():
    from src.evaluation.campaign import benchmark_plan
    from src.visualize.build_review import regression_scale_figure
    from src.visualize.feature_design import input_design_figure, predictor_width_figure

    fig, _, caption = input_design_figure()
    plt.close(fig)
    assert "seven fixed predictors" in caption and "six" not in caption
    summary = pd.DataFrame([{"dataset_id": d, "dataset": d, "task_type": t} for d, t in roster()[1].items()])
    fig, _, caption = predictor_width_figure(pd.DataFrame(benchmark_plan(cohort="open_weight")), summary)
    plt.close(fig)
    assert "16 selected decision models" in caption
    assert "excluded for Kumo" not in caption and "Experiment" not in caption
    fig, _, caption = regression_scale_figure({str(i): f"Level {i}: text" for i in range(-4, 5)})
    plt.close(fig)
    assert "probability-weighted mean" not in caption
