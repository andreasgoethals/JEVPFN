"""Scientific notebook summaries must not manufacture acceptance or duplicate costs."""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.evaluation import notebook_evidence as evidence


def test_fit_counts_use_dataset_tasks_not_provider_count(monkeypatch):
    monkeypatch.setattr(evidence, "load_yaml", lambda _: {"datasets": {
        "binary": {"task_type": "binary_classification"},
        "reg": {"task_type": "regression"}}})
    plan = [{"phase": "experiment_1", "dataset_id": dataset, "model": "catboost"}
            for dataset in ("binary", "reg", "binary")]
    row = evidence.study_grid(plan).iloc[0]
    assert row.configurations == 3 and row.outer_fits == 15
    assert row.binary_inner_fits == 30 and row.datasets == 2


def test_saved_accounting_is_read_only_and_provider_specific(tmp_path):
    path = tmp_path / "example.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE extraction_jobs (request_json, status, attempts, reserved_usd, spent_usd, input_tokens)")
        for provider, spent, attempts in [("jev", 0.02, 2), ("cloudflare", 100, 1)]:
            db.execute("INSERT INTO extraction_jobs VALUES (?,?,?,?,?,?)",
                       (json.dumps({"settings": {"provider": provider}}), "completed", attempts, 0, spent, 100))
    before = path.read_bytes()
    row = evidence.measured_jev_usage(tmp_path).iloc[0]
    assert row.stored_jobs == 1 and row.attempts == 2
    assert row.recorded_charges_usd == 0.02 and row.reported_input_tokens == 100
    assert before == path.read_bytes()


def test_prompt_analysis_realigns_options_and_preserves_full_distribution(tmp_path):
    path = tmp_path / "real.json"
    records = [{"probe": "real_REG_EXAMPLE__" + variant,
                "features": {**{f"p_{i}": float(i == 4) for i in range(9)}, "expected_direction": 0}}
               for variant in ("base", "paraphrase")]
    path.write_text(json.dumps({"status": "jev_dataset_robustness_requires_review", "attempts": records,
                                "created_utc": "2026-10-23T00:00:00Z"}), encoding="utf-8")
    table, cases, sources = evidence.prompt_evidence([path])
    assert table.total_variation_from_base.eq(0).all()
    assert cases["mean"].eq(0).all() and cases.variance.eq(0).all()
    assert np.array(cases.probabilities.tolist()).shape == (2, 9)
    assert sources[0]["sha256"] == evidence.file_sha256(path)
    report = json.loads(path.read_text())
    report["status"] = "failed"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="failed diagnostic"):
        evidence.prompt_evidence([path])


def test_no_local_receipt_is_unknown_not_provider_failure(isolated_output, monkeypatch):
    from src.cluster import token_audit

    monkeypatch.setattr(token_audit, "saved_exact_token_tables", lambda: ({"status": "not_counted"}, pd.DataFrame()))
    inventory = pd.DataFrame([{"provider": "jev", "representation": "joint", "dataset": "Example", "status": "not_created"}])
    table, _ = evidence.local_runtime_evidence(inventory)
    assert table.bounded_runtime.eq("unknown").all()
    assert table.configured.eq("observed").all()
    assert table.loc[table.model_id.eq("jev"), "complete_feature_datasets"].iloc[0] == "0/1 · 0/1"


@pytest.mark.notebook
def test_compact_table_retains_all_observations_in_report(monkeypatch, capsys, isolated_output):
    from src.utils import notebook_report as report
    from src.utils import run_notebooks
    from src.utils.notebook_display import display_table

    monkeypatch.setattr(run_notebooks, "discover", lambda: ("example",))
    report.begin_report()
    table = pd.DataFrame({"observation": range(100)},
                         index=pd.Index([f"case-{i}" for i in range(100)], name="case_identity"))
    display_table("Complete measurements", table, preview_rows=3)
    report.finish_report("example")
    printed = capsys.readouterr().out
    assert "99" in printed and "Complete measurements" in printed
    assert "case_identity" in printed and "case-99" in printed


@pytest.mark.parametrize("compact", [False, True])
def test_pca_view_excludes_inner_folds_and_duplicate_predictor_records(tmp_path, compact):
    path = tmp_path / "fold.json"
    diagnostic = {"requested_components": 32, "fitted_components": 32,
                  "training_nonempty_rows": 100, "training_explained_variance_fraction": 0.7}
    meta = {"prediction_role": "outer_test", "dataset_id": "example", "split_id": "split",
            "outer_fold": 0, "representation_recipe": {"embedding": {"encoder_id": "e5_small_v2", "pca_scope": "joint"}},
            "pca_diagnostics": [{"partition": "outer_train", "columns": [diagnostic]},
                                {"partition": "inner_train_0", "columns": [diagnostic | {"training_nonempty_rows": 50}]}]}
    if compact:
        from src.evaluation.store import RecordRef, connect

        path = tmp_path / "dataset.sqlite3"
        with connect(path) as db:
            db.execute("INSERT INTO records VALUES (?,?,?)",
                       ("fold", json.dumps({"metadata": meta}), b"not_read_by_metadata_analysis"))
        reference = str(RecordRef(path, "fold"))
    else:
        path.write_text(json.dumps({"metadata": meta}))
        reference = str(path)
    before = path.read_bytes()
    result = evidence.saved_pca_diagnostics(pd.DataFrame({"record_path": [reference, reference]}))
    assert len(result) == 1 and result.training_nonempty_rows.iloc[0] == 100
    assert before == path.read_bytes()


def test_prompt_panels_align_variants_even_when_reversal_is_inapplicable():
    import matplotlib.pyplot as plt

    from src.visualize.research import VARIANTS, prompt_sensitivity_figures

    rows = []
    for dataset in ["BIN_TEXT_FAKE_JOB_POSTING", "MUL_TEXT_DATA_SCIENTIST_SALARY", "REG_TEXT_BABIES_PRICES"]:
        for variant in VARIANTS:
            if variant == "option_order" and not dataset.startswith("MUL_"):
                continue
            rows.append({"provider": "jev", "source_sha256": "a" * 64, "dataset_id": dataset,
                         "variant": variant, "total_variation_from_base": 0.1, "same_top_option": True,
                         "created_utc": "2026-10-23", "expected_direction_change":
                         0.1 if dataset.startswith("REG_") else None})
    fig, _, _ = next(prompt_sensitivity_figures(pd.DataFrame(rows)))
    for ax in fig.axes[:3]:
        assert list(ax.get_yticks()) == [0, 1, 2, 3]
    assert any(t.get_text() == "n/a" for t in fig.axes[0].texts)
    assert not any(t.get_text() == "n/a" for t in fig.axes[1].texts)
    plt.close(fig)


def test_paired_plot_requires_five_defined_differences_per_dataset():
    import matplotlib.pyplot as plt

    from src.visualize.paired import paired_effect_figures

    rows = []
    for dataset in ["complete", "undefined"]:
        for fold in range(5):
            rows.append({"task_type": "binary_classification", "metric": "roc_auc",
                         "phase": "experiment_1", "feature_set": "non_text_plus_joint",
                         "provider": "jev", "model_id": "catboost", "reference": "non_text",
                         "dataset_id": dataset, "split_id": "split", "arm_id": dataset,
                         "outer_fold": fold,
                         "difference": 0.2 if dataset == "complete" else None})
    fig, _, caption = next(paired_effect_figures(pd.DataFrame(rows)))
    assert "· 1 dataset ·" in fig._suptitle.get_text()
    assert "five defined" in caption and "confidence interval" in caption
    plt.close(fig)


@pytest.mark.parametrize("task,metric,difference", [
    ("binary_classification", "roc_auc", 0.1),
    ("multiclass_classification", "roc_auc_ovr_macro", 0.1),
    ("regression", "r2", 0.1),
])
def test_embedding_plot_preserves_metric_direction_and_complete_paired_folds(task, metric, difference):
    import matplotlib.pyplot as plt

    from src.visualize.paired import embedding_effect_figures

    rows = []
    for scope, width in [("joint", w) for w in (8, 16, 32, 64, 128, 256)] + [("per_column", 32)]:
        for dataset in ["complete", "incomplete"]:
            for fold in range(5 if dataset == "complete" else 4):
                rows.append({"task_type": task, "metric": metric, "phase": "experiment_2",
                             "model_id": "catboost", "reference": "non_text", "dataset_id": dataset,
                             "split_id": "shared", "encoder": "e5_small_v2", "pca_scope": scope,
                             "pca_components": width, "arm_id": f"{dataset}_{scope}_{width}",
                             "outer_fold": fold, "reference_value": 1,
                             "difference": difference if dataset == "complete" else 100})
    fig, _, caption = next(embedding_effect_figures(pd.DataFrame(rows)))
    expected = 0.1
    assert np.allclose(fig.axes[0].lines[0].get_ydata(), [expected] * 6)
    assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()
    assert "not equal-budget" in caption and "missing/unsupported arms are not imputed" in caption
    plt.close(fig)


def test_embedding_curves_cannot_turn_changing_dataset_coverage_into_a_width_effect():
    import matplotlib.pyplot as plt

    from src.visualize.paired import embedding_effect_figures

    rows = []
    for dataset, widths, effect in [("matched", (8, 16, 32, 64, 128, 256), 0.1),
                                    ("partial_easy", (8,), 0.9)]:
        for width in widths:
            for fold in range(5):
                rows.append({"task_type": "binary_classification", "metric": "roc_auc",
                             "phase": "experiment_2", "model_id": "catboost",
                             "reference": "non_text", "dataset_id": dataset,
                             "split_id": "shared", "encoder": "e5_small_v2", "pca_scope": "joint",
                             "pca_components": width, "arm_id": f"{dataset}_{width}",
                             "outer_fold": fold, "difference": effect})
    fig, _, caption = next(embedding_effect_figures(pd.DataFrame(rows)))
    assert np.allclose(fig.axes[0].lines[0].get_ydata(), [0.1] * 6)
    assert "n=1" in fig.axes[0].lines[0].get_label()
    assert "fixed cohort" in caption
    plt.close(fig)


def test_regression_plot_effect_is_invariant_to_target_units_and_refuses_zero_reference():
    from src.visualize.paired import _dataset_effects

    rows = [{"dataset_id": name, "outer_fold": fold, "difference": difference, "metric": "rmse",
             "reference_value": reference}
            for name, difference, reference in [("small_units", -1, 10),
                                                 ("large_units", -1000, 10000),
                                                 ("zero_reference", -1, 0)]
            for fold in range(5)]
    effects = _dataset_effects(pd.DataFrame(rows), ["dataset_id"], "regression", -1)
    assert set(effects.dataset_id) == {"small_units", "large_units"}
    assert np.allclose(effects.effect, 10)


def test_empty_response_cache_remains_reportable(tmp_path):
    from src.visualize.research import pilot_output_figures

    table = evidence.pilot_decision_outputs(tmp_path)
    summary = table.groupby(["provider", "settings_sha256", "task_type"]).value.describe()
    assert summary.empty
    assert table.attrs["invalid_responses"] == 0
    assert list(pilot_output_figures(table)) == []


def test_effect_matrix_rejects_counting_a_dataset_twice():
    from src.visualize.paired import effect_matrix_figures

    rows = [{"phase": "experiment_1", "task_type": "binary_classification",
             "metric": "roc_auc", "model_id": "catboost", "reference": "non_text",
             "feature_set": "non_text_plus_joint", "provider": "jev",
             "dataset_id": "same_dataset", "split_id": split, "arm_id": split,
             "outer_fold": fold, "difference": 0.1}
            for split in ("old", "new") for fold in range(5)]
    with pytest.raises(ValueError, match="select one campaign"):
        list(effect_matrix_figures(pd.DataFrame(rows)))
