"""Complete-fold analysis, no outcome selection, and future figure layout fixtures."""

import ast
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.evaluation import notebook_study as study
from src.evaluation.endpoints import HEADLINES
from src.utils import paths
from src.utils.config import load_yaml
from src.visualize import style
from src.visualize.study import burden_figures, coverage_figures, native_figures


def fixture_folds():
    rows = []
    for task, metric in HEADLINES.items():
        dataset = next(d for d, s in load_yaml("dataset_semantics")["datasets"].items() if s["task_type"] == task)
        for model in ["catboost", "tabstar", "contexttab", "ag_mm"]:
            phase = "experiment_1" if model == "catboost" else "experiment_3"
            for fold in range(5):
                base = dict(arm_id=f"{dataset}/{model}", phase=phase, dataset_id=dataset, task_type=task,
                            model_id=model, checkpoint_id=model, feature_set="non_text" if phase == "experiment_1" else "original_features",
                            provider=None, encoder=None, pca_scope=None, pca_components=None,
                            split_id=dataset, outer_fold=fold, metric=metric, value=0.65+fold/100,
                            comparison_id=model, inference_status="prespecified", feature_artifact_ids="[]")
                rows.append(base)
    return pd.DataFrame(rows)


def test_headline_configuration_matches_user_and_multabench():
    protocol = load_yaml("experiment_1/protocol")["evaluation"]
    assert protocol["binary"]["primary_metric"] == HEADLINES["binary_classification"]
    assert protocol["multiclass"]["primary_metric"] == HEADLINES["multiclass_classification"]
    assert protocol["regression"]["primary_metric"] == HEADLINES["regression"]


def test_r2_differences_are_not_rescaled_or_dropped_for_negative_references():
    from src.visualize.paired import _dataset_effects
    frame = pd.DataFrame([dict(dataset_id=d, outer_fold=k, metric="r2", difference=.1,
                               reference_value=r) for d, r in [("negative", -.5), ("zero", 0)]
                          for k in range(5)])
    result = _dataset_effects(frame, ["dataset_id"], "regression", 1)
    assert len(result) == 2 and np.allclose(result.effect, .1)


def test_coverage_requires_five_defined_folds_and_rejects_duplicate_campaign():
    frame = fixture_folds()
    table = study.planned_coverage(frame, "experiment_1")
    assert table.planned.eq(34).all() and table.completed.sum() == 3
    broken = frame.drop(frame[frame.model_id.eq("catboost") & frame.outer_fold.eq(4)].index)
    assert study.planned_coverage(broken, "experiment_1").completed.sum() == 0
    bad = frame[frame.model_id.eq("catboost")].copy()
    bad["arm_id"] += "other"
    with pytest.raises(ValueError, match="Multiple campaigns"):
        study.planned_coverage(pd.concat([frame, bad]), "experiment_1")


def test_headline_summary_never_selects_best_recipe():
    table = study.headline_summary(fixture_folds())
    assert len(table) == 12 and table.defined_folds.eq(5).all()
    assert set(table.metric) == set(HEADLINES.values())
    with pytest.raises(ValueError, match="Duplicate headline"):
        study.headline_summary(pd.concat([fixture_folds(), fixture_folds().iloc[:1]]))


def test_cross_system_pairing_is_explicit_and_split_bound():
    frame = fixture_folds()
    paired = study.cross_system_contrasts(frame)
    assert len(paired) == 45 and paired.reference_model.nunique() == 3
    assert np.allclose(paired.difference, 0)
    frame.loc[frame.phase.eq("experiment_3"), "split_id"] = "other"
    assert study.cross_system_contrasts(frame).empty


def test_aligned_control_compares_same_artifacts_not_just_same_provider():
    data = fixture_folds().query("model_id == 'catboost'").copy()
    data["feature_set"] = "non_text_plus_both"
    data["provider"] = "jev"
    data["feature_artifact_ids"] = '["exact-artifact"]'
    control = data.copy()
    control["phase"] = "experiment_5"
    control["arm_id"] += "control"
    control["feature_set"] += "__permuted__main"
    control["value"] -= .1
    paired = study.aligned_control_contrasts(pd.concat([data, control]))
    assert len(paired) == 15 and np.allclose(paired.difference, -.1)
    assert study.contrast_summary(paired).datasets.eq(1).all()
    control["feature_artifact_ids"] = '["different"]'
    assert study.aligned_control_contrasts(pd.concat([data, control])).empty


def test_future_plots_draw_on_final_print_canvas_without_winner_selection():
    style.apply()
    summary = study.headline_summary(fixture_folds())
    figures = list(native_figures(summary)) + list(coverage_figures(study.planned_coverage(fixture_folds(), "experiment_1"), "Fixture"))
    for fig, _, _ in figures:
        fig.canvas.draw()
        assert fig.get_size_inches()[0] == style.WIDTH_FULL
        assert fig.get_size_inches()[1] <= style.MAX_HEIGHT
        plt.close(fig)


def test_no_empty_coverage_figures_before_results_exist():
    table = study.planned_coverage(pd.DataFrame(), "experiment_1")
    assert list(coverage_figures(table, "Pending")) == []


def test_every_experiment_has_thin_notebook_and_complete_printed_report():
    for folder in [f"{i+3}. experiment_{i}" for i in range(6)] + ["9. synthesis"]:
        notebooks = list((paths.REPO_ROOT / "notebooks" / folder).glob("*.ipynb"))
        assert notebooks
        for filename in notebooks:
            body = json.loads(filename.read_text(encoding="utf-8"))
            code = ["".join(c["source"]) for c in body["cells"] if c["cell_type"] == "code"]
            assert "finish_report" in code[-1]
            assert not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                           for block in code for n in ast.walk(ast.parse(block)))


def test_synthesis_uses_general_notebook_output_without_fake_experiment():
    assert paths.notebook_phase("9. synthesis/01_representation_comparison") == "synthesis"
    assert paths.phase_folder("synthesis") == "9. synthesis"


def test_burden_table_preserves_null_recipe_fields_and_complete_measurements():
    base = fixture_folds().query("model_id == 'catboost'")
    samples = []
    for metric, value in [("seconds.fit", 10), ("seconds.predict", 2), ("columns.representation", 4)]:
        samples.append(base.assign(metric=metric, value=value))
    table = study.burden_table(pd.concat(samples), ["experiment_1"])
    assert len(table) == 3 and table["seconds.fit"].eq(10).all()
    for fig, _, _ in burden_figures(table):
        fig.canvas.draw()
        plt.close(fig)
