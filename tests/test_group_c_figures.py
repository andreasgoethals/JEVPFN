"""Group C report views and figures on a deterministic fixture export (real export writer and
reader; synthetic values). Figures must render at final A4 width, never mix AUC and R² gains,
and yield nothing (a waiting state) for absent tables."""

import matplotlib

matplotlib.use("Agg")

import group_c_fixture as fx  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.evaluation import group_c_views as views  # noqa: E402
from src.visualize import group_c as figures  # noqa: E402
from src.visualize import style  # noqa: E402


@pytest.fixture(scope="module")
def exports(tmp_path_factory):
    folder = tmp_path_factory.mktemp("group_c_exports")
    c1, r1 = fx.read("C1", folder)
    c2, r2 = fx.read("C2", folder)
    assert r1["status"] == r2["status"] == "verified_export"
    return c1, c2


def test_overview_keeps_task_families_apart_and_adds_the_predictor_mean(exports):
    c1, _ = exports
    summary = views.overview(c1["dataset_gains"])
    assert set(summary.task_family) == {"classification", "regression"}
    mean = summary[summary.model_id.eq(views.PREDICTOR_MEAN)]
    assert set(mean.groupby("task_family").datasets.max()) == {9}
    one = mean[mean.representation.eq("hidden/joint/answer/L24") & mean.task_family.eq("regression")].iloc[0]
    gains = c1["dataset_gains"]
    expected = (gains[gains.representation.eq("hidden/joint/answer/L24") & gains.task_type.eq("regression")]
                .groupby("dataset_id").raw_gain.mean().mean())
    assert one.mean_raw_gain == pytest.approx(expected)
    assert one.raw_ci_low <= one.mean_raw_gain <= one.raw_ci_high


def test_predictor_average_requires_every_predictor():
    rows = [dict(dataset_id=d, task_type="regression", model_id=m, representation="adapted/joint/lora_head",
                 raw_gain=0.1 * i, headroom_share=0.2) for d in ("A", "B") for i, m in enumerate(("x", "y"))]
    gains = pd.DataFrame(rows[:-1])  # dataset B lacks predictor y
    out = views.predictor_average(gains, predictors=("x", "y"))
    assert views.predictor_average(gains).empty
    assert out.dataset_id.tolist() == ["A"] and out.raw_gain.iloc[0] == pytest.approx(0.05)
    assert views.predictor_average(pd.DataFrame()).empty and views.overview(pd.DataFrame()).empty


def test_audit_tables_for_head_contract_and_fit_diagnostics(exports):
    c1, c2 = exports
    contract = views.head_reproduction_contract(c1["extraction"])
    assert contract.passed.sum() == 17 and not contract.passed.iloc[-1]
    assert set(contract.tolerance) == {0.005}
    failing = c1["extraction"].assign(max_probability_deviation=0.006)
    assert not views.head_reproduction_contract(failing, tolerance=0.005).passed.any()
    fits = views.fit_diagnostics(c2["fits"])
    assert fits.set_index("cell").fits.to_dict() == {"head_only": 270, "lora_head": 270, "lora_only": 270}
    assert set(fits.planned_fits) == {270}
    assert views.fit_diagnostics(pd.DataFrame()).fits.eq(0).all()
    hours = views.training_gpu_hours(c2["fits"])
    assert hours["lora_head"] == pytest.approx(c2["fits"].query("cell == 'lora_head'").gpu_hours.sum())


def _generators(c1, c2):
    s1, s2 = views.overview(c1["dataset_gains"]), views.overview(c2["dataset_gains"])
    k1, k2 = views.overview(c1["contrasts"]), views.overview(c2["contrasts"])
    return {
        "c1_position_path": figures.position_path(s1, name="c1_position_path"),
        "c1_contrasts": figures.c1_contrasts(k1, name="c1_contrasts"),
        "c1_scatter": figures.c1_dataset_scatter(views.with_average(c1["dataset_gains"]), name="c1_scatter",
                                                 names=fx.NAMES),
        "c1_gains_joint": figures.c1_gains(s1, view="joint", name="c1_gains_joint"),
        "c2_factorial": figures.factorial_plot(s2, name="c2_factorial"),
        "c2_cells": figures.c2_cells(k2, name="c2_cells"),
        "c2_context": figures.c2_context(k2, name="c2_context"),
        "c2_datasets": figures.c2_datasets(views.with_average(c2["dataset_gains"]), name="c2_datasets",
                                           names=fx.NAMES),
        "c2_gains": figures.c2_gains(s2, name="c2_gains"),
        "strands_ladder": figures.strands_ladder(views.overview(views.combined_gains(c1, c2)),
                                                 name="strands_ladder",
                                                 gpu_hours=views.training_gpu_hours(c2["fits"])),
    }


def test_group_c_figures_render_at_final_width_with_captions(exports):
    c1, c2 = exports
    for key, generator in _generators(c1, c2).items():
        produced = list(generator)
        assert len(produced) == 1, key
        fig, name, caption = produced[0]
        width, height = fig.get_size_inches()
        assert width == pytest.approx(style.WIDTH_FULL) and height <= style.MAX_HEIGHT + 1e-9, key
        assert name and len(caption) > 150, key
        if key not in {"c1_scatter", "c2_datasets", "c1_position_path"}:
            assert "95%" in caption, key
        fig.canvas.draw()
        plt.close(fig)


def test_contrast_forest_rows_and_context_labels(exports):
    _, c2 = exports
    summary = views.overview(c2["contrasts"])
    fig, _, _ = next(figures.c2_context(summary, name="x"))
    ticks = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert ticks[0] == "Zero-shot" and ticks[1] == "minus TAR E5-small, per column PCA32"
    assert len(ticks) == 4 * 6  # four cells x (header + five references)
    plt.close(fig)
    assert figures.label("lora_head_vs_probe/probe/e5_large_v2/joint") == "LoRA + head vs classifier E5-large"
    assert figures.label("question_end_vs_state_end/joint/12") == "end of question vs end of text (joint, L12)"


@pytest.mark.parametrize("empty", [None, pd.DataFrame()])
def test_absent_tables_yield_no_figure(empty):
    for generator in (figures.position_path(empty, name="a"), figures.c1_contrasts(empty, name="b"),
                      figures.c1_dataset_scatter(empty, name="c"), figures.factorial_plot(empty, name="d"),
                      figures.c2_datasets(empty, name="e"), figures.c2_context(empty, name="f"),
                      figures.strands_ladder(empty, name="g"), figures.c2_gains(empty, name="h")):
        assert list(generator) == []


def test_ladder_annotates_missing_gpu_hours_explicitly(exports):
    c1, c2 = exports
    fig, _, _ = next(figures.strands_ladder(views.overview(views.combined_gains(c1, c2)), name="x"))
    texts = [t.get_text() for t in fig.axes[1].texts]
    assert texts.count("not in export") == 3 and "no training" in texts
    plt.close(fig)
    assert np.isfinite(views.coverage_summary(c1["coverage"]).share_complete).all()


def test_strands_ladder_on_the_normalized_score(exports):
    c1, c2 = exports
    per, summary = views.normalized_ladder(c1, c2, n_boot=50)
    rungs = {"baseline", *views.LADDER}
    assert set(per.representation) == rungs and per.normalized_score.between(0, 1).all()
    assert summary.configurations.max() == len(rungs) * 7
    produced = list(figures.strands_normalized_ladder(summary, name="n"))
    assert len(produced) == 1
    fig, name, caption = produced[0]
    fig.canvas.draw()
    assert fig.get_figwidth() == pytest.approx(style.WIDTH_FULL) and fig.get_figheight() <= style.MAX_HEIGHT
    assert name == "n" and "normalised" in caption
    plt.close(fig)
    assert list(figures.strands_normalized_ladder(pd.DataFrame(), name="x")) == []
    assert all(t.empty for t in views.normalized_ladder({}, {}))
