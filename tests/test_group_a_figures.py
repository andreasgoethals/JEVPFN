"""Group A summaries and figures on deterministic fixture exports (current export schemas)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from group_a_fixtures import a1_tables, a2_tables, a3_tables, design

from src.evaluation import group_a, model_properties
from src.visualize import style

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks" / "A"


@pytest.fixture(scope="module")
def properties():
    return model_properties.profile_table()


@pytest.fixture(scope="module")
def a1_values():
    a1 = a1_tables()
    return group_a.a1_dataset_values(a1["output_shape"], a1["loo_consensus"], a1["quality"])


def _check(figures, names):
    assert [n for _, n, _ in figures] == names
    for fig, _, caption in figures:
        fig.canvas.draw()
        width, height = fig.get_size_inches()
        assert width == style.WIDTH_FULL and height <= style.MAX_HEIGHT + 1e-9
        assert caption and "significant" not in caption.lower()
        plt.close(fig)


def test_bootstrap_summary_is_equal_dataset_and_reproducible():
    frame = pd.DataFrame({"k": ["a"] * 4 + ["b"], "v": [0.0, 1.0, 2.0, 3.0, 5.0]})
    out = group_a.summarize(frame, ["k"], "v").set_index("k")
    assert out.loc["a", "mean"] == 1.5 and out.loc["a", "median"] == 1.5 and out.loc["a", "datasets"] == 4
    assert 0 <= out.loc["a", "ci_low"] < 1.5 < out.loc["a", "ci_high"] <= 3
    assert np.isnan(out.loc["b", "ci_low"])  # One dataset: no interval.
    assert group_a.summarize(frame, ["k"], "v").equals(out.reset_index())


def test_a1_values_keep_task_families_apart(a1_values):
    datasets, providers = design()
    counts = a1_values.groupby("metric").dataset_id.nunique()
    assert counts["mean_top_probability"] == counts["consensus_agreement"] == 18
    assert counts["classification_auc"] == 9 and counts["regression_spearman"] == 9
    auc = a1_values[a1_values.metric.eq("classification_auc")]
    assert set(auc.task) <= set(group_a.CLASSIFICATION)
    assert a1_values.provider.nunique() == len(providers) == 17
    # Per-column fields are averaged within a dataset: one value per dataset, model and metric.
    assert not a1_values.duplicated(["dataset_id", "provider", "metric"]).any()


def test_model_order_groups_families_then_size(properties):
    order = group_a.model_order(properties.provider, properties)
    families = [group_a.family_of(p, properties) for p in order]
    known = list(style.ARCHITECTURE_MARKERS)
    ranks = [known.index(f) if f in known else len(known) for f in families]
    assert ranks == sorted(ranks) and len(order) == 17
    assert group_a.family_of("liquid_lfm3", properties) == "not stated"


def test_a2_excess_and_skill_follow_the_export_schema():
    a2 = a2_tables()
    per = group_a.excess_by_dataset(a2["mode_effects"])
    assert set(per.factor) == {"wording", "field_order", "option_order"}
    assert set(per.loc[per.factor.eq("field_order"), "mode"]) == {"joint"}
    summary = group_a.excess_summary(a2["mode_effects"])
    counts = summary.groupby("factor").datasets.max()
    assert counts["wording"] == 18 and counts["option_order"] == 6 and counts["field_order"] == 16
    skill = group_a.averaging_skill_summary(a2["averaging_quality"])
    assert skill.groupby("factor").datasets.max()["wording"] == 18  # Log loss and RPS pooled as skill.
    assert group_a.a2_margin_tests({"status": "awaiting_owner_approval"}) is None


def test_a2_margin_tests_read_an_approved_framework_record():
    table = pd.DataFrame({"learner": ["kev"], "versus": ["reference"], "p_holm": [0.01], "bayes_p_worse": [0.1],
                          "bayes_p_equivalent": [0.2], "bayes_p_better": [0.7]})
    record = {"status": "compared", "groups": [
        {"group": "wording/average_vs_original", "status": "compared", "table": table},
        {"group": "wording/average_vs_mean_single", "status": "compared", "table": table},
        {"group": "option_order/average_vs_original", "status": "stopped"}]}
    tests = group_a.a2_margin_tests(record)
    assert tests.to_dict("records") == [{"provider": "kev", "factor": "wording", "p_holm": 0.01, "bayes_p_worse": 0.1,
                                         "bayes_p_equivalent": 0.2, "bayes_p_better": 0.7}]


def test_geometry_summaries_leave_undefined_fields_out():
    comparisons = pd.DataFrame([
        {"dataset": "D", "field": "f1", "encoder": "e", "provider": "kev", "spearman": 0.4, "null_mean": 0.0,
         "neighbour_overlap": 0.3, "chance_overlap": 0.1, "status": "measured"},
        {"dataset": "D", "field": "f2", "encoder": "e", "provider": "kev", "spearman": None, "null_mean": None,
         "neighbour_overlap": None, "chance_overlap": None, "status": "undefined_insufficient_or_constant"},
        {"dataset": "E", "field": "f1", "encoder": "e", "provider": "kev", "spearman": 0.2, "null_mean": 0.02,
         "neighbour_overlap": 0.2, "chance_overlap": 0.1, "status": "measured"}])
    per = group_a.geometry_by_dataset(comparisons).set_index("dataset")
    assert per.loc["D", "spearman"] == 0.4 and per.loc["D", "undefined_fields"] == 1
    summary = group_a.geometry_summary(comparisons).iloc[0]
    assert summary["mean"] == pytest.approx(0.3) and summary.reference == pytest.approx(0.01)


def test_a1_figures_render_from_fixture_exports(a1_values, properties):
    from src.visualize import outputs, zero_shot

    style.apply()
    a1 = a1_tables()
    figures = [*outputs.model_profile_strip(a1_values, properties), *outputs.pairwise_figures(a1["pairs"], "joint"),
               *zero_shot.reliability_figure(a1["reliability"], highlight=("strands",))]
    _check(figures, ["model_profile_strip", "pairwise_agreement_binary_classification_joint",
                     "pairwise_agreement_multiclass_classification_joint", "pairwise_agreement_regression_joint",
                     "standalone_reliability"])
    with pytest.raises(ValueError, match="At most"):
        list(zero_shot.reliability_figure(a1["reliability"], highlight=("jev", "kev", "sii", "h2o")))
    assert list(outputs.model_profile_strip(a1_values.iloc[:0], properties)) == []


def test_clustered_order_puts_similar_models_together():
    from src.visualize.outputs import clustered_order

    means = pd.DataFrame([("a", "b", 0.95), ("a", "c", 0.2), ("b", "c", 0.25), ("a", "d", 0.3), ("b", "d", 0.3),
                          ("c", "d", 0.9)], columns=["model_a", "model_b", "top_agreement"])
    order = clustered_order(means, ["a", "c", "b", "d"])
    assert abs(order.index("a") - order.index("b")) == 1 and abs(order.index("c") - order.index("d")) == 1


def test_a2_figures_render_with_and_without_the_gated_margin(properties):
    from src.visualize import stability

    style.apply()
    a2 = a2_tables()
    order = group_a.model_order(properties.provider, properties)
    skill = group_a.averaging_skill_summary(a2["averaging_quality"])
    by_dataset = group_a.averaging_skill_by_dataset(a2["averaging_quality"])
    tests = skill[["provider", "factor"]].assign(p_holm=0.04, bayes_p_worse=0.1, bayes_p_equivalent=0.3,
                                                 bayes_p_better=0.6)
    plain = list(stability.averaging_skill_forest(skill, by_dataset, order=order))
    gated = list(stability.averaging_skill_forest(skill, by_dataset, order=order, tests=tests))
    assert len(gated[0][0].axes) == 2 * len(plain[0][0].axes)
    assert "Holm" in gated[0][2] and "Holm" not in plain[0][2]
    _check([*stability.excess_change_figure(group_a.excess_summary(a2["mode_effects"]), order=order), *plain, *gated],
           ["excess_change", "averaging_skill", "averaging_skill"])


def test_a3_and_synthesis_figures_render(a1_values, properties):
    from src.visualize import geometry, model_map

    style.apply()
    a3 = a3_tables()
    _check(list(geometry.geometry_figures(a3["comparisons"], a3["encoder_references"])),
           ["geometry_distance_correlation", "geometry_neighbour_overlap"])
    excess = group_a.excess_summary(a2_tables()["mode_effects"])
    table = group_a.model_map_table(a1_values, excess, properties)
    assert len(table) == 17 and table.classification_auc.notna().all()
    _check(list(model_map.decision_model_map(table)), ["decision_model_map"])
    waiting = group_a.model_map_table(a1_values.iloc[:0], excess.iloc[:0], properties)
    assert list(model_map.decision_model_map(waiting)) == []


def test_group_a_notebooks_are_thin_and_state_the_a1_regression_decision():
    import json

    for path in NOTEBOOKS.glob("*.ipynb"):
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
        assert "\ndef " not in "\n" + code and "\nclass " not in "\n" + code, path.name
        assert "finish_report" in "".join(cells[-1]["source"]), path.name
    exp01 = (NOTEBOOKS / "exp01_outputs.ipynb").read_text(encoding="utf-8")
    assert "log loss is undefined" in exp01 and "N = 9" in exp01
    exp02 = (NOTEBOOKS / "exp02_robustness.ipynb").read_text(encoding="utf-8")
    assert "robustness_matrix_figure" not in exp02
