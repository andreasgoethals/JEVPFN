"""Research contracts: test leakage boundaries and output information, not prose."""

import numpy as np
import pytest

from src.evaluation.qualification import profile_plan
from src.evaluation.research_questions import (
    adaptation_plan,
    align_permuted_probabilities,
    regression_readout_plan,
    regression_width_table,
    stable_probe_sample,
    training_defined_bins,
    validate_adaptation_partition,
)
from src.utils import paths
from src.utils.config import load_yaml


def test_research_layout_keeps_legacy_config_and_output_identities():
    assert paths.config_path("experiment_1/main") == paths.config_path("rq2/4. experiment_1/main")
    assert paths.notebook_phase("rq2/01_frozen_embeddings") == "experiment_2"
    assert paths.notebook_phase("rq1/04_decision_feature_comparison") == "feature_creation"
    assert paths.phase_folder("experiment_1") == "4. experiment_1"
    assert load_yaml("rq3/adaptation")["enabled"] is False


def test_readout_uses_paired_regression_arms_and_real_width_profiles():
    plan = regression_readout_plan()
    n = len(load_yaml("experiment_1/main")["candidate_extractors"])
    assert len(plan) == 9 * 2 * n
    assert {a["view"] for a in plan} == {"probabilities"}
    assert {a["feature_set"] for a in plan} == {"non_text_plus_per_column"}
    profiles = [a for a in profile_plan() if a["phase"] == "regression_readout"]
    assert {a["model"] for a in profiles} == {"linear", "catboost"}
    assert max(a["predictor_columns"] for a in profiles) == 132
    widths = regression_width_table().set_index("dataset_id")
    row = widths.loc["REG_TEXT_ROTTEN_TOMATOES"]
    assert (row.per_column_probabilities, row.both_probabilities, row.kumo_conservative_max) == (
        132,
        141,
        143,
    )
    assert widths.kumo_conservative_max.max() < 500


def test_same_mean_does_not_mean_same_representation():
    distributions = np.zeros((2, 9))
    distributions[0, 4] = 1
    distributions[1, [0, 8]] = 0.5
    assert np.array_equal(distributions @ np.arange(-4, 5), [0, 0])
    assert not np.array_equal(*distributions)


def test_probe_selection_cannot_depend_on_duplicate_rows_or_input_order():
    keys = ["a", "b", "c", "d"]
    assert stable_probe_sample(keys, limit=2) == stable_probe_sample(keys[::-1] * 3, limit=2)
    with pytest.raises(ValueError):
        stable_probe_sample(keys, limit=0)


def test_permutation_alignment_restores_class_meanings():
    p = np.array([[0.1, 0.2, 0.7], [0.3, 0.6, 0.1]])
    order = [2, 0, 1]
    np.testing.assert_array_equal(align_permuted_probabilities(p[:, order], order), p)
    with pytest.raises(ValueError):
        align_permuted_probabilities(p, [0, 0, 2])
    with pytest.raises(ValueError):
        align_permuted_probabilities(p * 2, [0, 1, 2])


def test_cohort_cutpoints_ignore_test_distribution():
    _, a = training_defined_bins([0, 1, 2, 3], [1, 2])
    bins, b = training_defined_bins([0, 1, 2, 3], [-1e9, 1e9, np.nan])
    np.testing.assert_array_equal(a, b)
    assert bins[-1] == -1


def test_adaptation_is_direct_and_train_validation_test_are_disjoint():
    plan = adaptation_plan()
    assert len(plan) == 90 and sum(a["pilot"] for a in plan) == 4
    assert not any(a["execution_enabled"] for a in plan)
    assert validate_adaptation_partition([1, 2], [3], [4])
    for parts in [([1, 2], [2], [3]), ([1], [2], [1]), ([1, 1], [2], [3])]:
        with pytest.raises(ValueError):
            validate_adaptation_partition(*parts)


def test_approved_additions_are_pinned_but_not_silently_executable():
    import re

    additions = load_yaml("feature_creation/approved_additions")
    assert additions["execution_enabled"] is False
    assert set(additions["models"]) == {"mammoth", "drex_max"}
    active = set(load_yaml("experiment_1/main")["candidate_extractors"])
    assert set(additions["models"]) <= active
    for model in additions["models"].values():
        assert re.fullmatch(r"[0-9a-f]{40}", model["revision"])
        assert re.fullmatch(r"[0-9a-f]{40}", model["source_revision"])
        assert model["status"] == "native_adapter_implemented_requires_real_0A"
        assert "token_audit" in model["required"]
        assert "sealed_B200_runtime" in model["required"]


def test_summary_discovery_follows_research_order():
    from src.utils.run_notebooks import discover

    names = discover()
    groups = ["general" if n.startswith("general/") else paths.notebook_phase(n) for n in names]
    order = {"general": 0, "rq1": 1, "rq2": 2, "rq3": 3}
    assert [order[g] for g in groups] == sorted(order[g] for g in groups)
