"""Research contracts: test leakage boundaries and output information, not prose."""

import numpy as np
import pytest

from src.evaluation.qualification import profile_plan
from src.evaluation.research_questions import (
    adaptation_plan,
    adaptation_prediction_plan,
    align_permuted_probabilities,
    random_probe_requests,
    random_probe_rows,
    regression_width_table,
    stable_probe_sample,
    training_defined_bins,
    validate_adaptation_partition,
)
from src.utils import paths
from src.utils.config import load_yaml


def test_research_layout_uses_one_name_per_experiment():
    assert paths.config_path("B1") == paths.REPO_ROOT / "config" / "B" / "B1.yaml"
    assert paths.notebook_phase("B2/01_frozen_embeddings") == "B2"
    assert paths.notebook_phase("feature_creation/04_decision_feature_comparison") == "feature_creation"
    assert paths.phase_folder("B1") == "B/B1"
    assert load_yaml("C2")["enabled"] is False


def test_removed_readout_is_absent_from_benchmark_and_qualification():
    from src.evaluation.campaign import benchmark_plan
    from src.evaluation.qualification import PHASE_CONFIGS
    plan = benchmark_plan()
    assert len(plan) == 81240 + 2142  # Group B grid (B1-B5) + Group C (C1, C2), 20-11-2026
    assert {a["phase"] for a in plan} == set(PHASE_CONFIGS) == {
        "B1", "B2", "B3", "B6", "B4", "B5",
        "C1", "C2"}
    assert all(a["phase"] != "regression_readout" for a in profile_plan())
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


def test_random_source_rows_are_uniform_reproducible_and_not_deduplicated_by_text():
    rows = [f"csv:{i}" for i in range(1000)]
    sample = random_probe_rows(rows, dataset_id="test")
    assert len(sample) == len(set(sample)) == 500
    assert sample == random_probe_rows(rows[::-1], dataset_id="test")
    assert sample != rows[:500]
    assert sample != random_probe_rows(rows, dataset_id="other")
    assert set(random_probe_rows(rows[:20], dataset_id="small")) == set(rows[:20])
    with pytest.raises(ValueError, match="unique"):
        random_probe_rows(rows + [rows[0]], dataset_id="test")


def test_primary_regression_probability_view_cannot_reuse_historical_mean_identity():
    from src.evaluation.campaign import benchmark_plan
    semantics = load_yaml("dataset_semantics")["datasets"]
    arms = [a for a in benchmark_plan() if a["phase"] == "B1"
            and a.get("provider") and semantics[a["dataset_id"]]["task_type"] == "regression"]
    assert arms and all(a["view"] == "probabilities" for a in arms)
    assert max(a["predictor_columns"] for a in arms) == 141


def test_cohort_cutpoints_ignore_test_distribution():
    _, a = training_defined_bins([0, 1, 2, 3], [1, 2])
    bins, b = training_defined_bins([0, 1, 2, 3], [-1e9, 1e9, np.nan])
    np.testing.assert_array_equal(a, b)
    assert bins[-1] == -1


def test_adaptation_shares_fields_and_predictors_but_never_outer_folds():
    # Group C2 (20-11-2026): dataset x outer fold x cross-fit part x trained cell.
    plan = adaptation_plan()
    assert len(plan) == 18 * 5 * 3 * 3
    assert len({(a['dataset_id'], a['fold'], a['part'], a['cell']) for a in plan}) == 810
    predictions = adaptation_prediction_plan()
    # Shares the benchmark's predictor roster: seven since LightGBM joined (20-11-2026).
    assert len(predictions) == 18 * 3 * 7
    assert len({a['model'] for a in predictions}) == 7
    assert all(a['binary_threshold'] == 0.5 for a in predictions)
    assert not any(a["execution_enabled"] for a in plan)
    assert not any(a["execution_enabled"] for a in predictions)
    assert validate_adaptation_partition([1, 2], [3], [4], [5])


def test_request_sampling_deduplicates_calls_and_separates_study_streams():
    keys = [f'request:{i}' for i in range(1000)]
    a = random_probe_requests(keys, dataset_id='x')
    assert len(a) == len(set(a)) == 500
    assert a == random_probe_requests(keys[::-1] * 2, dataset_id='x')
    assert a != random_probe_requests(keys, dataset_id='x', study='field_order')
    applicable = keys[:200]
    assert set(random_probe_requests(applicable, dataset_id='x')) == set(applicable)
    with pytest.raises(ValueError):
        random_probe_requests([''], dataset_id='x')


@pytest.mark.parametrize("left,right", [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
def test_adaptation_rejects_overlap_between_every_pair_of_roles(left, right):
    groups = [[1], [2], [3], [4]]
    groups[right].append(groups[left][0])
    with pytest.raises(ValueError, match="disjoint"):
        validate_adaptation_partition(*groups)


@pytest.mark.parametrize("role", range(4))
@pytest.mark.parametrize("invalid", [[], [9, 9]])
def test_adaptation_rejects_empty_or_duplicate_roles(role, invalid):
    groups = [[1], [2], [3], [4]]
    groups[role] = invalid
    with pytest.raises(ValueError, match="nonempty and unique"):
        validate_adaptation_partition(*groups)


def test_approved_additions_are_pinned_but_not_silently_executable():
    import re

    additions = load_yaml("feature_creation/approved_additions")
    assert additions["execution_enabled"] is False
    assert set(additions["models"]) == {"mammoth", "drex_max", "knowline", "sieve"}
    active = set(load_yaml("B1")["candidate_extractors"])
    assert set(additions["models"]) <= active
    for model in additions["models"].values():
        assert re.fullmatch(r"[0-9a-f]{40}", model["revision"])
        assert re.fullmatch(r"[0-9a-f]{40}", model["source_revision"])
        assert model["status"] == "native_adapter_implemented_requires_real_0A"
        assert "token_audit" in model["required"]
        assert "sealed_B200_runtime" in model["required"]


def test_summary_discovery_follows_research_order():
    from src.utils.run_notebooks import discover

    # Layout v4 (26-11-2026): general notebooks, then the letter groups A (RQ1), B (RQ2) and C,
    # each opened by its gate notebook (A0, B0, C0).
    names = discover()
    order = {"general": 0, "A": 1, "B": 2, "C": 3}
    groups = [n.split("/")[0] for n in names]
    assert [order[g] for g in groups] == sorted(order[g] for g in groups)
    assert paths.research_group(paths.notebook_phase("B/B5")) == "B"
    for group in "ABC":
        assert {paths.research_group(paths.notebook_phase(n)) for n in names if n.startswith(group + "/")} == {group}
        assert names[[n.split("/")[0] for n in names].index(group)] == f"{group}/{group}0"
    for group in "ABC":  # Each group ends with its synthesis notebook.
        assert [n for n in names if n.startswith(f"{group}/")][-1] == f"{group}/{group}_synthesis"


def test_replay_control_precedes_interventions():
    from src.utils.config import load_yaml

    studies = load_yaml("A2")["studies"]
    assert list(studies) == ["B1_replay", "B2_wording", "B3_field_order", "B4_option_order"]
    assert studies["B1_replay"]["eligible"] == "union_of_B2_B3_B4_samples"
    assert studies["B1_replay"]["total_fresh_executions"] == 3


def test_tied_cutpoints_keep_quartile_meaning_of_each_bin():
    bins, edges = training_defined_bins([1, 1, 1, 1, 1, 5, 9], [0, 1, 4, 10])
    assert len(edges) == 3  # Tied cut-offs are kept, never renumbered.
    # bin_k = number of training quartile cut-offs at or below the value.
    assert bins.tolist() == [0, 2, 3, 3]
