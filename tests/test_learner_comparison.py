"""Multi-learner comparisons over datasets follow Demsar (2006), Garcia et al. and Benavoli et al."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import friedmanchisquare

from src.evaluation import inference


def scores(matrix):
    rows = [
        {"dataset_id": f"d{i}", "learner": f"L{j}", "score": value}
        for i, row in enumerate(matrix)
        for j, value in enumerate(row)
    ]
    return pd.DataFrame(rows)


def test_friedman_matches_scipy_without_ties_and_iman_davenport_formula():
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(18, 5)) + np.arange(5) * 0.3
    result = inference.friedman(matrix)
    assert result["friedman_chi2"] == pytest.approx(friedmanchisquare(*matrix.T).statistic)
    n, k, chi = 18, 5, result["friedman_chi2"]
    assert result["iman_davenport_f"] == pytest.approx((n - 1) * chi / (n * (k - 1) - chi))
    assert result["iman_davenport_df"] == [4, 68]
    assert min(result["mean_ranks"].values()) == result["mean_ranks"][4]  # best gets rank ~1


def test_demsar_worked_example_ranks_statistics_and_critical_difference():
    # Demsar (2006), Section 3.2.2: N=14, k=4, mean ranks 3.143, 2.000, 2.893, 1.964.
    ranks = np.array([3.143, 2.000, 2.893, 1.964])
    n, k = 14, 4
    chi = 12 * n / (k * (k + 1)) * (np.sum(ranks**2) - k * (k + 1) ** 2 / 4)
    assert chi == pytest.approx(9.28, abs=0.02)
    assert (n - 1) * chi / (n * (k - 1) - chi) == pytest.approx(3.69, abs=0.01)
    assert inference.nemenyi_critical_difference(4, 14) == pytest.approx(1.25, abs=0.01)


def test_adjustments_holm_finner_and_shaffer():
    p = [0.01, 0.04, 0.03, 0.005]
    assert inference.holm(p).tolist() == pytest.approx([0.03, 0.06, 0.06, 0.02])
    finner = inference.finner(p)
    assert np.all(finner <= inference.holm(p) + 1e-12) and np.all(finner >= np.array(p))
    assert inference._shaffer_counts(4) == [0, 1, 2, 3, 6]
    six = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05]
    shaffer = inference.shaffer(six, 4)
    # First step multiplies by 6, the second by S(4) max <= 5, i.e. 3 (not Holm's 5).
    assert shaffer[0] == pytest.approx(0.006) and shaffer[1] == pytest.approx(0.03)
    assert np.all(shaffer <= inference.holm(six) + 1e-12)
    with pytest.raises(ValueError, match="every pairwise"):
        inference.shaffer(p[:3], 4)


def test_bayesian_signed_rank_separates_better_equivalent_and_worse():
    better = inference.bayesian_signed_rank(np.full(18, 0.05), rope=0.01, samples=5000)
    assert better["p_better"] > 0.95
    same = inference.bayesian_signed_rank(np.full(18, 0.001), rope=0.01, samples=5000)
    assert same["p_equivalent"] > 0.95
    worse = inference.bayesian_signed_rank(np.full(18, -0.05), rope=0.01, samples=5000)
    assert worse["p_worse"] > 0.95


def test_compare_learners_control_and_all_pairs_and_orientation():
    rng = np.random.default_rng(3)
    matrix = rng.normal(scale=0.01, size=(18, 3)) + np.array([0.70, 0.75, 0.70])
    omnibus, versus = inference.compare_learners(
        scores(matrix), higher_is_better=True, control="L0", rope=0.01)
    assert omnibus["learners"] == 3 and omnibus["datasets"] == 18
    assert set(versus.learner) == {"L1", "L2"} and "p_finner" in versus
    strong = versus.set_index("learner").loc["L1"]
    assert strong.wins == 18 and strong.p_holm < 0.01 and strong.bayes_p_better > 0.9
    _, pairs = inference.compare_learners(scores(matrix), higher_is_better=True, rope=0.01)
    assert len(pairs) == 3 and "p_shaffer" in pairs
    # Lower-is-better metrics (log loss) are oriented so that positive still means better.
    _, loss = inference.compare_learners(
        scores(-matrix), higher_is_better=False, control="L0", rope=0.01)
    assert loss.set_index("learner").loc["L1"].wins == 18


def test_incomplete_learner_scores_stop_inference():
    frame = scores(np.ones((4, 3))).iloc[:-1]
    with pytest.raises(ValueError, match="lack a score"):
        inference.compare_learners(frame, higher_is_better=True, rope=0.0)
