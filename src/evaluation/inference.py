"""Dataset-level paired summaries, with optional explicitly preregistered inference.

CV folds are not independent experimental units. Never turn their SD into an SE.
Raw metrics are summarized separately by task, predictor, recipe and reference.
"""

import numpy as np
import pandas as pd

GROUP = [
    "task_type",
    "model_id",
    "checkpoint_id",
    "feature_set",
    "comparison_id",
    "reference",
    "metric",
]
DIMENSIONLESS = {
    "r2",
    "explained_variance",
    "pearson_r",
    "spearman_rho",
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier",
    "brier_multiclass_sum",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "precision",
    "recall",
    "mcc",
    "cohen_kappa",
    "roc_auc_ovr_macro",
    "roc_auc_ovr_weighted",
    "roc_auc_ovo_macro",
    "roc_auc_ovo_weighted",
}


def exact_resolution(values):
    """Optimistic two-sided sign-pattern bound, not a power calculation or p-value.

    Wilcoxon drops zero differences here. Its auto method may use a different
    null distribution with ties/zeros, so this bound describes ideal exact
    enumeration only and must not be used to replace the actual test.
    """
    nonzero = int(np.count_nonzero(np.asarray(values, dtype=float)))
    return {
        "nonzero_datasets": nonzero,
        "ideal_exact_two_sided_p_floor_nonzero": min(1.0, 2.0 ** (1 - nonzero)),
    }


def dataset_differences(paired):
    if paired.empty:
        return pd.DataFrame(columns=GROUP + ["dataset_id", "mean_difference", "defined_folds"])
    keys = GROUP + ["dataset_id", "split_id"]
    if "inference_status" in paired:
        keys += ["inference_status"]
    if paired.duplicated(keys + ["outer_fold"]).any():
        raise ValueError("Duplicate fold comparison; choose one campaign")
    result = (
        paired.groupby(keys, dropna=False)
        .difference.agg(
            mean_difference="mean",
            descriptive_fold_sd="std",
            defined_folds="count",
        )
        .reset_index()
    )
    # Keep incomplete/undefined cases visible; they cannot enter inferential comparisons.
    result["complete"] = result.defined_folds.eq(5)
    return result


def across_datasets(datasets):
    if datasets.empty:
        return pd.DataFrame(
            columns=GROUP
            + ["datasets", "complete_datasets", "mean_difference", "median_difference"]
        )
    records = []
    for key, group in datasets.groupby(GROUP, dropna=False):
        if group.dataset_id.duplicated().any():
            raise ValueError("Multiple results for one dataset and contrast; select one campaign")
        values = group.loc[group.complete, "mean_difference"]
        comparable = key[-1].split(".")[-1] in DIMENSIONLESS
        records.append(
            dict(zip(GROUP, key))
            | {
                "datasets": len(group),
                "complete_datasets": len(values),
                "mean_difference": float(values.mean()) if len(values) and comparable else None,
                "median_difference": float(values.median()) if len(values) and comparable else None,
                "positive_deltas": int((values > 0).sum()),
                "zero_deltas": int((values == 0).sum()),
                "negative_deltas": int((values < 0).sum()),
                **exact_resolution(values),
                "minimum_two_sided_exact_sign_p_no_ties": 2.0 ** (1 - len(values))
                if len(values)
                else None,
                "power_note": "Ideal exact two-sided resolution before multiplicity correction: 3/6/9 nonzero datasets give .25/.03125/.00390625. This is not power or the auto-method p-value. Holm is step-down: compare ordered p-values with .05/(family_size-rank+1); later ranks have less strict thresholds. Keep fold SD descriptive and approve the family before outcomes.",
                "interpretation": "Descriptive; positive is better only for higher-is-better metrics. Raw magnitudes with dataset-specific units are not pooled.",
                "inference_status": ",".join(sorted(group.inference_status.unique()))
                if "inference_status" in group
                else "prespecified",
            }
        )
    return pd.DataFrame(records)


def confirmatory_tests(datasets, plan):
    """One complete family, declared before outcomes; Holm includes every planned contrast.

    Each contrast supplies GROUP identifiers, dataset IDs and direction (+1 or -1).
    Missing/undefined data blocks inference, rather than shrinking the testing family.
    """
    from scipy.stats import wilcoxon

    if (
        plan.get("approved_before_results") is not True
        or not plan.get("contrasts")
        or not plan.get("reviewed_by")
    ):
        raise ValueError("An explicit analysis plan approved before results is required")
    results, identities = [], set()
    for contrast in plan["contrasts"]:
        keys = contrast["match"]
        if set(keys) != set(GROUP) or contrast["direction"] not in {-1, 1}:
            raise ValueError("Specify one task, metric, predictor, recipe, reference and direction")
        if keys["metric"].split(".")[-1] not in DIMENSIONLESS:
            raise ValueError("Use a comparable dimensionless metric, not raw target-unit errors")
        identity = tuple(keys[k] for k in GROUP)
        if identity in identities:
            raise ValueError("Duplicate planned statistical contrast")
        identities.add(identity)
        chosen = datasets.loc[np.logical_and.reduce([datasets[k].eq(v) for k, v in keys.items()])]
        if "inference_status" in chosen and not chosen.inference_status.eq("prespecified").all():
            raise ValueError(
                "Post-selection exploratory combinations cannot enter confirmatory tests"
            )
        expected = contrast["datasets"]
        if (
            len(set(expected)) != len(expected)
            or len(expected) < 2
            or chosen.dataset_id.duplicated().any()
            or set(chosen.dataset_id) != set(expected)
            or not chosen.complete.all()
        ):
            raise ValueError("Every planned dataset requires exactly one complete paired result")
        differences = chosen.mean_difference.to_numpy() * contrast["direction"]
        if not np.isfinite(differences).all():
            raise ValueError("Nonfinite dataset difference")
        # Two-sided test avoids interpreting a harmful significant effect as an improvement.
        p = (
            1.0
            if np.all(differences == 0)
            else float(
                wilcoxon(
                    differences, alternative="two-sided", zero_method="wilcox", method="auto"
                ).pvalue
            )
        )
        results.append(
            {
                **keys,
                "datasets": len(chosen),
                "median_oriented_delta": float(np.median(differences)),
                "p_raw": p,
                **exact_resolution(differences),
                "wilcoxon_method": "auto",
                "zero_method": "wilcox",
                "holm_family_size": len(plan["contrasts"]),
            }
        )
    order = np.argsort([r["p_raw"] for r in results])
    adjusted = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, max(adjusted, results[index]["p_raw"] * (len(results) - rank)))
        results[index]["p_holm"] = adjusted
        results[index]["holm_rank"] = rank + 1
        results[index]["holm_step_threshold_at_0_05"] = 0.05 / (len(results) - rank)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Comparing several learners over the same datasets (owner's statistics collection).
#
# Unit: one score per dataset and learner (mean of the five outer folds), never folds.
# - Omnibus: Friedman (1937) with the Iman-Davenport (1980) F correction (Demsar 2006).
# - Mean ranks and the Nemenyi (1963) critical difference are DESCRIPTIVE (CD diagram):
#   mean-rank post-hoc tests depend on which other learners are in the pool
#   (Benavoli, Corani & Mangili 2016), so confirmatory pairwise claims use the paired
#   two-sided Wilcoxon (1945) signed-rank test on per-dataset differences instead.
# - Families of pairwise p-values: Holm (1979) step-down; Finner for comparisons with
#   one control and Shaffer's static procedure for all pairwise comparisons
#   (Garcia & Herrera 2008; Garcia et al. 2010) as reported sensitivity analyses.
# - Bayesian signed-rank test with a region of practical equivalence (Benavoli et al.
#   2014, 2017) reports P(worse), P(equivalent), P(better) next to every p-value.
# ---------------------------------------------------------------------------


def score_matrix(scores, *, higher_is_better):
    """Dataset x learner table from rows (dataset_id, learner, score); complete rows only."""
    if scores.duplicated(["dataset_id", "learner"]).any():
        raise ValueError("One score per dataset and learner is required")
    table = scores.pivot(index="dataset_id", columns="learner", values="score").astype(float)
    if table.isna().any().any():
        missing = table.columns[table.isna().any()].tolist()
        raise ValueError(f"Learners lack a score on some datasets: {missing}; inference stops")
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError("Nonfinite score")
    return table if higher_is_better else -table


def friedman(matrix):
    """Friedman chi-square and the Iman-Davenport F on a dataset x learner matrix.

    `matrix` is oriented so that larger is better; rank 1 is the best learner on a
    dataset and ties receive average ranks.
    """
    from scipy.stats import chi2, f

    values = np.asarray(matrix, dtype=float)
    n, k = values.shape
    if n < 2 or k < 2:
        raise ValueError("Friedman needs at least two datasets and two learners")
    ranks = pd.DataFrame(values).rank(axis=1, ascending=False, method="average").to_numpy()
    mean_ranks = ranks.mean(axis=0)
    chi_square = 12 * n / (k * (k + 1)) * (np.sum(mean_ranks**2) - k * (k + 1) ** 2 / 4)
    denominator = n * (k - 1) - chi_square
    iman_davenport = (n - 1) * chi_square / denominator if denominator > 0 else np.inf
    df1, df2 = k - 1, (k - 1) * (n - 1)
    return {
        "datasets": n,
        "learners": k,
        "mean_ranks": dict(zip(getattr(matrix, "columns", range(k)), mean_ranks.tolist())),
        "friedman_chi2": float(chi_square),
        "friedman_p": float(chi2.sf(chi_square, df1)),
        "iman_davenport_f": float(iman_davenport),
        "iman_davenport_df": [df1, df2],
        "iman_davenport_p": float(f.sf(iman_davenport, df1, df2)) if np.isfinite(iman_davenport) else 0.0,
    }


def nemenyi_critical_difference(learners, datasets, alpha=0.05):
    """Demsar's CD = q_alpha * sqrt(k(k+1)/(6N)), q from the studentized range / sqrt(2)."""
    from scipy.stats import studentized_range

    q = studentized_range.ppf(1 - alpha, learners, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(learners * (learners + 1) / (6 * datasets)))


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    adjusted, running = np.empty_like(p), 0.0
    for rank, index in enumerate(np.argsort(p, kind="stable")):
        running = min(1.0, max(running, p[index] * (len(p) - rank)))
        adjusted[index] = running
    return adjusted


def finner(p_values):
    """Finner step-down: APV_i = max_{j<=i} 1 - (1 - p_(j))^(m/j)."""
    p = np.asarray(p_values, dtype=float)
    adjusted, running, m = np.empty_like(p), 0.0, len(p)
    for rank, index in enumerate(np.argsort(p, kind="stable"), start=1):
        running = min(1.0, max(running, 1 - (1 - p[index]) ** (m / rank)))
        adjusted[index] = running
    return adjusted


def _shaffer_counts(k):
    """Shaffer's S(k): attainable numbers of simultaneously true pairwise hypotheses."""
    from math import comb

    sets = {0: {0}, 1: {0}}
    for size in range(2, k + 1):
        sets[size] = {comb(j, 2) + x for j in range(1, size + 1) for x in sets[size - j]}
    return sorted(sets[k])


def shaffer(p_values, learners):
    """Shaffer's static procedure for all m = k(k-1)/2 pairwise comparisons."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m != learners * (learners - 1) // 2:
        raise ValueError("Shaffer requires every pairwise comparison among the learners")
    counts = _shaffer_counts(learners)
    adjusted, running = np.empty_like(p), 0.0
    for rank, index in enumerate(np.argsort(p, kind="stable"), start=1):
        t = max(c for c in counts if c <= m - rank + 1)
        running = min(1.0, max(running, t * p[index]))
        adjusted[index] = running
    return adjusted


def wilcoxon_p(differences):
    from scipy.stats import wilcoxon

    differences = np.asarray(differences, dtype=float)
    if np.all(differences == 0):
        return 1.0
    return float(
        wilcoxon(differences, alternative="two-sided", zero_method="wilcox", method="auto").pvalue
    )


def bayesian_signed_rank(differences, rope, *, prior_strength=0.5, samples=50_000, seed=0):
    """Benavoli et al. (2014, 2017): Dirichlet-process signed-rank posterior.

    A pseudo-observation at zero carries the prior strength. Each posterior draw
    weights the Walsh averages (x_i + x_j)/2; the reported probabilities are the
    shares of draws in which left (< -rope), rope or right (> rope) has the most mass.
    Positive differences favour the first learner.
    """
    x = np.concatenate(([0.0], np.asarray(differences, dtype=float)))
    if not np.isfinite(x).all() or rope < 0:
        raise ValueError("Nonfinite difference or negative ROPE")
    weights = np.ones(len(x))
    weights[0] = prior_strength
    walsh = x[:, None] + x[None, :]
    right, left = (walsh > 2 * rope).astype(float), (walsh < -2 * rope).astype(float)
    draws = np.random.default_rng(seed).dirichlet(weights, samples)
    p_right = np.einsum("si,ij,sj->s", draws, right, draws)
    p_left = np.einsum("si,ij,sj->s", draws, left, draws)
    winner = np.argmax(np.stack([p_left, 1 - p_right - p_left, p_right]), axis=0)
    return {
        "p_worse": float(np.mean(winner == 0)),
        "p_equivalent": float(np.mean(winner == 1)),
        "p_better": float(np.mean(winner == 2)),
        "rope": float(rope),
        "prior_strength": prior_strength,
        "posterior_samples": samples,
    }


def compare_learners(scores, *, higher_is_better, control=None, rope, alpha=0.05):
    """Full multi-learner report on per-dataset scores (columns dataset_id, learner, score).

    Returns the omnibus test and one row per pairwise comparison (with `control`, only
    the comparisons against it). Wilcoxon + Holm is the confirmatory family; Finner or
    Shaffer, mean-rank z-tests and the Bayesian signed-rank test are reported alongside.
    """
    from itertools import combinations

    from scipy.stats import norm

    matrix = score_matrix(scores, higher_is_better=higher_is_better)
    n, k = matrix.shape
    omnibus = friedman(matrix) | {"nemenyi_cd": nemenyi_critical_difference(k, n, alpha), "alpha": alpha}
    ranks = omnibus["mean_ranks"]
    learners = list(matrix.columns)
    if control is not None and control not in learners:
        raise ValueError("The control learner has no scores")
    pairs = (
        [(learner, control) for learner in learners if learner != control]
        if control is not None
        else list(combinations(learners, 2))
    )
    se = np.sqrt(k * (k + 1) / (6 * n))
    rows = []
    for first, second in pairs:
        differences = (matrix[first] - matrix[second]).to_numpy()
        bayes = bayesian_signed_rank(differences, rope)
        rows.append({
            "learner": first,
            "versus": second,
            "datasets": n,
            "wins": int((differences > 0).sum()),
            "ties": int((differences == 0).sum()),
            "losses": int((differences < 0).sum()),
            "median_oriented_difference": float(np.median(differences)),
            "p_wilcoxon": wilcoxon_p(differences),
            "mean_rank_difference": float(ranks[first] - ranks[second]),
            "p_mean_rank_z": float(2 * norm.sf(abs(ranks[second] - ranks[first]) / se)),
            "bayes_p_worse": bayes["p_worse"],
            "bayes_p_equivalent": bayes["p_equivalent"],
            "bayes_p_better": bayes["p_better"],
        })
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table.p_wilcoxon)
    if control is not None:
        table["p_finner"] = finner(table.p_wilcoxon)
    else:
        table["p_shaffer"] = shaffer(table.p_wilcoxon, k)
    table["rope"] = rope
    return omnibus, table
