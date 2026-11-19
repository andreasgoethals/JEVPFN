"""Exp 8 (RQ2-E) statistics fixed before results: primary moderators, FDR, pooling.

The plan lives in `config/exp08_when_text_helps/when_text_helps.yaml` (status
``proposed_before_results``). Three analyses use the per-dataset gains of
``aggregate.dataset_gains`` restricted to the headline grid (median over decision models per
decision mode, and each embedding/TAR recipe), each predictor separately:

* ``moderator_tests``: Spearman rho of normalised and raw gain with every numeric dataset
  moderator, a dataset-bootstrap interval and a two-sided permutation p-value. The four
  PRIMARY moderators are flagged and reported unadjusted; every other (exploratory) cell is
  Benjamini–Hochberg adjusted across the whole exploratory grid.
* ``partial_primary``: partial Spearman rho of each primary moderator with gain, controlling
  for the other primary moderators (rank-transform, residualise on the controls, correlate).
* ``random_effects``: DerSimonian–Laird pooling of per-dataset gains with the fold-based
  within-dataset variance. Outer folds share training rows, so those variances are
  approximate (too small); tau² absorbs part of the shortfall, but intervals are optimistic.

About eighteen datasets give little power. Results are associations, never causal effects or
a basis for selecting a representation, and nothing here replaces the confirmatory plan in
`config/synthesis/analysis.yaml`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.aggregate import MEDIAN, RECIPE
from src.evaluation.moderators import bootstrap_spearman, moderator_data, spearman_rows

CONFIG = "exp08_when_text_helps/when_text_helps"
STATUS = "proposed_before_results"
TEST_COLUMNS = [
    "model_id", "checkpoint_id", "family", "representation", "outcome", "moderator", "role",
    "n_datasets", "spearman_rho", "ci_low", "ci_high", "p_value", "q_value", "significant_fdr",
    "status",
]
PARTIAL_COLUMNS = [
    "model_id", "checkpoint_id", "family", "representation", "outcome", "moderator", "controls",
    "n_datasets", "partial_spearman_rho", "ci_low", "ci_high", "status",
]
META_COLUMNS = [
    "model_id", "checkpoint_id", "family", "representation", "k", "excluded_datasets",
    "pooled_gain", "se", "ci_low", "ci_high", "tau2", "i2", "q", "status", "variance_note",
]
VARIANCE_NOTE = "fold-based within-dataset variance; folds overlap, so it is approximate"
_KEYS = ["model_id", "checkpoint_id", "family", "representation"]


def load_plan(raw: dict | None = None) -> dict:
    """The validated Exp 8 analysis plan."""
    if raw is None:
        from src.utils.config import load_yaml

        raw = load_yaml(CONFIG)
    if raw.get("schema") != "jevpfn.exp08_analysis.v1":
        raise ValueError("Unknown Exp 8 analysis-plan schema")
    if raw.get("status") not in {STATUS, "approved"}:
        raise ValueError("Exp 8 analysis plan must be proposed before results or approved")
    # Owner decision (19-11-2026): no moderator is singled out in advance. With an empty
    # list every moderator is exploratory and BH-adjusted across the whole grid.
    primary = raw.get("primary_moderators") or []
    names = [p["name"] for p in primary]
    if len(names) != len(set(names)):
        raise ValueError("Primary moderators must be unique")
    if raw.get("exploratory_correction") != "benjamini_hochberg_fdr_across_the_exploratory_grid":
        raise ValueError("Exploratory moderators require Benjamini-Hochberg FDR control")
    if not 0 < float(raw.get("fdr_level", 0)) < 1:
        raise ValueError("fdr_level must lie in (0, 1)")
    if raw.get("meta_analysis", {}).get("method") != "dersimonian_laird":
        raise ValueError("Only DerSimonian-Laird random-effects pooling is planned")
    return raw


def primary_columns(plan: dict) -> dict[str, str]:
    """Primary moderator name -> moderator-table column."""
    return {p["name"]: p["column"] for p in plan["primary_moderators"]}


def headline_gains(gains: pd.DataFrame) -> pd.DataFrame:
    """The planned grid: median-over-decision-models rows and embedding/TAR recipes."""
    if gains.empty:
        return gains
    return gains[gains.aggregation.isin([MEDIAN, RECIPE])].copy()


# ------------------------------------------------------------------------------ tests


def benjamini_hochberg(p_values) -> np.ndarray:
    """BH-adjusted q-values (monotone step-up); NaN p-values stay NaN and are not counted."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    m = len(finite)
    if not m:
        return q
    order = finite[np.argsort(p[finite], kind="stable")]
    ranked = p[order] * m / np.arange(1, m + 1)
    q[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    return q


def permutation_p(x, y, *, n_permutations: int, seed: int) -> float:
    """Two-sided permutation p-value of Spearman rho, (1 + #|rho*| >= |rho|) / (1 + B)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    observed = spearman_rows(x, y)[0]
    if not np.isfinite(observed) or n_permutations <= 0:
        return np.nan
    rng = np.random.default_rng(seed)
    shuffled = np.stack([x[rng.permutation(len(x))] for _ in range(n_permutations)])
    values = spearman_rows(shuffled, np.broadcast_to(y, shuffled.shape))
    extreme = np.sum(np.abs(values) >= abs(observed) - 1e-12)
    return float((1 + extreme) / (1 + n_permutations))


def moderator_tests(gains, moderators, plan=None):
    """Primary and exploratory moderator associations over the headline grid.

    Returns one row per predictor × representation × outcome × moderator with ``role``
    (``primary`` or ``exploratory``), rho, bootstrap interval, permutation p-value and, for
    exploratory rows, the BH q-value across all exploratory rows of the table.
    """
    plan = load_plan() if plan is None else plan
    grid = headline_gains(gains)
    if grid.empty or moderators.empty:
        return pd.DataFrame(columns=TEST_COLUMNS)
    data, names = moderator_data(grid, moderators)
    primary = primary_columns(plan)
    columns = {name: column for name, column in primary.items() if column in data}
    exploratory = [n for n in names if n not in set(primary.values())]
    seed, n_boot = int(plan["seed"]), int(plan["n_boot"])
    n_perm, minimum = int(plan["n_permutations"]), int(plan["min_datasets"])
    records = []
    for key, group in data.groupby(_KEYS, dropna=False, sort=True):
        for outcome in plan["outcomes"]:
            cells = [(n, c, "primary") for n, c in columns.items()]
            cells += [(n, n, "exploratory") for n in exploratory]
            for name, column, role in cells:
                pair = group[[outcome, column]].dropna()
                x, y = pair[column].to_numpy(float), pair[outcome].to_numpy(float)
                rho, low, high, _ = bootstrap_spearman(x, y, n_boot=n_boot, seed=seed)
                status = ("too_few_datasets" if len(x) < minimum
                          else "constant_input" if not np.isfinite(rho) else "estimated")
                ok = status == "estimated"
                records.append(dict(zip(_KEYS, key)) | {
                    "outcome": outcome, "moderator": name, "role": role, "n_datasets": len(x),
                    "spearman_rho": rho if ok else np.nan,
                    "ci_low": low if ok else np.nan, "ci_high": high if ok else np.nan,
                    "p_value": permutation_p(x, y, n_permutations=n_perm, seed=seed) if ok else np.nan,
                    "status": status,
                })
    table = pd.DataFrame(records)
    table["q_value"] = np.nan
    explore = table.role.eq("exploratory")
    table.loc[explore, "q_value"] = benjamini_hochberg(table.loc[explore, "p_value"])
    table["significant_fdr"] = explore & table.q_value.le(float(plan["fdr_level"]))
    return table[TEST_COLUMNS]


# ------------------------------------------------------------------- partial Spearman


def _residuals(values, controls):
    design = np.column_stack([np.ones(len(values)), controls])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def partial_spearman(x, y, controls) -> float:
    """Partial Spearman rho: Pearson correlation of rank residuals after the control ranks."""
    from scipy.stats import rankdata

    x, y = np.asarray(x, float), np.asarray(y, float)
    controls = np.asarray(controls, float).reshape(len(x), -1)
    if len(x) < controls.shape[1] + 3:
        return np.nan
    rx, ry = rankdata(x), rankdata(y)
    rc = np.column_stack([rankdata(c) for c in controls.T]) if controls.shape[1] else controls
    ex, ey = _residuals(rx, rc), _residuals(ry, rc)
    scale = np.sqrt((ex**2).sum() * (ey**2).sum())
    return float((ex * ey).sum() / scale) if scale > 1e-12 else np.nan


def partial_primary(gains, moderators, plan=None):
    """Each primary moderator against gain, controlling for the other primary moderators.

    Only datasets with every primary moderator defined enter. Percentile interval from a
    dataset bootstrap (seeded per cell); undersized or degenerate inputs are reported.
    """
    plan = load_plan() if plan is None else plan
    grid = headline_gains(gains)
    if grid.empty or moderators.empty:
        return pd.DataFrame(columns=PARTIAL_COLUMNS)
    data, _ = moderator_data(grid, moderators)
    primary = {n: c for n, c in primary_columns(plan).items() if c in data}
    seed, n_boot, minimum = int(plan["seed"]), int(plan["n_boot"]), int(plan["min_datasets"])
    records = []
    for key, group in data.groupby(_KEYS, dropna=False, sort=True):
        for outcome in plan["outcomes"]:
            complete = group[[outcome, *primary.values()]].dropna()
            for name, column in primary.items():
                others = [c for n, c in primary.items() if n != name]
                x = complete[column].to_numpy(float)
                y = complete[outcome].to_numpy(float)
                z = complete[others].to_numpy(float)
                rho = partial_spearman(x, y, z)
                low = high = np.nan
                if np.isfinite(rho) and n_boot > 0:
                    draws = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
                    values = np.array([partial_spearman(x[d], y[d], z[d]) for d in draws])
                    values = values[np.isfinite(values)]
                    if len(values):
                        low, high = (float(v) for v in np.percentile(values, [2.5, 97.5]))
                status = ("too_few_datasets" if len(x) < max(minimum, len(primary) + 2)
                          else "degenerate_input" if not np.isfinite(rho) else "estimated")
                ok = status == "estimated"
                records.append(dict(zip(_KEYS, key)) | {
                    "outcome": outcome, "moderator": name,
                    "controls": ", ".join(n for n in primary if n != name),
                    "n_datasets": len(x), "partial_spearman_rho": rho if ok else np.nan,
                    "ci_low": low if ok else np.nan, "ci_high": high if ok else np.nan,
                    "status": status,
                })
    return pd.DataFrame(records, columns=PARTIAL_COLUMNS)


# ------------------------------------------------------------------- random effects


def dersimonian_laird(effects, variances, *, z: float = 1.959963984540054) -> dict:
    """Random-effects pooled mean (DerSimonian–Laird tau²) with a normal interval.

    Studies need a finite effect and a positive finite variance. Returns pooled estimate,
    standard error, interval, tau², I² (share of total variation from heterogeneity) and Q.
    """
    y, v = np.asarray(effects, float), np.asarray(variances, float)
    keep = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = y[keep], v[keep]
    k = len(y)
    out = {"k": k, "excluded": int((~keep).sum()), "pooled": np.nan, "se": np.nan,
           "ci_low": np.nan, "ci_high": np.nan, "tau2": np.nan, "i2": np.nan, "q": np.nan}
    if k < 2:
        out["status"] = "too_few_datasets"
        return out
    w = 1 / v
    fixed = np.sum(w * y) / np.sum(w)
    q = float(np.sum(w * (y - fixed) ** 2))
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    w_star = 1 / (v + tau2)
    pooled = float(np.sum(w_star * y) / np.sum(w_star))
    se = float(np.sqrt(1 / np.sum(w_star)))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    out.update(pooled=pooled, se=se, ci_low=pooled - z * se, ci_high=pooled + z * se,
               tau2=float(tau2), i2=float(i2), q=q, status="estimated")
    return out


def random_effects(gains, plan=None, *, grid_only=True):
    """DerSimonian–Laird pooled normalised gain per predictor × representation."""
    if gains.empty or "gain_variance" not in gains:
        return pd.DataFrame(columns=META_COLUMNS)
    data = headline_gains(gains) if grid_only else gains
    records = []
    for key, group in data.groupby(_KEYS, dropna=False, sort=True):
        result = dersimonian_laird(group.gain, group.gain_variance)
        records.append(dict(zip(_KEYS, key)) | {
            "k": result["k"], "excluded_datasets": result["excluded"],
            "pooled_gain": result["pooled"], "se": result["se"], "ci_low": result["ci_low"],
            "ci_high": result["ci_high"], "tau2": result["tau2"], "i2": result["i2"],
            "q": result["q"], "status": result["status"], "variance_note": VARIANCE_NOTE,
        })
    return pd.DataFrame(records, columns=META_COLUMNS)
