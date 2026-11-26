"""Headroom-normalised representation gains over the no-text baseline; descriptive only.

Within each fixed predictor, every complete representation arm (five defined outer folds)
is compared with the same predictor's ``non_text`` arm on the same dataset and folds:

    gain = (mean augmented − mean baseline) / (1 − mean baseline)

for the prespecified headline metric (ROC-AUC, macro one-vs-rest ROC-AUC or R², each with a
ceiling of 1). The gain is the share of the remaining headroom that is closed, so datasets
with different baseline difficulty contribute on one scale. Datasets carry equal weight and
the percentile interval resamples datasets, never folds or rows. Nothing here is a
confirmatory test, selects a winning recipe or replaces the approved analysis plan.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from src.evaluation.endpoints import HEADLINES

DECISION_MODES = {
    "non_text_plus_per_column": "per_column",
    "non_text_plus_joint": "joint",
    "non_text_plus_both": "both",
}
FAMILIES = {"B1": "decision", "B2": "embedding", "B3": "tar"}
MEDIAN = "median_over_decision_models"
PER_MODEL = "per_decision_model"
RECIPE = "recipe"
SUMMARY_COLUMNS = [
    "model_id", "checkpoint_id", "family", "aggregation", "representation", "feature_set",
    "provider", "encoder", "pca_scope", "pca_components", "dataset_scope", "n_datasets",
    "mean_gain", "ci_low", "ci_high", "n_boot", "seed", "decision_models", "dataset_ids",
]
DATASET_COLUMNS = [
    "model_id", "checkpoint_id", "family", "aggregation", "representation", "feature_set",
    "provider", "encoder", "pca_scope", "pca_components", "dataset_id", "task_type",
    "baseline", "augmented", "gain", "decision_models", "raw_gain", "gain_variance",
]
RANKING_COLUMNS = ["model_a", "model_b", "n_representations", "kendall_tau_b"]
_KEYS = ["family", "aggregation", "representation", "feature_set", "provider", "encoder",
         "pca_scope", "pca_components"]


def catalog_text_columns():
    """Text-column counts from the reviewed dataset catalog (no data files are read)."""
    from src.data.loaders import load_catalog

    return {e["id"]: int(e["paper_summary"]["text_columns"]) for e in load_catalog()["datasets"]}


def _representation(row):
    from src.evaluation.group_b import representation

    name = representation(row._asdict())
    return name.replace("frozen/", "embedding/", 1) if name.startswith("frozen/") else name


def _headroom(baseline, eps):
    room = 1.0 - baseline
    return room.where(room > 0) if eps is None else room.clip(lower=eps)


def dataset_gains(paired, *, eps=None, roster=None):
    """One normalised gain per predictor, representation and dataset with five paired folds.

    ``raw_gain`` is the unnormalised five-fold mean difference (augmented − baseline).
    ``gain_variance`` is an approximate within-dataset sampling variance of ``gain``: the
    variance over the five fold differences divided by five and by the squared headroom.
    Outer folds share training rows, so this treats correlated folds as independent and
    understates the true variance; it serves only as a meta-analysis weight.
    """
    if paired.empty:
        return pd.DataFrame(columns=DATASET_COLUMNS)
    data = paired[paired.phase.isin(FAMILIES) & paired.metric.eq(paired.task_type.map(HEADLINES))]
    if "reference" in data:
        data = data[data.reference.eq("non_text")]
    if "inference_status" in data:
        # Post-selection exploratory combinations never enter a headline summary.
        data = data[data.inference_status.fillna("prespecified").eq("prespecified")]
    decision = data.phase.eq("B1") & data.feature_set.isin(DECISION_MODES) & data.provider.notna()
    embedded = data.phase.isin(["B2", "B3"]) & data.encoder.notna()
    data = data[decision | embedded]
    if data.empty:
        return pd.DataFrame(columns=DATASET_COLUMNS)
    if data.groupby("model_id").checkpoint_id.nunique().gt(1).any():
        raise ValueError("Select one checkpoint per predictor before aggregating gains")
    keys = ["phase", "model_id", "checkpoint_id", "dataset_id", "task_type", "split_id",
            "feature_set", "provider", "encoder", "pca_scope", "pca_components", "arm_id"]
    keys += [k for k in ("view", "probe_view") if k in data]
    from src.evaluation.group_b import validate_fold_identities

    validate_fold_identities(data)
    data = data.assign(_fold_difference=data.value - data.non_text_value)
    data = data[np.isfinite(data._fold_difference)]
    arms = data.groupby(keys, dropna=False).agg(
        augmented=("value", "mean"), baseline=("non_text_value", "mean"),
        defined=("value", "count"), defined_baseline=("non_text_value", "count"),
        folds=("outer_fold", "nunique"), fold_difference_var=("_fold_difference", "var"),
    ).reset_index()
    # Arms lacking five defined, paired folds are excluded rather than partially averaged.
    arms = arms[arms.defined.eq(5) & arms.defined_baseline.eq(5) & arms.folds.eq(5)].copy()
    if arms.empty:
        return pd.DataFrame(columns=DATASET_COLUMNS)
    arms["representation"] = [_representation(r) for r in arms.itertuples()]
    arms["family"] = arms.representation.str.split("/").str[0]
    arms["aggregation"] = np.where(arms.family.eq("decision"), PER_MODEL, RECIPE)
    if arms.duplicated(["model_id", "representation", "dataset_id"]).any():
        raise ValueError("Multiple arms/splits/views for one dataset and representation; select one campaign")
    headroom = _headroom(arms.baseline, eps)
    arms["gain"] = (arms.augmented - arms.baseline) / headroom
    arms["raw_gain"] = arms.augmented - arms.baseline
    arms["gain_variance"] = arms.fold_difference_var / 5 / headroom**2
    arms["decision_models"] = np.where(arms.family.eq("decision"), 1, np.nan)
    rows = arms[DATASET_COLUMNS].copy()
    if roster is None:
        from src.evaluation.benchmark_export import decision_roster

        roster = decision_roster()
    per_model = rows[rows.family.eq("decision") & rows.provider.isin(roster)]
    if not per_model.empty:
        median = per_model.groupby(["model_id", "checkpoint_id", "feature_set", "dataset_id",
                                    "task_type"], dropna=False).agg(
            gain=("gain", "median"), baseline=("baseline", "first"),
            augmented=("augmented", "median"), decision_models=("provider", "nunique"),
            raw_gain=("raw_gain", "median"),
        ).reset_index()
        median = median[median.decision_models.eq(len(roster))].copy()
        # Fold-level medians over decision models give the median arm's own fold variance.
        ident = ["model_id", "dataset_id", "split_id", "arm_id"]
        complete = pd.MultiIndex.from_frame(arms.loc[arms.family.eq("decision") & arms.provider.isin(roster), ident])
        fold_rows = data[pd.MultiIndex.from_frame(data[ident]).isin(complete)]
        per_fold = (fold_rows.groupby(["model_id", "checkpoint_id", "feature_set", "dataset_id",
                                       "outer_fold"], dropna=False)._fold_difference.median()
                    .groupby(["model_id", "checkpoint_id", "feature_set", "dataset_id"],
                             dropna=False).var().rename("fold_difference_var").reset_index())
        median = median.merge(per_fold, on=["model_id", "checkpoint_id", "feature_set",
                                            "dataset_id"], how="left")
        median["gain_variance"] = (median.fold_difference_var / 5
                                   / _headroom(median.baseline, eps) ** 2)
        median = median.assign(family="decision", aggregation=MEDIAN, provider=None, encoder=None,
                               pca_scope=None, pca_components=np.nan)
        median["representation"] = "decision/" + median.feature_set.map(DECISION_MODES) + "/median"
        rows = pd.concat([rows, median[DATASET_COLUMNS]], ignore_index=True)
    return rows.reset_index(drop=True)


def _bootstrap(values, n_boot, seed):
    values = np.asarray(values, dtype=float)
    if n_boot <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def normalized_gain_summary(paired, *, n_boot=2000, seed=0, eps=None, text_columns=None, roster=None):
    """Equal-dataset mean headroom-normalised gain with a dataset bootstrap 95% interval.

    ``dataset_scope='all'`` compares each representation with the no-text baseline on every
    complete dataset. Decision rows are repeated under ``'multi_text'`` for per-column versus
    joint versus both contrasts: those modes are identical by construction on one-text-column
    datasets (``single_text_rule``), which are therefore excluded only from that scope.
    """
    gains = dataset_gains(paired, eps=eps, roster=roster)
    if gains.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    counts = catalog_text_columns() if text_columns is None else text_columns
    unknown = set(gains.dataset_id) - set(counts)
    if unknown:
        raise ValueError(f"Text-column count unknown for {sorted(unknown)}")
    scoped = [gains.assign(dataset_scope="all")]
    decision = gains[gains.family.eq("decision")]
    scoped.append(decision[decision.dataset_id.map(counts).gt(1)].assign(dataset_scope="multi_text"))
    table = pd.concat(scoped, ignore_index=True)
    records = []
    for key, group in table.groupby(["model_id", "checkpoint_id", "dataset_scope", *_KEYS],
                                    dropna=False, sort=False):
        values = group.gain.to_numpy(dtype=float)
        low, high = _bootstrap(values, n_boot, seed)
        names = ["model_id", "checkpoint_id", "dataset_scope", *_KEYS]
        records.append(dict(zip(names, key)) | {
            "n_datasets": len(values), "mean_gain": float(values.mean()), "ci_low": low,
            "ci_high": high, "n_boot": n_boot, "seed": seed,
            "decision_models": group.decision_models.min() if group.family.iloc[0] == "decision" else np.nan,
            "dataset_ids": ", ".join(sorted(group.dataset_id)),
        })
    result = pd.DataFrame(records, columns=SUMMARY_COLUMNS)
    result["provider"] = result.provider.where(result.provider.notna(), None)
    return result.sort_values(["dataset_scope", "family", "representation", "model_id"],
                              kind="stable").reset_index(drop=True)


def headline_rows(summary):
    """Representation families shown in the headline: model-agnostic decision modes and recipes."""
    if summary.empty:
        return summary
    return summary[summary.dataset_scope.eq("all") & summary.aggregation.isin([MEDIAN, RECIPE])]


def kendall_tau_b(a, b):
    """Kendall tau-b with tie correction; NaN when either ranking is constant."""
    from scipy.stats import kendalltau

    if len(a) < 2:
        return np.nan
    return float(kendalltau(a, b, variant="b").statistic)


def ranking_stability(paired, *, summary=None, n_boot=0, seed=0, eps=None, text_columns=None, roster=None):
    """Kendall tau-b between predictors' rankings of the headline representations.

    Rankings use the equal-dataset mean normalised gain over the representations both
    predictors share. Returns the pairwise table and the mean of the defined taus.
    """
    if summary is None:
        summary = normalized_gain_summary(paired, n_boot=n_boot, seed=seed, eps=eps,
                                          text_columns=text_columns, roster=roster)
    rows = headline_rows(summary)
    if rows.empty:
        return pd.DataFrame(columns=RANKING_COLUMNS), np.nan
    wide = rows.pivot_table(index="representation", columns="model_id", values="mean_gain")
    records = []
    for left, right in combinations(list(wide.columns), 2):
        shared = wide[[left, right]].dropna()
        records.append({"model_a": left, "model_b": right, "n_representations": len(shared),
                        "kendall_tau_b": kendall_tau_b(shared[left], shared[right])})
    table = pd.DataFrame(records, columns=RANKING_COLUMNS)
    defined = table.kendall_tau_b.dropna()
    return table, float(defined.mean()) if len(defined) else np.nan
