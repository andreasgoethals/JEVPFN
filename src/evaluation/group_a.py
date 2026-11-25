"""Group A (RQ1) summaries for the notebooks: compact-export tables only, nothing recomputed.

Every function reads tables of the three Group A exports in their CURRENT schemas
(`comparison_report.SCHEMA` for A1, `robustness_export.SCHEMA` for A2, `geometry_export.SCHEMA`
for A3) and returns one value per dataset first, then equal-dataset summaries. Uncertainty is a
percentile bootstrap over datasets (folds do not exist in Group A; text fields are averaged
within a dataset before any summary).

A1 stand-alone quality (owner question of 23 November 2026, decided here): the A1 export scores
the joint-request probabilities with log loss, Brier, ECE and ROC-AUC for the nine
classification datasets, and only with the expected-level Spearman rho (a rank measure) for the
nine regression datasets; log loss is undefined there and the export carries no ranked
probability score. A1 therefore never pools the two families: classification quality has
N = 9 (ROC-AUC in the figures, log loss in the descriptive CD diagram, as
`learner_scores.a1_groups`), and regression is the separate rank-only rho with N = 9. The
pooled proper-score skill (log loss or RPS) belongs to A2, where the export provides both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_BOOT = 2000
SEED = 0
CLASSIFICATION = ("binary_classification", "multiclass_classification")
#: A1 strip metrics: (export table, column) and the task families each one covers.
A1_METRICS = {
    "mean_top_probability": "all",
    "mean_normalized_entropy": "all",
    "consensus_agreement": "all",
    "classification_auc": "classification",
    "regression_spearman": "regression",
}
#: A2 factors: the variants of each factor in `robustness_export` stores.
A2_FACTORS = {"wording": ("paraphrase_1", "paraphrase_2", "paraphrase_3"),
              "field_order": ("field_order",), "option_order": ("option_order",)}
A2_VARIANT_FACTOR = {variant: factor for factor, variants in A2_FACTORS.items() for variant in variants}


def bootstrap_ci(values, statistic=np.mean, *, n_boot: int = N_BOOT, seed: int = SEED):
    """Percentile 95% interval of `statistic` over resampled datasets (NaN below two values)."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = statistic(values[rng.integers(0, len(values), (n_boot, len(values)))], axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize(frame: pd.DataFrame, keys, value: str, *, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Equal-dataset mean and median with dataset-bootstrap 95% intervals, per `keys`."""
    columns = [*keys, "datasets", "mean", "ci_low", "ci_high", "median", "median_ci_low", "median_ci_high"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for key, group in frame.groupby(list(keys), sort=False):
        values = group[value].to_numpy(float)
        values = values[np.isfinite(values)]
        low, high = bootstrap_ci(values, n_boot=n_boot, seed=seed)
        mlow, mhigh = bootstrap_ci(values, np.median, n_boot=n_boot, seed=seed)
        rows.append(dict(zip(keys, key if isinstance(key, tuple) else (key,)))
                    | {"datasets": len(values), "mean": float(values.mean()) if len(values) else np.nan,
                       "ci_low": low, "ci_high": high,
                       "median": float(np.median(values)) if len(values) else np.nan,
                       "median_ci_low": mlow, "median_ci_high": mhigh})
    return pd.DataFrame(rows, columns=columns)


# ------------------------------------------------------------------------------ A1


def _per_dataset(table: pd.DataFrame, mode: str, metric: str) -> pd.DataFrame:
    """One value per dataset and model: distinct inputs, text fields with equal weight."""
    if table.empty or metric not in table:
        return pd.DataFrame(columns=["dataset_id", "task", "provider", "value"])
    frame = table[table.representation.eq(mode) & table.weighting.eq("unique_inputs")]
    return (frame.groupby(["dataset_id", "task", "provider"], sort=False)[metric].mean()
            .rename("value").reset_index())


def a1_dataset_values(shape: pd.DataFrame, loo: pd.DataFrame, quality: pd.DataFrame,
                      mode: str = "joint") -> pd.DataFrame:
    """Long table (dataset_id, task, provider, metric, value) of the A1 profile metrics.

    Shape and consensus come from ``mode`` requests; stand-alone quality is joint-only by
    design (classification ROC-AUC or macro OvR AUC; regression expected-level Spearman).
    """
    parts = []
    for metric, table, column in (("mean_top_probability", shape, "mean_top_probability"),
                                  ("mean_normalized_entropy", shape, "mean_normalized_entropy"),
                                  ("consensus_agreement", loo, "top_agreement")):
        parts.append(_per_dataset(table, mode, column).assign(metric=metric))
    if not quality.empty:
        scored = quality[quality.status.eq("descriptive")]
        auc = scored[scored.metric.isin(["roc_auc", "roc_auc_ovr_macro"])].assign(metric="classification_auc")
        rho = scored[scored.metric.eq("spearman_expected_level")].assign(metric="regression_spearman")
        for part in (auc, rho):
            parts.append(part.rename(columns={"task_type": "task"})[["dataset_id", "task", "provider", "metric", "value"]])
    columns = ["dataset_id", "task", "provider", "metric", "value"]
    parts = [p[columns] for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def model_order(providers, properties: pd.DataFrame | None = None) -> list[str]:
    """Group A row order: architecture family (style order), then size (largest first), then roster.

    ``properties`` is `model_properties.profile_table()`; a family no local source states is
    grouped last as ``not stated``.
    """
    from src.visualize import style

    providers = list(dict.fromkeys(providers))
    roster = {p: i for i, p in enumerate(style.DECISION_SHORT)}
    families = {f: i for i, f in enumerate(style.ARCHITECTURE_MARKERS)}
    props = (properties.set_index("provider") if properties is not None and not properties.empty
             else pd.DataFrame())

    def key(provider):
        family = props.architecture_family.get(provider) if "architecture_family" in props else None
        size = props.parameters_billion.get(provider) if "parameters_billion" in props else None
        size = float(size) if size is not None and pd.notna(size) else -1.0
        return (families.get(family, len(families)), -size, roster.get(provider, len(roster)), provider)

    return sorted(providers, key=key)


def family_of(provider: str, properties: pd.DataFrame | None) -> str:
    """Architecture family from the verified properties; ``not stated`` when missing."""
    if properties is None or properties.empty:
        return "not stated"
    value = properties.set_index("provider").architecture_family.get(provider)
    return value if isinstance(value, str) and value else "not stated"


# ------------------------------------------------------------------------------ A2


def excess_by_dataset(mode_effects: pd.DataFrame) -> pd.DataFrame:
    """Per dataset, model, request mode and factor: mean change and its excess over replay.

    Wording averages its three rewordings within the dataset. The export's change is the total
    variation distance of the full probability vectors (nine levels for regression); the excess
    subtracts the mean change of the two unchanged replays on the same requests.
    """
    columns = ["dataset", "provider", "mode", "factor", "mean_tv", "excess_tv", "flip_rate",
               "min_matched_requests_per_variant", "max_matched_requests_per_variant", "matched_presentations"]
    if mode_effects.empty:
        return pd.DataFrame(columns=columns)
    frame = mode_effects.assign(factor=mode_effects.variant.map(A2_VARIANT_FACTOR)).dropna(subset=["factor"])
    return (frame.groupby(["dataset", "provider", "mode", "factor"], sort=False)
            .agg(mean_tv=("mean_tv", "mean"), excess_tv=("excess_tv", "mean"),
                 flip_rate=("flip_rate", "mean"), min_matched_requests_per_variant=("matched_requests", "min"),
                 max_matched_requests_per_variant=("matched_requests", "max"),
                 matched_presentations=("matched_requests", "sum"))
            .reset_index()[columns])


def excess_summary(mode_effects: pd.DataFrame, **bootstrap) -> pd.DataFrame:
    """Equal-dataset excess change over replay per model, mode and factor (dataset-bootstrap CI)."""
    return summarize(excess_by_dataset(mode_effects), ["provider", "mode", "factor"], "excess_tv", **bootstrap)


def averaging_skill_by_dataset(averaging_quality: pd.DataFrame,
                               contrast: str = "average_vs_original") -> pd.DataFrame:
    """Per dataset, model and factor: proper-score skill 1 - S_average / S_reference.

    The score is each dataset's own primary proper score (log loss for classification, ranked
    probability score for regression), as the A2 framework application (`learner_scores.a2_skill`).
    Positive means the averaged presentation scored better.
    """
    from src.evaluation.learner_scores import a2_skill

    skill = a2_skill(averaging_quality)
    if skill.empty:
        return skill
    return skill[skill.contrast.eq(contrast)].reset_index(drop=True)


def averaging_skill_summary(averaging_quality: pd.DataFrame, contrast: str = "average_vs_original",
                            **bootstrap) -> pd.DataFrame:
    return summarize(averaging_skill_by_dataset(averaging_quality, contrast), ["provider", "factor"], "skill",
                     **bootstrap)


# ------------------------------------------------------------------------------ A3


def geometry_by_dataset(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Per dataset, encoder and model: measured fields averaged with equal weight.

    Undefined correlations (constant or too few texts) are not zero; they leave the field out
    and are counted in ``undefined_fields``.
    """
    columns = ["dataset", "encoder", "provider", "spearman", "null_mean", "neighbour_overlap",
               "chance_overlap", "fields", "undefined_fields"]
    if comparisons.empty:
        return pd.DataFrame(columns=columns)
    frame = comparisons.copy()
    for column in ("spearman", "null_mean", "neighbour_overlap", "chance_overlap"):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["measured"] = frame.status.eq("measured") if "status" in frame else frame.spearman.notna()
    measured = frame[frame.measured]
    out = (measured.groupby(["dataset", "encoder", "provider"], sort=False)
           [["spearman", "null_mean", "neighbour_overlap", "chance_overlap"]].mean())
    out["fields"] = measured.groupby(["dataset", "encoder", "provider"], sort=False).size()
    out = out.reset_index()
    undefined = (frame[~frame.measured].groupby(["dataset", "encoder", "provider"]).size()
                 .rename("undefined_fields").reset_index())
    out = out.merge(undefined, how="left", on=["dataset", "encoder", "provider"])
    out["undefined_fields"] = out.undefined_fields.fillna(0).astype(int)
    return out[columns]


def geometry_summary(comparisons: pd.DataFrame, metric: str = "spearman", **bootstrap) -> pd.DataFrame:
    """Equal-dataset mean of `metric` per model and encoder, with its reference (null or chance)."""
    by_dataset = geometry_by_dataset(comparisons)
    out = summarize(by_dataset, ["provider", "encoder"], metric, **bootstrap)
    reference = "null_mean" if metric == "spearman" else "chance_overlap"
    if out.empty:
        return out.assign(reference=pd.Series(dtype=float))
    means = by_dataset.groupby(["provider", "encoder"], sort=False)[reference].mean().rename("reference")
    return out.merge(means.reset_index(), on=["provider", "encoder"], how="left")


def encoder_reference_summary(references: pd.DataFrame, metric: str = "spearman") -> pd.DataFrame:
    """Encoder-encoder agreement on the same field samples: fields within dataset, then datasets."""
    columns = ["left_encoder", "right_encoder", "datasets", "mean"]
    if references.empty or metric not in references:
        return pd.DataFrame(columns=columns)
    frame = references.copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    per = frame.dropna(subset=[metric]).groupby(["left_encoder", "right_encoder", "dataset"])[metric].mean()
    out = per.groupby(["left_encoder", "right_encoder"]).agg(datasets="size", mean="mean").reset_index()
    return out[columns]


# ------------------------------------------------------------------------------ A synthesis


def model_map_table(a1_values: pd.DataFrame, excess: pd.DataFrame, properties: pd.DataFrame) -> pd.DataFrame:
    """One row per decision model: the Group A profile used by the synthesis map and table.

    Medians over datasets of the A1 joint-request metrics (classification AUC and regression
    rho over their nine datasets each), the A2 mean wording excess on joint requests, and the
    verified size, readout and architecture. Missing measurements stay missing.
    """
    columns = ["provider", "architecture_family", "readout_type", "parameters_billion",
               *A1_METRICS, "wording_excess_joint", "wording_excess_datasets"]
    providers = list(dict.fromkeys([*properties.provider] if not properties.empty else []))
    medians = (a1_values.groupby(["provider", "metric"]).value.median().unstack("metric")
               if not a1_values.empty else pd.DataFrame())
    wording = excess[excess["mode"].eq("joint") & excess.factor.eq("wording")].set_index("provider") \
        if not excess.empty else pd.DataFrame()
    rows = []
    for provider in providers:
        props = properties.set_index("provider").loc[provider]
        row = {"provider": provider, "architecture_family": props.get("architecture_family"),
               "readout_type": props.get("readout_type"),
               "parameters_billion": pd.to_numeric(props.get("parameters_billion"), errors="coerce")}
        for metric in A1_METRICS:
            row[metric] = medians.loc[provider, metric] if provider in medians.index and metric in medians else np.nan
        row["wording_excess_joint"] = wording["mean"].get(provider, np.nan) if not wording.empty else np.nan
        row["wording_excess_datasets"] = wording["datasets"].get(provider, 0) if not wording.empty else 0
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def a2_margin_tests(framework: dict, contrast: str = "average_vs_original") -> pd.DataFrame | None:
    """Holm p and Bayesian posteriors per model and factor from an approved A2 framework run.

    ``framework`` is `learner_scores.run_application("A2_stability")`. None until the framework
    is approved and at least one factor group was compared, so the forest margin stays empty.
    """
    if not framework or framework.get("status") != "compared":
        return None
    parts = []
    for group in framework.get("groups", []):
        factor, _, name = group["group"].partition("/")
        if name != contrast or group.get("status") != "compared":
            continue
        table = group["table"]
        parts.append(table.assign(provider=table.learner, factor=factor)[
            ["provider", "factor", "p_holm", "bayes_p_worse", "bayes_p_equivalent", "bayes_p_better"]])
    return pd.concat(parts, ignore_index=True) if parts else None


def cohort_frames(frames, roster):
    """Select per-provider A2/A3 rows for the named cohort, retaining their measured coverage."""
    return {key: table[table.provider.isin(roster)].copy() if "provider" in table else table
            for key, table in frames.items()}
