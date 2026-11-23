"""Read the compact A2/A3 exports locally. Validation pilots never enter the exports."""

import numpy as np
import pandas as pd

from src.evaluation import geometry_export, robustness_export


def robustness_export_tables():
    """All A2 tables and the export receipt (waiting state when the ZIP is absent)."""
    return robustness_export.read_export()


def robustness_tables():
    frames, _ = robustness_export_tables()
    return frames.get("effects", pd.DataFrame()), frames.get("coverage", pd.DataFrame())


def geometry_export_tables():
    return geometry_export.read_export()


def geometry_tables():
    frames, _ = geometry_export_tables()
    return frames.get("comparisons", pd.DataFrame()), frames.get("coverage", pd.DataFrame())


def encoder_references():
    frames, _ = geometry_export_tables()
    return frames.get("encoder_references", pd.DataFrame())


def equal_dataset_replay(effects):
    """Each dataset contributes one matched perturbation-minus-replay contrast."""
    if effects.empty or "excess_tv" not in effects:
        return pd.DataFrame(columns=["provider", "variant", "datasets", "mean_excess_tv"])
    return (effects.dropna(subset=["excess_tv"]).groupby(["provider", "variant"])
            .agg(datasets=("dataset", "nunique"), mean_excess_tv=("excess_tv", "mean")).reset_index())


def equal_dataset_mode_effects(mode_effects):
    """Per model, request mode and factor: within-dataset values, then equal dataset weight."""
    columns = ["provider", "mode", "variant", "datasets", "mean_tv", "excess_tv", "flip_rate"]
    if mode_effects.empty:
        return pd.DataFrame(columns=columns)
    return (mode_effects.groupby(["provider", "mode", "variant"])
            .agg(datasets=("dataset", "nunique"), mean_tv=("mean_tv", "mean"),
                 excess_tv=("excess_tv", "mean"), flip_rate=("flip_rate", "mean")).reset_index())


def wording_averaging_summary(table):
    """Equal dataset weight: single-version versus averaged-pair wording distance."""
    columns = ["provider", "mode", "datasets", "single_tv", "averaged_tv", "averaged_to_single_tv",
               "replay_tv"]
    if table.empty:
        return pd.DataFrame(columns=columns)
    out = (table.groupby(["provider", "mode"])
           .agg(datasets=("dataset", "nunique"), single_tv=("single_tv", "mean"),
                averaged_tv=("averaged_tv", "mean"), replay_tv=("replay_tv", "mean")).reset_index())
    out["averaged_to_single_tv"] = np.where(out.single_tv > 0, out.averaged_tv / out.single_tv, np.nan)
    return out[columns]


#: Pooled across task types: log loss (classification) or RPS (regression) as a relative skill.
POOLED_SKILL = "proper_score_skill"
POOLED_SKILL_ROLE = {POOLED_SKILL: {"role": "primary_pooled", "lower_is_better": False, "convex": True}}
_TASK_PRIMARY = ("log_loss", "ranked_probability_score")


def _pooled_skill(part):
    """Skill 1 - S_average / S_reference of each dataset's own primary proper score.

    The nine classification datasets use log loss and the nine regression datasets the
    ranked probability score, so one dimensionless, commensurable value per dataset lets a
    signed-rank test use all 18 datasets (raw losses differ in scale between tasks). A ratio
    of proper scores is generally not itself proper (Gneiting & Raftery 2007); it only
    rescales a paired difference of two fixed forecasts, and its sign equals the raw
    difference's sign. A zero reference loss leaves the skill undefined (NaN), never zero.
    """
    primary = part[part.metric.isin(_TASK_PRIMARY)]
    keys = ["dataset", "provider", "factor"]
    if primary.duplicated(keys).any():
        raise ValueError("A dataset has both log loss and RPS; one primary proper score per task")
    pooled = primary[keys].copy()
    reference = primary["reference"].astype(float)
    pooled["metric"] = POOLED_SKILL
    pooled["reference"], pooled["average"] = 0.0, (reference - primary["average"]) / reference.where(
        reference > 0)
    pooled["difference"] = pooled["improvement"] = pooled["average"]
    return pooled


def averaging_quality_summary(table):
    """Per contrast: averaged minus reference presentation per dataset, then equal dataset weight.

    ``average_vs_original`` is primary and not guaranteed; ``average_vs_mean_single`` is the
    Jensen gap. ``improvement`` is signed so that positive means averaging scored better;
    ``guaranteed`` marks the convex proper scores in the Jensen contrast (never negative there).
    """
    from src.evaluation.robustness_export import CONTRASTS, METRIC_ROLES

    columns = ["provider", "factor", "contrast", "metric", "role", "guaranteed", "datasets",
               "reference", "average", "difference", "improvement", "datasets_improved"]
    if table.empty:
        return pd.DataFrame(columns=columns)
    wide = table.pivot_table(index=["dataset", "provider", "factor", "metric"], columns="presentation",
                             values="value").reset_index()
    out = []
    for contrast, presentation, reference in CONTRASTS:
        if not {presentation, reference} <= set(wide.columns):
            continue
        part = wide[["dataset", "provider", "factor", "metric"]].copy()
        part["reference"], part["average"] = wide[reference], wide[presentation]
        part["difference"] = part["average"] - part["reference"]
        lower = part.metric.map(lambda m: METRIC_ROLES.get(m, {}).get("lower_is_better", True))
        part["improvement"] = np.where(lower, -part["difference"], part["difference"])
        part = pd.concat([part, _pooled_skill(part)], ignore_index=True)
        part = part.dropna(subset=["improvement"])
        summary = (part.groupby(["provider", "factor", "metric"])
                   .agg(datasets=("dataset", "nunique"), reference=("reference", "mean"),
                        average=("average", "mean"), difference=("difference", "mean"),
                        improvement=("improvement", "mean"),
                        datasets_improved=("improvement", lambda v: int((v > 0).sum())))
                   .reset_index())
        summary["contrast"] = contrast
        summary["role"] = summary.metric.map(lambda m: (METRIC_ROLES | POOLED_SKILL_ROLE).get(m, {}).get(
            "role", "descriptive"))
        summary["guaranteed"] = (contrast == "average_vs_mean_single") & summary.metric.map(
            lambda m: (METRIC_ROLES | POOLED_SKILL_ROLE).get(m, {}).get("convex", False))
        out.append(summary)
    return pd.concat(out, ignore_index=True)[columns] if out else pd.DataFrame(columns=columns)
