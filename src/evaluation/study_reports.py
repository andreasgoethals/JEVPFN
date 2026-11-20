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


def averaging_quality_summary(table):
    """Average-minus-original stand-alone score per dataset, then equal dataset weight."""
    columns = ["provider", "factor", "metric", "datasets", "original", "average", "difference"]
    if table.empty:
        return pd.DataFrame(columns=columns)
    wide = table.pivot_table(index=["dataset", "provider", "factor", "metric"], columns="presentation",
                             values="value").reset_index()
    if not {"original", "average"} <= set(wide.columns):
        return pd.DataFrame(columns=columns)
    wide["difference"] = wide["average"] - wide["original"]
    return (wide.groupby(["provider", "factor", "metric"])
            .agg(datasets=("dataset", "nunique"), original=("original", "mean"),
                 average=("average", "mean"), difference=("difference", "mean")).reset_index())[columns]
