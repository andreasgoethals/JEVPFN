"""RQ2-E dataset-level moderators of the headroom-normalised gain; descriptive only.

Covariates are label-free dataset properties (size, text-field composition, text length,
missing and repeated text, task type and number of classes) plus the predictor's own
no-text headroom, 1 − baseline headline metric. Optional decision-output properties come
from RQ1A compact comparison tables when present; otherwise they are reported as pending.

``moderator_associations`` computes, per predictor × representation, Spearman's rank
correlation between per-dataset normalised gain (``aggregate.dataset_gains``) and each
moderator across datasets, with a dataset-bootstrap percentile interval. Datasets carry
equal weight. With about eighteen datasets these are exploratory associations, never
p-values, causal claims or a basis for selecting a winning recipe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.field_types import FIELD_TYPES, classify_text_column
from src.utils.serialization import text_value

DECISION_PROPERTIES = (
    "decision_mean_max_probability",
    "decision_mean_entropy",
    "decision_mean_pairwise_agreement",
)
ASSOCIATION_COLUMNS = [
    "model_id", "checkpoint_id", "family", "aggregation", "representation", "moderator",
    "n_datasets", "spearman_rho", "ci_low", "ci_high", "n_boot", "defined_boot", "seed",
    "status", "dataset_ids",
]
_REPRESENTATION_KEYS = ["model_id", "checkpoint_id", "family", "aggregation", "representation"]


def dataset_covariates(dataset) -> dict:
    """Label-free covariates of one loaded dataset; the target column is never read."""
    metadata = dataset.metadata
    frame = dataset.features
    text_columns = list(metadata.text_columns)
    types = [classify_text_column(frame[c])["field_type"] for c in text_columns]
    lengths, cells, missing, repeated, present = [], 0, 0, 0, 0
    for column in text_columns:
        values = frame[column].map(text_value)
        available = values.notna()
        cells += len(values)
        missing += int((~available).sum())
        text = values[available].astype(str)
        present += len(text)
        repeated += int(text.duplicated(keep=False).sum())
        lengths.append(text.str.len().to_numpy())
    lengths = np.concatenate(lengths) if lengths else np.array([])
    classification = metadata.task_type != "regression"
    row = {
        "dataset_id": metadata.dataset_id,
        "task_type": metadata.task_type,
        "n_rows": len(frame),
        "n_text_fields": len(text_columns),
        "n_non_text_fields": len(list(metadata.non_text_columns)),
        "n_classes": len(metadata.classes) if classification else np.nan,
        "median_text_chars": float(np.median(lengths)) if len(lengths) else np.nan,
        "missing_text_share": missing / cells if cells else np.nan,
        # Within each column: non-missing cells whose exact text occurs more than once.
        "repeated_text_share": repeated / present if present else np.nan,
    }
    for kind in (*FIELD_TYPES, "empty"):
        row[f"share_{kind}"] = (
            sum(t == kind for t in types) / len(types) if types else np.nan
        )
    return row


def decision_properties(tables: Path | None = None):
    """Per dataset and provider from RQ1A ``distributions.csv`` and ``pairs.csv``.

    Row-weighted joint-text summaries only. Returns ``(table, status)``; an absent or
    unreadable folder gives an empty table and a ``pending`` status, never an error.
    """
    columns = ["dataset_id", "provider", *DECISION_PROPERTIES]
    if tables is None:
        from src.utils import paths

        root = paths.notebook_outputs_dir() / "results" / "notebooks"
        found = sorted(root.glob("**/tables/distributions.csv")) if root.is_dir() else []
        if len(found) != 1:
            reason = "no RQ1A comparison tables" if not found else "several RQ1A table folders"
            return pd.DataFrame(columns=columns), f"pending: {reason}"
        tables = found[0].parent
    tables = Path(tables)
    try:
        distributions = pd.read_csv(tables / "distributions.csv")
        pairs = pd.read_csv(tables / "pairs.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=columns), f"pending: incomplete tables in {tables}"
    needed = {"dataset_id", "provider", "representation", "weighting", "variable", "mean"}
    if not needed <= set(distributions) or not {
        "dataset_id", "model_a", "model_b", "representation", "weighting", "agreement"
    } <= set(pairs):
        return pd.DataFrame(columns=columns), f"pending: unexpected table schema in {tables}"
    rows = distributions[distributions.representation.eq("joint")
                         & distributions.weighting.eq("rows")]
    wide = (rows[rows.variable.isin(["max_probability", "entropy"])]
            .pivot_table(index=["dataset_id", "provider"], columns="variable", values="mean")
            .rename(columns={"max_probability": DECISION_PROPERTIES[0],
                             "entropy": DECISION_PROPERTIES[1]}))
    joint = pairs[pairs.representation.eq("joint") & pairs.weighting.eq("rows")]
    both = pd.concat([joint.rename(columns={"model_a": "provider"}),
                      joint.rename(columns={"model_b": "provider"})])
    agreement = both.groupby(["dataset_id", "provider"]).agreement.mean().rename(
        DECISION_PROPERTIES[2])
    table = wide.join(agreement, how="outer").reset_index()
    for column in columns:
        if column not in table:
            table[column] = np.nan
    return table[columns], f"available: {tables}"


def dataset_moderators(datasets=None, *, gains=None, decision_tables=None):
    """One row per dataset (or dataset × predictor when ``gains`` supplies headroom).

    ``datasets`` are loaded datasets (default: the Experiment 1 selection, read locally).
    Headroom is predictor-specific: 1 − that predictor's mean no-text headline metric,
    taken from ``aggregate.dataset_gains``. Decision properties are dataset medians over
    providers; the per-provider table is kept in ``attrs['decision_properties']``.
    """
    if datasets is None:
        from src.data.loaders import load_catalog, load_dataset
        from src.data.selection import selected_entries
        from src.utils.config import load_yaml

        entries = selected_entries(load_catalog(), load_yaml("experiment_1/main"))
        datasets = (load_dataset(e, allow_download=False) for e in entries)
    table = pd.DataFrame([dataset_covariates(d) for d in datasets])
    properties, status = decision_properties(decision_tables)
    if properties.empty:
        for column in DECISION_PROPERTIES:
            table[column] = np.nan
    else:
        medians = properties.groupby("dataset_id")[list(DECISION_PROPERTIES)].median()
        table = table.join(medians, on="dataset_id")
    if gains is not None and not gains.empty:
        headroom = (gains[["dataset_id", "model_id", "baseline"]].drop_duplicates()
                    .assign(baseline_headroom=lambda f: 1.0 - f.baseline)
                    .drop(columns="baseline"))
        if headroom.duplicated(["dataset_id", "model_id"]).any():
            raise ValueError("Several no-text baselines for one dataset and predictor")
        table = table.merge(headroom, on="dataset_id", how="left")
    else:
        table["baseline_headroom"] = np.nan
    table.attrs["decision_properties"] = properties
    table.attrs["decision_properties_status"] = status
    table.attrs["headroom_status"] = "available" if gains is not None and not gains.empty else "pending"
    return table


def spearman_rows(x, y):
    """Spearman rho of each row pair (average ranks for ties); NaN for a constant row."""
    from scipy.stats import rankdata

    x, y = np.atleast_2d(x), np.atleast_2d(y)
    if x.shape[1] < 3:
        return np.full(x.shape[0], np.nan)
    rx, ry = rankdata(x, axis=1), rankdata(y, axis=1)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    scale = np.sqrt((rx**2).sum(axis=1) * (ry**2).sum(axis=1))
    return np.divide((rx * ry).sum(axis=1), scale, out=np.full(x.shape[0], np.nan),
                     where=scale > 0)


def moderator_associations(gains, moderators, *, n_boot=2000, seed=0, min_datasets=5):
    """Spearman rho per predictor × representation × moderator with a dataset bootstrap.

    Headroom always comes from ``gains.baseline`` (the same predictor's no-text arm).
    Per-provider decision rows use that provider's own decision properties when the
    moderator table carries them. Constant or undersized inputs are reported, not dropped.
    """
    if gains.empty or moderators.empty:
        return pd.DataFrame(columns=ASSOCIATION_COLUMNS)
    per_dataset = moderators.drop(columns=["model_id", "baseline_headroom"], errors="ignore")
    per_dataset = per_dataset.drop_duplicates("dataset_id")
    names = [c for c in per_dataset if c not in {"dataset_id", "task_type"}
             and pd.api.types.is_numeric_dtype(per_dataset[c])]
    data = gains.merge(per_dataset, on="dataset_id", how="left", validate="many_to_one")
    data["baseline_headroom"] = 1.0 - data.baseline
    properties = moderators.attrs.get("decision_properties")
    if properties is not None and not properties.empty and "provider" in data:
        specific = data[["dataset_id", "provider"]].merge(
            properties, on=["dataset_id", "provider"], how="left")
        for column in DECISION_PROPERTIES:
            shared = data[column].to_numpy(float) if column in data else np.full(len(data), np.nan)
            data[column] = np.where(data.provider.notna(), specific[column].to_numpy(), shared)
            names = names if column in names else [*names, column]
    names = [*names, "baseline_headroom"]
    records = []
    for key, group in data.groupby(_REPRESENTATION_KEYS, dropna=False, sort=True):
        for moderator in names:
            pair = group[["dataset_id", "gain", moderator]].dropna()
            x, y = pair[moderator].to_numpy(float), pair.gain.to_numpy(float)
            rho = float(spearman_rows(x, y)[0])
            low = high = np.nan
            defined = 0
            if np.isfinite(rho) and n_boot > 0:
                # Fixed seed per cell: moderators over the same datasets share resamples.
                draws = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
                values = spearman_rows(x[draws], y[draws])
                values = values[np.isfinite(values)]
                defined = len(values)
                if defined:
                    low, high = (float(v) for v in np.percentile(values, [2.5, 97.5]))
            status = (
                "too_few_datasets" if len(x) < min_datasets
                else "constant_input" if not np.isfinite(rho)
                else "descriptive"
            )
            records.append(dict(zip(_REPRESENTATION_KEYS, key)) | {
                "moderator": moderator, "n_datasets": len(x),
                "spearman_rho": rho if status == "descriptive" else np.nan,
                "ci_low": low if status == "descriptive" else np.nan,
                "ci_high": high if status == "descriptive" else np.nan,
                "n_boot": n_boot, "defined_boot": defined, "seed": seed, "status": status,
                "dataset_ids": ", ".join(sorted(pair.dataset_id)),
            })
    return pd.DataFrame(records, columns=ASSOCIATION_COLUMNS)
