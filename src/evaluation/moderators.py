"""RQ2-E dataset-level moderators of the headroom-normalised gain; descriptive only.

Covariates are label-free dataset properties (size, text-field composition, text length,
missing and repeated text, task type and number of classes) plus the predictor's own
no-text headroom, 1 − baseline headline metric. Optional decision-output properties come
from RQ1A compact comparison tables when present; otherwise they are reported as pending.

``moderator_associations`` computes, per predictor × representation, Spearman's rank
correlation between per-dataset normalised gain (``aggregate.dataset_gains``) and each
moderator across datasets, with a dataset-bootstrap percentile interval. Datasets carry
equal weight. With about eighteen datasets these are associations, never causal claims or
a basis for selecting a winning recipe.

Optional model-level covariates (``model_covariates``: verified parameter count, architecture
family and readout type from `config/02_feature_creation/model_properties.yaml`) describe the
decision models rather than the datasets. ``model_level_associations`` relates them to each
provider's equal-dataset mean gain across decision models, per predictor and decision feature
set (``model_gain_table`` ranks the decision models by that mean). When Exp 2 compact
robustness summaries exist, ``model_robustness`` adds each model's replay noise and
presentation sensitivity as further covariates; otherwise they are pending. Size, training
data, readout and robustness are confounded across releases, so these too are descriptive.

Every association is reported for the normalised gain and the raw (unnormalised) gain:
normalisation divides by headroom, so normalised gain against ``baseline_headroom`` is
partly mechanical, while raw gain against headroom shows the ceiling effect directly.
Prespecified primary moderators, FDR control, meta-analysis and partial correlations live
in ``moderator_statistics``.
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
    "model_id", "checkpoint_id", "family", "aggregation", "representation", "outcome",
    "moderator",
    "n_datasets", "spearman_rho", "ci_low", "ci_high", "n_boot", "defined_boot", "seed",
    "status", "dataset_ids",
]
_REPRESENTATION_KEYS = ["model_id", "checkpoint_id", "family", "aggregation", "representation"]
#: Model-level covariates (one value per decision model), all optional.
MODEL_COVARIATES = ("log10_parameters", "architecture_family", "readout_type")
MODEL_ASSOCIATION_COLUMNS = [
    "model_id", "checkpoint_id", "feature_set", "covariate", "level", "statistic", "value",
    "n_providers", "ci_low", "ci_high", "n_boot", "defined_boot", "seed", "status", "providers",
]


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
    # Joint row text: a row whose combined text occurs in no other row is unseen in every
    # training partition, whatever the split (a fold-free, label-free novelty share).
    joint = (frame[text_columns].map(text_value).astype(object)
             .apply(lambda row: None if row.isna().all() else
                    "\x1f".join("" if v is None or v != v else str(v) for v in row), axis=1)
             if text_columns else pd.Series(dtype=object))
    joint = joint.dropna()
    unique_rows = int((~joint.duplicated(keep=False)).sum())
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
        "unseen_text_share": unique_rows / len(joint) if len(joint) else np.nan,
    }
    for kind in (*FIELD_TYPES, "empty"):
        row[f"share_{kind}"] = (
            sum(t == kind for t in types) / len(types) if types else np.nan
        )
    row["free_text_share"] = (row["share_long_text"] + row["share_short_text"]
                              if types else np.nan)
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


def moderator_data(gains, moderators):
    """Gains joined to dataset moderators; returns ``(data, moderator names)``.

    Headroom always comes from ``gains.baseline`` (the same predictor's no-text arm).
    Per-provider decision rows use that provider's own decision properties when the
    moderator table carries them.
    """
    per_dataset = moderators.drop(columns=["model_id", "baseline_headroom"], errors="ignore")
    per_dataset = per_dataset.drop_duplicates("dataset_id")
    names = [c for c in per_dataset if c not in {"dataset_id", "task_type"}
             and pd.api.types.is_numeric_dtype(per_dataset[c])]
    data = gains.merge(per_dataset, on="dataset_id", how="left", validate="many_to_one")
    data["baseline_headroom"] = 1.0 - data.baseline
    if "raw_gain" not in data:
        data["raw_gain"] = data.augmented - data.baseline
    properties = moderators.attrs.get("decision_properties")
    if properties is not None and not properties.empty and "provider" in data:
        specific = data[["dataset_id", "provider"]].merge(
            properties, on=["dataset_id", "provider"], how="left")
        for column in DECISION_PROPERTIES:
            shared = data[column].to_numpy(float) if column in data else np.full(len(data), np.nan)
            data[column] = np.where(data.provider.notna(), specific[column].to_numpy(), shared)
            names = names if column in names else [*names, column]
    return data, [*names, "baseline_headroom"]


def bootstrap_spearman(x, y, *, n_boot, seed):
    """Spearman rho with a percentile bootstrap over paired units; ``(rho, low, high, defined)``."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = float(spearman_rows(x, y)[0]) if len(x) else np.nan
    low = high = np.nan
    defined = 0
    if np.isfinite(rho) and n_boot > 0:
        # Fixed seed per cell: moderators over the same units share resamples.
        draws = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
        values = spearman_rows(x[draws], y[draws])
        values = values[np.isfinite(values)]
        defined = len(values)
        if defined:
            low, high = (float(v) for v in np.percentile(values, [2.5, 97.5]))
    return rho, low, high, defined


def moderator_associations(gains, moderators, *, n_boot=2000, seed=0, min_datasets=5,
                           outcomes=("gain", "raw_gain")):
    """Spearman rho per predictor × representation × outcome × moderator, dataset bootstrap.

    ``outcome`` is the headroom-normalised ``gain`` or the unnormalised ``raw_gain``
    (augmented − baseline headline metric). Both are reported because normalising by
    headroom couples the gain to ``baseline_headroom`` by construction; raw gain against
    headroom shows the ceiling effect directly. Constant or undersized inputs are
    reported, not dropped.
    """
    if gains.empty or moderators.empty:
        return pd.DataFrame(columns=ASSOCIATION_COLUMNS)
    data, names = moderator_data(gains, moderators)
    records = []
    for key, group in data.groupby(_REPRESENTATION_KEYS, dropna=False, sort=True):
        for outcome in outcomes:
            for moderator in names:
                pair = group[["dataset_id", outcome, moderator]].dropna()
                x, y = pair[moderator].to_numpy(float), pair[outcome].to_numpy(float)
                rho, low, high, defined = bootstrap_spearman(x, y, n_boot=n_boot, seed=seed)
                status = (
                    "too_few_datasets" if len(x) < min_datasets
                    else "constant_input" if not np.isfinite(rho)
                    else "descriptive"
                )
                ok = status == "descriptive"
                records.append(dict(zip(_REPRESENTATION_KEYS, key)) | {
                    "outcome": outcome, "moderator": moderator, "n_datasets": len(x),
                    "spearman_rho": rho if ok else np.nan,
                    "ci_low": low if ok else np.nan, "ci_high": high if ok else np.nan,
                    "n_boot": n_boot, "defined_boot": defined, "seed": seed, "status": status,
                    "dataset_ids": ", ".join(sorted(pair.dataset_id)),
                })
    return pd.DataFrame(records, columns=ASSOCIATION_COLUMNS)


def model_covariates(properties=None):
    """Verified model-level covariates per decision provider, or an empty pending table.

    Returns ``(table, status)`` with columns ``provider`` and ``MODEL_COVARIATES``; an
    unverified value is missing, never estimated.
    """
    columns = ["provider", *MODEL_COVARIATES]
    try:
        from src.evaluation.model_properties import covariates

        table = covariates(properties)
    except (FileNotFoundError, KeyError, ValueError) as error:
        return pd.DataFrame(columns=columns), f"pending: {error}"
    return table[columns], "available"


ROBUSTNESS_COVARIATES = ("replay_noise_tv", "sensitivity_excess_tv")
NUMERIC_MODEL_COVARIATES = ("log10_parameters", *ROBUSTNESS_COVARIATES)
MODEL_GAIN_COLUMNS = [
    "model_id", "checkpoint_id", "feature_set", "provider", "n_datasets", "mean_gain",
    "mean_raw_gain", "rank", "dataset_ids",
]


def model_robustness(effects=None):
    """Per decision model, Exp 2 replay noise and presentation sensitivity, or pending.

    From the compact RQ1B summaries (``study_reports.robustness_tables``; validation pilots
    excluded): ``replay_noise_tv`` is the equal-dataset mean total variation between unchanged
    repeats (``repeat_1``/``repeat_2``); ``sensitivity_excess_tv`` is the equal-dataset mean,
    over the controlled presentation variants, of their matched excess over replay noise.
    Returns ``(table, status)``.
    """
    columns = ["provider", *ROBUSTNESS_COVARIATES, "robustness_datasets"]
    if effects is None:
        from src.evaluation.study_reports import robustness_tables

        try:
            effects, _ = robustness_tables()
        except (OSError, ValueError, KeyError) as error:
            return pd.DataFrame(columns=columns), f"pending: {error}"
    if effects is None or effects.empty or not {"dataset", "provider", "variant"} <= set(effects):
        return pd.DataFrame(columns=columns), "pending: no completed Exp 2 robustness summaries"
    repeats = effects[effects.variant.isin(["repeat_1", "repeat_2"])]
    noise = (repeats.groupby(["provider", "dataset"]).mean_tv.mean().groupby("provider").mean()
             .rename("replay_noise_tv"))
    perturbed = effects[~effects.variant.isin(["repeat_1", "repeat_2"])]
    if "excess_tv" in perturbed:
        excess = (perturbed.dropna(subset=["excess_tv"]).groupby(["provider", "dataset"])
                  .excess_tv.mean().groupby("provider").mean().rename("sensitivity_excess_tv"))
    else:
        excess = pd.Series(dtype=float, name="sensitivity_excess_tv")
    datasets = effects.groupby("provider").dataset.nunique().rename("robustness_datasets")
    table = pd.concat([noise, excess, datasets], axis=1).reset_index().rename(
        columns={"index": "provider"})
    for column in columns:
        if column not in table:
            table[column] = np.nan
    return table[columns], f"available: {len(table)} decision models"


def model_gain_table(gains):
    """Per predictor × decision feature set × decision model: equal-dataset mean gains.

    Only datasets observed for every decision model of the cell enter, so all models are
    ranked on the same datasets. ``mean_raw_gain`` is the unnormalised headline difference.
    Rank 1 is the largest mean normalised gain. Descriptive; no model is selected.
    """
    if gains.empty:
        return pd.DataFrame(columns=MODEL_GAIN_COLUMNS)
    per_model = gains[gains.family.eq("decision") & gains.provider.notna()]
    if per_model.empty:
        return pd.DataFrame(columns=MODEL_GAIN_COLUMNS)
    if "raw_gain" not in per_model:
        raw = (per_model.augmented - per_model.baseline
               if {"augmented", "baseline"} <= set(per_model) else np.nan)
        per_model = per_model.assign(raw_gain=raw)
    keys = ["model_id", "checkpoint_id", "feature_set"]
    tables = []
    for key, cell in per_model.groupby(keys, dropna=False, sort=True):
        providers = cell.provider.nunique()
        observed = cell.groupby("dataset_id").provider.nunique()
        shared = sorted(observed.index[observed.eq(providers)])
        table = (cell[cell.dataset_id.isin(shared)].groupby("provider")
                 .agg(mean_gain=("gain", "mean"), mean_raw_gain=("raw_gain", "mean"),
                      n_datasets=("dataset_id", "nunique")).reset_index())
        table = table.assign(**dict(zip(keys, key)), dataset_ids=", ".join(shared))
        table["rank"] = table.mean_gain.rank(ascending=False, method="min")
        tables.append(table)
    result = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return result[MODEL_GAIN_COLUMNS].sort_values([*keys, "rank"], kind="stable").reset_index(drop=True)


def model_level_associations(gains, covariates=None, *, robustness=None, n_boot=2000, seed=0,
                             min_providers=5):
    """Decision-model covariates against each provider's equal-dataset mean gain.

    Per predictor × decision feature set, over ``model_gain_table`` (shared datasets only):
    Spearman rho of mean normalised gain with each numeric covariate (``log10_parameters``
    and, when Exp 2 summaries exist, ``replay_noise_tv`` and ``sensitivity_excess_tv``), with
    a percentile bootstrap over decision models; and the mean gain per architecture family and
    readout type (``level``). Undersized or constant inputs are reported, not dropped. Size,
    data, readout and robustness are confounded across releases: descriptive only.
    """
    if covariates is None:
        covariates, _ = model_covariates()
    table = model_gain_table(gains)
    if table.empty or covariates.empty:
        return pd.DataFrame(columns=MODEL_ASSOCIATION_COLUMNS)
    if robustness is not None and not robustness.empty:
        covariates = covariates.merge(robustness, on="provider", how="left")
    numeric = [c for c in NUMERIC_MODEL_COVARIATES if c in covariates]
    records = []
    keys = ["model_id", "checkpoint_id", "feature_set"]
    for key, cell in table.groupby(keys, dropna=False, sort=True):
        means = cell[["provider", "mean_gain"]].rename(columns={"mean_gain": "gain"}).merge(
            covariates, on="provider", how="left")
        base = dict(zip(keys, key)) | {"n_boot": n_boot, "seed": seed}
        for covariate in numeric:
            pair = means[["provider", "gain", covariate]].dropna()
            x, y = pair[covariate].to_numpy(float), pair.gain.to_numpy(float)
            rho, low, high, defined = bootstrap_spearman(x, y, n_boot=n_boot, seed=seed)
            status = ("too_few_providers" if len(x) < min_providers
                      else "constant_input" if not np.isfinite(rho) else "descriptive")
            ok = status == "descriptive"
            records.append(base | {
                "covariate": covariate, "level": None, "statistic": "spearman_rho",
                "value": rho if ok else np.nan, "n_providers": len(x),
                "ci_low": low if ok else np.nan, "ci_high": high if ok else np.nan,
                "defined_boot": defined, "status": status,
                "providers": ", ".join(sorted(pair.provider)),
            })
        for covariate in ("architecture_family", "readout_type"):
            if covariate not in means:
                continue
            for level, group in means.dropna(subset=[covariate]).groupby(covariate, sort=True):
                records.append(base | {
                    "covariate": covariate, "level": level, "statistic": "mean_gain",
                    "value": float(group.gain.mean()), "n_providers": len(group),
                    "ci_low": np.nan, "ci_high": np.nan, "defined_boot": 0,
                    "status": "descriptive",
                    "providers": ", ".join(sorted(group.provider)),
                })
    return pd.DataFrame(records, columns=MODEL_ASSOCIATION_COLUMNS)
