"""Group B analyses ("When do decision features improve tabular prediction?").

Pure functions over the compact fold tables of the per-experiment exports (`benchmark_export`);
nothing here fits a predictor. Common protocol: datasets are the unit. Every comparison pairs
an arm with the same predictor's baseline (non-text columns only) on identical rows and folds;
per dataset the five-fold mean gives the raw gain and its share of the baseline's headroom
(`raw gain / (1 - baseline)`); across datasets the equal-weight mean has a percentile
bootstrap interval over datasets, and win/tie/loss counts use a fixed tie band. Headline
metrics are fixed (binary ROC-AUC, multiclass macro one-vs-rest ROC-AUC, regression R2).

B3 adds label-free dataset properties (including decision redundancy), exploratory
model-level relations (n = selected decision models), text-column refit importance and a
leave-one-dataset-out (LODO) rule that predicts each dataset's gain in advance. The rule runs
on three nested input sets (owner decision 23-11-2026): label-free properties only ("should I
acquire?"), plus redundancy on a fixed 200-row label-free pilot ("should I acquire after a
200-request pilot?") and plus redundancy on all rows ("should I use features I already have?").
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.evaluation.endpoints import HEADLINES

#: |raw gain| at or below this (headline-metric units) counts as a tie. Fixed before results.
TIE_TOLERANCE = 0.001
N_BOOT = 2000
SEED = 20261120

MODE_LABELS = {"non_text_plus_joint": "joint", "non_text_plus_per_column": "per_column",
               "non_text_plus_both": "both"}


# ------------------------------------------------------------------ representation labels


def representation(row) -> str:
    """One stable, human-readable arm label (no winner selection, every arm kept)."""
    fs, provider = row["feature_set"], row.get("provider")
    if fs == "non_text":
        return "baseline"
    if fs in MODE_LABELS:
        return f"decision/{MODE_LABELS[fs]}/{provider}"
    if fs == "non_text_plus_frozen_e5":
        return f"frozen/{row.get('encoder')}/{row.get('pca_scope')}{int(row.get('pca_components'))}"
    if fs == "non_text_plus_both_plus_frozen_e5":
        return f"decision+frozen/both/{provider}"
    if fs == "non_text_plus_tar":
        return f"tar/{row.get('encoder')}/{row.get('pca_scope')}{int(row.get('pca_components'))}"
    if fs == "non_text_plus_supervised_probe":
        return f"probe/{row.get('encoder')}/{row.get('probe_view')}"
    if fs == "non_text_plus_per_column_without_one_text":
        return f"without_text/{provider}/{int(row.get('dropped_text_column'))}"
    if fs == "original_features":
        return f"native/{row.get('model_id')}"
    # Group C (Strands only): C1 hidden states, C1b hidden + probabilities, C2 adapted.
    if fs == "non_text_plus_hidden":
        return f"hidden/{row.get('hidden_view')}/{row.get('hidden_position')}/L{int(row.get('hidden_layer'))}"
    if fs in ("non_text_plus_per_column_plus_hidden", "non_text_plus_joint_plus_hidden"):
        return (f"hidden+decision/{row.get('hidden_view')}/{row.get('hidden_position')}/"
                f"L{int(row.get('hidden_layer'))}/{provider}")
    if fs == "non_text_plus_adapted_joint":
        return f"adapted/joint/{row.get('adaptation_cell')}"
    raise ValueError(f"Unknown feature set {fs}")


def family(label: str) -> str:
    return label.split("/", 1)[0]


def headline(folds: pd.DataFrame) -> pd.DataFrame:
    """Fold rows of the fixed headline metric, with a representation label."""
    if folds.empty:
        return folds.assign(representation=pd.Series(dtype=str))
    # Export fold tables store labels as `category` (collect_folds, Parquet round trip): compare
    # and group on plain values so no unobserved category combination enters the analyses.
    folds = folds.astype({c: object for c in folds.columns
                          if isinstance(folds[c].dtype, pd.CategoricalDtype)})
    data = folds[folds.metric.eq(folds.task_type.map(HEADLINES))].copy()
    data["representation"] = [representation(r) for r in data.to_dict("records")]
    return data


# ------------------------------------------------------------------------ paired gains

PAIR_KEYS = ["dataset_id", "task_type", "model_id", "checkpoint_id", "split_id", "outer_fold"]


def validate_fold_identities(head: pd.DataFrame, *, extra=()) -> None:
    """Require one split per dataset and one checkpoint per predictor at a given size/draw."""
    if head.empty:
        return
    keys = ["dataset_id", *extra]
    identities = [("split_id", keys), ("checkpoint_id", [*keys, "model_id"])]
    if "pairing_identity" in head:
        identities.append(("pairing_identity", [*keys, "model_id"]))
    for column, by in identities:
        if head[column].isna().any() or head.groupby(by, dropna=False)[column].nunique().gt(1).any():
            raise ValueError(f"Cannot compare folds with missing or different {column} identities")
    if not head.outer_fold.isin(range(5)).all():
        raise ValueError("Expected outer folds numbered 0 to 4")


def fold_gains(rows: pd.DataFrame, baseline: pd.DataFrame | None = None, *, extra_keys=()) -> pd.DataFrame:
    """Fold-level gain of every arm over the same predictor/split/fold baseline."""
    if rows.empty:
        return rows.assign(baseline=pd.Series(dtype=float), raw_gain=pd.Series(dtype=float))
    keys = [*PAIR_KEYS, *extra_keys]
    base = rows if baseline is None else baseline
    base = base[base.representation.eq("baseline")][keys + ["value"]]
    if base.duplicated(keys).any():
        raise ValueError("Several baselines match one predictor, split and fold")
    arms = rows[~rows.representation.eq("baseline")]
    paired = arms.merge(base.rename(columns={"value": "baseline"}), on=keys, how="inner")
    paired["raw_gain"] = paired.value - paired.baseline
    return paired


def dataset_gains(paired: pd.DataFrame, extra_keys=()) -> pd.DataFrame:
    """Per dataset x predictor x arm: five-fold mean raw gain and share of headroom."""
    keys = ["dataset_id", "task_type", "model_id", "representation", *extra_keys]
    if paired.empty:
        return pd.DataFrame(columns=[*keys, "augmented", "baseline", "raw_gain", "fold_variance",
                                     "folds", "headroom", "headroom_share", "family"])
    identity = [c for c in ("split_id", "checkpoint_id", "pairing_identity") if c in paired]
    mixed = paired.groupby(keys, dropna=False)[identity].nunique(dropna=False).gt(1).any(axis=1)
    if mixed.any():
        raise ValueError("A dataset gain must pair five folds of one split and checkpoint: "
                         f"{list(mixed[mixed].index[:3])}")
    paired = paired.assign(_finite=np.isfinite(paired.raw_gain.to_numpy(float)))
    out = paired.groupby(keys, as_index=False, dropna=False).agg(
        augmented=("value", "mean"), baseline=("baseline", "mean"),
        raw_gain=("raw_gain", "mean"), fold_variance=("raw_gain", "var"),
        folds=("outer_fold", "nunique"), pairs=("outer_fold", "size"),
        finite=("_finite", "sum"), **{c: (c, "first") for c in identity})
    if out.pairs.gt(out.folds).any():
        raise ValueError("Several fold records of one arm entered one dataset gain")
    # Exactly five finite paired differences; an undefined fold leaves the dataset incomplete.
    out = out[out.folds.eq(5) & out.finite.eq(5)].drop(columns=["pairs", "finite"]).copy()
    out["headroom"] = 1 - out.baseline
    out["headroom_share"] = out.raw_gain / out.headroom.where(out.headroom > 0)
    out["family"] = out.representation.map(family)
    return out.reset_index(drop=True)


def _decision_modes(gains: pd.DataFrame) -> pd.DataFrame:
    data = gains[gains.family.isin(["decision", "decision+frozen"])].copy()
    if data.empty:
        return data
    parts = data.representation.str.split("/")
    data["mode"], data["provider"] = parts.str[:2].str.join("/"), parts.str[2]
    return data


def median_over_decision_models(gains: pd.DataFrame, *, roster) -> pd.DataFrame:
    """Per dataset, predictor and decision mode: median over ALL decision models of the
    configured roster (never a selected winner, never inferred from the results present);
    rows lacking any roster model on that dataset are dropped (see `roster_gaps`)."""
    data = _decision_modes(gains)
    roster = sorted(set(roster))
    if data.empty or not roster:
        return pd.DataFrame(columns=["dataset_id", "task_type", "model_id", "raw_gain", "headroom_share",
                                     "baseline", "decision_models", "representation", "family"])
    data = data[data.provider.isin(roster)]
    keys = ["dataset_id", "task_type", "model_id", "mode"]
    if data.duplicated([*keys, "provider"]).any():
        raise ValueError("Several gains for one member of the decision-model cohort")
    out = data.groupby(keys, as_index=False).agg(
        raw_gain=("raw_gain", "median"), headroom_share=("headroom_share", "median"),
        baseline=("baseline", "first"), decision_models=("provider", "nunique"))
    out = out[out.decision_models.eq(len(roster))].copy()
    out["representation"] = out["mode"] + "/median"
    out["family"] = out.representation.map(family)
    return out.drop(columns="mode").reset_index(drop=True)


def roster_gaps(gains: pd.DataFrame, *, roster) -> pd.DataFrame:
    """Dataset x predictor x decision mode cells whose median lacks configured models, by name."""
    data = _decision_modes(gains)
    columns = ["dataset_id", "task_type", "model_id", "mode", "decision_models",
               "missing_decision_models"]
    if data.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for key, group in data.groupby(["dataset_id", "task_type", "model_id", "mode"], sort=True):
        missing = sorted(set(roster) - set(group.provider))
        if missing:
            rows += [dict(zip(columns[:4], key), decision_models=int(group.provider.nunique()),
                          missing_decision_models=name) for name in missing]
    return pd.DataFrame(rows, columns=columns)


def _bootstrap_mean(values, rng, n_boot, statistic=np.mean):
    values = np.asarray(values, float)
    if len(values) < 2:
        return np.nan, np.nan
    draws = rng.integers(0, len(values), (n_boot, len(values)))
    means = statistic(values[draws], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(gains: pd.DataFrame, *, keys=("model_id", "representation"), n_boot=N_BOOT,
              seed=SEED, tie=TIE_TOLERANCE) -> pd.DataFrame:
    """Equal-dataset means and medians with dataset-bootstrap 95% intervals, and win/tie/loss counts."""
    if gains.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    for key, group in gains.groupby(list(keys), sort=True):
        raw, share = group.raw_gain.to_numpy(float), group.headroom_share.to_numpy(float)
        same_units = group.task_type.map(task_family).nunique() == 1
        low, high = _bootstrap_mean(raw, rng, n_boot) if same_units else (np.nan, np.nan)
        share_low, share_high = _bootstrap_mean(share, rng, n_boot)
        # The median sits next to the mean: a near-perfect baseline makes the headroom share
        # very large on one dataset, which moves the mean but not the median.
        median_low, median_high = _bootstrap_mean(share, rng, n_boot, np.median)
        rows.append(dict(zip(keys, key if isinstance(key, tuple) else (key,))) | dict(
            datasets=len(group), mean_raw_gain=float(raw.mean()) if same_units else np.nan, raw_ci_low=low, raw_ci_high=high,
            mean_headroom_share=float(share.mean()), share_ci_low=share_low,
            share_ci_high=share_high, median_headroom_share=float(np.median(share)),
            median_share_ci_low=median_low, median_share_ci_high=median_high,
            median_raw_gain=float(np.median(raw)) if same_units else np.nan, wins=int((raw > tie).sum()),
            ties=int((np.abs(raw) <= tie).sum()), losses=int((raw < -tie).sum()),
            family=family(group.representation.iloc[0]) if "representation" in group else None))
    return pd.DataFrame(rows)


def task_family(task_type: str) -> str:
    """classification (binary and multiclass, AUC) or regression (R^2): raw-gain units."""
    if task_type in {"binary_classification", "multiclass_classification"}:
        return "classification"
    if task_type == "regression":
        return "regression"
    raise ValueError(f"Unknown task type: {task_type}")


def summarize_by_family(gains: pd.DataFrame, *, keys=("model_id", "representation"), **kwargs) -> pd.DataFrame:
    """`summarize` within each task family, so raw AUC and R^2 gains are never averaged together.

    The headroom share remains comparable across families; pool it with `summarize` (N = 18).
    """
    if gains.empty:
        return pd.DataFrame()
    frame = gains.assign(task_family=gains.task_type.map(task_family))
    return summarize(frame, keys=(*keys, "task_family"), **kwargs)


# ------------------------------------------- normalised score (MulTaBench-style pooling)

NORMALIZED_COLUMNS = ["dataset_id", "task_type", "normalized_score", "folds", "tied_folds"]
NORMALIZED_SUMMARY = ["task_family", "datasets", "configurations", "mean_normalized_score",
                      "normalized_ci_low", "normalized_ci_high", "median_normalized_score",
                      "median_normalized_ci_low", "median_normalized_ci_high"]


def normalized_scores(scores: pd.DataFrame, *, keys=("model_id", "representation"), within=(),
                      clip_negative_r2=True) -> pd.DataFrame:
    """Per dataset and configuration: the min-max normalised headline score in [0, 1].

    MulTaBench (Arazi et al. 2026) puts classification AUC and regression R^2 on one scale by
    min-max normalising each dataset's scores across the compared configurations and then
    averaging over datasets. Here, per (dataset, outer fold) cell and comparison set:
    ``(score - min) / (max - min)`` over every configuration of that set, then the mean over the
    dataset's usable folds. A configuration is one combination of `keys` (default predictor x
    arm); `within` names extra keys that form separate comparison sets (e.g. ``("model_id",)`` to
    normalise each predictor's arms on their own).

    `scores` holds the raw headline metric in ``value`` (binary ROC-AUC, multiclass macro OvR
    ROC-AUC, regression R^2; `headline`) with ``dataset_id``, ``task_type`` and, when fold scores
    exist, ``outer_fold`` (otherwise each dataset is one cell). Rules, fixed before results:

    * negative R^2 is clipped at 0 before normalising (MulTaBench Tables 11/12): worse than the
      mean of the evaluated targets counts as "no skill", so one failed configuration cannot stretch the scale;
      AUC is never clipped;
    * the comparison set is every configuration present anywhere in the input for that set; a
      cell lacking a finite score for any of them is excluded for all (no configuration is
      normalised against a different set), so the normalisation set is the same on every cell;
    * a cell on which all configurations tie (max = min; e.g. every R^2 clipped to 0) carries no
      ordering and is excluded rather than given an arbitrary value; ``tied_folds`` counts them;
    * a dataset with no usable cell is absent from the result. Excluded cells are listed in
      ``attrs["excluded_cells"]`` (reason ``incomplete`` or ``all_tied``).

    The normalised score is RELATIVE: it depends on which configurations are compared (adding
    or removing one can change every other score), so it is a descriptive pooled view next to,
    never instead of, the per-task-family raw scores and gains. 1 = best compared configuration
    on that dataset, 0 = worst.
    """
    within = list(within)
    keys = [k for k in keys if k not in within]  # a set key identifies configurations already
    columns = ["dataset_id", "task_type", *within, *keys, "normalized_score", "folds", "tied_folds"]
    excluded_columns = ["dataset_id", *within, "outer_fold", "reason"]
    empty = pd.DataFrame(columns=columns)
    empty.attrs["excluded_cells"] = pd.DataFrame(columns=excluded_columns)
    if scores is None or scores.empty:
        return empty
    data = scores.astype({c: object for c in scores.columns
                          if isinstance(scores[c].dtype, pd.CategoricalDtype)})
    if "outer_fold" not in data:
        data = data.assign(outer_fold=0)
    if "split_id" in data or "checkpoint_id" in data:
        validate_fold_identities(data)
    cell = ["dataset_id", *within, "outer_fold"]
    if data.duplicated([*cell, *keys]).any():
        raise ValueError("Several scores for one configuration in one dataset and fold")
    if data.groupby("dataset_id").task_type.nunique().gt(1).any():
        raise ValueError("A dataset has several task types")
    value = data["value"].astype(float).to_numpy()
    if clip_negative_r2:
        value = np.where(data.task_type.eq("regression").to_numpy() & np.isfinite(value) & (value < 0), 0.0, value)
    data = data.assign(clipped=value)
    rows, excluded = [], []
    sets = data.groupby(within, dropna=False) if within else [((), data)]
    for _, part in sets:
        configurations = part[keys].drop_duplicates().reset_index(drop=True)
        n_config = len(configurations)
        for cell_key, group in part.groupby(cell, dropna=False):
            finite = group[np.isfinite(group.clipped.to_numpy(float))]
            if len(finite) < n_config or n_config < 2:
                excluded.append(dict(zip(cell, cell_key)) | {"reason": "incomplete"})
                continue
            low, high = finite.clipped.min(), finite.clipped.max()
            if not high > low:
                excluded.append(dict(zip(cell, cell_key)) | {"reason": "all_tied"})
                continue
            rows.append(finite.assign(normalized_score=(finite.clipped - low) / (high - low)))
    excluded = pd.DataFrame(excluded, columns=excluded_columns)
    if not rows:
        empty.attrs["excluded_cells"] = excluded
        return empty
    cells = pd.concat(rows, ignore_index=True)
    out = cells.groupby(["dataset_id", "task_type", *within, *keys], as_index=False, dropna=False).agg(
        normalized_score=("normalized_score", "mean"), folds=("outer_fold", "nunique"))
    tied = excluded[excluded.reason.eq("all_tied")].groupby(["dataset_id", *within]).size()
    if within:
        index = pd.MultiIndex.from_frame(out[["dataset_id", *within]])
        out["tied_folds"] = tied.reindex(index).fillna(0).astype(int).to_numpy()
    else:
        out["tied_folds"] = out.dataset_id.map(tied).fillna(0).astype(int)
    out = out[columns].reset_index(drop=True)
    out.attrs["excluded_cells"] = excluded
    return out


def summarize_normalized(per_dataset: pd.DataFrame, *, keys=("model_id", "representation"), within=(),
                         n_boot=N_BOOT, seed=SEED) -> pd.DataFrame:
    """Equal-dataset mean and median normalised score per configuration, pooled over all
    datasets (``task_family = "all"``) and within each task family, with 95% percentile
    intervals from resampling datasets (all configurations of a set share each resample).

    `per_dataset` is `normalized_scores` output. Only datasets scored for every configuration of
    the set enter (they all are, by construction of `normalized_scores`). The scores depend on
    the compared set; see `normalized_scores`.
    """
    within = list(within)
    keys = [k for k in keys if k not in within]  # a set key identifies configurations already
    columns = [*within, *keys, *NORMALIZED_SUMMARY]
    if per_dataset is None or per_dataset.empty:
        return pd.DataFrame(columns=columns)
    data = per_dataset.assign(task_family=per_dataset.task_type.map(task_family))
    rows = []
    sets = data.groupby(within, dropna=False) if within else [((), data)]
    for set_key, part in sets:
        set_key = set_key if isinstance(set_key, tuple) else (set_key,)
        for scope in ("all", "classification", "regression"):
            frame = part if scope == "all" else part[part.task_family.eq(scope)]
            if frame.empty:
                continue
            table = frame.pivot_table(index="dataset_id", columns=keys, values="normalized_score",
                                      aggfunc="first").dropna()
            if table.empty:
                continue
            values = table.to_numpy(float)
            n = len(values)
            rng = np.random.default_rng(seed)
            draws = rng.integers(0, n, (n_boot, n)) if n >= 2 else None
            for j, config in enumerate(table.columns):
                column = values[:, j]
                if draws is not None:
                    means, medians = column[draws].mean(axis=1), np.median(column[draws], axis=1)
                    ci = (*np.quantile(means, [0.025, 0.975]), *np.quantile(medians, [0.025, 0.975]))
                else:
                    ci = (np.nan,) * 4
                config = config if isinstance(config, tuple) else (config,)
                rows.append(dict(zip(within, set_key)) | dict(zip(keys, config)) | {
                    "task_family": scope, "datasets": n, "configurations": table.shape[1],
                    "mean_normalized_score": float(column.mean()), "normalized_ci_low": float(ci[0]),
                    "normalized_ci_high": float(ci[1]), "median_normalized_score": float(np.median(column)),
                    "median_normalized_ci_low": float(ci[2]), "median_normalized_ci_high": float(ci[3])})
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------- B4: text columns


def text_column_importance(head: pd.DataFrame) -> pd.DataFrame:
    """Per dataset x predictor x decision model x text column: per-column arm minus its refit
    without that column's block (fold-paired five-fold mean; positive = the column helps)."""
    full = head[head.representation.str.startswith("decision/per_column/")].copy()
    refit = head[head.representation.str.startswith("without_text/")].copy()
    if full.empty or refit.empty:
        return pd.DataFrame()
    full["provider"] = full.representation.str.split("/").str[2]
    refit["provider"] = refit.representation.str.split("/").str[1]
    refit["text_column"] = refit.representation.str.split("/").str[2].astype(int)
    keys = ["dataset_id", "task_type", "model_id", "checkpoint_id", "split_id", "outer_fold",
            "provider"]
    if full.duplicated(keys).any() or refit.duplicated([*keys, "text_column"]).any():
        raise ValueError("Several fold records of one arm entered the text-column importance")
    paired = refit.merge(full[keys + ["value"]].rename(columns={"value": "full"}), on=keys)
    paired["importance"] = paired.full - paired.value
    out = paired.groupby(["dataset_id", "task_type", "model_id", "provider", "text_column"],
                         as_index=False).agg(
        importance=("importance", "mean"), folds=("outer_fold", "nunique"),
        finite=("importance", lambda v: int(np.isfinite(v.to_numpy(float)).sum())))
    # Five finite fold-paired differences of the same split, or no importance.
    return out[out.folds.eq(5) & out.finite.eq(5)].drop(columns="finite").reset_index(drop=True)


# ------------------------------------------------------------- B3: associations, models


def associations(outcomes: pd.DataFrame, properties: pd.DataFrame, *, by=("model_id",),
                 unit="dataset_id", outcome="raw_gain", n_permutations=5000, seed=SEED,
                 min_units=5) -> pd.DataFrame:
    """Spearman rho of an outcome with every numeric property, permutation p-value and one
    Benjamini-Hochberg adjustment across ALL rows (no prespecified primary property)."""
    from src.evaluation.moderator_statistics import benjamini_hochberg, permutation_p
    from src.evaluation.moderators import spearman_rows

    numeric = [c for c in properties.columns if c != unit and pd.api.types.is_numeric_dtype(properties[c])]
    data = outcomes.merge(properties, on=unit, how="inner")
    rows = []
    for key, group in data.groupby(list(by), sort=True):
        for name in numeric:
            pair = group[[outcome, name]].dropna()
            row = dict(zip(by, key if isinstance(key, tuple) else (key,))) | dict(
                property=name, units=len(pair), spearman_rho=np.nan, p_value=np.nan)
            if len(pair) >= min_units and pair[name].nunique() > 1 and pair[outcome].nunique() > 1:
                row["spearman_rho"] = float(spearman_rows(pair[name].to_numpy(float),
                                                          pair[outcome].to_numpy(float))[0])
                row["p_value"] = permutation_p(pair[name], pair[outcome],
                                               n_permutations=n_permutations, seed=seed)
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_bh"] = benjamini_hochberg(out.p_value)
    return out


def model_diagnostics(quality=None, output_shape=None, consensus=None, profile=None,
                      stability=None) -> pd.DataFrame:
    """One row per decision model from the Group A exports (joint requests), equal dataset weight.

    stand-alone quality (AUC; regression Spearman), calibration (ECE), mean top probability,
    distance to the leave-one-out consensus, excess change over replay (stability) and size.
    Missing exports leave their columns missing.
    """
    parts = []

    def distinct(table):
        # A1 tables hold "rows" and "unique_inputs" weightings; each distinct input counts once.
        # Joint requests only (owner decision 20 November 2026), as A1 stand-alone quality.
        if "weighting" in table:
            table = table[table.weighting.eq("unique_inputs")]
        return table[table.representation.eq("joint")] if "representation" in table else table

    output_shape = None if output_shape is None else distinct(output_shape)
    consensus = None if consensus is None else distinct(consensus)
    if quality is not None and not quality.empty:
        q = quality.dropna(subset=["value"])
        for name, metrics in (("standalone_classification_auc", ("roc_auc", "roc_auc_ovr_macro")),
                              ("standalone_regression_spearman", ("spearman_expected_level", "spearman"))):
            scores = q[q.metric.isin(metrics)]
            if not scores.empty:
                parts.append(scores.groupby(["provider", "dataset_id"]).value.mean()
                             .groupby("provider").mean().rename(name))
        ece = q[q.metric.eq("ece")]
        if not ece.empty:
            parts.append(ece.groupby(["provider", "dataset_id"]).value.mean()
                         .groupby("provider").mean().rename("calibration_ece"))
    if output_shape is not None and not output_shape.empty and "mean_top_probability" in output_shape:
        parts.append(output_shape.groupby(["provider", "dataset_id"]).mean_top_probability.mean()
                     .groupby("provider").mean().rename("mean_top_probability"))
    if consensus is not None and not consensus.empty and "distance" in consensus:
        parts.append(consensus.groupby(["provider", "dataset_id"]).distance.mean()
                     .groupby("provider").mean().rename("consensus_distance"))
    if stability is not None and "mode" in stability:  # A2 mode_effects: joint requests only
        stability = stability[stability["mode"].eq("joint")]
    if stability is not None and not stability.empty and "excess_tv" in stability:
        if "variant" in stability:
            from src.evaluation.group_a import A2_VARIANT_FACTOR

            stability = (stability.assign(factor=stability.variant.map(A2_VARIANT_FACTOR))
                         .groupby(["provider", "dataset", "factor"], as_index=False).excess_tv.mean())
        parts.append(stability.groupby(["provider", "dataset"]).excess_tv.mean()
                     .groupby("provider").mean().rename("stability_excess_tv"))
    if profile is not None and not profile.empty and "parameters_billion" in profile:
        parts.append(np.log10(profile.set_index("provider").parameters_billion.astype(float))
                     .rename("log10_parameters_billion"))
    if not parts:
        return pd.DataFrame(columns=["provider"])
    return pd.concat(parts, axis=1).rename_axis("provider").reset_index()


def per_model_usefulness(gains: pd.DataFrame, *, datasets=None) -> pd.DataFrame:
    """Equal-dataset mean gain on the declared dataset roster, per provider, predictor and mode.

    Raw gain is in task-specific units (AUC or R^2), so it is averaged only within a task
    family (``task_family`` classification/regression). Rows with ``task_family == "all"``
    average the unit-free headroom share over all datasets and leave ``raw_gain`` empty.
    Incomplete outcomes keep their coverage counts and have no comparable mean.
    """
    data = gains[gains.family.eq("decision")].copy()
    if data.empty:
        return pd.DataFrame()
    if datasets is None:
        from src.data.loaders import load_catalog
        from src.data.selection import selected_entries
        from src.utils.config import load_yaml

        semantics = load_yaml("dataset_semantics")["datasets"]
        entries = selected_entries(load_catalog(), load_yaml("B1"))
        datasets = pd.DataFrame([{"dataset_id": e["id"], "task_type": semantics[e["id"]]["task_type"]}
                                 for e in entries])
    tasks = datasets.set_index("dataset_id", verify_integrity=True).task_type
    data = data[data.dataset_id.isin(tasks.index)].copy()
    if not data.task_type.eq(data.dataset_id.map(tasks)).all():
        raise ValueError("Decision gains disagree with the declared dataset task types")
    parts = data.representation.str.split("/")
    data["mode"], data["provider"] = parts.str[1], parts.str[2]
    data["task_family"] = data.task_type.map(task_family)
    keys = ["model_id", "mode", "provider"]
    if data.duplicated([*keys, "dataset_id"]).any():
        raise ValueError("Several usefulness scores for one decision model, predictor, mode and dataset")
    values = ["raw_gain", "headroom_share"]
    data[values] = data[values].replace([np.inf, -np.inf], np.nan)
    scoped = pd.concat([data, data.assign(task_family="all", raw_gain=np.nan)], ignore_index=True)
    out = scoped.groupby([*keys, "task_family"], as_index=False).agg(
        raw_gain=("raw_gain", "mean"), headroom_share=("headroom_share", "mean"),
        datasets=("dataset_id", "nunique"), raw_gain_datasets=("raw_gain", "count"),
        headroom_share_datasets=("headroom_share", "count"))
    expected = tasks.map(task_family).value_counts().to_dict() | {"all": len(tasks)}
    out["expected_datasets"] = out.task_family.map(expected)
    for value in ("raw_gain", "headroom_share"):
        out[value] = out[value].where(out[f"{value}_datasets"].eq(out.expected_datasets))
    return out


def _gain_associations(units, properties, *, unit, by, n_permutations, seed, min_units):
    """Raw gain within each family and headroom share across families, with one BH adjustment."""
    from src.evaluation.moderator_statistics import benjamini_hochberg

    parts = []
    for outcome, rows in (("raw_gain", units[units.task_family.ne("all")]),
                          ("headroom_share", units[units.task_family.eq("all")])):
        if rows.empty:
            continue
        out = associations(rows, properties, by=by, unit=unit,
                           outcome=outcome, n_permutations=n_permutations, seed=seed,
                           min_units=min_units)
        if not out.empty:
            parts.append(out.assign(outcome=outcome))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["q_value_bh"] = benjamini_hochberg(out.p_value)
    return out


def dataset_associations(medians, properties, *, n_permutations=5000, seed=SEED, min_units=5):
    """Relate each dataset property to comparable gains for each predictor and representation."""
    if medians.empty:
        return pd.DataFrame()
    units = pd.concat([medians.assign(task_family=medians.task_type.map(task_family)),
                       medians.assign(task_family="all")], ignore_index=True)
    return _gain_associations(units, properties, unit="dataset_id", by=("model_id", "representation", "task_family"),
                              n_permutations=n_permutations, seed=seed, min_units=min_units)


def model_relations(usefulness: pd.DataFrame, diagnostics: pd.DataFrame, *,
                    n_permutations=5000, seed=SEED, min_units=5) -> pd.DataFrame:
    """Relate comparable per-model usefulness to Group A diagnostics, with one BH adjustment."""
    if usefulness.empty or diagnostics is None or len(diagnostics.columns) <= 1:
        return pd.DataFrame()
    return _gain_associations(usefulness, diagnostics, unit="provider", by=("model_id", "mode", "task_family"),
                              n_permutations=n_permutations, seed=seed, min_units=min_units)


# ------------------------------------------------------------------ B3: predict in advance


def _ridge_fit_predict(x_train, y_train, x_test, alpha):
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    z = (x_train - mean) / scale
    zt = (x_test - mean) / scale
    y_mean = y_train.mean()
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y_train - y_mean))
    return zt @ coef + y_mean


def lodo_predictions(outcome: pd.DataFrame, properties: pd.DataFrame, *, features,
                     alpha=1.0, unit="dataset_id", value="raw_gain") -> pd.DataFrame:
    """Leave-one-dataset-out predicted gain from <= 2 prespecified label-free properties."""
    if len(features) > 2:
        raise ValueError("The prespecified LODO rule uses at most two properties")
    if value == "raw_gain" and "task_type" in outcome and outcome.task_type.map(task_family).nunique() > 1:
        raise ValueError("Raw AUC and R² gains cannot be pooled by a LODO rule")
    data = (outcome[[unit, value]].merge(properties[[unit, *features]], on=unit, validate="one_to_one")
            .replace([np.inf, -np.inf], np.nan).dropna())
    rows = []
    for i in range(len(data)):
        train = data.drop(index=data.index[i])
        test = data.iloc[[i]]
        if len(train) < 3:
            continue
        predicted = _ridge_fit_predict(train[features].to_numpy(float), train[value].to_numpy(float),
                                       test[features].to_numpy(float), alpha)[0]
        rows.append({unit: test[unit].iloc[0], "actual": float(test[value].iloc[0]),
                     "predicted": float(predicted)})
    return pd.DataFrame(rows, columns=[unit, "actual", "predicted"])


def lodo_evaluation(predictions: pd.DataFrame, *, threshold=0.0, n_shuffles=2000, seed=SEED,
                    outcome=None, properties=None, features=None, alpha=1.0, unit="dataset_id", value="raw_gain") -> dict:
    """Rank correlation, use/don't-use accuracy, regret against always/never and a shuffle null.

    Use the decision features when the predicted gain exceeds `threshold`. Realised gain of a
    policy = mean over datasets of (actual gain if used else 0); regret = oracle - realised.
    The null refits the whole LODO rule after shuffling gains across datasets.
    """
    from scipy.stats import spearmanr

    if len(predictions) < 3:
        return {"datasets": len(predictions), "status": "too_few_datasets"}

    def score(table):
        use = table.predicted > threshold
        helpful = table.actual > threshold
        rho = (spearmanr(table.predicted, table.actual).statistic
               if table.predicted.nunique() > 1 and table.actual.nunique() > 1 else np.nan)
        return float(rho), float((use == helpful).mean()), float(np.where(use, table.actual, 0).mean())

    rho, accuracy, realised = score(predictions)
    actual = predictions.actual.to_numpy(float)
    oracle = float(np.maximum(actual, 0).mean())
    result = dict(datasets=len(predictions), spearman_rho=rho, use_accuracy=accuracy,
                  realised_gain=realised, always_gain=float(actual.mean()), never_gain=0.0,
                  oracle_gain=oracle, regret=oracle - realised,
                  regret_always=oracle - float(actual.mean()), regret_never=oracle,
                  status="estimated" if np.isfinite(rho) else "rank_correlation_undefined")
    if outcome is not None and properties is not None and n_shuffles:
        rng = np.random.default_rng(seed)
        outcome = outcome.sort_values(unit, kind="stable")
        null_rho, null_acc = [], []
        for _ in range(n_shuffles):
            shuffled = outcome.assign(**{value: rng.permutation(outcome[value].to_numpy())})
            table = lodo_predictions(shuffled, properties, features=features, alpha=alpha, unit=unit, value=value)
            if len(table) >= 3:
                r, a, _ = score(table)
                null_rho.append(r)
                null_acc.append(a)
        null_rho, null_acc = np.asarray(null_rho, float), np.asarray(null_acc, float)
        valid = null_rho[np.isfinite(null_rho)]  # undefined null correlations are not exceedances
        result.update(shuffle_p_rho=float((1 + np.sum(valid >= rho)) / (1 + len(valid)))
                      if np.isfinite(rho) and len(valid) else np.nan,
                      shuffle_p_accuracy=float((1 + np.sum(null_acc >= accuracy)) / (1 + len(null_acc))),
                      shuffles=len(null_rho), valid_rho_shuffles=len(valid))
    return result


def lodo_input_sets(outcome: pd.DataFrame, properties: pd.DataFrame, *, input_sets, alpha=1.0,
                    threshold=0.0, n_shuffles=2000, seed=SEED, unit="dataset_id", value="raw_gain"):
    """The same LODO rule, outcome and shuffle null on several prespecified input sets.

    All sets use one common dataset cohort: datasets with the outcome and every property of
    every set defined, so the sets differ only in their inputs. Returns (predictions, summary);
    both carry `input_set` and its `claim`, and the summary lists excluded datasets.
    """
    needed = sorted({f for spec in input_sets.values() for f in spec["properties"]})
    missing = [f for f in needed if f not in properties]
    if missing:
        raise ValueError(f"Prespecified LODO properties are absent: {missing}")
    data = outcome[[unit, value]].merge(properties[[unit, *needed]], on=unit, how="outer", validate="one_to_one")
    complete = np.isfinite(data[[value, *needed]].to_numpy(dtype=float)).all(axis=1)
    cohort = set(data.loc[complete, unit])
    excluded = sorted(set(data[unit]) - cohort)
    outcome = outcome[outcome[unit].isin(cohort)]
    properties = properties[properties[unit].isin(cohort)]
    rows, summary = [], []
    for name, spec in input_sets.items():
        predictions = lodo_predictions(outcome, properties, features=spec["properties"], alpha=alpha,
                                       unit=unit, value=value)
        rows.append(predictions.assign(input_set=name, claim=spec["claim"], outcome=value))
        summary.append({"input_set": name, "claim": spec["claim"], "outcome": value,
                        "properties": ",".join(spec["properties"]),
                        "excluded_datasets": ",".join(excluded),
                        **lodo_evaluation(predictions, threshold=threshold, n_shuffles=n_shuffles,
                                          seed=seed, outcome=outcome, properties=properties,
                                          features=spec["properties"], alpha=alpha, unit=unit, value=value)})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summary)


# ------------------------------------------------------------------ B3: redundancy


def pilot_rows(n_rows: int, *, dataset_id: str, size: int = 200, seed: int = SEED) -> np.ndarray:
    """Fixed label-free pilot: the `size` row positions with the smallest seeded hash.

    The hash of (seed, dataset ID, row position in the loaded dataset) never reads a target,
    feature value or decision output, so the same rows would be requested before extraction.
    In a real deployment the pilot costs `size` decision requests per model.
    """
    keys = [hashlib.sha256(f"{seed}:{dataset_id}:{i}".encode()).digest() for i in range(n_rows)]
    order = sorted(range(n_rows), key=keys.__getitem__)
    return np.sort(np.asarray(order[:min(size, n_rows)], dtype=int))


def pilot_redundancy(non_text: pd.DataFrame, decision: np.ndarray, *, dataset_id, task_type,
                     size=200, seed=SEED, folds=5, alpha=1.0) -> dict:
    """`decision_redundancy` on the fixed pilot only, with `folds`-fold CV inside the pilot.

    Folds assign pilot rows by their position in hash order, so they are label-free too.
    """
    rows = pilot_rows(len(non_text), dataset_id=dataset_id, size=size, seed=seed)
    rank = np.argsort(np.argsort([hashlib.sha256(f"{seed}:{dataset_id}:{r}".encode()).digest()
                                  for r in rows]))
    splits = [{"train": [i for i, q in enumerate(rank) if q % folds != k],
               "test": [i for i, q in enumerate(rank) if q % folds == k]} for k in range(folds)]
    result = decision_redundancy(non_text.iloc[rows].reset_index(drop=True),
                                 np.asarray(decision, float)[rows], splits, task_type=task_type,
                                 alpha=alpha)
    # `rows` of the result counts the pilot rows that have a decision output; keep it apart
    # from the pilot size so a sparse-text dataset shows how many rows were scored.
    scored = {("pilot_scored_rows" if k == "rows" else f"pilot_{k}"): v for k, v in result.items()}
    return {"pilot_rows": len(rows), **scored}




def decision_redundancy(non_text: pd.DataFrame, decision: np.ndarray, splits, *, task_type,
                        alpha=1.0) -> dict:
    """How well the non-text columns predict a decision output (label-free).

    A ridge regression (training-fold imputation, scaling and one-hot encoding; alpha fixed)
    is fitted on the training rows of each label-independent fold and predicts the decision
    probabilities of its test rows. Only rows with a decision output enter. Returns the
    pooled out-of-fold R2 (variance-weighted over probability columns) and, for
    classification, the share of rows whose predicted top class equals the decision's.
    The target column is never read.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from src.evaluation.models import FoldInput

    decision = np.asarray(decision, float)
    available = np.isfinite(decision).all(axis=1)
    predicted = np.full(decision.shape, np.nan)
    for fold in splits:
        train = np.asarray([i for i in fold["train"] if available[i]])
        test = np.asarray([i for i in fold["test"] if available[i]])
        if len(train) < 10 or not len(test):
            continue
        encoder = FoldInput().fit(non_text.iloc[train])
        x_train = encoder.transform(non_text.iloc[train], strings=True)
        x_test = encoder.transform(non_text.iloc[test], strings=True)
        numeric = encoder.numeric_ or ([] if encoder.categorical_ else
                                       ["__constant_no_non_text_features__"])
        model = make_pipeline(
            ColumnTransformer([
                ("numeric", make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler()),
                 numeric),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), encoder.categorical_),
            ]),
            Ridge(alpha=alpha),
        ).fit(x_train, decision[train])
        predicted[test] = np.asarray(model.predict(x_test)).reshape(len(test), -1)
    rows = available & np.isfinite(predicted).all(axis=1)
    if rows.sum() < 10:
        return {"rows": int(rows.sum()), "r2": np.nan, "top_agreement": np.nan}
    truth, guess = decision[rows], predicted[rows]
    residual = ((truth - guess) ** 2).sum()
    total = ((truth - truth.mean(axis=0)) ** 2).sum()
    result = {"rows": int(rows.sum()), "r2": float(1 - residual / total) if total > 0 else np.nan}
    if task_type == "binary_classification":
        result["top_agreement"] = float(((truth[:, 0] > 0.5) == (guess[:, 0] > 0.5)).mean())
    else:
        result["top_agreement"] = float((truth.argmax(axis=1) == guess.argmax(axis=1)).mean())
    return result
