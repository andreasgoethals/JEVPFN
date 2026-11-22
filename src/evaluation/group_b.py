"""Group B analyses ("When do decision features improve tabular prediction?").

Pure functions over the compact fold tables of the per-experiment exports (`benchmark_export`);
nothing here fits a predictor. Common protocol: datasets are the unit. Every comparison pairs
an arm with the same predictor's baseline (non-text columns only) on identical rows and folds;
per dataset the five-fold mean gives the raw gain and its share of the baseline's headroom
(`raw gain / (1 - baseline)`); across datasets the equal-weight mean has a percentile
bootstrap interval over datasets, and win/tie/loss counts use a fixed tie band. Headline
metrics are fixed (binary ROC-AUC, multiclass macro one-vs-rest ROC-AUC, regression R2).

B3 adds label-free dataset properties (including decision redundancy), exploratory
model-level relations (n = 17 decision models), text-column refit importance and a
leave-one-dataset-out (LODO) rule that predicts each dataset's gain in advance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.endpoints import HEADLINES

#: |raw gain| at or below this (headline-metric units) counts as a tie. Fixed before results.
TIE_TOLERANCE = 0.001
N_BOOT = 2000
SEED = 20261120
HEADROOM_FLOOR = 1e-3

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


def fold_gains(rows: pd.DataFrame, baseline: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fold-level gain of every arm over the same predictor/split/fold baseline."""
    if rows.empty:
        return pd.DataFrame()
    base = rows if baseline is None else baseline
    base = base[base.representation.eq("baseline")][PAIR_KEYS + ["value"]]
    if base.duplicated(PAIR_KEYS).any():
        raise ValueError("Several baselines match one predictor, split and fold")
    arms = rows[~rows.representation.eq("baseline")]
    paired = arms.merge(base.rename(columns={"value": "baseline"}), on=PAIR_KEYS, how="inner")
    paired["raw_gain"] = paired.value - paired.baseline
    return paired


def dataset_gains(paired: pd.DataFrame, extra_keys=()) -> pd.DataFrame:
    """Per dataset x predictor x arm: five-fold mean raw gain and share of headroom."""
    if paired.empty:
        return pd.DataFrame()
    keys = ["dataset_id", "task_type", "model_id", "representation", *extra_keys]
    identity = [c for c in ("split_id", "checkpoint_id") if c in paired]
    mixed = paired.groupby(keys, dropna=False)[identity].nunique(dropna=False).gt(1).any(axis=1)
    if mixed.any():
        raise ValueError("A dataset gain must pair five folds of one split and checkpoint: "
                         f"{list(mixed[mixed].index[:3])}")
    out = paired.groupby(keys, as_index=False, dropna=False).agg(
        augmented=("value", "mean"), baseline=("baseline", "mean"),
        raw_gain=("raw_gain", "mean"), fold_variance=("raw_gain", "var"),
        folds=("outer_fold", "nunique"), pairs=("outer_fold", "size"),
        finite=("raw_gain", lambda v: int(np.isfinite(v.to_numpy(float)).sum())))
    if out.pairs.gt(out.folds).any():
        raise ValueError("Several fold records of one arm entered one dataset gain")
    # Exactly five finite paired differences; an undefined fold leaves the dataset incomplete.
    out = out[out.folds.eq(5) & out.finite.eq(5)].drop(columns=["pairs", "finite"]).copy()
    out["headroom"] = 1 - out.baseline
    out["headroom_share"] = out.raw_gain / np.maximum(out.headroom, HEADROOM_FLOOR)
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
        return pd.DataFrame()
    data = data[data.provider.isin(roster)]
    out = data.groupby(["dataset_id", "task_type", "model_id", "mode"], as_index=False).agg(
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


def _bootstrap_mean(values, rng, n_boot):
    values = np.asarray(values, float)
    if len(values) < 2:
        return np.nan, np.nan
    draws = rng.integers(0, len(values), (n_boot, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(gains: pd.DataFrame, *, keys=("model_id", "representation"), n_boot=N_BOOT,
              seed=SEED, tie=TIE_TOLERANCE) -> pd.DataFrame:
    """Equal-dataset means with dataset-bootstrap 95% intervals, and win/tie/loss counts."""
    if gains.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    for key, group in gains.groupby(list(keys), sort=True):
        raw, share = group.raw_gain.to_numpy(float), group.headroom_share.to_numpy(float)
        low, high = _bootstrap_mean(raw, rng, n_boot)
        share_low, share_high = _bootstrap_mean(share, rng, n_boot)
        rows.append(dict(zip(keys, key if isinstance(key, tuple) else (key,))) | dict(
            datasets=len(group), mean_raw_gain=float(raw.mean()), raw_ci_low=low, raw_ci_high=high,
            mean_headroom_share=float(share.mean()), share_ci_low=share_low,
            share_ci_high=share_high, wins=int((raw > tie).sum()),
            ties=int((np.abs(raw) <= tie).sum()), losses=int((raw < -tie).sum()),
            family=family(group.representation.iloc[0]) if "representation" in group else None))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- B3: text columns


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
        headline_q = q[q.metric.isin(["roc_auc", "roc_auc_ovr_macro", "spearman_expected_level",
                                      "spearman"]) | q.metric.str.startswith("spearman")]
        parts.append(headline_q.groupby(["provider", "dataset_id"]).value.mean()
                     .groupby("provider").mean().rename("standalone_quality"))
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
        parts.append(stability.groupby(["provider", "dataset"]).excess_tv.mean()
                     .groupby("provider").mean().rename("stability_excess_tv"))
    if profile is not None and not profile.empty and "parameters_billion" in profile:
        parts.append(np.log10(profile.set_index("provider").parameters_billion.astype(float))
                     .rename("log10_parameters_billion"))
    if not parts:
        return pd.DataFrame(columns=["provider"])
    return pd.concat(parts, axis=1).rename_axis("provider").reset_index()


def per_model_usefulness(gains: pd.DataFrame) -> pd.DataFrame:
    """Equal-dataset mean raw gain of each decision model, per predictor and mode."""
    data = gains[gains.family.eq("decision")].copy()
    if data.empty:
        return pd.DataFrame()
    parts = data.representation.str.split("/")
    data["mode"], data["provider"] = parts.str[1], parts.str[2]
    return data.groupby(["model_id", "mode", "provider"], as_index=False).agg(
        raw_gain=("raw_gain", "mean"), headroom_share=("headroom_share", "mean"),
        datasets=("dataset_id", "nunique"))


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
    data = outcome[[unit, value]].merge(properties[[unit, *features]], on=unit).dropna()
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
    return pd.DataFrame(rows)


def lodo_evaluation(predictions: pd.DataFrame, *, threshold=0.0, n_shuffles=2000, seed=SEED,
                    outcome=None, properties=None, features=None, alpha=1.0) -> dict:
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
        rho = spearmanr(table.predicted, table.actual).statistic
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
        null_rho, null_acc = [], []
        for _ in range(n_shuffles):
            shuffled = outcome.assign(raw_gain=rng.permutation(outcome.raw_gain.to_numpy()))
            table = lodo_predictions(shuffled, properties, features=features, alpha=alpha)
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


# ------------------------------------------------------------------ B3: redundancy


def decision_redundancy(non_text: pd.DataFrame, decision: np.ndarray, splits, *, task_type,
                        alpha=1.0) -> dict:
    """How well the non-text columns predict a decision output (label-free).

    A ridge regression (training-fold imputation, scaling and one-hot encoding; alpha fixed)
    is fitted on the training rows of each shared outer fold and predicts the decision
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
