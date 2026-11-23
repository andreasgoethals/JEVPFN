"""Derived Group B tables for the B notebooks' figures (descriptive; nothing is fitted here).

Every function reads the tables of one `benchmark_export` ZIP (or several) and returns a tidy
frame that one figure in `src/visualize/` draws. No result selects a winner: decision models
enter either one by one or as the per-dataset median over the whole configured roster
(`benchmark_export.decision_roster`). Units follow `group_b`: raw gains stay inside one task
family (classification: ROC-AUC or macro AUC; regression: R^2) and only the headroom share is
pooled over all 18 datasets. Dataset intervals resample datasets (percentile bootstrap);
fold-level win rates use `fold_wins` (dataset-cluster bootstrap), with each predictor's five
folds as separate cells of the same dataset when predictors are pooled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import group_b

MODES = ("per_column", "joint", "both")
N_BOOT = group_b.N_BOOT
SEED = group_b.SEED
WIN_RESAMPLES = 2000
#: Arm order of every Group B figure: no text, decision, frozen, label-trained, native.
FAMILY_ORDER = ("baseline", "decision", "decision+frozen", "frozen", "probe", "tar", "native")


def with_family(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(task_family=frame.task_type.map(group_b.task_family))


def decision_rows(gains: pd.DataFrame, *, family="decision") -> pd.DataFrame:
    """Single-model decision rows (no median rows) with `mode` and `provider` columns."""
    data = gains[gains.family.eq(family) & ~gains.representation.str.endswith("/median")].copy()
    parts = data.representation.str.split("/")
    data["mode"], data["provider"] = parts.str[1], parts.str[2]
    return data


def text_column_counts() -> dict[str, int]:
    """Text columns per selected dataset (catalog metadata; label-free)."""
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries
    from src.utils.config import load_yaml

    return {e["id"]: int(e["paper_summary"]["text_columns"])
            for e in selected_entries(load_catalog(), load_yaml("experiment_1/main"))}


def dataset_rows() -> dict[str, int]:
    """Rows per selected dataset (catalog metadata)."""
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries
    from src.utils.config import load_yaml

    return {e["id"]: int(e["paper_summary"]["rows"])
            for e in selected_entries(load_catalog(), load_yaml("experiment_1/main"))}


def _spread(frame, keys, outcome):
    """Median over predictors with the predictor range (min, max)."""
    return with_family(frame).groupby([*keys, "task_type", "task_family"], as_index=False).agg(
        value=(outcome, "median"), low=(outcome, "min"), high=(outcome, "max"),
        predictors=("model_id", "nunique"))


# ----------------------------------------------------------------------- B1: per model


def model_gains(gains: pd.DataFrame, *, roster, outcome="raw_gain", modes=MODES) -> pd.DataFrame:
    """Per dataset x decision model x view: median gain over predictors and its range."""
    data = decision_rows(gains)
    data = data[data.provider.isin(roster) & data["mode"].isin(modes)]
    return _spread(data, ["dataset_id", "provider", "mode"], outcome)


def median_model_gains(medians: pd.DataFrame, *, outcome="raw_gain") -> pd.DataFrame:
    """Per dataset x view: the median decision model's gain, median over predictors, range."""
    data = medians[medians.representation.str.startswith("decision/")].copy()
    data["mode"] = data.representation.str.split("/").str[1]
    return _spread(data, ["dataset_id", "mode"], outcome)


def _bootstrap(values, *, n_boot=N_BOOT, seed=SEED, statistic=np.mean):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    draws = np.random.default_rng(seed).integers(0, len(values), (n_boot, len(values)))
    stats = statistic(values[draws], axis=1)
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def model_ranks(gains: pd.DataFrame, *, roster, mode="both", outcome="raw_gain", n_boot=N_BOOT,
                seed=SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank of every decision model within each dataset (1 = largest gain).

    Ranks are taken within dataset x predictor (ties share the mean rank) over cells where the
    whole roster is present, then averaged over predictors. Returns (per dataset ranks, per
    model mean rank over datasets with a dataset-bootstrap 95% interval).
    """
    data = decision_rows(gains)
    data = data[data["mode"].eq(mode) & data.provider.isin(roster)]
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    complete = data.groupby(["dataset_id", "model_id"]).provider.transform("nunique").eq(len(set(roster)))
    data = data[complete].copy()
    data["rank"] = data.groupby(["dataset_id", "model_id"])[outcome].rank(ascending=False, method="average")
    per = with_family(data).groupby(["dataset_id", "task_type", "task_family", "provider"],
                                    as_index=False).agg(rank=("rank", "mean"),
                                                        predictors=("model_id", "nunique"))
    rows = []
    for provider, group in per.groupby("provider"):
        low, high = _bootstrap(group["rank"], n_boot=n_boot, seed=seed)
        rows.append({"provider": provider, "mean_rank": float(group["rank"].mean()),
                     "median_rank": float(group["rank"].median()), "rank_ci_low": low,
                     "rank_ci_high": high, "datasets": int(group.dataset_id.nunique())})
    summary = pd.DataFrame(rows).sort_values(["mean_rank", "provider"]).reset_index(drop=True)
    return per, summary


def predictor_rank_agreement(gains: pd.DataFrame, *, roster, mode="both", outcome="raw_gain"):
    """Kendall tau_b between predictors of the decision-model ranking (mean within-dataset rank).

    Returns a long table (predictor_a, predictor_b, tau_b, models) over complete roster cells.
    """
    from scipy.stats import kendalltau

    data = decision_rows(gains)
    data = data[data["mode"].eq(mode) & data.provider.isin(roster)].copy()
    if data.empty:
        return pd.DataFrame(columns=["predictor_a", "predictor_b", "tau_b", "models", "datasets"])
    complete = data.groupby(["dataset_id", "model_id"]).provider.transform("nunique").eq(len(set(roster)))
    data = data[complete].copy()
    data["rank"] = data.groupby(["dataset_id", "model_id"])[outcome].rank(ascending=False, method="average")
    # Only datasets every predictor ranked, so all predictors rank on the same datasets.
    shared = data.groupby("dataset_id").model_id.nunique()
    data = data[data.dataset_id.isin(shared[shared.eq(data.model_id.nunique())].index)]
    table = data.pivot_table(index="provider", columns="model_id", values="rank", aggfunc="mean")
    rows = []
    for a in table.columns:
        for b in table.columns:
            tau = kendalltau(table[a], table[b]).statistic if a != b else 1.0
            rows.append({"predictor_a": a, "predictor_b": b, "tau_b": float(tau),
                         "models": len(table), "datasets": int(data.dataset_id.nunique())})
    return pd.DataFrame(rows)


def view_contrasts(gains: pd.DataFrame, *, roster, text_columns: dict) -> pd.DataFrame:
    """Per dataset x decision model: joint - per column and both - per column (raw gain, median
    over predictors of the fold-mean difference). Only datasets with >= 2 text columns; with
    one text column the three views are the same block by construction."""
    data = decision_rows(gains)
    data = data[data.provider.isin(roster)]
    data = data[data.dataset_id.map(text_columns).fillna(0).ge(2)]
    if data.empty:
        return pd.DataFrame()
    wide = data.pivot_table(index=["dataset_id", "task_type", "model_id", "provider"], columns="mode",
                            values="raw_gain").reset_index()
    parts = []
    for view in ("joint", "both"):
        if view in wide and "per_column" in wide:
            parts.append(wide.assign(contrast=f"{view}_minus_per_column",
                                     difference=wide[view] - wide["per_column"]))
    long = pd.concat(parts, ignore_index=True).dropna(subset=["difference"])
    return _spread(long, ["dataset_id", "provider", "contrast"], "difference")


# ----------------------------------------------------------------- fold-level win rates


def fold_long(folds: pd.DataFrame) -> pd.DataFrame:
    """Headline fold scores with a representation label (categories made plain)."""
    return group_b.headline(folds)


def median_fold_scores(head: pd.DataFrame, *, roster, modes=MODES, family="decision") -> pd.DataFrame:
    """Fold-level score of the median decision model: per dataset x predictor x fold, the median
    over ALL roster models' fold scores (cells lacking any roster model are dropped)."""
    data = head[head.representation.str.startswith(f"{family}/")].copy()
    parts = data.representation.str.split("/")
    data["mode"], data["provider"] = parts.str[1], parts.str[2]
    data = data[data.provider.isin(roster) & data["mode"].isin(modes)]
    if data.empty:
        return data
    keys = ["dataset_id", "task_type", "model_id", "checkpoint_id", "split_id", "outer_fold", "mode"]
    out = data.groupby(keys, as_index=False).agg(value=("value", "median"), models=("provider", "nunique"))
    out = out[out.models.eq(len(set(roster)))]
    out["representation"] = f"{family}/" + out["mode"] + "/median"
    return out.drop(columns=["mode", "models"])


def _scores(head, learners, *, model=None):
    """Long (dataset, cell, learner, score) frame; pooled predictors become separate cells."""
    data = head[head.representation.isin(learners)]
    if model is not None:
        data = data[data.model_id.eq(model)]
    return pd.DataFrame({"dataset_id": data.dataset_id.astype(str),
                         "outer_fold": data.model_id.astype(str) + ":" + data.outer_fold.astype(str),
                         "learner": data.representation, "score": data.value.astype(float)})


def fold_win_rates(head: pd.DataFrame, learners, *, versus="baseline", by_predictor=True,
                   resamples=WIN_RESAMPLES, seed=0) -> pd.DataFrame:
    """Fold-level win rate of each learner against `versus` (higher headline = better).

    One row per learner and predictor (``model_id``) and one pooled row (``model_id = "all"``)
    whose cells are dataset x predictor x fold; every interval resamples datasets.
    """
    from src.evaluation import fold_wins

    rows = []
    models = sorted(set(head.model_id)) if by_predictor else []
    for model in [*models, None]:
        frame = _scores(head, [*learners, versus], model=model)
        for learner in learners:
            if learner not in set(frame.learner) or versus not in set(frame.learner):
                continue
            try:
                result = fold_wins.fold_win_rate(frame, learner=learner, versus=versus,
                                                 higher_is_better=True, resamples=resamples, seed=seed)
            except ValueError:
                continue
            rows.append({"model_id": model or "all", **result})
    return pd.DataFrame(rows)


def pairwise_win_rates(head: pd.DataFrame, learners, *, resamples=WIN_RESAMPLES, seed=0) -> pd.DataFrame:
    """Fold-level win rate of every ordered learner pair, pooled over predictors."""
    from src.evaluation import fold_wins

    frame = _scores(head, learners)
    rows = []
    for a in learners:
        for b in learners:
            if a == b or not {a, b} <= set(frame.learner):
                continue
            try:
                rows.append(fold_wins.fold_win_rate(frame, learner=a, versus=b, higher_is_better=True,
                                                    resamples=resamples, seed=seed))
            except ValueError:
                continue
    return pd.DataFrame(rows)


def model_pama(head: pd.DataFrame, *, roster, mode="both", resamples=WIN_RESAMPLES, seed=0) -> pd.DataFrame:
    """PAMA over the decision models of one view, cells = dataset x predictor x fold."""
    from src.evaluation import fold_wins

    learners = [f"decision/{mode}/{p}" for p in roster]
    frame = _scores(head, learners)
    if frame.empty:
        return pd.DataFrame()
    present = [learner for learner in learners if learner in set(frame.learner)]
    table = fold_wins.pama(frame, higher_is_better=True, learners=present, resamples=resamples, seed=seed)
    return table.assign(provider=table.learner.str.split("/").str[2])


# --------------------------------------------------------------------- B2: paired views


def combined_gains(frames: dict) -> pd.DataFrame:
    """Single-arm dataset gains plus the median-over-models rows of one export."""
    parts = [frames.get("dataset_gains"), frames.get("median_gains")]
    parts = [p for p in parts if p is not None and not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def paired_gains(gains: pd.DataFrame, *, x, y, outcome="raw_gain") -> pd.DataFrame:
    """Per dataset: arm `y` against arm `x` (median over predictors, with predictor ranges),
    pairing the two arms within each predictor first."""
    keep = gains[gains.representation.isin([x, y])]
    wide = keep.pivot_table(index=["dataset_id", "task_type", "model_id"], columns="representation",
                            values=outcome).reset_index()
    if x not in wide or y not in wide:
        return pd.DataFrame()
    wide = wide.dropna(subset=[x, y])
    out = with_family(wide).groupby(["dataset_id", "task_type", "task_family"], as_index=False).agg(
        x=(x, "median"), x_low=(x, "min"), x_high=(x, "max"),
        y=(y, "median"), y_low=(y, "min"), y_high=(y, "max"), predictors=("model_id", "nunique"))
    return out.assign(x_representation=x, y_representation=y)


def arm_widths(folds: pd.DataFrame, *, roster) -> pd.DataFrame:
    """Added representation columns per dataset and arm (median over predictors and folds);
    median-model rows take the median width over the roster."""
    data = folds[folds.metric.astype(str).eq("columns.representation")]
    if data.empty:
        return pd.DataFrame(columns=["dataset_id", "representation", "columns"])
    data = data.astype({c: object for c in data.columns if isinstance(data[c].dtype, pd.CategoricalDtype)})
    data = data.assign(representation=[group_b.representation(r) for r in data.to_dict("records")])
    widths = data.groupby(["dataset_id", "representation"], as_index=False).value.median()
    widths = widths.rename(columns={"value": "columns"})
    medians = []
    for family in ("decision", "decision+frozen"):
        part = widths[widths.representation.str.startswith(f"{family}/")].copy()
        if part.empty:
            continue
        bits = part.representation.str.split("/")
        part = part[bits.str[2].isin(roster)]
        part["representation"] = family + "/" + bits.str[1] + "/median"
        medians.append(part.groupby(["dataset_id", "representation"], as_index=False)["columns"].median())
    return pd.concat([widths, *medians], ignore_index=True)


def compactness(gains: pd.DataFrame, widths: pd.DataFrame, representations, *, outcome="raw_gain"):
    """Per dataset x arm: median gain over predictors against the number of added columns."""
    data = gains[gains.representation.isin(representations)]
    if data.empty or widths.empty:
        return pd.DataFrame()
    out = _spread(data, ["dataset_id", "representation"], outcome)
    return out.merge(widths, on=["dataset_id", "representation"], how="inner")


def family_summaries(gains: pd.DataFrame, representations, *, n_boot=N_BOOT) -> pd.DataFrame:
    """Raw gain per task family (`summarize_by_family`) and the pooled headroom share
    (`summarize`, all datasets; ``task_family = "all"``), per predictor and arm."""
    data = gains[gains.representation.isin(representations)]
    if data.empty:
        return pd.DataFrame()
    by_family = group_b.summarize_by_family(data, n_boot=n_boot)
    pooled = group_b.summarize(data, n_boot=n_boot).assign(task_family="all")
    return pd.concat([by_family, pooled], ignore_index=True)


# --------------------------------------------------------------------------- B3


def headroom_table(medians: pd.DataFrame, *, representation="decision/both/median") -> pd.DataFrame:
    """Per dataset: the median model's raw gain and the baseline headroom (1 - baseline),
    median over predictors with ranges."""
    data = medians[medians.representation.eq(representation)].assign(
        headroom=lambda t: 1 - t.baseline)
    gain = _spread(data, ["dataset_id"], "raw_gain")
    room = _spread(data, ["dataset_id"], "headroom")[["dataset_id", "value", "low", "high"]]
    return gain.merge(room.rename(columns={"value": "headroom", "low": "headroom_low",
                                           "high": "headroom_high"}), on="dataset_id")


DIAGNOSTICS = ("standalone_quality", "calibration_ece", "consensus_distance", "stability_excess_tv")


def model_usefulness_relations(gains: pd.DataFrame, diagnostics: pd.DataFrame, *, roster, mode="both",
                               outcome="headroom_share", n_permutations=5000, seed=SEED):
    """Per decision model: equal-dataset mean headroom share (all 18 datasets; mean over
    predictors) beside each Group A diagnostic, with Spearman rho, permutation p and one BH
    adjustment over the diagnostics shown (n = models). Returns (table, relations)."""
    data = decision_rows(gains)
    data = data[data["mode"].eq(mode) & data.provider.isin(roster)]
    if data.empty or diagnostics is None or diagnostics.empty:
        return pd.DataFrame(), pd.DataFrame()
    per_predictor = data.groupby(["provider", "model_id"], as_index=False)[outcome].mean()
    usefulness = per_predictor.groupby("provider", as_index=False).agg(
        usefulness=(outcome, "mean"), low=(outcome, "min"), high=(outcome, "max"))
    keep = ["provider", *[c for c in DIAGNOSTICS if c in diagnostics]]
    table = usefulness.merge(diagnostics[keep], on="provider", how="left")
    relations = group_b.associations(
        table[["provider", "usefulness"]].assign(group="models").rename(columns={"provider": "unit"}),
        table[keep].rename(columns={"provider": "unit"}), by=("group",), unit="unit",
        outcome="usefulness", n_permutations=n_permutations, seed=seed, min_units=5)
    return table, relations


def column_refits(importance: pd.DataFrame, *, field_types: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per dataset x text column: median over decision models and predictors of the per-column
    arm minus its refit without that column (headline units; positive = the column helps)."""
    if importance is None or importance.empty:
        return pd.DataFrame()
    per = importance.groupby(["dataset_id", "task_type", "text_column", "model_id"],
                             as_index=False).importance.median()
    out = _spread(per, ["dataset_id", "text_column"], "importance")
    if field_types is not None and not field_types.empty:
        types = field_types.assign(text_column=field_types.groupby("dataset_id").cumcount())
        out = out.merge(types[["dataset_id", "text_column", "field", "field_type"]],
                        on=["dataset_id", "text_column"], how="left")
    return out


#: Row cohort axes kept in B3 (novelty and prediction-shift cohorts are not part of B3).
COHORT_AXES = ("availability", "length", "baseline_entropy", "decision_confidence", "decision_agreement")


def cohort_benefits(balanced: pd.DataFrame) -> pd.DataFrame:
    """Equal-dataset mean row benefit per task family, decision view, axis and cohort (mean over
    predictors); only the pre-specified B3 axes."""
    if balanced is None or balanced.empty:
        return pd.DataFrame()
    data = balanced[balanced.axis.isin(COHORT_AXES) & balanced.phase.eq("experiment_1")].copy()
    data["task_family"] = data.task.map(group_b.task_family)
    data["mode"] = data.features.map(group_b.MODE_LABELS)
    data = data.dropna(subset=["mode"])
    return data.groupby(["task_family", "mode", "axis", "cohort"], as_index=False).agg(
        mean_benefit=("mean_benefit", "mean"), low=("mean_benefit", "min"),
        high=("mean_benefit", "max"), predictors=("model", "nunique"), datasets=("datasets", "max"))


# --------------------------------------------------------------------------- B4


def size_draw_gains(head: pd.DataFrame, *, roster) -> pd.DataFrame:
    """Per dataset x predictor x arm x size x draw: five-fold mean gain over the same-size,
    same-draw baseline, plus the median decision model (median over the whole roster)."""
    keys = ["dataset_id", "task_type", "model_id", "training_size", "draw", "outer_fold"]
    data = head.dropna(subset=["training_size"]).copy()
    base = data[data.representation.eq("baseline")][keys + ["value"]].rename(columns={"value": "baseline"})
    data = data[~data.representation.eq("baseline")].merge(base, on=keys, how="inner")
    data["raw_gain"] = data.value - data.baseline
    data["headroom_share"] = data.raw_gain / np.maximum(1 - data.baseline, group_b.HEADROOM_FLOOR)
    per = data.groupby(["dataset_id", "task_type", "model_id", "representation", "training_size", "draw"],
                       as_index=False).agg(raw_gain=("raw_gain", "mean"),
                                           headroom_share=("headroom_share", "mean"),
                                           folds=("outer_fold", "nunique"))
    per = per[per.folds.eq(5)].drop(columns="folds")
    decision = per[per.representation.str.startswith("decision/")]
    parts = decision.representation.str.split("/")
    decision = decision[parts.str[2].isin(roster)].assign(mode=parts.str[1])
    median = decision.groupby(["dataset_id", "task_type", "model_id", "training_size", "draw", "mode"],
                              as_index=False).agg(raw_gain=("raw_gain", "median"),
                                                  headroom_share=("headroom_share", "median"),
                                                  models=("representation", "nunique"))
    median = median[median.models.eq(len(set(roster)))]
    median["representation"] = "decision/" + median["mode"] + "/median"
    return pd.concat([per, median.drop(columns=["mode", "models"])], ignore_index=True)


def _over_draws(per_draw):
    out = per_draw.groupby(["dataset_id", "task_type", "model_id", "representation", "training_size"],
                           as_index=False).agg(
        raw_gain=("raw_gain", "mean"), headroom_share=("headroom_share", "mean"),
        draws=("draw", "nunique"), draw_variance=("raw_gain", "var"),
        headroom_share_draw_variance=("headroom_share", "var"))
    out["log2_size"] = np.log2(out.training_size)
    return out


def size_curves(head: pd.DataFrame, *, roster, curves, representations, outcome="raw_gain"):
    """B4 curves per task family: DerSimonian-Laird pooled gain over datasets per size, for each
    predictor and for the mean over predictors (``model_id = "all"``: per dataset and draw the
    mean over the seven predictors, so its draw variance is that of the average). Each curve
    keeps one fixed dataset cohort (`data_efficiency.cohort_gains`)."""
    from src.evaluation import data_efficiency

    per_draw = size_draw_gains(head, roster=roster)
    per_draw = per_draw[per_draw.representation.isin(representations)]
    if per_draw.empty:
        return pd.DataFrame()
    counts = per_draw.groupby(["dataset_id", "representation", "training_size", "draw"]).model_id.transform("nunique")
    averaged = (per_draw[counts.eq(per_draw.model_id.nunique())]
                .groupby(["dataset_id", "task_type", "representation", "training_size", "draw"], as_index=False)
                [["raw_gain", "headroom_share"]].mean().assign(model_id="all"))
    sizes = _over_draws(pd.concat([per_draw, averaged], ignore_index=True))
    cohorts = data_efficiency.cohort_gains(sizes, curves)
    parts = []
    for family, group in with_family(cohorts).groupby("task_family"):
        pooled = data_efficiency.pooled_curve(group.drop(columns="task_family"), outcome=outcome)
        parts.append(pooled.assign(task_family=family))
    return pd.concat(parts, ignore_index=True)


def size_slopes(head: pd.DataFrame, *, roster, representations, max_size=2048, outcome="raw_gain"):
    """Per dataset x arm: least-squares slope of the predictor-averaged gain per doubling of
    labelled rows over the main sizes (<= `max_size`), with the predictor range of slopes."""
    from src.evaluation import data_efficiency

    per_draw = size_draw_gains(head, roster=roster)
    per_draw = per_draw[per_draw.representation.isin(representations) & per_draw.training_size.le(max_size)]
    if per_draw.empty:
        return pd.DataFrame()
    sizes = _over_draws(per_draw)
    slopes = data_efficiency.gain_slopes(sizes, outcome=outcome)
    return _spread(slopes, ["dataset_id", "representation"], "slope_per_doubling")


# --------------------------------------------------------------------------- B5


def native_scores(head: pd.DataFrame, *, roster) -> pd.DataFrame:
    """Per dataset: five-fold mean headline score of each native system, each predictor's
    no-text baseline and each predictor's median decision model (both view, absolute score)."""
    if head.empty:
        return pd.DataFrame()
    fold = pd.concat([head, median_fold_scores(head, roster=roster, modes=("both",))], ignore_index=True)
    keep = fold[fold.representation.str.startswith("native/")
                | fold.representation.isin(["baseline", "decision/both/median"])]
    out = keep.groupby(["dataset_id", "task_type", "model_id", "representation"], as_index=False).agg(
        value=("value", "mean"), folds=("outer_fold", "nunique"))
    return with_family(out[out.folds.eq(5)])


# --------------------------------------------------------------------- B synthesis


LADDER = (
    "decision/per_column/median", "decision/joint/median", "decision/both/median",
    "decision+frozen/both/median",
    "frozen/e5_small_v2/per_column32", "frozen/e5_small_v2/joint64",
    "frozen/e5_large_v2/per_column32", "frozen/e5_large_v2/joint64",
    "frozen/qwen3_embedding_0_6b/per_column32", "frozen/qwen3_embedding_0_6b/joint64",
    "probe/e5_small_v2/per_column", "probe/e5_small_v2/joint",
    "probe/e5_large_v2/per_column", "probe/e5_large_v2/joint",
    "probe/qwen3_embedding_0_6b/per_column", "probe/qwen3_embedding_0_6b/joint",
    "tar/e5_small_v2_tar/per_column32", "tar/e5_small_v2_tar/joint64",
)


def native_gains(head: pd.DataFrame) -> pd.DataFrame:
    """Native system minus the median no-text baseline over the seven predictors, per dataset
    (native systems share no predictor; ``model_id = "native"``)."""
    per = head.groupby(["dataset_id", "task_type", "model_id", "representation"], as_index=False).agg(
        value=("value", "mean"), folds=("outer_fold", "nunique"))
    per = per[per.folds.eq(5)]
    base = per[per.representation.eq("baseline")].groupby("dataset_id").value.median().rename("baseline")
    native = per[per.representation.str.startswith("native/")].merge(base.reset_index(), on="dataset_id")
    if native.empty:
        return pd.DataFrame()
    native = native.assign(raw_gain=native.value - native.baseline, model_id="native")
    native["headroom_share"] = native.raw_gain / np.maximum(1 - native.baseline, group_b.HEADROOM_FLOOR)
    native["family"] = "native"
    return native.drop(columns=["value", "folds"])


def ladder(b1: dict, b2: dict, b5: dict, *, n_boot=N_BOOT) -> pd.DataFrame:
    """Every main arm's per-family raw gain and pooled headroom share (`family_summaries`),
    from the B1, B2 and B5 exports that exist (absent exports contribute nothing)."""
    parts = []
    if b1:
        parts.append(b1["median_gains"])
    if b2:
        parts += [b2["dataset_gains"], b2["median_gains"][b2["median_gains"].representation.eq(
            "decision+frozen/both/median")]]
    if b5:
        parts.append(native_gains(group_b.headline(b5["folds"])))
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame()
    gains = pd.concat(parts, ignore_index=True)
    arms = [*LADDER, *sorted(r for r in set(gains.representation) if r.startswith("native/"))]
    return family_summaries(gains, arms, n_boot=n_boot)


MAIN_ARMS = ("baseline", "decision/both/median", "decision+frozen/both/median",
             "frozen/e5_small_v2/per_column32", "probe/e5_small_v2/per_column",
             "tar/e5_small_v2_tar/per_column32")


def main_arm_fold_scores(b1: dict, b2: dict, *, roster) -> pd.DataFrame:
    """Fold scores of the main arms (median decision models at the fold level)."""
    heads = []
    if b2:
        heads.append(group_b.headline(b2["folds"]))
    elif b1:
        heads.append(group_b.headline(b1["folds"]))
    if not heads:
        return pd.DataFrame()
    head = heads[0]
    medians = [median_fold_scores(head, roster=roster, modes=("both",))]
    medians.append(median_fold_scores(head, roster=roster, modes=("both",), family="decision+frozen"))
    head = pd.concat([head, *medians], ignore_index=True)
    return head[head.representation.isin(MAIN_ARMS)]


def compute_costs(folds: pd.DataFrame, *, rows: dict, extraction: pd.DataFrame | None = None,
                  roster=()) -> pd.DataFrame:
    """Per arm: predictor fit hours per 1,000 training rows (median over datasets, predictors and
    folds of `seconds.fit`) plus decision extraction GPU-hours per 1,000 rows where measured
    (`model_properties.compute_per_thousand`); unmeasured costs stay missing."""
    fit = folds[folds.metric.astype(str).eq("seconds.fit")]
    if fit.empty:
        return pd.DataFrame()
    fit = fit.astype({c: object for c in fit.columns if isinstance(fit[c].dtype, pd.CategoricalDtype)})
    fit = fit.assign(representation=[group_b.representation(r) for r in fit.to_dict("records")])
    training = fit.dataset_id.map(rows).astype(float) * 0.8
    fit = fit.assign(hours_per_1000=fit.value.astype(float) / training * 1000 / 3600)
    out = fit.groupby("representation", as_index=False).hours_per_1000.median().rename(
        columns={"hours_per_1000": "fit_hours_per_1000_rows"})
    out["extraction_gpu_hours_per_1000_rows"] = np.nan
    if extraction is not None and not extraction.empty:
        lookup = extraction.set_index("provider").gpu_hours_per_1000_requests
        provider = out.representation.str.split("/").str[2]
        decision = out.representation.str.startswith("decision/")
        out.loc[decision, "extraction_gpu_hours_per_1000_rows"] = provider[decision].map(lookup).to_numpy()
    out["total_hours_per_1000_rows"] = out.fit_hours_per_1000_rows + out.extraction_gpu_hours_per_1000_rows
    medians = []
    for family in ("decision", "decision+frozen"):
        part = out[out.representation.str.startswith(f"{family}/")].copy()
        bits = part.representation.str.split("/")
        part = part[bits.str[2].isin(roster)]
        if part.empty:
            continue
        part["representation"] = family + "/" + bits.str[1] + "/median"
        medians.append(part.groupby("representation", as_index=False).median(numeric_only=True))
    return pd.concat([out, *medians], ignore_index=True)


def pareto_points(summaries, costs: pd.DataFrame) -> pd.DataFrame:
    """Per arm: mean headroom share (median over predictors, with range) beside its compute cost.

    `summaries`: `ladder` output and/or a `group_b.summarize` table (single decision models);
    only pooled (all-dataset) rows are used.
    """
    frames = [s for s in summaries if s is not None and not s.empty]
    if not frames or costs is None or costs.empty:
        return pd.DataFrame()
    pooled = pd.concat([f[f.task_family.eq("all")] if "task_family" in f else f for f in frames],
                       ignore_index=True).drop_duplicates(["model_id", "representation"])
    points = pooled.groupby("representation", as_index=False).agg(
        value=("mean_headroom_share", "median"), low=("mean_headroom_share", "min"),
        high=("mean_headroom_share", "max"))
    return points.merge(costs, on="representation", how="left")
