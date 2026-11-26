"""Study-at-a-glance table for `general/4_synthesis` from the verified Group B/C exports.

Each prespecified arm is reduced per dataset to the mean gain over the seven predictors
(datasets are the unit), then summarised with `group_b.summarize`: the headroom-normalised
gain pools all 18 datasets; raw gains stay within a task family (ΔAUC N = 9, ΔR² N = 9).
When the fold tables are present, a MulTaBench-style normalised score (`group_b.normalized_scores`:
per dataset and fold, min-max over every predictor x glance arm plus the no-text baseline; negative
R² clipped at 0) is added as a second pooled scope; it is relative to that comparison set.
No arm is selected by its outcome: the Strands row is the prespecified LoRA + head cell.
"""

from __future__ import annotations

import pandas as pd

from src.evaluation import group_b

#: (label, export study, table, representation) in the display order of the figure.
GLANCE_ARMS = (
    ("Decision, median model (both)", "B1", "median_gains", "decision/both/median"),
    ("Decision + E5, median model", "B2", "median_gains", "decision+frozen/both/median"),
    ("Frozen E5-small, per-column PCA32", "B2", "dataset_gains", "frozen/e5_small_v2/per_column32"),
    ("Label-trained classifier (E5-small)", "B3", "dataset_gains", "probe/e5_small_v2/per_column"),
    ("TAR E5-small, per-column PCA32", "B3", "dataset_gains", "tar/e5_small_v2_tar/per_column32"),
    ("Strands LoRA + head (C2)", "C2", "dataset_gains", "adapted/joint/lora_head"),
)
COLUMNS = ["arm", "scope", "statistic", "estimate", "low", "high", "datasets", "predictors", "predictors_max"]
#: Reference row of the normalised scope (drawn as a line, not as an arm).
NO_TEXT = "No text (same predictor)"


def _dataset_level(gains: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset: mean over the predictors present (their count is kept)."""
    return gains.groupby(["dataset_id", "task_type"], as_index=False).agg(
        raw_gain=("raw_gain", "mean"), headroom_share=("headroom_share", "mean"),
        predictors=("model_id", "nunique"))


def glance_table(frames_by_study: dict, *, arms=GLANCE_ARMS, n_boot=group_b.N_BOOT) -> pd.DataFrame:
    """Mean and median with dataset-bootstrap 95% intervals per arm and scope (pooled share,
    raw gain per task family). Arms whose export is absent are omitted."""
    rows = []
    for label, study, table, representation in arms:
        frame = (frames_by_study.get(study) or {}).get(table)
        if frame is None or frame.empty or "representation" not in frame:
            continue
        gains = frame[frame.representation.eq(representation)]
        if gains.empty:
            continue
        level = _dataset_level(gains).assign(representation=label)
        pooled = group_b.summarize(level, keys=("representation",), n_boot=n_boot)
        families = group_b.summarize_by_family(level, keys=("representation",), n_boot=n_boot)
        predictors = (int(level.predictors.min()), int(level.predictors.max()))
        for s in pooled.itertuples():
            rows.append([label, "pooled", "mean", s.mean_headroom_share, s.share_ci_low, s.share_ci_high,
                         s.datasets, *predictors])
            rows.append([label, "pooled", "median", s.median_headroom_share, s.median_share_ci_low,
                         s.median_share_ci_high, s.datasets, *predictors])
        for s in families.itertuples():
            counts = level.loc[level.task_type.map(group_b.task_family).eq(s.task_family), "predictors"]
            predictors = (int(counts.min()), int(counts.max()))
            rows.append([label, s.task_family, "mean", s.mean_raw_gain, s.raw_ci_low, s.raw_ci_high,
                         s.datasets, *predictors])
            rows.append([label, s.task_family, "median", s.median_raw_gain, None, None, s.datasets,
                         *predictors])
    rows += normalized_rows(frames_by_study, arms=arms, n_boot=n_boot)
    return pd.DataFrame(rows, columns=COLUMNS)


def normalized_rows(frames_by_study: dict, *, arms=GLANCE_ARMS, n_boot=group_b.N_BOOT, roster=None) -> list:
    """Glance rows of scope ``normalized``: per arm (and the no-text baseline), the equal-dataset
    mean and median of the predictor-averaged normalised score with dataset-bootstrap intervals.

    The comparison set is every predictor x (glance arm present, no-text baseline); each arm's fold
    scores come from its own export (baseline from B1). Empty without the B1 fold table.
    """
    from src.evaluation import group_b_views

    b1 = frames_by_study.get("B1") or {}
    if b1.get("folds") is None or b1["folds"].empty:
        return []
    if roster is None:
        from src.evaluation.benchmark_export import decision_roster

        roster = decision_roster()
    labels = {"baseline": NO_TEXT}
    sources = {"B1": b1["folds"]}
    for label, study, _, representation in arms:
        folds = (frames_by_study.get(study) or {}).get("folds")
        if folds is not None and not folds.empty:
            sources[study] = folds
            labels[representation] = label
    # Overlapping B1, B2, B3 and C2 scores must refer to the same split, predictor and recipe.
    head = group_b_views.merged_heads(*sources.values())
    parts = [head[head.representation.isin(labels)]]
    for representation in labels:
        if representation.endswith("/median"):
            family, mode, _ = representation.split("/")
            parts.append(group_b_views.median_fold_scores(head, roster=roster, modes=(mode,), family=family))
    fold = pd.concat(parts, ignore_index=True)
    if fold.representation.nunique() < 2:
        return []
    per_dataset = group_b.normalized_scores(fold)
    level = per_dataset.groupby(["dataset_id", "task_type", "representation"], as_index=False).agg(
        normalized_score=("normalized_score", "mean"), predictors=("model_id", "nunique"))
    summary = group_b.summarize_normalized(level, keys=("representation",), n_boot=n_boot)
    rows = []
    for s in summary[summary.task_family.eq("all")].itertuples():
        counts = level.loc[level.representation.eq(s.representation), "predictors"]
        predictors = (int(counts.min()), int(counts.max()))
        label = labels[s.representation]
        rows.append([label, "normalized", "mean", s.mean_normalized_score, s.normalized_ci_low,
                     s.normalized_ci_high, s.datasets, *predictors])
        rows.append([label, "normalized", "median", s.median_normalized_score, s.median_normalized_ci_low,
                     s.median_normalized_ci_high, s.datasets, *predictors])
    return rows
