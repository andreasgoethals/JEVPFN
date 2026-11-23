"""Study-at-a-glance table for `general/5_synthesis` from the verified Group B/C exports.

Each prespecified arm is reduced per dataset to the mean gain over the seven predictors
(datasets are the unit), then summarised with `group_b.summarize`: the headroom-normalised
gain pools all 18 datasets; raw gains stay within a task family (ΔAUC N = 9, ΔR² N = 9).
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
    ("Label-trained classifier (E5-small)", "B2", "dataset_gains", "probe/e5_small_v2/per_column"),
    ("TAR E5-small, per-column PCA32", "B2", "dataset_gains", "tar/e5_small_v2_tar/per_column32"),
    ("Strands LoRA + head (C2)", "C2", "dataset_gains", "adapted/joint/lora_head"),
)
COLUMNS = ["arm", "scope", "statistic", "estimate", "low", "high", "datasets", "predictors"]


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
        predictors = int(level.predictors.min())
        for s in pooled.itertuples():
            rows.append([label, "pooled", "mean", s.mean_headroom_share, s.share_ci_low, s.share_ci_high,
                         s.datasets, predictors])
            rows.append([label, "pooled", "median", s.median_headroom_share, s.median_share_ci_low,
                         s.median_share_ci_high, s.datasets, predictors])
        for s in families.itertuples():
            rows.append([label, s.task_family, "mean", s.mean_raw_gain, s.raw_ci_low, s.raw_ci_high,
                         s.datasets, predictors])
            rows.append([label, s.task_family, "median", s.median_raw_gain, None, None, s.datasets,
                         predictors])
    return pd.DataFrame(rows, columns=COLUMNS)
