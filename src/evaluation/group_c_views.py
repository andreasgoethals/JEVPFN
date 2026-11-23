"""Group C report views: pure functions over the verified C1/C2 export tables.

Kept apart from `src.evaluation.group_c` so the export code, and with it the export's
analysis identity, is unchanged. Nothing here reads a store, fits a model or alters a
computation of the export; it regroups the exported per-dataset gains for the notebooks:

* raw gains are summarised WITHIN a task family (classification ΔAUC, regression ΔR²), never
  averaged across families (`group_b.summarize_by_family`);
* the "mean over predictors" row averages, per dataset and arm, the seven predictors' five-fold
  gains (only where every predictor of that arm is present), so its interval still resamples
  datasets;
* the C1 head-reproduction contract and the C2 fit diagnostics are exposed as tables for the
  study audit (`general/4_study_audit`).

References (Strands' own probabilities = B1 Strands arm, TAR, the B2 label-trained classifier)
are read from the Group C exports themselves, which carry those B arms on identical folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import group_b

PREDICTOR_MEAN = "predictor_mean"
ZERO_SHOT = "decision/joint/strands"
CELLS = {"zero_shot": ZERO_SHOT, "head_only": "adapted/joint/head_only",
         "lora_only": "adapted/joint/lora_only", "lora_head": "adapted/joint/lora_head"}
TAR = ("tar/e5_small_v2_tar/per_column32", "tar/e5_small_v2_tar/joint64")
PROBES = ("probe/e5_small_v2/joint", "probe/e5_large_v2/joint", "probe/qwen3_embedding_0_6b/joint")
#: The Strands ladder of C_synthesis: label-free first, then trained, then label-trained references.
LADDER = (ZERO_SHOT, "hidden/joint/answer/L24", CELLS["head_only"], CELLS["lora_only"],
          CELLS["lora_head"], *TAR, PROBES[0])
#: C2 cells as (LoRA trained, head trained).
FACTORIAL = {"zero_shot": (False, False), "head_only": (False, True), "lora_only": (True, False),
             "lora_head": (True, True)}


def predictor_average(gains: pd.DataFrame) -> pd.DataFrame:
    """Per dataset x arm: the mean over predictors of the five-fold raw gain and headroom share.

    Only dataset x arm cells holding every predictor seen for that arm enter, so a missing
    predictor never changes the average silently; the result has ``model_id = PREDICTOR_MEAN``.
    """
    if gains is None or gains.empty:
        return pd.DataFrame()
    keys = ["dataset_id", "task_type", "representation"]
    expected = gains.groupby("representation").model_id.nunique()
    out = gains.groupby(keys, as_index=False).agg(
        raw_gain=("raw_gain", "mean"), headroom_share=("headroom_share", "mean"),
        predictors=("model_id", "nunique"))
    out = out[out.predictors.eq(out.representation.map(expected))].copy()
    out["model_id"] = PREDICTOR_MEAN
    return out.reset_index(drop=True)


def with_average(gains: pd.DataFrame) -> pd.DataFrame:
    """Per-predictor gains plus the predictor-average rows."""
    if gains is None or gains.empty:
        return pd.DataFrame()
    return pd.concat([gains, predictor_average(gains)], ignore_index=True)


def overview(gains: pd.DataFrame) -> pd.DataFrame:
    """Equal-dataset summaries within each task family, per predictor and for the predictor mean.

    Works for the export's ``dataset_gains`` (arm minus no text) and ``contrasts`` (A minus B).
    """
    data = with_average(gains)
    return group_b.summarize_by_family(data) if not data.empty else pd.DataFrame()


def predictor_range(summary: pd.DataFrame, column: str = "mean_raw_gain") -> pd.DataFrame:
    """Per arm and task family: smallest and largest per-predictor equal-dataset value."""
    if summary is None or summary.empty:
        return pd.DataFrame()
    data = summary[summary.model_id.ne(PREDICTOR_MEAN)]
    return data.groupby(["representation", "task_family"], as_index=False).agg(
        low=(column, "min"), high=(column, "max"), predictors=("model_id", "nunique"))


def combined_gains(c1: dict, c2: dict) -> pd.DataFrame:
    """C1 and C2 per-dataset gains in one table; the shared zero-shot arm is kept once (C2)."""
    parts = []
    if c2 and not c2["dataset_gains"].empty:
        parts.append(c2["dataset_gains"])
    if c1 and not c1["dataset_gains"].empty:
        c1_gains = c1["dataset_gains"]
        if parts:
            c1_gains = c1_gains[~c1_gains.representation.isin(set(parts[0].representation))]
        parts.append(c1_gains)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def dataset_names() -> dict[str, str]:
    """Catalog display names of the selected datasets (labels only)."""
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries
    from src.utils.config import load_yaml

    return {e["id"]: e.get("name") or e["id"]
            for e in selected_entries(load_catalog(), load_yaml("experiment_1/main"))}


def coverage_summary(coverage: pd.DataFrame) -> pd.DataFrame:
    """Planned versus complete five-fold arms per phase, over all predictors."""
    if coverage is None or coverage.empty:
        return pd.DataFrame(columns=["phase", "planned_arms", "complete_arms", "share_complete"])
    out = coverage.groupby("phase", as_index=False)[["planned_arms", "complete_arms"]].sum()
    out["share_complete"] = out.complete_arms / out.planned_arms.where(out.planned_arms > 0)
    return out


# ----------------------------------------------------------------- study-audit tables


def head_reproduction_contract(extraction: pd.DataFrame, *, tolerance: float | None = None) -> pd.DataFrame:
    """C1 per-dataset contract: does the released head on the saved float16 vectors reproduce
    Strands' float32 probabilities (and the publisher's engine on the first requests)?

    Reads the C1 export's ``extraction`` table; ``passed`` requires a complete store and both
    maximum deviations at or below the configured tolerance (0.005).
    """
    if tolerance is None:
        from src.utils.config import load_yaml

        tolerance = float(load_yaml("representations/representations")["extraction"]["probability_tolerance"])
    columns = ["dataset_id", "status", "unique_inputs", "joint_inputs", "max_probability_deviation",
               "engine_check_max_difference", "tolerance", "passed", "store_gb"]
    if extraction is None or extraction.empty:
        return pd.DataFrame(columns=columns)
    data = extraction.copy()
    for column in columns:
        if column not in data:
            data[column] = np.nan
    data["tolerance"] = tolerance
    deviation = pd.to_numeric(data.max_probability_deviation, errors="coerce")
    engine = pd.to_numeric(data.engine_check_max_difference, errors="coerce")
    data["passed"] = data.status.eq("complete") & deviation.le(tolerance) & engine.le(tolerance)
    return data[columns].reset_index(drop=True)


def fit_diagnostics(fits: pd.DataFrame, *, by=("cell",)) -> pd.DataFrame:
    """C2 cross-fitted training fits: count against the plan, epochs run, best validation loss,
    GPU hours and peak GPU memory, grouped by ``by`` (cell; or dataset_id, cell; ...).

    Planned fits per group come from `strands_adaptation.training_plan` (18 x 5 folds x 3 parts
    per cell). Reads the C2 export's ``fits`` table only.
    """
    from src.evaluation.strands_adaptation import training_plan

    by = list(by)
    plan = pd.DataFrame(training_plan())
    planned = plan.groupby(by).size().rename("planned_fits") if set(by) <= set(plan) else None
    columns = [*by, "planned_fits", "fits", "epochs_median", "epochs_max", "best_validation_loss_median",
               "gpu_hours", "gpu_hours_per_fit", "peak_gpu_gb_max"]
    if fits is None or fits.empty:
        out = planned.reset_index() if planned is not None else pd.DataFrame(columns=by)
        return out.reindex(columns=columns).fillna({"fits": 0})
    out = fits.groupby(by).agg(
        fits=("part", "size"), epochs_median=("epochs", "median"), epochs_max=("epochs", "max"),
        best_validation_loss_median=("best_validation_loss", "median"), gpu_hours=("gpu_hours", "sum"),
        gpu_hours_per_fit=("gpu_hours", "mean"), peak_gpu_gb_max=("peak_gpu_gb", "max"))
    if planned is not None:
        out = planned.to_frame().join(out, how="outer")
    return out.reset_index().reindex(columns=columns).fillna({"fits": 0})


def training_gpu_hours(fits: pd.DataFrame) -> dict[str, float]:
    """Total measured GPU hours of the cross-fitted fits per C2 cell (empty when absent)."""
    if fits is None or fits.empty:
        return {}
    return {cell: float(hours) for cell, hours in fits.groupby("cell").gpu_hours.sum().items()}
