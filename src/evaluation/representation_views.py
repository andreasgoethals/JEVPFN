"""Descriptive contrasts of complete saved arms; no tuning or winner selection."""

import pandas as pd

from src.evaluation.design import pca_grid
from src.evaluation.endpoints import HEADLINES

PAIR_KEYS = ["dataset_id", "task_type", "model_id", "checkpoint_id", "split_id", "outer_fold", "metric"]
MODES = ["non_text_plus_joint", "non_text_plus_per_column", "non_text_plus_both"]


def decision_vs_embedding(folds, phase="experiment_2"):
    """Pair every recipe for the three displayed descriptive metrics, without ranking.

    All other metrics remain in collect()'s tables. Avoid a cartesian expansion of
    timings, cohort counts and every logged metric in a local plotting notebook.
    """
    if folds.empty:
        return pd.DataFrame()
    complete = folds.groupby("arm_id").outer_fold.nunique()
    chosen = folds[folds.arm_id.isin(complete[complete.eq(5)].index)]
    metric = chosen.task_type.map(HEADLINES)
    chosen = chosen[chosen.metric.eq(metric)]
    decisions = chosen[chosen.phase.eq("experiment_1") & chosen.feature_set.isin(MODES)]
    embeddings = chosen[chosen.phase.eq(phase) & chosen.encoder.notna()]
    embedding_recipe = pd.MultiIndex.from_frame(embeddings[["pca_scope", "pca_components"]])
    embeddings = embeddings[embedding_recipe.isin(pca_grid())]
    if decisions.empty or embeddings.empty:
        return pd.DataFrame()
    for table in (decisions, embeddings):
        if table.duplicated(PAIR_KEYS + ["comparison_id"]).any():
            raise ValueError("Multiple recipes/records for a contrast; select one campaign")
    other = embeddings[PAIR_KEYS + ["arm_id", "comparison_id", "encoder", "pca_scope", "pca_components", "value"]]
    other = other.rename(columns={"arm_id": "embedding_arm_id", "comparison_id": "embedding_comparison_id", "value": "reference_value"})
    # Cartesian product is intentional ONLY within the exact pair keys. All recipe
    # dimensions stay explicit; no best-performing recipe is selected or averaged away.
    result = decisions.drop(columns=["encoder", "pca_scope", "pca_components"]).merge(other, on=PAIR_KEYS, how="inner")
    result["difference"] = result.value - result.reference_value
    result["reference"] = "matched_embedding_recipe"
    return result


def measured_extraction():
    """Read full-extraction timing summaries separately from pilots and predictions.

    No hardware-normalized price is invented. Reports retain run/hardware identity;
    partial runs are not automatically joined to a completed feature artifact.
    """
    import json

    from src.utils import paths

    rows = []
    for file in sorted((paths.manifests_dir("feature_creation") / "timings").glob("*.json")):
        report = json.loads(file.read_text(encoding="utf-8"))
        for stage in report.get("stages", []):
            rows.append({"run_id": report["run_id"], "extractor": report["name"],
                         "context": json.dumps(report.get("context", {}), sort_keys=True),
                         **stage, "source": str(file)})
    return pd.DataFrame(rows)


def tar_vs_frozen(folds):
    """Prespecified same-E5 and PCA comparison; input policies are not identical."""
    if folds.empty:
        return pd.DataFrame()
    chosen = folds[folds.metric.eq(folds.task_type.map(HEADLINES))]
    if chosen.duplicated(["arm_id", "outer_fold", "metric"]).any():
        raise ValueError("Multiple records for a TAR/frozen fold; select one campaign")
    complete = chosen.groupby("arm_id").value.count()
    chosen = chosen[chosen.arm_id.isin(complete[complete.eq(5)].index)]
    keys = PAIR_KEYS + ["pca_scope", "pca_components"]
    tar = chosen[chosen.phase.eq("experiment_3")]
    frozen = chosen[chosen.phase.eq("experiment_2") & chosen.encoder.eq("e5_small_v2")]
    if tar.empty or frozen.empty:
        return pd.DataFrame()
    if tar.duplicated(keys).any() or frozen.duplicated(keys).any():
        raise ValueError("Multiple TAR/frozen campaigns match a comparison")
    result = tar.merge(frozen[keys + ["value"]].rename(columns={"value": "reference_value"}),
                       on=keys, how="inner", validate="one_to_one")
    result["difference"] = result.value - result.reference_value
    result["reference"] = "frozen_e5_small_same_pca"
    return result
