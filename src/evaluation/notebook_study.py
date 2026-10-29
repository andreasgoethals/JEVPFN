"""Read-only, phase-specific notebook tables; planned work is never predictive evidence."""

import json

import pandas as pd

from src.data.loaders import load_catalog
from src.evaluation.campaign import benchmark_plan
from src.evaluation.endpoints import HEADLINES
from src.evaluation.features import DECISION_FEATURE_SETS
from src.utils import paths
from src.utils.config import load_yaml

RECIPE = ["phase", "dataset_id", "model_id", "feature_set", "provider", "encoder",
          "pca_scope", "pca_components"]
IDENTITY = ["arm_id", "phase", "dataset_id", "task_type", "model_id", "checkpoint_id",
            "feature_set", "provider", "encoder", "pca_scope", "pca_components", "split_id"]


def planned_coverage(folds, phase):
    plan = pd.DataFrame(benchmark_plan())
    plan = plan[plan.phase.eq(phase)].rename(columns={"model": "model_id"}).copy()
    for name in RECIPE:
        if name not in plan:
            plan[name] = None
    plan = plan[RECIPE].copy()
    plan["complete"] = False
    if not folds.empty:
        observed = folds[folds.phase.isin(plan.phase.unique()) & folds.metric.eq(folds.task_type.map(HEADLINES))]
        observed = observed.copy()
        observed["defined"] = observed.value.notna()
        by_arm = observed.groupby(RECIPE + ["arm_id"], dropna=False).agg(
            outer_folds=("outer_fold", "nunique"), defined_folds=("defined", "sum")
        ).reset_index()
        if by_arm.duplicated(RECIPE).any():
            raise ValueError("Multiple campaigns match a planned recipe; select one campaign")
        # groupby turns all-null object keys into floats. Match nullable recipe
        # fields explicitly instead of making completion depend on pandas dtypes.
        for name in RECIPE:
            if name == "pca_components":
                plan[name] = pd.to_numeric(plan[name], errors="raise").astype(float)
                by_arm[name] = pd.to_numeric(by_arm[name], errors="raise").astype(float)
            else:
                plan[name] = plan[name].astype("string")
                by_arm[name] = by_arm[name].astype("string")
        plan = plan.drop(columns="complete").merge(by_arm, on=RECIPE, how="left", validate="many_to_one")
        plan["complete"] = plan.outer_folds.eq(5) & plan.defined_folds.eq(5)
    return plan.groupby(["dataset_id", "model_id"], dropna=False).agg(
        planned=("complete", "size"), completed=("complete", "sum")
    ).reset_index()


def headline_summary(folds):
    if folds.empty:
        return pd.DataFrame(columns=IDENTITY + ["metric", "mean", "descriptive_fold_sd", "defined_folds"])
    table = folds[folds.metric.eq(folds.task_type.map(HEADLINES))]
    if table.duplicated(["arm_id", "outer_fold", "metric"]).any():
        raise ValueError("Duplicate headline record within a fold")
    return table.groupby(IDENTITY + ["metric"], dropna=False).value.agg(
        mean="mean", descriptive_fold_sd="std", defined_folds="count"
    ).reset_index()




def artifact_coverage():
    from src.data.selection import selected_entries
    from src.jev.features import feature_inventory

    main = load_yaml("experiment_1/main")
    rows = []
    for entry in selected_entries(load_catalog(), main):
        inventory = feature_inventory(paths.dataset_cache_path(entry["id"]))
        for provider in main["candidate_extractors"]:
            selected = inventory[inventory.provider.eq(provider)]
            rows.append({"dataset_id": entry["id"], "model_id": provider, "planned": 2,
                         "completed": selected.representation.nunique()})
    return pd.DataFrame(rows)


def burden_table(folds, phases):
    if folds.empty:
        return pd.DataFrame()
    wanted = ["seconds.fit", "seconds.predict", "columns.representation"]
    table = folds[folds.phase.isin(phases) & folds.metric.isin(wanted)]
    if table.empty:
        return pd.DataFrame()
    means = table.groupby(IDENTITY + ["metric"], dropna=False).value.agg(mean="mean", n="count").reset_index()
    means = means[means.n.eq(5)]
    return means.pivot(index=IDENTITY, columns="metric", values="mean").reset_index()


def cross_system_contrasts(folds):
    """Every fixed representation versus each native system; descriptive, no winner."""
    if folds.empty:
        return pd.DataFrame()
    data = folds[folds.metric.eq(folds.task_type.map(HEADLINES))].copy()
    counts = data.groupby("arm_id").value.count()
    data = data[data.arm_id.isin(counts[counts.eq(5)].index)]
    keys = ["dataset_id", "task_type", "split_id", "outer_fold", "metric"]
    native = data[data.phase.eq("experiment_3") & data.feature_set.eq("original_features")]
    other = data[(data.phase.eq("experiment_1") & data.feature_set.isin(DECISION_FEATURE_SETS))
                 | data.phase.eq("experiment_2")]
    if native.empty or other.empty:
        return pd.DataFrame()
    if native.duplicated(keys + ["model_id"]).any():
        raise ValueError("Multiple native reference recipes; select one campaign")
    matched = other.merge(native[keys + ["model_id", "checkpoint_id", "value"]].rename(
        columns={"model_id": "reference_model", "checkpoint_id": "reference_checkpoint",
                 "value": "reference_value"}), on=keys, how="inner", validate="many_to_many")
    matched["difference"] = matched.value - matched.reference_value
    matched["family"] = matched.phase.map({"experiment_1": "Decision", "experiment_2": "Embedding"})
    matched.loc[matched.feature_set.eq("non_text"), "family"] = "Non-text"
    return matched




def contrast_summary(table):
    """Small complete-dataset rollups, retaining recipe/cohort identities, no selection."""
    if table.empty:
        return pd.DataFrame()
    names = ["model_id", "checkpoint_id", "task_type", "provider", "feature_set", "encoder", "pca_scope",
             "pca_components", "reference_model", "reference", "comparison_id",
             "reference_checkpoint", "embedding_comparison_id", "metric"]
    keys = [k for k in names if k in table]
    means = table.groupby(keys + ["dataset_id", "split_id"], dropna=False).agg(
        effect=("difference", "mean"), defined=("difference", "count"), folds=("outer_fold", "nunique")
    ).reset_index()
    means = means[means.defined.eq(5) & means.folds.eq(5)]
    if means.empty:
        return pd.DataFrame()
    if means.duplicated(keys + ["dataset_id"]).any():
        raise ValueError("Multiple splits match a comparison; select one campaign")
    return means.groupby(keys, dropna=False).agg(
        mean_difference=("effect", "mean"), median_difference=("effect", "median"),
        datasets=("dataset_id", "nunique"), dataset_ids=("dataset_id", lambda v: ", ".join(sorted(v)))
    ).reset_index()


def predictor_evidence():
    from src.evaluation.acceptance import evaluation_identity
    current = evaluation_identity()
    catalog = load_yaml("experiment_1/models")["models"]
    folder = paths.manifests_dir("experiment_0") / "after_extraction"
    rows = []
    for model in (*load_yaml("experiment_1/main")["candidate_models"],
                  *load_yaml("experiment_3/native_text")["candidate_models"]):
        reports = []
        for filename in folder.glob(f"predictor_{model}_*.json"):
            body = json.loads(filename.read_text(encoding="utf-8"))
            reports.append((filename, body))
        matching = [(p, b) for p, b in reports if b.get("evaluation_identity") == current]
        chosen = max(matching or reports, key=lambda r: r[0].stat().st_mtime_ns) if reports else (None, {})
        path, body = chosen
        rows.append({"model": model, "configured_packages": json.dumps(catalog[model]["runtime"]["packages"]),
                     "contract_status": body.get("status", "not found locally"),
                     "current_evaluation_identity": body.get("evaluation_identity") == current if body else None,
                     "full_fold_profile": body.get("full_dataset_profile", False),
                     "source": str(path) if path else None})
    return pd.DataFrame(rows)
