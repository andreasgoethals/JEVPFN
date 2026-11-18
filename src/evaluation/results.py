"""Aggregate only complete, checksummed CV arms; never fit or rank models.

Individual predictions remain immutable on PROJECT. This produces per-dataset
metrics and paired feature gains within a fixed predictor/split/runtime.
"""

import argparse
import json

import pandas as pd

from src.evaluation.records import completed_fold
from src.evaluation.store import all_recipes
from src.utils import paths
from src.utils.serialization import digest, write_json


def collect(phases=("experiment_1", "experiment_2", "experiment_3", "experiment_4")):
    rows = []
    for phase in phases:
        for recipe in all_recipes(phase):
            if type(recipe["outer_fold"]) is not int or recipe["outer_fold"] not in range(5):
                raise ValueError("Stored outer fold is outside the registered five-fold protocol")
            records = completed_fold(recipe, phase)
            arm_id = digest({k: v for k, v in recipe.items() if k != "outer_fold"})
            for path in records:
                record = json.loads(path.read_text(encoding="utf-8"))
                if record["metadata"]["prediction_role"] != "outer_test":
                    continue
                base = {
                    "arm_id": arm_id,
                    "phase": phase,
                    "pairing_identity": digest({k: recipe.get(k) for k in
                        ("data_identity", "task_schema_sha256", "protocol_sha256", "estimator_settings")}),
                    "dataset_id": recipe["dataset_id"],
                    "task_type": recipe["task_type"],
                    "model_id": recipe["model_id"],
                    "checkpoint_id": recipe["checkpoint_id"],
                    "feature_set": recipe["feature_set"],
                    "feature_artifact_ids": json.dumps(recipe["feature_artifact_ids"]),
                    "provider": (recipe.get("representation_recipe") or {}).get("provider"),
                    "view": (recipe.get("representation_recipe") or {}).get("view", "main"),
                    "inference_status": (recipe.get("representation_recipe") or {}).get(
                        "inference_status", "prespecified"
                    ),
                    "encoder": (
                        (recipe.get("representation_recipe") or {}).get("embedding") or {}
                    ).get("encoder_id"),
                    "pca_components": (
                        (recipe.get("representation_recipe") or {}).get("embedding") or {}
                    ).get("pca_components_per_column"),
                    "pca_scope": (
                        (recipe.get("representation_recipe") or {}).get("embedding") or {}
                    ).get("pca_scope"),
                    "binary_threshold_policy": recipe.get("binary_threshold_policy", "inner_oof_f1"),
                    "embedding_recipe_id": digest(
                        (recipe.get("representation_recipe") or {}).get("embedding")
                    ),
                    "decision_recipe_id": digest(
                        {
                            k: v
                            for k, v in (
                                recipe.get("representation_recipe")
                                or {"legacy_artifacts": recipe["feature_artifact_ids"]}
                            ).items()
                            if k not in {"embedding", "selection", "inference_status"}
                        }
                    ),
                    # A representation recipe is shared across datasets; row artifact IDs are not.
                    # Legacy records without one remain separate rather than silently pooling providers.
                    "comparison_id": digest(
                        {"feature_set": recipe["feature_set"],
                         "representation": recipe.get("representation_recipe")
                         or {"legacy_artifacts": recipe["feature_artifact_ids"]}}
                    ),
                    "split_id": recipe["split_id"],
                    "outer_fold": recipe["outer_fold"],
                    "record_path": str(path),
                }
                metrics = dict(record["scores"]["metrics"])
                # Before predictor-specific category encoding; after embedding PCA.
                # Older records lack this field: unknown is not zero/raw embedding width.
                metrics["columns.representation"] = record["metadata"].get("representation_feature_count")
                for name, values in record["scores"].get("decisions", {}).items():
                    metrics.update(
                        {f"{name}.{key}": value for key, value in values["metrics"].items()}
                    )
                for cohort, values in (
                    record["scores"].get("text_overlap", {}).get("cohorts", {}).items()
                ):
                    prefix = f"text.{cohort}."
                    metrics[prefix + "n_observations"] = values["n_observations"]
                    metrics[prefix + "fraction_of_outer_test"] = values["fraction_of_outer_test"]
                    metrics.update(
                        {prefix + key: value for key, value in values["metrics"].items()}
                    )
                    for decision, scores in values.get("decisions", {}).items():
                        metrics.update(
                            {
                                prefix + decision + "." + key: value
                                for key, value in scores["metrics"].items()
                            }
                        )
                metrics.update(
                    {
                        f"seconds.{name}": value["wall_seconds"]
                        for name, value in record["timing"]["stages"].items()
                    }
                )
                metrics.update(
                    {
                        f"resources.{name}": value
                        for name, value in record["resources"].items()
                        if isinstance(value, (float, int))
                    }
                )
                rows.extend(
                    {**base, "metric": key, "value": value}
                    for key, value in metrics.items()
                    if value is None or isinstance(value, (float, int))
                )
    folds = pd.DataFrame(rows)
    if folds.empty:
        return {"status": "no_completed_folds"}, folds, pd.DataFrame(), pd.DataFrame()
    counts = folds.groupby("arm_id").outer_fold.nunique()
    complete = counts[counts == 5].index
    chosen = folds.loc[folds.arm_id.isin(complete)]
    group = [
        "arm_id",
        "phase",
        "dataset_id",
        "task_type",
        "model_id",
        "checkpoint_id",
        "feature_set",
        "feature_artifact_ids",
        "provider",
        "view",
        "inference_status",
        "binary_threshold_policy",
        "embedding_recipe_id",
        "encoder",
        "pca_components",
        "pca_scope",
        "decision_recipe_id",
        "comparison_id",
        "split_id",
        "metric",
    ]
    summary = (
        chosen.groupby(group, dropna=False)
        .value.agg(mean="mean", std="std", defined_folds="count")
        .reset_index()
    )
    keys = [
        "dataset_id",
        "task_type",
        "model_id",
        "checkpoint_id",
        "split_id",
        "outer_fold",
        "pairing_identity",
        "metric",
    ]
    baseline = chosen.loc[
        (chosen.phase == "experiment_1") & (chosen.feature_set == "non_text"), keys + ["value"]
    ]
    if baseline.duplicated(keys).any():
        raise ValueError("Multiple baseline recipes match; select one campaign before comparison")
    augmented = chosen.loc[
        ~chosen.feature_set.isin(["non_text", "original_features"])
        & ~chosen.feature_set.str.startswith("original_features")
    ]
    paired = augmented.merge(
        baseline.rename(columns={"value": "non_text_value"}),
        on=keys,
        how="left",
        validate="many_to_one",
    )
    paired["difference_from_non_text"] = paired.value - paired.non_text_value
    paired["reference"] = "non_text"
    paired["reference_value"] = paired.non_text_value
    paired["difference"] = paired.difference_from_non_text
    # Arms whose run identity (e.g. estimator settings or code hash) matches no baseline
    # would otherwise vanish from effect tables without a trace.
    unpaired = paired.loc[paired.non_text_value.isna(), "arm_id"].nunique()
    return (
        {
            "status": "collected",
            "complete_arms": len(complete),
            "incomplete_arms": int((counts != 5).sum()),
            "augmented_arms_without_matching_baseline": int(unpaired),
            "difference_definition": "augmented minus the same predictor and split using non_text; improvement direction depends on metric",
            "fold_sd": "descriptive variation; not a standard error or confidence interval",
            "native_text_results": "descriptive original-text system references; cross-system comparisons in synthesis",
            "single_text_rule": "Separate/joint/both use identical information for one-text-column datasets; exclude those datasets from representation contrasts.",
        },
        folds,
        summary,
        paired,
    )



def collect_for_notebook(phases=("experiment_1", "experiment_2", "experiment_3", "experiment_4")):
    """Verify compact prediction stores once per notebook run and share their metric tables."""
    import os

    from src.utils.locking import file_lock
    from src.utils.serialization import file_sha256

    if not os.environ.get("JEVPFN_NOTEBOOK_SCRATCH"):
        return collect(phases)
    # Legacy fold files have external prediction paths. Always verify their complete chain.
    if any(any(paths.results_dir("cv_folds", phase=p).glob("*.json")) for p in phases):
        return collect(phases)

    def inventory():
        files = []
        for phase in phases:
            for path in sorted(paths.results_dir("evaluation", phase=phase).glob("*.sqlite3*")):
                stat = path.stat()
                files.append((str(path.resolve()), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        return files

    initial = inventory()
    cache = paths.notebook_cache_dir("evaluation", digest({"phases": phases, "files": initial}))
    cache.mkdir(parents=True, exist_ok=True)
    marker = cache / "receipt.json"
    files = [cache / (name + ".parquet") for name in ("folds", "datasets", "paired")]
    with file_lock(cache / "collect.lock", timeout=7200):
        if marker.is_file():
            receipt = json.loads(marker.read_text(encoding="utf-8"))
            if [file_sha256(p) for p in files] != receipt["checksums"]:
                raise ValueError("Notebook metric cache checksum mismatch")
            result = (receipt["status"], *(pd.read_parquet(p) for p in files))
        else:
            result = collect(phases)
            for frame, path in zip(result[1:], files, strict=True):
                frame.to_parquet(path, index=False)
            write_json(marker, {"status": result[0], "checksums": [file_sha256(p) for p in files]})
        if inventory() != initial:
            marker.unlink(missing_ok=True)
            raise ValueError("Published evaluation stores changed during notebook collection; rerun on a stable snapshot")
        return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    parser.add_argument(
        "--infer", action="store_true", help="Requires the approved pre-results analysis plan"
    )
    args = parser.parse_args()
    report, folds, summary, paired = collect()
    if args.infer:
        from src.evaluation.inference import confirmatory_tests, dataset_differences
        from src.utils.config import load_yaml

        inference = confirmatory_tests(
            dataset_differences(paired), load_yaml("synthesis/analysis")
        )
        print(inference.to_string(index=False))
    if args.save and not folds.empty:
        destination = paths.results_dir("paired_summary", phase="synthesis")
        destination.mkdir(parents=True, exist_ok=True)
        for name, table in (("folds", folds), ("per_dataset", summary), ("paired", paired)):
            table.to_parquet(destination / f"{name}.parquet", index=False)
        from src.evaluation.inference import across_datasets, dataset_differences

        datasets = dataset_differences(paired)
        datasets.to_parquet(destination / "dataset_differences.parquet", index=False)
        across_datasets(datasets).to_parquet(destination / "across_datasets.parquet", index=False)
        if args.infer:
            inference.to_parquet(destination / "confirmatory_tests.parquet", index=False)
        summary.loc[summary.feature_set.eq("original_features")].to_parquet(
            destination / "native_text_references.parquet", index=False
        )
        write_json(paths.manifests_dir("synthesis") / "paired_summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
