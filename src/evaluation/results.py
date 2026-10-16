"""Aggregate only complete, checksummed CV arms; never fit or rank models.

Individual predictions remain immutable on PROJECT. This produces per-dataset
metrics and paired feature gains within a fixed predictor/split/runtime.
"""

import argparse
import json

import pandas as pd

from src.evaluation.features import EMBEDDING_BASELINE, decision_feature_set, uses_embeddings
from src.evaluation.records import completed_fold
from src.utils import paths
from src.utils.serialization import digest, write_json


def collect(phases=("experiment_1", "experiment_2", "experiment_3")):
    rows = []
    for phase in phases:
        for marker in sorted(paths.results_dir("cv_folds", phase=phase).glob("*.json")):
            stored = json.loads(marker.read_text(encoding="utf-8"))
            recipe = stored["recipe"]
            records = completed_fold(recipe, phase)
            arm_id = digest({k: v for k, v in recipe.items() if k != "outer_fold"})
            for path in records:
                record = json.loads(path.read_text(encoding="utf-8"))
                if record["metadata"]["prediction_role"] != "outer_test":
                    continue
                base = {
                    "arm_id": arm_id,
                    "phase": phase,
                    "dataset_id": recipe["dataset_id"],
                    "task_type": recipe["task_type"],
                    "model_id": recipe["model_id"],
                    "checkpoint_id": recipe["checkpoint_id"],
                    "feature_set": recipe["feature_set"],
                    "feature_artifact_ids": json.dumps(recipe["feature_artifact_ids"]),
                    "provider": (recipe.get("representation_recipe") or {}).get("provider"),
                    "decision_recipe_id": digest(
                        {
                            k: v
                            for k, v in (
                                recipe.get("representation_recipe")
                                or {"legacy_artifacts": recipe["feature_artifact_ids"]}
                            ).items()
                            if k != "embedding"
                        }
                    ),
                    # A representation recipe is shared across datasets; row artifact IDs are not.
                    # Legacy records without one remain separate rather than silently pooling providers.
                    "comparison_id": digest(
                        recipe.get("representation_recipe")
                        or {
                            "legacy_artifacts": recipe["feature_artifact_ids"],
                            "feature_set": recipe["feature_set"],
                        }
                    ),
                    "split_id": recipe["split_id"],
                    "outer_fold": recipe["outer_fold"],
                    "record_path": str(path),
                }
                metrics = dict(record["scores"]["metrics"])
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
        "metric",
    ]
    baseline = chosen.loc[
        (chosen.phase == "experiment_1") & (chosen.feature_set == "non_text"), keys + ["value"]
    ]
    if baseline.duplicated(keys).any():
        raise ValueError("Multiple baseline recipes match; select one campaign before comparison")
    augmented = chosen.loc[~chosen.feature_set.isin(["non_text", "original_features"])]
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
    embedding = chosen.loc[
        (chosen.phase == "experiment_1") & chosen.feature_set.eq(EMBEDDING_BASELINE),
        keys + ["value"],
    ]
    if embedding.duplicated(keys).any():
        raise ValueError(
            "Multiple embedding baselines match; select one campaign before comparison"
        )
    if not embedding.empty:
        compared = augmented.loc[~augmented.feature_set.eq(EMBEDDING_BASELINE)].merge(
            embedding.rename(columns={"value": "reference_value"}),
            on=keys,
            how="inner",
            validate="many_to_one",
        )
        compared["reference"] = "frozen_e5"
        compared["difference"] = compared.value - compared.reference_value
        paired = pd.concat([paired, compared], ignore_index=True)
    # The combined arm also needs its SAME extractor/mode without E5 as reference.
    # This measures the benefit of adding embeddings, while comparison with E5 alone
    # measures the benefit of adding decision features. Never match another provider.
    combined = chosen.loc[
        (chosen.phase == "experiment_1")
        & chosen.feature_set.map(uses_embeddings)
        & ~chosen.feature_set.eq(EMBEDDING_BASELINE)
    ].copy()
    if not combined.empty:
        combined["reference_feature_set"] = combined.feature_set.map(decision_feature_set)
        decision_keys = keys + ["decision_recipe_id", "reference_feature_set"]
        decisions = chosen.loc[
            (chosen.phase == "experiment_1")
            & chosen.feature_set.isin(
                ["non_text_plus_joint", "non_text_plus_per_column", "non_text_plus_both"]
            )
        ].rename(columns={"feature_set": "reference_feature_set", "value": "reference_value"})
        if decisions.duplicated(decision_keys).any():
            raise ValueError("Multiple matching decision baselines; select one campaign")
        compared = combined.merge(
            decisions[decision_keys + ["reference_value"]],
            on=decision_keys,
            how="left",
            validate="many_to_one",
        )
        compared["reference"] = "same_decision_features_without_e5"
        compared["difference"] = compared.value - compared.reference_value
        paired = pd.concat([paired, compared], ignore_index=True)
    return (
        {
            "status": "collected",
            "complete_arms": len(complete),
            "incomplete_arms": int((counts != 5).sum()),
            "difference_definition": "augmented minus named reference (non_text, frozen_e5, or same decision features without E5); improvement direction depends on metric",
            "fold_sd": "descriptive variation; not a standard error or confidence interval",
            "native_text_results": "descriptive reference results; no matched same-predictor non-text arm",
            "single_text_rule": "Separate/joint/both use identical information for one-text-column datasets; exclude those datasets from representation contrasts.",
        },
        folds,
        summary,
        paired,
    )


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
            dataset_differences(paired), load_yaml("experiment_4/analysis")
        )
        print(inference.to_string(index=False))
    if args.save and not folds.empty:
        destination = paths.results_dir("paired_summary", phase="experiment_4")
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
        write_json(paths.manifests_dir("experiment_4") / "paired_summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
