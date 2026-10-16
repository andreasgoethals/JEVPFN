"""Aggregate only complete, checksummed CV arms; never fit or rank models.

Individual predictions remain immutable on PROJECT. This produces per-dataset
metrics and paired feature gains within a fixed predictor/split/runtime.
"""

import argparse
import json

import pandas as pd

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
    paired = chosen.loc[chosen.feature_set != "non_text"].merge(
        baseline.rename(columns={"value": "non_text_value"}),
        on=keys,
        how="left",
        validate="many_to_one",
    )
    paired["difference_from_non_text"] = paired.value - paired.non_text_value
    return (
        {
            "status": "collected",
            "complete_arms": len(complete),
            "incomplete_arms": int((counts != 5).sum()),
            "difference_definition": "augmented minus non-text; improvement direction depends on metric",
            "single_text_rule": "Separate/joint/both use identical information for one-text-column datasets; exclude those datasets from representation contrasts.",
        },
        folds,
        summary,
        paired,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    report, folds, summary, paired = collect()
    if args.save and not folds.empty:
        destination = paths.results_dir("paired_summary", phase="experiment_4")
        destination.mkdir(parents=True, exist_ok=True)
        for name, table in (("folds", folds), ("per_dataset", summary), ("paired", paired)):
            table.to_parquet(destination / f"{name}.parquet", index=False)
        write_json(paths.manifests_dir("experiment_4") / "paired_summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
