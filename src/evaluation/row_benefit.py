"""RQ2E. Pair saved held-out predictions with the structured baseline, without fitting."""

import argparse
import json

import numpy as np
import pandas as pd

from src.data.loaders import load_catalog, load_dataset
from src.evaluation.protocol import shared_splits
from src.evaluation.records import completed_fold
from src.evaluation.research_questions import training_defined_bins
from src.evaluation.store import all_recipes, read_predictions
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import canonical_json, digest, scalar, text_value, write_json


def row_losses(table, classes, variance):
    if not len(classes):
        residual = table.y_true.to_numpy(dtype=float) - table.prediction.to_numpy(dtype=float)
        if not np.isfinite(residual).all() or not np.isfinite(variance) or variance < 0:
            raise ValueError("Invalid regression prediction or training variance")
        return (residual**2 / variance if variance else np.full(len(table), np.nan)), None
    names = [f"p_c{i:03d}" for i in range(len(classes))]
    if set(names) != {c for c in table if c.startswith("p_c")}:
        raise ValueError("Probability columns differ from the saved class order")
    p = table[names].to_numpy(dtype=float)
    indices = {canonical_json(scalar(c)): i for i, c in enumerate(classes)}
    true = [indices[canonical_json(scalar(v))] for v in table.y_true]
    from src.evaluation.metrics import observed_log_losses

    loss = observed_log_losses(p, true)
    entropy = -np.sum(p * np.log(np.maximum(p, 1e-300)), axis=1) / np.log(len(classes))
    return loss, entropy


# Fixed before any result: how far the representation moved each held-out prediction.
# Label-free (predictions only): classification total variation between the two
# probability vectors; regression |change| in units of the training-target SD.
SHIFT_EDGES = {"classification": (0.02, 0.1, 0.25), "regression": (0.05, 0.25, 0.5)}


def prediction_shift_bands(reference, augmented, classes, variance):
    if len(classes):
        names = [f"p_c{i:03d}" for i in range(len(classes))]
        shift = np.abs(reference[names].to_numpy(float) - augmented[names].to_numpy(float)).sum(axis=1) / 2
        edges = SHIFT_EDGES["classification"]
    else:
        if not np.isfinite(variance) or variance <= 0:
            return np.repeat("undefined", len(reference))
        shift = np.abs(reference.prediction.to_numpy(float) - augmented.prediction.to_numpy(float))
        shift = shift / np.sqrt(variance)
        edges = SHIFT_EDGES["regression"]
    return np.array([f"shift_{b}" for b in np.searchsorted(edges, shift, side="right")])


def paired_cohorts(reference, augmented, classes, variance, train_lengths, test_lengths):
    for col in ("row_id", "y_true", "text_present", "joint_text_seen_in_train"):
        if not reference[col].equals(augmented[col]):
            raise ValueError("Unmatched held-out rows, labels or overlap indicators")
    if reference.row_id.duplicated().any() or len(test_lengths) != len(reference):
        raise ValueError("Duplicate or misaligned evaluation rows")
    left, uncertainty = row_losses(reference, classes, variance)
    right, _ = row_losses(augmented, classes, variance)
    present = reference.text_present.to_numpy(dtype=bool)
    if not np.array_equal(present, np.isfinite(test_lengths)):
        raise ValueError("Saved missingness differs from the source text")
    if np.isfinite(train_lengths).any():
        bins, edges = training_defined_bins(train_lengths, test_lengths)
        length_codes = [f"bin_{b}" if b >= 0 else "missing" for b in bins]
    else:
        edges = []
        length_codes = np.where(present, "no_training_text", "missing")
    cohorts = dict(
        all=np.repeat("all", len(left)),
        availability=np.where(present, "present", "absent"),
        length=length_codes,
        novelty=np.where(
            present, np.where(reference.joint_text_seen_in_train, "seen", "unseen"), "missing"
        ),
    )
    if uncertainty is not None:
        cohorts["baseline_entropy"] = np.searchsorted(
            [0.25, 0.5, 0.75], uncertainty, side="right"
        ).astype(str)
    cohorts["prediction_shift"] = prediction_shift_bands(reference, augmented, classes, variance)
    rows = []
    for axis, memberships in cohorts.items():
        memberships = np.asarray(memberships)
        for group in np.unique(memberships):
            mask = memberships == group
            valid = mask & np.isfinite(left) & np.isfinite(right)
            rows.append(
                dict(
                    axis=axis,
                    cohort=str(group),
                    rows=int(mask.sum()),
                    defined=int(valid.sum()),
                    baseline_loss_sum=float(left[valid].sum()),
                    augmented_loss_sum=float(right[valid].sum()),
                )
            )
    return rows, list(edges)


def outer_record(recipe, phase):
    records = completed_fold(recipe, phase)
    (record,) = [
        r
        for r in records
        if json.loads(r.read_text())["metadata"]["prediction_role"] == "outer_test"
    ]
    return json.loads(record.read_text())["metadata"], read_predictions(record)


def run(dataset_id):
    entry = next(e for e in load_catalog()["datasets"] if e["id"] == dataset_id)
    ds = load_dataset(entry, allow_download=False)
    spec = shared_splits(ds, seed=load_yaml("experiment_1/main")["random_seed"], folds=5)
    lengths = np.zeros(len(ds.frame))
    present = np.zeros(len(ds.frame), dtype=bool)
    for col in ds.metadata.text_columns:
        text = ds.frame[col].map(text_value)
        lengths += text.map(lambda v: 0 if v is None else len(str(v))).to_numpy()
        present |= text.notna().to_numpy()
    lengths[~present] = np.nan

    def matching_key(r):
        return digest(
            {
                k: r[k]
                for k in (
                    "data_identity",
                    "task_schema_sha256",
                    "model_id",
                    "checkpoint_id",
                    "split_id",
                    "outer_fold",
                    "protocol_sha256",
                    "estimator_settings",
                )
            }
        )

    arms, baselines = {}, {}
    for phase in ("experiment_1", "experiment_2", "experiment_3"):
        for r in all_recipes(phase, dataset_id=dataset_id):
            if r["dataset_id"] != dataset_id:
                continue
            if (
                ds.source["data_identity"] != r["data_identity"]
                or spec["split_id"] != r["split_id"]
            ):
                raise ValueError(
                    "Select a single campaign matching the current source rows and splits"
                )
            if phase == "experiment_1" and r["feature_set"] == "non_text":
                key = matching_key(r)
                if key in baselines:
                    raise ValueError("Ambiguous structured baseline")
                baselines[key] = r
            else:
                arm = digest({k: v for k, v in r.items() if k != "outer_fold"})
                arms.setdefault((phase, arm), []).append(r)
    from functools import lru_cache

    @lru_cache(maxsize=5)
    def baseline_record(key):
        return outer_record(baselines[key], "experiment_1")

    output, pending = [], []
    for (phase, arm), folds in arms.items():
        if sorted(r["outer_fold"] for r in folds) != list(range(5)) or any(
            matching_key(r) not in baselines for r in folds
        ):
            pending.append(arm)
            continue
        for r in folds:
            fold = r["outer_fold"]
            train, test = (spec["folds"][fold][k] for k in ("train", "test"))
            meta, table = outer_record(r, phase)
            base, reference = baseline_record(matching_key(r))
            classes = list(ds.metadata.classes)
            target = ds.frame[ds.metadata.target]
            if (
                meta["class_order"] != classes
                or base["class_order"] != classes
                or table.row_id.tolist() != [f"csv:{i}" for i in test]
                or table.y_true.tolist() != target.iloc[test].tolist()
            ):
                raise ValueError("Prediction rows or class order differ from the frozen dataset")
            variance = None if classes else float(np.var(target.iloc[train].astype(float)))
            rows, edges = paired_cohorts(
                reference, table, classes, variance, lengths[train], lengths[test]
            )
            output.append(
                dict(
                    phase=phase,
                    arm=arm,
                    fold=fold,
                    model=r["model_id"],
                    checkpoint=r["checkpoint_id"],
                    feature_set=r["feature_set"],
                    recipe=r["representation_recipe"],
                    training_variance=variance,
                    length_cutoffs=edges,
                    cohorts=rows,
                )
            )
    file = paths.results_dir("row_benefit", dataset_id + ".json", phase="rq2")
    return write_json(
        file,
        dict(
            dataset=dataset_id,
            split_id=spec["split_id"],
            data_identity=ds.source["data_identity"],
            task=ds.metadata.task_type,
            folds=output,
            incomplete_or_unmatched=pending,
        ),
    )


def summaries():
    frames = []
    for path in paths.results_dir("row_benefit", phase="rq2").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for fold in data["folds"]:
            frame = pd.DataFrame(fold["cohorts"])
            frame["dataset"], frame["arm"] = data["dataset"], fold["arm"]
            frame["phase"] = fold["phase"]
            frame["task"] = data["task"]
            frame["checkpoint"] = fold["checkpoint"]
            frame["representation"] = canonical_json(fold["recipe"])
            frame["model"], frame["features"] = fold["model"], fold["feature_set"]
            frames.append(frame)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    keys = ["task", "phase", "model", "checkpoint", "features", "representation", "axis", "cohort"]
    by = (
        pd.concat(frames)
        .groupby(["dataset", "arm", *keys], as_index=False)[
            ["rows", "defined", "baseline_loss_sum", "augmented_loss_sum"]
        ]
        .sum()
    )
    by["benefit"] = (by.baseline_loss_sum - by.augmented_loss_sum) / by.defined.replace(0, np.nan)
    by["baseline_loss"] = by.baseline_loss_sum / by.defined.replace(0, np.nan)
    by["augmented_loss"] = by.augmented_loss_sum / by.defined.replace(0, np.nan)
    if by.duplicated(["dataset", *keys]).any():
        raise ValueError("Multiple arms map to the same RQ2E comparison")
    return by, by.groupby(keys, as_index=False).agg(
        mean_benefit=("benefit", "mean"), datasets=("benefit", "count")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    print(run(parser.parse_args().dataset))
