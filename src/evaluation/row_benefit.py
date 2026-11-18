"""RQ2E. Pair saved held-out predictions with the structured baseline, without fitting."""

import argparse
import json
import re

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


# Fixed before any result (approved 18 Nov 2026), label-free decision-input cohorts:
# confidence = the augmented arm's own decision blocks, max probability per available block
# (classification classes, regression evidence levels), averaged over the row's blocks;
# agreement = share of available decision models whose argmax equals the plurality choice,
# averaged over the arm's text groups. Exact top ties abstain (decision_comparison.TIED).
CONFIDENCE_EDGES = (0.5, 0.7, 0.9)
CONFIDENCE_LABELS = ("lt_0.5", "0.5_to_0.7", "0.7_to_0.9", "ge_0.9")
AGREEMENT_EDGES = (0.5, 0.75)
AGREEMENT_LABELS = ("lt_0.5", "0.5_to_0.75", "0.75_to_1", "unanimous")
UNAVAILABLE = -100  # No decision input for this row (missing text or no artifact).


def block_probabilities(block, task, n_rows):
    """Per text group: full-dataset probability matrix in csv row order, NaN rows if missing."""
    columns = block.manifest["feature_columns"]
    groups = (
        [f"text_{i:03d}" for i in range(len(block.manifest["text_columns"]))]
        if block.manifest["identity"]["representation"] == "per_column"
        else ["all_text"]
    )
    positions = block.values.row_id.str.removeprefix("csv:").astype(int).to_numpy()
    if len(positions) != n_rows or set(positions) != set(range(n_rows)):
        raise ValueError("Decision block rows differ from the frozen dataset rows")
    result = {}
    for group in groups:
        named = {c.rsplit("__", 1)[-1]: c for c in columns if f"__{group}__" in c}
        if task == "binary_classification":
            suffixes = ["p_positive"]
        elif task == "regression":
            suffixes = [f"p_{i}" for i in range(9)]
        else:
            suffixes = sorted(k for k in named if re.fullmatch(r"p_c\d{3}", k))
        if not suffixes or not set(suffixes) <= set(named):
            raise ValueError("Decision block lacks its complete probability vector")
        values = block.values[[named[s] for s in suffixes]].to_numpy(dtype=float)
        if task == "binary_classification":
            values = np.column_stack([1 - values[:, 0], values[:, 0]])
        matrix = np.full((n_rows, values.shape[1]), np.nan)
        matrix[positions] = values
        result[group] = matrix
    return result


def load_decision_block(dataset_id, provider, mode, cfg):
    """The configured artifact from the canonical store, else the portable notebook copy.

    Returns ``(block, source)``; an absent artifact is ``(None, reason)``. Checksum,
    schema or ambiguity errors in a matching artifact are never treated as absence.
    """
    from src.evaluation.features import load_feature_selection, match_artifact_pins
    from src.evaluation.portable_features import portable_path
    from src.evaluation.qualification import decision_artifact_pins, decision_selector
    from src.jev.features import feature_inventory

    try:
        selector = decision_selector(cfg, provider)
    except ValueError:
        return None, "no_configured_feature_selector"
    pins = decision_artifact_pins(cfg, dataset_id, provider)
    for path in (paths.dataset_cache_path(dataset_id), portable_path(dataset_id)):
        if not path.is_file():
            continue
        inventory = match_artifact_pins(feature_inventory(path), pins)
        for key, value in selector.items():
            inventory = inventory.loc[inventory[key].isna() if value is None else inventory[key].eq(value)]
        if inventory.representation.eq(mode).any():
            (block,) = load_feature_selection(path, selector, (mode,), artifact_pins=pins)
            return block, str(path)
    return None, "artifact_not_found"


class DecisionInputs:
    """Label-free per-row decision summaries for one dataset, loaded lazily per provider/mode.

    Only the max probability and argmax choice per row and text group are retained, so a
    full roster fits in memory on a CPU allocation; probability matrices are discarded.
    """

    def __init__(self, dataset, cfg, *, loader=load_decision_block):
        self.dataset, self.cfg, self.loader = dataset, cfg, loader
        self.task = dataset.metadata.task_type
        self.roster = list(cfg["candidate_extractors"])
        self.blocks, self.status = {}, {}

    def groups(self, provider, mode):
        key = (provider, mode)
        if key not in self.blocks:
            block, source = self.loader(self.dataset.metadata.dataset_id, provider, mode, self.cfg)
            self.status[f"{provider}/{mode}"] = source if block is not None else f"missing:{source}"
            self.blocks[key] = None if block is None else {
                group: decision_summaries(p)
                for group, p in block_probabilities(block, self.task, len(self.dataset.frame)).items()
            }
        return self.blocks[key]

    def confidence(self, provider, modes):
        values = [top for mode in modes for top, _ in (self.groups(provider, mode) or {}).values()]
        return row_nanmean(values) if values else np.full(len(self.dataset.frame), np.nan)

    def agreement(self, modes):
        per_group = []
        for mode in modes:
            loaded = {p: self.groups(p, mode) for p in self.roster}
            names = sorted({g for groups in loaded.values() if groups for g in groups})
            for group in names:
                choices = np.stack([
                    groups[group][1] if groups and group in groups
                    else np.full(len(self.dataset.frame), UNAVAILABLE)
                    for groups in loaded.values()
                ])
                per_group.append(group_agreement(choices))
        return (row_nanmean(per_group) if per_group
                else np.full(len(self.dataset.frame), np.nan))


def decision_modes(recipe, feature_set, phase):
    """Decision modes of a decision arm; None for baselines and embedding arms."""
    from src.evaluation.aggregate import DECISION_MODES

    mode = DECISION_MODES.get(feature_set)
    if phase != "experiment_1" or mode is None or not (recipe or {}).get("provider"):
        return None
    return ("per_column", "joint") if mode == "both" else (mode,)


def decision_summaries(p):
    """Row max probability and argmax choice; exact top ties abstain, missing rows unavailable."""
    from src.evaluation.decision_comparison import TIED

    available = np.isfinite(p).all(axis=1)
    top = np.where(available, np.nanmax(np.where(np.isfinite(p), p, -np.inf), axis=1), np.nan)
    tied = available & ((p == top[:, None]).sum(axis=1) > 1)
    choice = np.where(available, np.argmax(np.where(np.isfinite(p), p, -np.inf), axis=1), UNAVAILABLE)
    return top, np.where(tied, TIED, choice).astype(int)


def row_nanmean(arrays):
    """Per-row mean over the available (finite) values; NaN when none is available."""
    stack = np.vstack(arrays)
    present = np.isfinite(stack)
    total = np.where(present, stack, 0).sum(axis=0)
    count = present.sum(axis=0)
    return np.divide(total, count, out=np.full(stack.shape[1], np.nan), where=count > 0)


def group_agreement(choices):
    """Plurality share among available models (rows x models -> rows); < 2 models is NaN."""
    from src.evaluation.decision_comparison import TIED

    choices = np.asarray(choices)
    available = choices != UNAVAILABLE
    n = available.sum(axis=0)
    options = [c for c in np.unique(choices) if c not in (UNAVAILABLE, TIED)]
    votes = (
        np.max(np.stack([(choices == c).sum(axis=0) for c in options]), axis=0)
        if options else np.zeros(choices.shape[1], dtype=int)
    )
    return np.divide(votes, n, out=np.full(choices.shape[1], np.nan), where=n >= 2)


def confidence_bands(values):
    values = np.asarray(values, dtype=float)
    codes = np.searchsorted(CONFIDENCE_EDGES, np.nan_to_num(values), side="right")
    return np.where(np.isfinite(values), np.asarray(CONFIDENCE_LABELS)[codes], "missing")


def agreement_bands(values):
    values = np.asarray(values, dtype=float)
    codes = np.searchsorted(AGREEMENT_EDGES, np.nan_to_num(values), side="right")
    labels = np.where(values >= 1, AGREEMENT_LABELS[-1], np.asarray(AGREEMENT_LABELS[:-1])[codes])
    return np.where(np.isfinite(values), labels, "missing")


def paired_cohorts(reference, augmented, classes, variance, train_lengths, test_lengths, *,
                   decision=None):
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
    if decision is not None:
        for name in ("confidence", "agreement"):
            if len(decision[name]) != len(reference):
                raise ValueError("Decision cohort inputs are misaligned with held-out rows")
        cohorts["decision_confidence"] = confidence_bands(decision["confidence"])
        cohorts["decision_agreement"] = agreement_bands(decision["agreement"])
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


def run(dataset_id, *, decision_loader=load_decision_block):
    """Pair saved predictions; decision arms also join their decision inputs by row ID.

    Decision inputs come from the canonical per-dataset feature store (VSC CPU) or the
    portable notebook copy (local). Absent artifacts put rows in the "missing" cohort.
    """
    entry = next(e for e in load_catalog()["datasets"] if e["id"] == dataset_id)
    ds = load_dataset(entry, allow_download=False)
    main = load_yaml("experiment_1/main")
    spec = shared_splits(ds, seed=main["random_seed"], folds=5)
    inputs = DecisionInputs(ds, main, loader=decision_loader)
    decision_rows = {}
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
            modes = decision_modes(r["representation_recipe"], r["feature_set"], phase)
            decision = None
            if modes is not None:
                provider = r["representation_recipe"]["provider"]
                if (provider, modes) not in decision_rows:
                    decision_rows[provider, modes] = (
                        inputs.confidence(provider, modes), inputs.agreement(modes)
                    )
                confidence, agreement = decision_rows[provider, modes]
                decision = {"confidence": confidence[test], "agreement": agreement[test]}
            rows, edges = paired_cohorts(
                reference, table, classes, variance, lengths[train], lengths[test],
                decision=decision,
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
            decision_inputs=dict(sorted(inputs.status.items())),
            decision_cohorts=dict(
                confidence_edges=list(CONFIDENCE_EDGES),
                agreement_edges=list(AGREEMENT_EDGES),
                agreement_minimum_models=2,
                ties="exact top-probability ties abstain",
            ),
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
