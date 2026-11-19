"""RQ2-E row-level complementarity and data maps from paired held-out predictions.

**Complementarity.** For each dataset and fixed predictor, every decision-feature arm is
compared with every frozen-embedding and TAR arm on the same outer-test rows and folds. The
row benefit of an arm is the structured baseline's per-row loss minus the arm's loss (log
loss; regression: squared error over the outer-training target variance), exactly as in
``row_benefit``. Per pair this module records Spearman's rho between the two benefit vectors
and the Jaccard overlap of their top-10% benefit rows, with the overlap expected for two
independent random top sets. High rho or overlap means the representations help the same
rows (substitutes); low values mean they help different rows (potential complements).
``complementarity_summary`` averages per dataset first, then gives datasets equal weight.

**Data map** (classification only). Per dataset, predictor and decision mode, outer-test rows
are binned by the baseline's confidence (maximum class probability, or 1 − normalised
entropy) against the decision arm's own decision confidence (``row_benefit.DecisionInputs``),
storing row counts and summed benefit per cell. Cells pool row × decision-model pairs over
the decision models of that mode.

``run`` reads the canonical prediction and feature stores and is designed for a VSC CPU
allocation (``python -m src.evaluation.complementarity --dataset ID``); it writes one compact
JSON per dataset. ``summaries`` reads those compact files locally. Everything is descriptive:
no fit, no selection, no causal claim.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

from src.evaluation.attribution import _representation
from src.evaluation.moderators import spearman_rows
from src.evaluation.row_benefit import Pairing, load_decision_block, row_losses
from src.utils import paths
from src.utils.serialization import write_json

SCHEMA = "jevpfn.complementarity.v1"
TOP_SHARE = 0.10
BINS = 20
X_AXES = ("baseline_max_probability", "baseline_certainty")
PAIR_COLUMNS = [
    "dataset_id", "task_type", "model_id", "checkpoint_id", "decision_representation",
    "decision_mode", "provider", "other_representation", "other_family", "n_rows",
    "spearman_rho", "top_k", "jaccard_top", "jaccard_chance",
]
SUMMARY_COLUMNS = [
    "aggregation", "model_id", "checkpoint_id", "decision", "other", "n_datasets",
    "mean_spearman_rho", "rho_ci_low", "rho_ci_high", "mean_jaccard_top",
    "mean_jaccard_chance", "dataset_ids",
]


def row_benefits(reference, augmented, classes, variance) -> np.ndarray:
    """Per-row baseline loss minus augmented loss on matched rows (positive = helped)."""
    for column in ("row_id", "y_true"):
        if not reference[column].equals(augmented[column]):
            raise ValueError("Unmatched held-out rows or labels")
    left, _ = row_losses(reference, classes, variance)
    right, _ = row_losses(augmented, classes, variance)
    return left - right


def top_set(values: np.ndarray, k: int) -> set[int]:
    """Indices of the k largest values; ties broken by position, so the set is deterministic."""
    return set(np.argsort(-values, kind="stable")[:k].tolist())


def pair_statistics(a, b, *, top_share: float = TOP_SHARE) -> dict:
    """Spearman rho and top-share Jaccard overlap of two row-benefit vectors."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("Benefit vectors must cover the same rows")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    n = len(a)
    if n < 3:
        return {"n_rows": n, "spearman_rho": np.nan, "top_k": 0, "jaccard_top": np.nan,
                "jaccard_chance": np.nan}
    k = max(1, math.ceil(top_share * n))
    left, right = top_set(a, k), top_set(b, k)
    shared = k * k / n  # expected intersection of two independent random k-subsets
    return {
        "n_rows": n,
        "spearman_rho": float(spearman_rows(a, b)[0]),
        "top_k": k,
        "jaccard_top": len(left & right) / len(left | right),
        "jaccard_chance": shared / (2 * k - shared),
    }


def complementarity_pairs(benefits: dict, *, dataset_id: str, task_type: str,
                          top_share: float = TOP_SHARE) -> pd.DataFrame:
    """All decision × (embedding | TAR) pairs within each predictor of one dataset.

    ``benefits`` maps ``(model_id, checkpoint_id)`` to ``{representation: {"family",
    "values"}}`` where ``values`` is a full-length row-benefit vector in dataset row order.
    """
    records = []
    for (model_id, checkpoint_id), arms in sorted(benefits.items()):
        decisions = {r: a for r, a in arms.items() if a["family"] == "decision"}
        others = {r: a for r, a in arms.items() if a["family"] in ("embedding", "tar")}
        for decision, left in sorted(decisions.items()):
            _, mode, provider = decision.split("/", 2)
            for other, right in sorted(others.items()):
                records.append({
                    "dataset_id": dataset_id, "task_type": task_type, "model_id": model_id,
                    "checkpoint_id": checkpoint_id, "decision_representation": decision,
                    "decision_mode": mode, "provider": provider,
                    "other_representation": other, "other_family": right["family"],
                    **pair_statistics(left["values"], right["values"], top_share=top_share),
                })
    return pd.DataFrame(records, columns=PAIR_COLUMNS)


def complementarity_summary(pairs: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0):
    """Equal-dataset summaries: per decision × other representation, and per mode × family.

    ``per_pair`` averages each exact pair over datasets. ``mode_by_family`` first averages,
    within each dataset, over decision models of one mode and recipes of one family, then
    gives datasets equal weight. Intervals are dataset-bootstrap percentiles of mean rho.
    """
    if pairs.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    defined = pairs.dropna(subset=["spearman_rho"])
    views = {
        "per_pair": defined.assign(decision=defined.decision_representation,
                                   other=defined.other_representation),
        "mode_by_family": defined.assign(decision="decision/" + defined.decision_mode,
                                         other=defined.other_family),
    }
    records = []
    for aggregation, table in views.items():
        per_dataset = (table.groupby(["model_id", "checkpoint_id", "decision", "other",
                                      "dataset_id"], dropna=False)
                       [["spearman_rho", "jaccard_top", "jaccard_chance"]].mean().reset_index())
        for key, group in per_dataset.groupby(["model_id", "checkpoint_id", "decision", "other"],
                                              dropna=False, sort=True):
            rho = group.spearman_rho.to_numpy(float)
            low = high = np.nan
            if n_boot > 0 and len(rho):
                draws = np.random.default_rng(seed).integers(0, len(rho), size=(n_boot, len(rho)))
                low, high = (float(v) for v in np.percentile(rho[draws].mean(axis=1), [2.5, 97.5]))
            records.append(dict(zip(["model_id", "checkpoint_id", "decision", "other"], key)) | {
                "aggregation": aggregation, "n_datasets": len(group),
                "mean_spearman_rho": float(rho.mean()), "rho_ci_low": low, "rho_ci_high": high,
                "mean_jaccard_top": float(group.jaccard_top.mean()),
                "mean_jaccard_chance": float(group.jaccard_chance.mean()),
                "dataset_ids": ", ".join(sorted(group.dataset_id)),
            })
    return pd.DataFrame(records, columns=SUMMARY_COLUMNS)


# --------------------------------------------------------------------------- data map


def baseline_confidence(reference: pd.DataFrame, n_classes: int) -> dict[str, np.ndarray]:
    """Baseline maximum class probability and certainty (1 − normalised entropy) per row."""
    names = [f"p_c{i:03d}" for i in range(n_classes)]
    p = reference[names].to_numpy(float)
    entropy = -np.sum(p * np.log(np.maximum(p, 1e-300)), axis=1) / np.log(n_classes)
    return {"baseline_max_probability": p.max(axis=1), "baseline_certainty": 1.0 - entropy}


def data_map_grid(x, y, benefit, *, bins: int = BINS) -> dict:
    """Counts and summed benefit on a bins × bins grid over [0, 1]²; non-finite rows skipped."""
    x, y, benefit = (np.asarray(v, float) for v in (x, y, benefit))
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(benefit)
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _, _ = np.histogram2d(x[keep], y[keep], bins=[edges, edges])
    sums, _, _ = np.histogram2d(x[keep], y[keep], bins=[edges, edges], weights=benefit[keep])
    return {"bins": bins, "counts": counts.astype(int).tolist(), "benefit_sum": sums.tolist(),
            "rows": int(keep.sum())}


def merge_grids(grids) -> dict:
    """Pool several grids of equal shape (counts and benefit sums add)."""
    grids = list(grids)
    counts = np.sum([np.asarray(g["counts"]) for g in grids], axis=0)
    sums = np.sum([np.asarray(g["benefit_sum"]) for g in grids], axis=0)
    return {"bins": grids[0]["bins"], "counts": counts.astype(int).tolist(),
            "benefit_sum": sums.tolist(), "rows": int(sum(g["rows"] for g in grids))}


# ------------------------------------------------------------------------ VSC runner


def run(dataset_id, *, decision_loader=load_decision_block, top_share=TOP_SHARE, bins=BINS):
    """Row-level complementarity and data map of one dataset; writes compact JSON."""
    pairing = Pairing(dataset_id, decision_loader=decision_loader)
    ds = pairing.ds
    n = len(ds.frame)
    classes = list(ds.metadata.classes)
    benefits, baseline, modes_of = {}, {}, {}
    for item in pairing.folds():
        r, test = item["recipe"], item["test"]
        family, representation = _representation(r["representation_recipe"], r["feature_set"],
                                                 item["phase"])
        key = (r["model_id"], r["checkpoint_id"])
        arm = benefits.setdefault(key, {}).setdefault(
            representation, {"family": family, "values": np.full(n, np.nan), "folds": set()})
        if item["fold"] in arm["folds"]:
            raise ValueError(f"Two arms map to {representation} for {key}; select one campaign")
        arm["folds"].add(item["fold"])
        arm["values"][test] = row_benefits(item["reference"], item["table"], classes,
                                           item["variance"])
        if item["modes"] is not None:
            modes_of[key, representation] = (r["representation_recipe"]["provider"], item["modes"],
                                             r["feature_set"])
        if classes:
            confidence = baseline.setdefault(key, {a: np.full(n, np.nan) for a in X_AXES})
            for axis, values in baseline_confidence(item["reference"], len(classes)).items():
                confidence[axis][test] = values
    pairs = complementarity_pairs(benefits, dataset_id=dataset_id, task_type=ds.metadata.task_type,
                                  top_share=top_share)
    grids = []
    if classes:
        pooled = {}
        for (key, representation), (provider, modes, feature_set) in sorted(modes_of.items()):
            decision = pairing.inputs.confidence(provider, modes)
            values = benefits[key][representation]["values"]
            for axis in X_AXES:
                grid = data_map_grid(baseline[key][axis], decision, values, bins=bins)
                pooled.setdefault((*key, feature_set, axis), []).append((provider, grid))
        for (model_id, checkpoint_id, feature_set, axis), items in sorted(pooled.items()):
            grids.append({
                "model_id": model_id, "checkpoint_id": checkpoint_id, "feature_set": feature_set,
                "x_axis": axis, "y_axis": "decision_confidence",
                "providers": sorted(p for p, _ in items), **merge_grids(g for _, g in items),
            })
    file = paths.results_dir("complementarity", dataset_id + ".json", phase="rq2")
    return write_json(file, {
        "schema": SCHEMA,
        "dataset": dataset_id,
        "task": ds.metadata.task_type,
        "split_id": pairing.spec["split_id"],
        "data_identity": ds.source["data_identity"],
        "top_share": top_share,
        "pairs": json.loads(pairs.to_json(orient="records")),
        "data_map": grids,
        "data_map_status": "available" if classes else "not_applicable_regression",
        "incomplete_or_unmatched": pairing.pending,
        "decision_inputs": dict(sorted(pairing.inputs.status.items())),
    })


def summaries(folder=None):
    """Compact per-dataset exports -> ``(pairs, data_map)``; empty frames when pending."""
    folder = paths.results_dir("complementarity", phase="rq2") if folder is None else folder
    pairs, grids = [], []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        body = json.loads(path.read_text(encoding="utf-8"))
        if body.get("schema") != SCHEMA:
            continue
        pairs.append(pd.DataFrame(body["pairs"], columns=PAIR_COLUMNS))
        grids.extend({"dataset_id": body["dataset"], "task_type": body["task"], **g}
                     for g in body["data_map"])
    table = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame(columns=PAIR_COLUMNS)
    return table, pd.DataFrame(grids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    print(run(parser.parse_args().dataset))
