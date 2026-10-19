"""Label-free comparisons of complete, immutable, saved decision-model features.

Read only: no provider imports, requests, model loads, target labels or cache writes.
Metrics use all aligned inputs; only illustrative scatter points are subsampled.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loaders import load_catalog
from src.data.selection import selected_entries
from src.jev.features import FeatureBlock, feature_inventory, load_block
from src.utils import paths
from src.utils.serialization import canonical_json, digest, write_json


@dataclass
class GroupFeatures:
    provider: str
    row_ids: np.ndarray
    probabilities: np.ndarray
    input_hashes: np.ndarray
    labels: tuple
    task: str
    signature: str


@dataclass
class Comparison:
    records: dict[str, list[dict]] = field(default_factory=dict)

    def add(self, table: str, record: dict):
        self.records.setdefault(table, []).append(record)

    def table(self, name: str) -> pd.DataFrame:
        return pd.DataFrame(self.records.get(name, []))


def payload_hashes(path: Path, block: FeatureBlock) -> dict[str, str]:
    """Check raw cache identities and hash the provider-neutral state/question, in batches.

    Provider settings deliberately differ, but the actual semantic payload must match.
    Reading stored payloads also detects provider-specific shortening of a source row.
    """
    identity = block.manifest["identity"]
    keys = block.requests.request_key.dropna().unique().tolist()
    result = {}
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        for offset in range(0, len(keys), 400):
            batch = keys[offset : offset + 400]
            query = (
                "SELECT request_key,payload_json,settings_json FROM responses "
                "WHERE kind='verified_response' AND request_key IN ("
                + ",".join("?" for _ in batch)
                + ")"
            )
            for key, payload_json, settings_json in db.execute(query, batch):
                payload, settings = json.loads(payload_json), json.loads(settings_json)
                if digest({"payload": payload, "settings": settings}) != key:
                    raise ValueError("Stored request checksum mismatch")
                for name in ("provider", "model", "version"):
                    if settings.get(name, "jev" if name == "provider" else None) != identity[name]:
                        raise ValueError("Feature artifact and cached provider settings differ")
                result[key] = digest(payload)
    if set(result) != set(keys):
        raise ValueError("Feature artifact is missing its source responses")
    return result


def normalize_group(block: FeatureBlock, group: str, hashes: dict[str, str]) -> GroupFeatures:
    """Normalize retained probabilities without silently renormalizing invalid output."""
    manifest, identity = block.manifest, block.manifest["identity"]
    labels = tuple(manifest["class_labels"])
    task = (
        "regression"
        if not labels
        else ("binary_classification" if len(labels) == 2 else "multiclass_classification")
    )
    columns = {c.rsplit("__", 1)[-1]: c for c in manifest["feature_columns"] if f"__{group}__" in c}
    refs = block.requests[block.requests.group.eq(group)].set_index("row_id")
    refs = refs.reindex(block.values.row_id)
    if refs.status.isna().any():
        raise ValueError("Incomplete row/group correspondence")
    missing = refs.status.eq("missing_text").to_numpy()
    suffixes = (
        ["p_positive"]
        if task == "binary_classification"
        else [f"p_c{i:03d}" for i in range(len(labels))]
        if labels
        else [f"p_{i}" for i in range(9)]
    )
    if not set(suffixes) <= set(columns):
        raise ValueError("Comparison requires complete retained probability vectors")
    p = block.values[[columns[s] for s in suffixes]].to_numpy(dtype=float)
    if task == "binary_classification":
        p = np.column_stack([1 - p[:, 0], p[:, 0]])
    valid = p[~missing]
    if (
        not np.isnan(p[missing]).all()
        or not np.isfinite(valid).all()
        or (valid < 0).any()
        or (valid > 1).any()
        or not np.allclose(valid.sum(axis=1), 1, rtol=0, atol=1e-6)
    ):
        raise ValueError("Invalid or incomplete probability distribution")
    if task == "regression":
        if manifest["regression_index_to_direction"] != {str(i): i - 4 for i in range(9)}:
            raise ValueError("Regression probability/score order differs")
        score = block.values[columns["expected_direction"]].to_numpy(dtype=float)
        if not np.allclose(score[~missing], valid @ np.arange(-4, 5), atol=1e-6, rtol=0):
            raise ValueError("Saved regression score differs from its retained probabilities")
    input_ids = refs.request_key.map(hashes).to_numpy()
    if pd.isna(input_ids[~missing]).any():
        raise ValueError("Missing source input hash")
    input_ids[missing] = None
    # A repeated identical request must reuse the identical cached vector.
    if len(valid):
        _, first, inverse = np.unique(input_ids[~missing], return_index=True, return_inverse=True)
        if not np.array_equal(valid, valid[first[inverse]]):
            raise ValueError("Repeated input has inconsistent cached features")
    common = {
        k: identity.get(k)
        for k in (
            "dataset_id",
            "data_identity",
            "task_schema_sha256",
            "question",
            "question_bank_id",
            "question_variant",
            "documentary_definition",
            "representation",
            "include_non_text_features",
            "missing_text_policy",
            "text_policy",
        )
    }
    common.update(labels=labels, text_columns=manifest["text_columns"], group=group)
    return GroupFeatures(
        identity["provider"],
        block.values.row_id.to_numpy(),
        p,
        input_ids,
        labels,
        task,
        digest(common),
    )


def entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy normalized by log(number of options); 0=certain, 1=uniform."""
    return -(p * np.log(np.maximum(p, np.finfo(float).tiny))).sum(axis=-1) / np.log(p.shape[-1])


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence in bits: 0=identical, 1=disjoint support."""
    mid = (p + q) / 2
    tiny = np.finfo(float).tiny
    return np.maximum(
        0,
        0.5
        * (
            (p * np.log2(np.maximum(p, tiny) / np.maximum(mid, tiny))).sum(axis=-1)
            + (q * np.log2(np.maximum(q, tiny) / np.maximum(mid, tiny))).sum(axis=-1)
        ),
    )


def _choices(p: np.ndarray, task: str, neutral: float) -> np.ndarray:
    if task == "regression":
        score = p @ np.arange(-4, 5)
        return np.where(score > neutral, 1, np.where(score < -neutral, -1, 0))
    if task == "binary_classification":
        return (p[:, 1] >= 0.5).astype(int)
    return p.argmax(axis=1)  # Canonical class order breaks exact ties; ties are counted below.


def _scalar(p: np.ndarray, task: str) -> np.ndarray:
    return (
        p @ np.arange(-4, 5)
        if task == "regression"
        else (p[..., 1] if task == "binary_classification" else p.max(axis=-1))
    )


def _correlation(x, y, *, rank=False):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    if rank:
        x, y = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(x, y)[0, 1])


def _weighting_indices(mask, hashes, weightings):
    rows = np.flatnonzero(mask)
    unique = rows[np.unique(hashes[rows], return_index=True)[1]] if len(rows) else rows
    for weighting in weightings:
        yield weighting, rows if weighting == "rows" else unique, len(rows), len(unique)


def analyze_group(groups: list[GroupFeatures], context: dict, cfg: dict, result: Comparison):
    """Compare the same row/text/question. Reject any mismatched nonmissing input."""
    if not groups:
        return
    reference = groups[0]
    aligned = []
    for group in groups:
        if (
            group.signature != reference.signature
            or group.task != reference.task
            or group.labels != reference.labels
            or len(np.unique(group.row_ids)) != len(group.row_ids)
        ):
            raise ValueError("Cannot compare different data, questions, classes or text policies")
        order = pd.Index(group.row_ids).get_indexer(reference.row_ids)
        if len(group.row_ids) != len(reference.row_ids) or (order < 0).any():
            raise ValueError("Cannot compare different source row sets")
        aligned.append(
            GroupFeatures(
                group.provider,
                reference.row_ids,
                group.probabilities[order],
                group.input_hashes[order],
                group.labels,
                group.task,
                group.signature,
            )
        )
    names = [g.provider for g in aligned]
    if len(names) != len(set(names)):
        raise ValueError("Select one artifact per provider and representation")
    ps = np.stack([g.probabilities for g in aligned])
    present = np.isfinite(ps).all(axis=2)
    if not (present == present[0]).all():
        raise ValueError("Complete feature blocks disagree about which text inputs are missing")
    hashes = np.array([None] * len(reference.row_ids), dtype=object)
    for g, good in zip(aligned, present):
        both = good & pd.notna(hashes)
        if (hashes[both] != g.input_hashes[both]).any():
            raise ValueError("Providers received different text/question payloads for the same row")
        hashes[good] = g.input_hashes[good]
    base = {**context, "task": reference.task}
    neutral = cfg["regression_neutral_band"]
    tolerance = cfg["regression_agreement_tolerance"]
    choices = np.stack([_choices(p, reference.task, neutral) for p in ps])
    scalars = np.stack([_scalar(p, reference.task) for p in ps])
    for i, g in enumerate(aligned):
        result.add(
            "coverage",
            {
                **base,
                "provider": g.provider,
                "rows": len(hashes),
                "available_rows": int(present[i].sum()),
                "missing_text_rows": int((~present[i]).sum()),
            },
        )
        for weighting, idx, n, u in _weighting_indices(present[i], hashes, cfg["weightings"]):
            if not len(idx):
                continue
            for variable, values in {
                "main_scalar": scalars[i, idx],
                "entropy": entropy(ps[i, idx]),
                "max_probability": ps[i, idx].max(axis=1),
            }.items():
                q = np.quantile(values, [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])
                result.add(
                    "distributions",
                    {
                        **base,
                        "provider": g.provider,
                        "weighting": weighting,
                        "variable": variable,
                        "n": len(idx),
                        "source_rows": n,
                        "unique_inputs": u,
                        **dict(zip(("min", "p01", "q1", "median", "q3", "p99", "max"), q)),
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "constant": bool(np.ptp(values) == 0),
                    },
                )
    for i, j in combinations(range(len(aligned)), 2):
        mask = present[i] & present[j]
        for weighting, idx, n, u in _weighting_indices(mask, hashes, cfg["weightings"]):
            if not len(idx):
                continue
            p, q = ps[i, idx], ps[j, idx]
            delta = np.abs(scalars[i, idx] - scalars[j, idx])
            same = choices[i, idx] == choices[j, idx]
            record = {
                **base,
                "model_a": names[i],
                "model_b": names[j],
                "weighting": weighting,
                "n": len(idx),
                "source_rows": n,
                "unique_inputs": u,
                "agree_count": int(same.sum()),
                "disagree_count": int((~same).sum()),
                "agreement": float(same.mean()),
                "probability_mae": float(np.abs(p - q).mean()),
                "total_variation": float((np.abs(p - q).sum(axis=1) / 2).mean()),
                "js_divergence": float(js_divergence(p, q).mean()),
                "scalar_mae": float(delta.mean()),
                "scalar_difference_p95": float(np.quantile(delta, 0.95)),
                "pearson": _correlation(scalars[i, idx], scalars[j, idx]),
                "spearman": _correlation(scalars[i, idx], scalars[j, idx], rank=True),
            }
            if reference.task == "regression":
                record.update(
                    within_tolerance_count=int((delta <= tolerance).sum()),
                    within_tolerance=float((delta <= tolerance).mean()),
                )
            result.add("pairs", record)
        idx = np.flatnonzero(mask)
        if len(idx):
            rng = np.random.default_rng(cfg["random_seed"])
            selected = rng.choice(idx, min(len(idx), cfg["scatter_rows_per_group"]), replace=False)
            for k in selected:
                result.add(
                    "scatter",
                    {
                        **base,
                        "model_a": names[i],
                        "model_b": names[j],
                        "row_id": str(reference.row_ids[k]),
                        "x": scalars[i, k],
                        "y": scalars[j, k],
                    },
                )
    roster_complete = set(names) == set(cfg["providers"])
    if len(names) < 2 or (cfg["require_complete_roster_for_consensus"] and not roster_complete):
        return
    common = present.all(axis=0)
    for weighting, idx, n, u in _weighting_indices(common, hashes, cfg["weightings"]):
        if not len(idx):
            continue
        p = ps[:, idx]
        others = (p.sum(axis=0)[None, :, :] - p) / (len(names) - 1)
        tv = np.abs(p - others).sum(axis=2) / 2
        distances = (
            np.abs(scalars[:, idx] - _scalar(others, reference.task))
            if (reference.task == "regression")
            else tv
        )
        farthest = np.isclose(distances, distances.max(axis=0), atol=1e-12, rtol=0)
        tie_shares = farthest / farthest.sum(axis=0)
        tied = (p == p.max(axis=2, keepdims=True)).sum(axis=2) > 1
        for i, name in enumerate(names):
            result.add(
                "consensus",
                {
                    **base,
                    "provider": name,
                    "weighting": weighting,
                    "n": len(idx),
                    "source_rows": n,
                    "unique_inputs": u,
                    "roster": ", ".join(names),
                    "distance_to_others": float(distances[i].mean()),
                    "tv_to_others": float(tv[i].mean()),
                    "js_to_others": float(js_divergence(p[i], others[i]).mean()),
                    "farthest_share": float(tie_shares[i].mean()),
                    "farthest_tie_adjusted_count": float(tie_shares[i].sum()),
                    "tied_top_probability_count": int(tied[i].sum()),
                },
            )
        decisions = choices[:, idx]
        max_votes = np.max(
            np.stack([(decisions == c).sum(axis=0) for c in np.unique(decisions)]), axis=0
        )
        for count in range(1, len(names) + 1):
            result.add(
                "votes",
                {
                    **base,
                    "weighting": weighting,
                    "models": len(names),
                    "largest_agreeing_group": count,
                    "count": int((max_votes == count).sum()),
                    "fraction": float((max_votes == count).mean()),
                    "n": len(idx),
                },
            )
        if weighting == "unique_inputs":
            disagreement = (
                scalars[:, idx].max(axis=0) - scalars[:, idx].min(axis=0)
                if reference.task == "regression"
                else tv.max(axis=0)
            )
            top = np.argsort(-disagreement, kind="stable")[: cfg["disagreement_examples_per_group"]]
            for k in top:
                result.add(
                    "examples",
                    {
                        **base,
                        "row_id": str(reference.row_ids[idx[k]]),
                        "input_hash": str(hashes[idx[k]]),
                        "disagreement": float(disagreement[k]),
                        "labels": canonical_json(
                            list(reference.labels) if reference.labels else list(range(-4, 5))
                        ),
                        "probabilities": canonical_json(
                            {name: p[i, k].tolist() for i, name in enumerate(names)}
                        ),
                    },
                )


def compare_saved_features(cfg: dict, *, progress=print) -> Comparison:
    """Load one dataset/representation at a time; never pick a 'latest' model version."""
    options = cfg["comparison"]
    if (
        not options["providers"]
        or len(set(options["providers"])) != len(options["providers"])
        or set(options["weightings"]) != {"rows", "unique_inputs"}
        or options["regression_agreement_tolerance"] < 0
        or options["regression_neutral_band"] < 0
        or options["scatter_rows_per_group"] < 0
        or options["disagreement_examples_per_group"] < 0
    ):
        raise ValueError("Invalid comparison configuration")
    result = Comparison()
    entries = selected_entries(load_catalog(cfg["catalog"]), cfg)
    for number, entry in enumerate(entries, 1):
        dataset_id = entry["id"]
        path = paths.dataset_cache_path(dataset_id)
        inventory = feature_inventory(path)
        if progress:
            progress(
                f"[{number}/{len(entries)}] {dataset_id}: {len(inventory)} saved feature artifacts",
                flush=True,
            )
        for representation in options["representations"]:
            blocks, hash_maps = [], []
            for provider in options["providers"]:
                matches = inventory[
                    inventory.provider.eq(provider) & inventory.representation.eq(representation)
                ]
                pinned = (
                    options.get("artifacts", {})
                    .get(dataset_id, {})
                    .get(provider, {})
                    .get(representation)
                )
                if pinned:
                    matches = matches[matches.artifact_id.eq(pinned)]
                    if len(matches) != 1:
                        raise ValueError(
                            f"Pinned artifact absent: {dataset_id}/{provider}/{representation}"
                        )
                if len(matches) > 1:
                    raise ValueError(
                        f"Multiple versions: pin comparison.artifacts for {dataset_id}/{provider}/{representation}"
                    )
                row = {
                    "dataset_id": dataset_id,
                    "dataset": entry.get("name", dataset_id),
                    "provider": provider,
                    "representation": representation,
                    "status": "available" if len(matches) else "not_created",
                }
                if len(matches):
                    artifact_id = matches.iloc[0].artifact_id
                    block = load_block(path, artifact_id=artifact_id)
                    if block.manifest["identity"]["dataset_id"] != dataset_id:
                        raise ValueError("Feature file belongs to a different dataset")
                    row.update(matches.iloc[0].to_dict())
                    blocks.append(block)
                    hash_maps.append(payload_hashes(path, block))
                result.add("inventory", row)
            if not blocks:
                continue
            group_names = (
                [f"text_{i:03d}" for i in range(len(blocks[0].manifest["text_columns"]))]
                if (representation == "per_column")
                else ["all_text"]
            )
            for group in group_names:
                title = (
                    blocks[0].manifest["text_columns"][int(group[5:])]
                    if group != "all_text"
                    else "All available text columns"
                )
                normalized = [
                    normalize_group(block, group, hashes)
                    for block, hashes in zip(blocks, hash_maps)
                ]
                analyze_group(
                    normalized,
                    {
                        "dataset_id": dataset_id,
                        "dataset": entry.get("name", dataset_id),
                        "representation": representation,
                        "group": group,
                        "text_column": title,
                    },
                    options,
                    result,
                )
    return result


def dataset_balanced(table: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    """Equal text-group weight within dataset, then equal dataset weight; tasks stay separate."""
    if table.empty:
        return table.copy()
    per_dataset = table.groupby([*by, "dataset_id"], dropna=False)[metrics].mean().reset_index()
    means = per_dataset.groupby(by, dropna=False)[metrics].mean()
    means["datasets"] = per_dataset.groupby(by, dropna=False).size()
    return means.reset_index()


def save_comparison(result: Comparison, cfg: dict, notebook: str):
    """Small reproducible analysis tables beside local notebook figures, never in data/."""
    folder = paths.notebook_artifacts(notebook, kind="results") / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    for name in (
        "inventory",
        "coverage",
        "pairs",
        "consensus",
        "votes",
        "distributions",
        "examples",
        "scatter",
    ):
        result.table(name).to_csv(folder / f"{name}.csv", index=False)
    write_json(
        folder / "manifest.json",
        {
            "schema": "jevpfn.decision_feature_comparison.v1",
            "configuration": cfg["comparison"],
            "config_sha256": digest(cfg),
            "source_artifacts": result.table("inventory").fillna("").to_dict("records"),
            "target_labels_used": False,
            "all_rows_used_for_metrics": True,
            "scatter_only_sampled": True,
            "distance_is_not_accuracy": True,
        },
    )
    return folder
