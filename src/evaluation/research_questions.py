"""Reviewed research plans and paired-analysis contracts; never dispatch model calls."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.data.loaders import load_catalog
from src.data.selection import selected_entries
from src.evaluation.widths import feature_width, width_status
from src.utils.config import load_yaml
from src.utils.serialization import digest


def regression_readout_plan():
    """New full-vector fits only; mean comparators reuse RQ2's exact paired arms."""
    cfg = load_yaml("rq2/readout")
    semantics = load_yaml("dataset_semantics")["datasets"]
    result = []
    for entry in selected_entries(load_catalog(), cfg):
        if semantics[entry["id"]]["task_type"] != "regression":
            continue
        for model in cfg["candidate_models"]:
            for provider in cfg["candidate_extractors"]:
                for mode in cfg["modes"]:
                    arm = dict(
                        dataset_id=entry["id"],
                        model=model,
                        provider=provider,
                        phase="regression_readout",
                        feature_set="non_text_plus_" + mode,
                        view="probabilities",
                    )
                    arm.update(width_status(_metadata(entry), arm))
                    result.append(arm)
    return result


def _metadata(entry):
    stats = entry["paper_summary"]
    return SimpleNamespace(
        text_columns=range(stats["text_columns"]),
        non_text_columns=range(stats["non_text_columns"]),
        classes=[],
        task_type="regression",
    )


def regression_width_table():
    """Catalog widths, plus conservative Kumo expansion; actual fits still check width."""
    cfg = load_yaml("rq2/readout")
    semantics = load_yaml("dataset_semantics")["datasets"]
    rows = []
    for entry in selected_entries(load_catalog(), cfg):
        if semantics[entry["id"]]["task_type"] != "regression":
            continue
        meta = _metadata(entry)
        row = dict(
            dataset=entry["name"],
            dataset_id=entry["id"],
            text_columns=len(meta.text_columns),
            non_text_columns=len(meta.non_text_columns),
        )
        for mode in ("joint", "per_column", "both"):
            for view in ("main", "probabilities"):
                row[f"{mode}_{view}"] = feature_width(
                    meta, dict(feature_set="non_text_plus_" + mode, provider="strands", view=view)
                )
        row["kumo_conservative_max"] = row["both_probabilities"] + len(meta.non_text_columns)
        rows.append(row)
    return pd.DataFrame(rows)


def adaptation_plan():
    """Direct classifier evaluation: no reuse as supervised downstream training features."""
    cfg = load_yaml("rq3/adaptation")
    main = load_yaml("experiment_1/main")
    semantics = load_yaml("dataset_semantics")["datasets"]
    return [
        dict(
            dataset_id=e["id"],
            fold=fold,
            provider=cfg["provider"],
            variant=variant,
            pilot=e["id"] in cfg["pilot_datasets"] and fold in cfg["pilot_folds"],
            execution_enabled=False,
        )
        for e in selected_entries(load_catalog(), main)
        if semantics[e["id"]]["task_type"] in cfg["tasks"]
        for fold in range(cfg["outer_folds"])
        for variant in ("head_only", "lora_and_head")
    ]


def stable_probe_sample(input_keys, *, limit=100, seed=42):
    """Unique request keys, independent of target values, row order or model outcomes."""
    if type(limit) is not int or limit < 1:
        raise ValueError("Probe limit must be positive")
    return sorted(set(input_keys), key=lambda key: digest([seed, key]))[:limit]


def align_permuted_probabilities(probabilities, original_indices):
    """Map permuted option positions back to their original semantic class indices."""
    values = np.asarray(probabilities, dtype=float)
    order = np.asarray(original_indices)
    if (
        values.ndim != 2
        or order.ndim != 1
        or values.shape[1] != len(order)
        or not np.issubdtype(order.dtype, np.integer)
        or sorted(order.tolist()) != list(range(len(order)))
    ):
        raise ValueError("A complete one-to-one option permutation is required")
    if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(1), 1):
        raise ValueError("Invalid probability vectors")
    return values[:, np.argsort(order)]


def training_defined_bins(training_values, query_values, *, quantiles=(0.25, 0.5, 0.75)):
    """Cohort boundaries use training data only; missing values retain a separate code."""
    training = np.asarray(training_values, dtype=float)
    query = np.asarray(query_values, dtype=float)
    observed = training[np.isfinite(training)]
    if not len(observed):
        raise ValueError("No observed training values for cohort boundaries")
    edges = np.unique(np.quantile(observed, quantiles))
    bins = np.searchsorted(edges, query, side="right")
    bins[~np.isfinite(query)] = -1
    return bins, edges


def validate_adaptation_partition(optimizer_rows, validation_rows, test_rows):
    """Reject leakage before training, calibration or checkpoint selection."""
    groups = [list(x) for x in (optimizer_rows, validation_rows, test_rows)]
    if any(not x or len(x) != len(set(x)) for x in groups):
        raise ValueError("Adaptation partitions must be nonempty and unique")
    sets = [set(x) for x in groups]
    if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Optimizer, validation and test rows must be disjoint")
    return True
