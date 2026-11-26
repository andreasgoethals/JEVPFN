"""Reviewed research plans and paired-analysis contracts; never dispatch model calls."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.data.loaders import load_catalog
from src.data.selection import selected_entries
from src.evaluation.widths import feature_width
from src.utils.config import load_yaml
from src.utils.serialization import digest


def _metadata(entry):
    stats = entry["paper_summary"]
    return SimpleNamespace(
        text_columns=range(stats["text_columns"]),
        non_text_columns=range(stats["non_text_columns"]),
        classes=[],
        task_type="regression",
    )


def regression_width_table():
    """Catalog widths, plus conservative Kumo expansion; fits widen Kumo past 500 columns."""
    cfg = load_yaml("B1")
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
            for view in ("probabilities",):
                row[f"{mode}_{view}"] = feature_width(
                    meta, dict(feature_set="non_text_plus_" + mode, provider="strands", view=view)
                )
        row["kumo_conservative_max"] = row["both_probabilities"] + len(meta.non_text_columns)
        rows.append(row)
    return pd.DataFrame(rows)


def adaptation_plan():
    """Group C2 (approved 20 November 2026): every cross-fitted Strands training fit
    (dataset x outer fold x part x trained cell); zero-shot reuses B1. No dispatch."""
    from src.evaluation.strands_adaptation import training_plan

    return training_plan()


def adaptation_prediction_plan():
    """C2 predictor configurations: every trained cell x dataset x predictor (five outer fits
    each, fixed 0.5 binary threshold); they reuse each fold's cross-fitted probabilities."""
    from src.evaluation.campaign import benchmark_plan

    return [dict(arm, binary_threshold=0.5, execution_enabled=False)
            for arm in benchmark_plan() if arm["phase"] == "C2"]


def random_probe_requests(request_keys, *, dataset_id, limit=500, seed=42, study="wording"):
    """Uniform distinct semantic requests, shared across providers; never sample outcomes.

    Keys must bind the frozen bank, text group, definitions and input bytes. Repeated
    row occurrences of an identical request do not give it extra sampling weight.
    Field-order callers first restrict eligibility to >=2 nonempty named fields.
    """
    keys = list(request_keys)
    if not dataset_id or not study or any(not isinstance(k, str) or not k for k in keys):
        raise ValueError("Nonempty semantic request keys are required")
    return random_probe_rows(sorted(set(keys)), dataset_id=f"{dataset_id}:{study}",
                             limit=limit, seed=seed)


def stable_probe_sample(input_keys, *, limit=100, seed=42):
    """Unique request keys, independent of target values, row order or model outcomes."""
    if type(limit) is not int or limit < 1:
        raise ValueError("Probe limit must be positive")
    return sorted(set(input_keys), key=lambda key: digest([seed, key]))[:limit]


def random_probe_rows(row_ids, *, dataset_id, limit=500, seed=42):
    """Uniform source-row sample without replacement; duplicates of text remain eligible.

    Canonical row IDs, not text keys or targets, define the sampling population.
    Store the returned IDs with the source fingerprint before running any variants.
    """
    rows = sorted(row_ids)
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("Source row IDs must be nonempty and unique")
    if type(limit) is not int or limit < 1 or not dataset_id:
        raise ValueError("A positive limit and dataset identity are required")
    rng = np.random.Generator(np.random.PCG64(int(digest([seed, dataset_id])[:16], 16)))
    indices = rng.choice(len(rows), size=min(limit, len(rows)), replace=False)
    return [rows[i] for i in indices]


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
    # Keep tied cut-offs: dropping them would renumber the bins, so "bin_2" would
    # mean a different quartile in different folds/datasets before pooling.
    # Ties simply leave some bins empty.
    edges = np.quantile(observed, quantiles)
    bins = np.searchsorted(edges, query, side="right")
    bins[~np.isfinite(query)] = -1
    return bins, edges


def validate_adaptation_partition(optimizer_rows, validation_rows, calibration_rows, test_rows):
    """Require four distinct roles before training or fitting a temperature.

    Checkpoint selection and temperature fitting must not silently share rows.
    The caller must additionally bind these identities to the saved outer split.
    """
    groups = [list(x) for x in (optimizer_rows, validation_rows, calibration_rows, test_rows)]
    if any(not x or len(x) != len(set(x)) for x in groups):
        raise ValueError("Adaptation partitions must be nonempty and unique")
    sets = [set(x) for x in groups]
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i + 1, len(sets))):
        raise ValueError("Optimizer, validation, calibration and test rows must be disjoint")
    return True
