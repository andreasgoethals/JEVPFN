"""Upper-tail diagnostics using actual, nonempty text lengths from the audit."""

import numpy as np
import pandas as pd

from src.data.audit import _atomic_csv, audit_lock


def distribution(values):
    values = np.asarray(values)
    values = values[values > 0]
    if not len(values):
        return {k: None for k in ("mean", "p01", "p25", "median", "p75", "p95", "p99", "max")}
    return {
        "mean": float(values.mean()),
        "max": int(values.max()),
        **{
            name: float(np.quantile(values, q))
            for name, q in (
                ("p01", 0.01),
                ("p25", 0.25),
                ("median", 0.5),
                ("p75", 0.75),
                ("p95", 0.95),
                ("p99", 0.99),
            )
        },
    }


def long_text_audit(audit, cfg):
    """Return dataset/column tables and one longest-cell showcase per dataset.

    Text-only length excludes JSON escaping, task metadata and questions. A context-limit
    comparison here is a screening diagnostic, not an exact request-validity decision.
    """
    ratio = cfg["token_estimation"]["characters_per_token"]
    limit = cfg["provider_reference"]["state_plus_longest_question_tokens"]
    datasets, columns, examples = [], [], []
    with audit_lock(audit.folder):
        for dataset_id in audit.summary.dataset_id:
            meta = audit.metadata(dataset_id)
            detail = audit.details[dataset_id]
            with np.load(audit.folder / f"{dataset_id}_lengths.npz") as arrays:
                lengths = [arrays[f"characters_{i}"] for i in range(len(meta.text_columns))]
            joint = np.sum(lengths, axis=0)
            cell_maxima = []
            for col, chars in zip(meta.text_columns, lengths):
                tokens = np.ceil(chars / ratio).astype(np.int64)
                stats = {
                    "dataset_id": dataset_id,
                    "dataset": meta.dataset_name,
                    "text_column": col,
                    "nonempty_cells": int((chars > 0).sum()),
                    "missing_or_empty_cells": int((chars == 0).sum()),
                    **{f"characters_{k}": v for k, v in distribution(chars).items()},
                    **{f"tokens_{k}": v for k, v in distribution(tokens).items()},
                    "cells_above_text_only_screen": int((tokens > limit).sum()),
                }
                example = detail["longest_text_examples"].get(col)
                stats["longest_row_id"] = example["row_id"] if example else None
                columns.append(stats)
                if example:
                    cell_maxima.append(
                        {"dataset": meta.dataset_name, "text_column": col, **example}
                    )
            pooled = np.concatenate(lengths)
            joint_tokens = np.ceil(joint / ratio).astype(np.int64)
            longest = max(cell_maxima, key=lambda x: x["characters"]) if cell_maxima else {}
            datasets.append(
                {
                    "dataset_id": dataset_id,
                    "dataset": meta.dataset_name,
                    "text_columns": len(lengths),
                    "nonempty_text_cells": int((pooled > 0).sum()),
                    **{f"cell_characters_{k}": v for k, v in distribution(pooled).items()},
                    **{
                        f"cell_tokens_{k}": v
                        for k, v in distribution(np.ceil(pooled / ratio)).items()
                    },
                    **{f"joint_characters_{k}": v for k, v in distribution(joint).items()},
                    **{f"joint_tokens_{k}": v for k, v in distribution(joint_tokens).items()},
                    "cells_above_text_only_screen": int((np.ceil(pooled / ratio) > limit).sum()),
                    "joint_rows_above_text_only_screen": int((joint_tokens > limit).sum()),
                    "longest_text_column": longest.get("text_column"),
                    "longest_row_id": longest.get("row_id"),
                }
            )
            if longest:
                examples.append(longest)
    dataset_table, column_table = pd.DataFrame(datasets), pd.DataFrame(columns)
    _atomic_csv(dataset_table, audit.folder / "long_text_datasets.csv")
    _atomic_csv(column_table, audit.folder / "long_text_columns.csv")
    return dataset_table, column_table, examples
