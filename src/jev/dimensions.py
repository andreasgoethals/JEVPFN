"""Numeric feature widths from task metadata, independent of API requests or labels."""

import pandas as pd


def numeric_feature_dimensions(audit, *, excluded=()):
    """Count model-input columns; repeated text saves calls, never shrinks matrix width.

    The combined arm removes the identical joint block on single-text datasets.
    Float64 sizes describe only an uncompressed dense numeric array: no row IDs,
    original features, responses, confidence, Parquet/SQLite overhead or compression.
    """
    excluded = set(excluded)
    records = []
    for row in audit.summary.itertuples(index=False):
        meta = audit.metadata(row.dataset_id)
        texts = len(meta.text_columns)
        width = len(meta.classes) if meta.task_type == "multiclass_classification" else 1
        separate, joint = texts * width, width
        combined = separate + (joint if texts > 1 else 0)
        records.append(
            {
                "dataset_id": meta.dataset_id,
                "dataset": meta.dataset_name,
                "status": "REMOVED" if meta.dataset_id in excluded else "Included",
                "task": meta.task_type,
                "rows": int(row.rows),
                "text_columns": texts,
                "non_text_columns": len(meta.non_text_columns),
                "classes": len(meta.classes) if meta.classes else None,
                "features_per_text_input": width,
                "per_column_features": separate,
                "joint_features": joint,
                "combined_features": combined,
                "predictor_columns_with_combined": len(meta.non_text_columns) + combined,
                "combined_numeric_values": int(row.rows) * combined,
                "combined_float64_mib": int(row.rows) * combined * 8 / 1024**2,
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values(["task", "combined_features", "dataset_id"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
