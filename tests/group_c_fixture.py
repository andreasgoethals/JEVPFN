"""Deterministic Group C export fixture (no GPU, no store): synthetic fold records with the real
C1/C2 arm grammar, passed through the actual export code (`group_c.build_tables`,
`study_export.write_export`) and read back with `group_c.read`, so figures see the real schema.

Shapes follow the study: 18 datasets (3 binary, 6 multiclass, 9 regression), the seven main
predictors and five outer folds. Values are synthetic and only exercise the figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import benchmark_export, group_c, study_export
from src.evaluation.endpoints import HEADLINES
from src.evaluation.features import (
    ADAPTED,
    HIDDEN,
    JOINT_PLUS_HIDDEN,
    PER_COLUMN_PLUS_HIDDEN,
    SUPERVISED_PROBE,
)
from src.visualize import style

DATASETS = (
    [(f"BIN_TEXT_FIXTURE_{i}", "binary_classification") for i in range(3)]
    + [(f"MUL_TEXT_FIXTURE_{i}", "multiclass_classification") for i in range(6)]
    + [(f"REG_TEXT_FIXTURE_{i}", "regression") for i in range(9)]
)
NAMES = {d: f"{d.split('_')[0].title()} fixture {d[-1]}" for d, _ in DATASETS}
BLANK = {c: None for c in benchmark_export.RECIPE_COLUMNS}


def _arm(phase, feature_set, effect, **extra):
    return dict(phase=phase, feature_set=feature_set, effect=effect, **extra)


def c1_arms():
    arms = [_arm("B1", "non_text", 0.0),
            _arm("B1", "non_text_plus_per_column", 0.030, provider="strands", view="probabilities"),
            _arm("B1", "non_text_plus_joint", 0.022, provider="strands", view="probabilities")]
    path = {"state_end": 0.008, "question_end": 0.016, "answer": 0.026}
    for view, components in (("per_column", 32), ("joint", 64)):
        for position, base in path.items():
            for layer in (12, 24):
                arms.append(_arm("C1", HIDDEN, base + (0.006 if layer == 24 else 0.0)
                                 + (0.004 if view == "per_column" else 0.0),
                                 pca_scope="per_column", pca_components=components, hidden_view=view,
                                 hidden_position=position, hidden_layer=layer))
        arms.append(_arm("C1", PER_COLUMN_PLUS_HIDDEN if view == "per_column" else JOINT_PLUS_HIDDEN,
                         0.036 if view == "per_column" else 0.030, provider="strands", view="probabilities",
                         pca_scope="per_column", pca_components=components, hidden_view=view,
                         hidden_position="answer", hidden_layer=24))
    return arms


def c2_arms():
    arms = [_arm("B1", "non_text", 0.0),
            _arm("B1", "non_text_plus_joint", 0.022, provider="strands", view="probabilities")]
    for cell, effect in (("head_only", 0.032), ("lora_only", 0.040), ("lora_head", 0.046)):
        arms.append(_arm("C2", ADAPTED, effect, adaptation_cell=cell))
    arms += [_arm("B3", "non_text_plus_tar", 0.050, encoder="e5_small_v2_tar", pca_scope="per_column",
                  pca_components=32),
             _arm("B3", "non_text_plus_tar", 0.044, encoder="e5_small_v2_tar", pca_scope="joint",
                  pca_components=64)]
    for encoder, effect in (("e5_small_v2", 0.038), ("e5_large_v2", 0.041), ("qwen3_embedding_0_6b", 0.043)):
        arms.append(_arm("B3", SUPERVISED_PROBE, effect, encoder=encoder, probe_view="joint"))
    return arms


def folds(arms, seed=7) -> pd.DataFrame:
    """Five headline-metric fold records per dataset x predictor x arm."""
    import hashlib

    rows = []
    predictor_shift = dict(zip(style.MAIN_PREDICTORS, np.linspace(-0.012, 0.012, len(style.MAIN_PREDICTORS))))
    for d_index, (dataset, task) in enumerate(DATASETS):
        regression = task == "regression"
        base = (0.45 if regression else 0.80) + 0.02 * (d_index % 5)
        scale = 2.2 if regression else 1.0  # R2 gains are larger than AUC gains
        dataset_rng = np.random.default_rng(seed + d_index)
        dataset_shift = dataset_rng.normal(0, 0.012) * scale
        for model in style.MAIN_PREDICTORS:
            model_base = base + dataset_rng.normal(0, 0.02)
            for arm in arms:
                recipe_key = repr((seed, dataset, model, sorted(arm.items()))).encode()
                recipe_id = hashlib.sha256(recipe_key).hexdigest()
                rng = np.random.default_rng(int(recipe_id[:16], 16))
                effect = 0.0 if arm["effect"] == 0 else scale * (arm["effect"] + predictor_shift[model]) + dataset_shift
                for fold in range(5):
                    value = model_base + effect + rng.normal(0, 0.004)
                    rows.append({**BLANK, **{k: v for k, v in arm.items() if k != "effect"},
                                 "dataset_id": dataset, "task_type": task, "model_id": model,
                                 "checkpoint_id": f"ckpt-{model}", "split_id": f"split-{dataset}",
                                 "outer_fold": fold, "recipe_id": recipe_id, "metric": HEADLINES[task],
                                 "value": min(value, 0.999)})
    return pd.DataFrame(rows)


def extraction() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for i, (dataset, _) in enumerate(DATASETS):
        if i == 17:
            rows.append({"dataset_id": dataset, "status": "FileNotFoundError"})
            continue
        rows.append({"dataset_id": dataset, "status": "complete", "artifact_id": f"hidden-{i}",
                     "unique_inputs": int(rng.integers(5_000, 200_000)), "joint_inputs": int(rng.integers(2_000, 60_000)),
                     "max_probability_deviation": float(rng.uniform(0.0002, 0.003)),
                     "engine_check_max_difference": float(rng.uniform(0.00005, 0.0009)),
                     "store_gb": float(rng.uniform(0.3, 9.0))})
    return pd.DataFrame(rows)


def fits() -> pd.DataFrame:
    rng = np.random.default_rng(13)
    rows = []
    for dataset, _ in DATASETS:
        for fold in range(5):
            for cell in ("head_only", "lora_only", "lora_head"):
                for part in range(3):
                    lora = cell != "head_only"
                    rows.append({"dataset_id": dataset, "fold": fold, "cell": cell, "part": part,
                                 "epochs": int(rng.integers(1, 4)), "best_validation_loss": float(rng.uniform(0.4, 1.4)),
                                 "gpu_hours": float(rng.uniform(0.2, 1.5) if lora else rng.uniform(0.01, 0.05)),
                                 "peak_gpu_gb": float(rng.uniform(30, 70) if lora else rng.uniform(2, 6))})
    return pd.DataFrame(rows)


def write(study: str, folder: Path) -> Path:
    """Build and publish the fixture export of one study with the real export code."""
    saved = group_c.extraction_table, group_c.fits_table
    group_c.extraction_table, group_c.fits_table = extraction, fits
    try:
        records = benchmark_export._parquet_safe(folds(c1_arms() if study == "C1" else c2_arms()))
        tables = {k: benchmark_export._parquet_safe(v) for k, v in group_c.build_tables(study, records).items()}
    finally:
        group_c.extraction_table, group_c.fits_table = saved
    target = Path(folder) / group_c.STUDIES[study]["file"]
    return study_export.write_export(target, {name: tables[name] for name in group_c.TABLES[study]},
                                     schema=group_c.SCHEMA, identity=group_c.analysis_identity(study),
                                     origin="fixture", extra={"study": study, "headlines": HEADLINES})


def read(study: str, folder: Path):
    """(frames, receipt) of the fixture export, read with the notebooks' own reader."""
    target = Path(folder) / group_c.STUDIES[study]["file"]
    if not target.exists():
        write(study, folder)
    return group_c.read(study, target)
