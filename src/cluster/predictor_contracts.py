"""Opt-in Experiment 0B: real predictor fits on bounded synthetic task fixtures.

These establish adapter/task compatibility, not full-dataset resource feasibility.
Run each model in its separately prepared environment and allocated compute job.
"""

import argparse
import copy
import os
from uuid import uuid4

import numpy as np
import pandas as pd

from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.evaluation.design import pca_grid
from src.evaluation.embedding_control import EmbeddingControl
from src.evaluation.features import add_embeddings, add_text_presence, original_features
from src.evaluation.models import (
    MAIN_MODELS,
    TEXT_MODELS,
    estimator_factory,
    predictor_settings_identity,
)
from src.evaluation.protocol import evaluate_cv, shared_splits
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import digest, write_json

FIXTURE_CLASSES = {"binary": 2, "multiclass": 3, "wide_multiclass": 30, "regression": 0}
CONTRACT_VERSION = 7


def embedding_fixture_recipes():
    encoders = load_yaml("experiment_2/embeddings")["encoders"]
    configs = load_yaml("feature_creation/embedding_models")["models"]
    dimensions = sorted({configs[encoder]["dimensions"] for encoder in encoders})
    return [
        (f"embedding_{width}_{scope}_pca_{count}", width, scope, count)
        for width in dimensions
        for scope, count in pca_grid()
    ]


def required_fixture_cases(model):
    cases = (
        ("original_text",)
        if model in TEXT_MODELS
        else (
            "non_text",
            "decision_per_column",
            "decision_joint",
            "decision_both",
            *(case for case, *_ in embedding_fixture_recipes()),
        )
    )
    return {
        (task, classes, case)
        for task, classes in (
            ("binary_classification", 2),
            ("multiclass_classification", 3),
            ("multiclass_classification", 30),
            ("regression", 0),
        )
        for case in cases
    }


def fixtures():
    for kind, classes in FIXTURE_CLASSES.items():
        # Binary inner training must have more than 64 rows after missing blocks.
        count = max(180, 6 * classes)
        values = np.arange(count)
        frame = pd.DataFrame(
            {
                "number": values.astype(float),
                "category": ["a" if i % 2 else "b" for i in values],
                "text": [f"Description of observation {i}." for i in values],
                "notes": [f"Notes on observation {i}." for i in values],
                "target": values % classes if classes else values * 0.1,
            }
        )
        frame.loc[::11, "number"] = np.nan
        frame.loc[::13, "category"] = None
        frame.loc[::17, "text"] = None
        frame.loc[::19, "notes"] = None
        official = {"target": "target", "task_type": "cls" if classes else "reg"}
        meta = extract_metadata(
            frame,
            official,
            dataset_id=f"EXP0_PREDICTOR_{kind}",
            dataset_name=kind,
            text_columns=["text", "notes"],
        )
        yield LoadedDataset(
            frame, meta, official,
            {"data_identity": digest({"fixture": kind, "version": CONTRACT_VERSION})}, [],
        )


def feature_cases(dataset, model):
    """Real adapter tests with explicitly synthetic vectors; no encoder/API execution."""
    original = original_features(dataset)
    if model in TEXT_MODELS:
        yield "original_text", original, None
        return
    base = add_text_presence(original.drop(columns=list(dataset.metadata.text_columns)), dataset)
    base.attrs["feature_artifact_ids"] = []
    yield "non_text", base, None
    rng = np.random.default_rng(20260929)
    recipes = embedding_fixture_recipes()
    for width in sorted({width for _, width, _, _ in recipes}):
        groups = []
        for i, text in enumerate(dataset.metadata.text_columns):
            names = [f"frozen_e5__text_{i:03d}__{j:03d}" for j in range(width)]
            vectors = rng.normal(size=(len(base), width)).astype("float32")
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors[dataset.frame[text].isna().to_numpy()] = np.nan
            groups.append(pd.DataFrame(vectors, index=base.index, columns=names))
        encoded = pd.concat(groups, axis=1)
        encoded.attrs.update(feature_artifact_ids=[], embedding_groups=[list(g) for g in groups])
        combined = add_embeddings(base, encoded)
        for case, expected_width, scope, count in recipes:
            if expected_width == width:
                features = combined.copy(deep=False)
                features.attrs["pca_scope"] = scope
                yield case, features, count
    # Decision-only fixtures exercise the three actual Experiment 1 augmentations.
    blocks = {}
    text_missing = dataset.frame[list(dataset.metadata.text_columns)].isna().to_numpy()
    task = dataset.metadata.task_type
    width = len(dataset.metadata.classes) if task == "multiclass_classification" else 1
    for mode in ("per_column", "joint"):
        masks = text_missing.T if mode == "per_column" else [text_missing.all(axis=1)]
        columns = {}
        for group, missing in enumerate(masks):
            values = rng.uniform(size=(len(base), width))
            if task == "multiclass_classification":
                values /= values.sum(axis=1, keepdims=True)
            elif task == "regression":
                values = values * 8 - 4
            values[missing] = np.nan
            columns.update({f"synthetic_decision__{mode}__{group}_{i}": values[:, i]
                            for i in range(width)})
        blocks[mode] = pd.DataFrame(columns, index=base.index)
        result = base.join(blocks[mode])
        result.attrs.update(feature_artifact_ids=[])
        yield "decision_" + mode, result, None
    result = base.join(blocks["per_column"]).join(blocks["joint"])
    result.attrs.update(feature_artifact_ids=[])
    yield "decision_both", result, None


def case_factory(base_factory, features, components):
    if components is None:
        return base_factory
    return lambda: EmbeddingControl(
        base_factory(),
        features.attrs["embedding_groups"],
        components=components,
        scope=features.attrs.get("pca_scope", "per_column"),
    )


def run(model, *, execute=False):
    cfg = load_yaml("experiment_0/debug")
    if not execute or not cfg["predictor_execution_enabled"]:
        raise RuntimeError("Enable bounded predictor tests explicitly before execution")
    if paths.on_vsc() and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Predictor tests belong in a compute allocation")
    settings = copy.deepcopy(load_yaml("experiment_1/models")["models"][model]["runtime"])
    settings["threads"] = min(
        settings["threads"], int(os.environ.get("SLURM_CPUS_PER_TASK", settings["threads"]))
    )
    if model == "tabstar":
        settings["output_phase"] = "experiment_0"
    run_id = f"predictor_{model}_{uuid4().hex}"
    manifest = paths.manifests_dir("experiment_0") / "after_extraction" / f"{run_id}.json"
    report = {
        "model": model,
        "settings": settings,
        "status": "running",
        "fixtures": [],
        "full_dataset_profile": False,
        "embedding_vectors": "synthetic_interface_fixtures_not_real_E5_encoding",
        "contract_version": CONTRACT_VERSION,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
    }
    from src.cluster.containers import active_identity
    from src.cluster.evidence import current_identity
    from src.evaluation.qualification import execution_runtime

    report["execution_container"] = active_identity() or {}
    report["runtime_identity"] = execution_runtime()
    report["settings_sha256"] = digest(settings)
    report["validation_identity"] = current_identity()
    from src.evaluation.acceptance import evaluation_identity

    report["evaluation_identity"] = evaluation_identity()
    write_json(manifest, report)
    try:
        for dataset, (case, features, components) in (
            (item, case) for item in fixtures() for case in feature_cases(item, model)
        ):
            base_factory = estimator_factory(
                model, dataset.metadata, settings, allow_model_runs=True
            )
            factory = case_factory(base_factory, features, components)
            sync = None
            if settings["device"].startswith("cuda"):
                import torch

                sync = torch.cuda.synchronize
            args = dict(
                specification=shared_splits(dataset),
                model_id=model,
                checkpoint_id=predictor_settings_identity(settings),
                feature_set=f"predictor_fixture__{case}",
                representation_recipe={
                    "contract_version": CONTRACT_VERSION,
                    "pca_components": components,
                    "synthetic_vectors": components is not None,
                },
                allow_model_runs=True,
                phase="experiment_0",
                estimator_settings=settings,
                feature_artifact_ids=[],
                device=settings["device"],
                synchronize=sync,
            )
            records = evaluate_cv(dataset, features, factory, **args)

            def forbidden_factory():
                raise AssertionError("Completed predictor fold was fitted again")

            assert evaluate_cv(dataset, features, forbidden_factory, **args) == records
            report["fixtures"].append(
                {
                    "task_type": dataset.metadata.task_type,
                    "classes": len(dataset.metadata.classes),
                    "rows": len(features),
                    "feature_case": case,
                    "pca_components": components,
                    "pca_scope": features.attrs.get("pca_scope", "per_column")
                    if components
                    else None,
                    "records": [str(p) for p in records],
                    "resume_without_refitting": True,
                }
            )
            write_json(manifest, report)
        report["status"] = "predictor_contracts_passed_requires_full_fold_profile"
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        write_json(manifest, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MAIN_MODELS + TEXT_MODELS)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        run(args.model, execute=True)
    else:
        print(
            "Disabled: four tasks; main predictors test non-text, per-column PCA32 and joint "
            "PCA64 on two text fields at each encoder width, plus decision-only features. "
            "Native-text models test original text. Five-fold CV, inner binary thresholds and "
            "resume checks. No inference performed."
        )


if __name__ == "__main__":
    main()
