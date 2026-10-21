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
from src.evaluation.embedding_control import EmbeddingControl
from src.evaluation.features import add_embeddings, original_features
from src.evaluation.models import MAIN_MODELS, TEXT_MODELS, estimator_factory
from src.evaluation.protocol import evaluate_cv, shared_splits
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import digest, write_json


def fixtures():
    for kind, classes in (
        ("binary", 2),
        ("multiclass", 3),
        ("wide_multiclass", 30),
        ("regression", 0),
    ):
        # Even binary inner training folds must have enough present rows for PCA32.
        count = max(90, 6 * classes)
        values = np.arange(count)
        frame = pd.DataFrame(
            {
                "number": values.astype(float),
                "category": ["a" if i % 2 else "b" for i in values],
                "text": [f"Description of observation {i}." for i in values],
                "target": values % classes if classes else values * 0.1,
            }
        )
        frame.loc[::11, "number"] = np.nan
        frame.loc[::13, "category"] = None
        frame.loc[::17, "text"] = None
        official = {"target": "target", "task_type": "cls" if classes else "reg"}
        meta = extract_metadata(
            frame,
            official,
            dataset_id=f"EXP0_PREDICTOR_{kind}",
            dataset_name=kind,
            text_columns=["text"],
        )
        yield LoadedDataset(
            frame, meta, official, {"data_identity": digest({"fixture": kind, "version": 2})}, []
        )


def feature_cases(dataset, model):
    """Real adapter tests with explicitly synthetic vectors; no encoder/API execution."""
    original = original_features(dataset)
    if model in TEXT_MODELS:
        yield "original_text", original, None
        return
    base = original.drop(columns=list(dataset.metadata.text_columns))
    base.attrs["feature_artifact_ids"] = []
    yield "non_text", base, None
    rng = np.random.default_rng(20260929)
    names = [f"frozen_e5__text_000__{i:03d}" for i in range(384)]
    vectors = rng.normal(size=(len(base), 384)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    missing = dataset.frame["text"].isna().to_numpy()
    vectors[missing] = np.nan
    encoded = pd.DataFrame(vectors, index=base.index, columns=names)
    encoded.attrs.update(feature_artifact_ids=[], embedding_groups=[names])
    combined = add_embeddings(base, encoded)
    for count in (16, 32):
        yield f"e5_pca_{count}", combined, count
    combined = combined.copy()
    for mode in ("per_column", "joint"):
        task = dataset.metadata.task_type
        width = len(dataset.metadata.classes) if task == "multiclass_classification" else 1
        values = rng.uniform(size=(len(base), width))
        if task == "multiclass_classification":
            values /= values.sum(axis=1, keepdims=True)
        elif task == "regression":
            values = values * 8 - 4
        values[missing] = np.nan
        combined = combined.join(
            pd.DataFrame(
                values,
                index=base.index,
                columns=[f"synthetic_decision__{mode}__{i}" for i in range(width)],
            )
        )
    combined.attrs.update(feature_artifact_ids=[], embedding_groups=[names])
    yield "e5_pca_32_plus_both", combined, 32


def case_factory(base_factory, features, components):
    if components is None:
        return base_factory
    return lambda: EmbeddingControl(
        base_factory(),
        features.attrs["embedding_groups"],
        components=components,
    )


def run(model, *, execute=False):
    cfg = load_yaml("experiment_0/debug")
    if not execute or not cfg["predictor_execution_enabled"]:
        raise RuntimeError("Enable bounded predictor tests explicitly before execution")
    if paths.on_vsc() and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Predictor tests belong in a compute allocation")
    settings = copy.deepcopy(load_yaml("experiment_1/models")["models"][model]["runtime"])
    run_id = f"predictor_{model}_{uuid4().hex}"
    manifest = paths.manifests_dir("experiment_0") / "after_extraction" / f"{run_id}.json"
    report = {
        "model": model,
        "settings": settings,
        "status": "running",
        "fixtures": [],
        "full_dataset_profile": False,
        "embedding_vectors": "synthetic_interface_fixtures_not_real_E5_encoding",
        "contract_version": 2,
    }
    from src.cluster.containers import active_identity
    from src.cluster.evidence import current_identity

    report["execution_container"] = active_identity() or {}
    report["settings_sha256"] = digest(settings)
    report["validation_identity"] = current_identity()
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
                checkpoint_id=digest(settings),
                feature_set=f"predictor_fixture__{case}",
                representation_recipe={
                    "contract_version": 2,
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
                    "pca_components_per_column": components,
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
            "Disabled: four tasks; main predictors test non-text, PCA16, PCA32 and PCA32+decision features; native-text models test original text. Five-fold CV, inner binary thresholds and resume checks. No inference performed."
        )


if __name__ == "__main__":
    main()
