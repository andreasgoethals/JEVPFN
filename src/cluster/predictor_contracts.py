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
from src.evaluation.features import original_features
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
        count = max(30, 6 * classes)
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
            frame, meta, official, {"data_identity": digest({"fixture": kind, "version": 1})}, []
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
    }
    write_json(manifest, report)
    try:
        for dataset in fixtures():
            features = original_features(dataset)
            if model in MAIN_MODELS:
                features = features.drop(columns=list(dataset.metadata.text_columns))
                features.attrs["feature_artifact_ids"] = []
            factory = estimator_factory(model, dataset.metadata, settings, allow_model_runs=True)
            sync = None
            if settings["device"].startswith("cuda"):
                import torch

                sync = torch.cuda.synchronize
            args = dict(
                specification=shared_splits(dataset),
                model_id=model,
                checkpoint_id=digest(settings),
                feature_set="predictor_fixture",
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
            "Disabled: four task fixtures, five-fold CV, inner binary thresholds and resume checks; no inference performed."
        )


if __name__ == "__main__":
    main()
