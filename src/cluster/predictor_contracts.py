"""Opt-in B0: real predictor fits on bounded synthetic task fixtures.

These establish adapter/task compatibility, not full-dataset resource feasibility.
Run each model in its separately prepared environment and allocated compute job.
"""

import argparse
import copy
import json
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
from src.utils.serialization import digest, text_value, write_json

FIXTURE_CLASSES = {"binary": 2, "multiclass": 3, "wide_multiclass": 30, "regression": 0}
CONTRACT_VERSION = 8


def embedding_fixture_recipes():
    encoders = load_yaml("B2")["encoders"]
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
        for case in (cases + (("decision_per_column_probabilities",)
                              if task == "regression" and model not in TEXT_MODELS else ()))
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
        frame.loc[1, "text"] = "   "
        frame.loc[2, "notes"] = "Récit 新 – observation " * 30
        frame.loc[3, "category"] = "appears in one row"
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
            vectors[dataset.frame[text].map(text_value).isna().to_numpy()] = np.nan
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
    text_missing = np.array([dataset.frame[c].map(text_value).isna().to_numpy()
                             for c in dataset.metadata.text_columns]).T
    task = dataset.metadata.task_type
    width = 9 if task == "regression" else len(dataset.metadata.classes) if task == "multiclass_classification" else 1
    for mode in ("per_column", "joint"):
        masks = text_missing.T if mode == "per_column" else [text_missing.all(axis=1)]
        columns = {}
        for group, missing in enumerate(masks):
            values = rng.uniform(size=(len(base), width))
            if width > 1:
                values /= values.sum(axis=1, keepdims=True)
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
    if task == "regression":
        # Exercise the standalone nine-probability fixture as well as the three production modes.
        columns = {}
        for group, missing in enumerate(text_missing.T):
            values = rng.dirichlet(np.ones(9), size=len(base))
            values[missing] = np.nan
            columns.update({f"synthetic_probabilities__{group}__p_{i}": values[:, i]
                            for i in range(9)})
        result = base.join(pd.DataFrame(columns, index=base.index))
        result.attrs.update(feature_artifact_ids=[])
        yield "decision_per_column_probabilities", result, None


def case_factory(base_factory, features, components):
    if components is None:
        return base_factory
    return lambda: EmbeddingControl(
        base_factory(),
        features.attrs["embedding_groups"],
        components=components,
        scope=features.attrs.get("pca_scope", "per_column"),
    )


def interrupt_fixture(dataset, features, factory, args):
    """Complete fold 0, then interrupt fold 1 immediately after its first durable write."""
    from unittest.mock import patch

    from src.evaluation import protocol

    first = evaluate_cv(dataset, features, factory, fold_numbers=[0], **args)

    class FixtureInterruption(Exception):
        pass

    save_record = protocol.save_evaluation

    def interrupt_after_outer_record(**kwargs):
        raise FixtureInterruption(save_record(**kwargs))

    try:
        with patch.object(protocol, "save_evaluation", interrupt_after_outer_record):
            evaluate_cv(dataset, features, factory, fold_numbers=[1], **args)
    except FixtureInterruption as exc:
        return first, exc.args[0]
    raise RuntimeError("Unfinished fixture fold was unexpectedly reused")


def run(model, *, execute=False):
    cfg = load_yaml("0/debug")
    if not execute or not cfg["predictor_execution_enabled"]:
        raise RuntimeError("Enable bounded predictor tests explicitly before execution")
    if paths.on_vsc() and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Predictor tests belong in a compute allocation")
    settings = copy.deepcopy(load_yaml("B1/models")["models"][model]["runtime"])
    settings["threads"] = min(
        settings["threads"], int(os.environ.get("SLURM_CPUS_PER_TASK", settings["threads"]))
    )
    if model == "tabstar":
        settings["output_phase"] = "0"
    run_id = f"predictor_{model}_{uuid4().hex}"
    manifest = paths.phase_path("B0", "manifests") / f"{run_id}.json"
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
                    "evaluation_identity": report["evaluation_identity"],
                    "runtime_identity": report["runtime_identity"],
                    "pca_components": components,
                    "synthetic_vectors": components is not None,
                },
                allow_model_runs=True,
                phase="0",
                estimator_settings=settings,
                feature_artifact_ids=[],
                device=settings["device"],
                synchronize=sync,
            )
            # One distinct trial per runtime tests restarting with a completed fold and an
            # interrupted next fold. A rerun must not inherit a previously completed trial.
            partial_trial = dataset.metadata.task_type == "binary_classification" and case in {
                "original_text", "non_text"
            }
            if partial_trial:
                args["representation_recipe"]["partial_resume_trial"] = run_id
                first, incomplete = interrupt_fixture(dataset, features, factory, args)
            records = evaluate_cv(dataset, features, factory, **args)
            if partial_trial and (not first or records[:len(first)] != first or incomplete in records):
                raise RuntimeError("Interrupted fixture resume changed a completed fold or reused an incomplete record")

            def forbidden_factory():
                raise AssertionError("Completed predictor fold was fitted again")

            # An explicit check, not `assert`: `python -O` must not skip the resume contract.
            if not records or evaluate_cv(dataset, features, forbidden_factory, **args) != records:
                raise RuntimeError("Completed predictor folds were not reused exactly on resume")
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
                    "partial_fold_resume_verified": partial_trial,
                    **({"interrupted_record_id": incomplete.key} if partial_trial else {}),
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


def verify_published_fixtures(report):
    """Verify immutable fold markers, predictions and held-out identities on durable PROJECT."""
    from src.evaluation.metrics import classification_metrics, regression_metrics
    from src.evaluation.store import RecordRef, completed, database, read_predictions, readonly

    if paths.on_vsc() and paths.staging_override() != paths.durable_project_override():
        raise ValueError("Verify published B0 fixtures outside scratch on durable PROJECT")
    datasets = {(d.metadata.task_type, len(d.metadata.classes)): d for d in fixtures()}
    for case in report["fixtures"]:
        task = case["task_type"]
        data = datasets[task, case["classes"]]
        split = shared_splits(data)
        dataset_id = data.metadata.dataset_id
        db = database(dataset_id, "0")
        expected = {(i, role) for i in range(len(split["folds"])) for role in
                    (("outer_test", "inner_oof_binary_threshold") if task == "binary_classification"
                     else ("outer_test",))}
        marker_records = {}
        for locator in case["records"]:
            original, sep, record_id = locator.rpartition("#")
            if not sep or not original.replace("\\", "/").endswith("/" + db.name):
                raise ValueError("B0 fixture record locator has a different dataset")
            ref = RecordRef(db, record_id)
            body = json.loads(ref.read_text())
            meta = body["metadata"]
            fold, role = meta["outer_fold"], meta["prediction_role"]
            if (fold, role) not in expected:
                raise ValueError("B0 fixture has duplicate or unexpected fold/role")
            expected.remove((fold, role))
            recipe_id = meta["fold_recipe_id"]
            if recipe_id not in marker_records:
                with readonly(db) as conn:
                    marker = conn.execute("SELECT recipe FROM folds WHERE id=?", (recipe_id,)).fetchone()
                if not marker or digest(recipe := json.loads(marker[0])) != recipe_id:
                    raise ValueError("B0 fixture lacks a current completed fold")
                marker_records[recipe_id] = set(completed(recipe, "0"))
            if ref not in marker_records[recipe_id]:
                raise ValueError("B0 fixture record differs from its completed fold")
            table = read_predictions(ref)
            positions = split["folds"][fold]["test" if role == "outer_test" else "train"]
            spec = meta["representation_recipe"]
            if (meta["dataset_id"] != dataset_id or meta["data_identity"] != data.source["data_identity"]
                    or meta["split_id"] != split["split_id"] or meta["model_id"] != report["model"]
                    or meta["checkpoint_id"] != predictor_settings_identity(report["settings"])
                    or meta["feature_set"] != "predictor_fixture__" + case["feature_case"]
                    or spec.get("contract_version") != CONTRACT_VERSION
                    or spec.get("runtime_identity") != report["runtime_identity"]
                    or spec.get("evaluation_identity") != report["evaluation_identity"]
                    or (case.get("partial_fold_resume_verified") and not spec.get("partial_resume_trial"))
                    or table.row_id.tolist() != [f"csv:{i}" for i in positions]
                    or table.y_true.tolist() != data.frame[data.metadata.target].iloc[positions].tolist()):
                raise ValueError("Published B0 fixture does not match its runtime, case or expected rows")
            if task == "regression":
                scores = regression_metrics(table.y_true, table.prediction)
            else:
                columns = [f"p_c{i:03d}" for i in range(case["classes"])]
                if ([col for col in table if col.startswith("p_c")] != columns
                        or meta["class_order"] != list(data.metadata.classes)):
                    raise ValueError("Published fixture probability columns differ from its class order")
                scores = classification_metrics(table.y_true, table[columns], classes=meta["class_order"],
                                                positive_class=meta["positive_class"],
                                                selected_threshold=meta.get("selected_threshold"))
            published = body["scores"]["metrics"]
            reproduced = scores["metrics"]
            if (set(reproduced) != set(published)
                    or {k for k,v in published.items() if v is None} != {k for k,v in reproduced.items() if v is None}
                    or any(not np.isclose(value, published[k], rtol=1e-10, atol=1e-12)
                           for k, value in reproduced.items() if value is not None)):
                raise ValueError("Published fixture scores do not reproduce from its held-out predictions")
        if expected or len(marker_records) != len(split["folds"]):
            raise ValueError("B0 fixture omitted a published fold/role")
        if case.get("partial_fold_resume_verified"):
            orphan = RecordRef(db, case["interrupted_record_id"])
            interrupted = json.loads(orphan.read_text())["metadata"]
            fold_records = marker_records.get(interrupted["fold_recipe_id"])
            if (interrupted["outer_fold"] != 1 or interrupted["prediction_role"] != "outer_test"
                    or not fold_records or orphan in fold_records or len(read_predictions(orphan)) != len(split["folds"][1]["test"])):
                raise ValueError("Completed fixture fold reused an interrupted prediction record")


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
