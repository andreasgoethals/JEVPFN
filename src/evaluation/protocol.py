"""Shared CV and inner-OOF threshold evaluation with an explicitly supplied estimator.

Nothing runs at import. This module does not download models or select checkpoints.
Preprocessing must be inside the supplied estimator/pipeline, fitted by its fit method.
"""

from __future__ import annotations

import hashlib
import warnings
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from src.evaluation.metrics import classification_metrics, regression_metrics
from src.evaluation.records import (
    PhaseTimer,
    ResourceMonitor,
    complete_fold,
    completed_fold,
    save_evaluation,
)
from src.evaluation.text_overlap import (
    OVERLAP_VERSION,
    cohort_scores,
    fold_overlap,
    text_identities,
)
from src.utils import paths
from src.utils.locking import file_lock
from src.utils.provenance import code_identity
from src.utils.serialization import canonical_json, digest, scalar, write_json


def feature_input_identity(features: pd.DataFrame) -> dict:
    """Fingerprint the exact values, schema and row order supplied to an estimator.

    Pandas' stable per-row hashes keep this bounded by rows rather than serializing a
    potentially large text/feature matrix into one JSON string. Record its version
    so a changed hashing implementation cannot masquerade as the same convention.
    """
    if not features.columns.is_unique:
        raise ValueError("Feature column names must be unique")
    schema = []
    for column, dtype in features.dtypes.items():
        field = {"name": scalar(column), "dtype": str(dtype)}
        if isinstance(dtype, pd.CategoricalDtype):
            field.update(categories=[scalar(v) for v in dtype.categories], ordered=dtype.ordered)
        schema.append(field)
    hashed = pd.util.hash_pandas_object(features, index=True).to_numpy(dtype="<u8")
    return {
        "algorithm": "pandas-hash-pandas-object-then-sha256-v1",
        "pandas_version": pd.__version__,
        "rows": len(features),
        "schema": schema,
        "row_ids_sha256": digest(features.index.tolist()),
        "values_sha256": hashlib.sha256(hashed.tobytes()).hexdigest(),
    }


def shared_splits(dataset, *, seed: int = 20260929, folds: int = 5) -> dict:
    """One positional partition specification shared by all feature/model comparisons."""
    y = dataset.frame[dataset.metadata.target]
    if y.isna().any() or len(y) < folds or folds < 2:
        raise ValueError("Missing targets or insufficient rows for CV")
    classification = dataset.metadata.task_type != "regression"
    encoded = np.array([canonical_json(scalar(v)) for v in y]) if classification else None
    if classification and set(encoded) != {canonical_json(c) for c in dataset.metadata.classes}:
        raise ValueError("Every class in the declared vocabulary must be observed")
    if classification and pd.Series(encoded).value_counts().min() < folds:
        raise ValueError("Every class requires at least one observation per outer fold")
    splitter = (
        StratifiedKFold(folds, shuffle=True, random_state=seed)
        if classification
        else KFold(folds, shuffle=True, random_state=seed)
    )
    specification = {
        "dataset_id": dataset.metadata.dataset_id,
        "data_identity": dataset.source["data_identity"],
        "task_schema_sha256": dataset.metadata.fingerprint,
        "rows": len(y),
        "seed": seed,
        "strategy": "stratified_kfold" if classification else "kfold",
        "folds": [
            {"train": train.tolist(), "test": test.tolist()}
            for train, test in splitter.split(np.zeros(len(y)), encoded)
        ],
    }
    return {**specification, "split_id": digest(specification)}


def validate_splits(dataset, specification):
    body = {k: v for k, v in specification.items() if k != "split_id"}
    if specification["split_id"] != digest(body):
        raise ValueError("Split checksum mismatch")
    if (
        body["dataset_id"] != dataset.metadata.dataset_id
        or body["data_identity"] != dataset.source["data_identity"]
        or body["task_schema_sha256"] != dataset.metadata.fingerprint
        or body["rows"] != len(dataset.frame)
    ):
        raise ValueError("Split belongs to another dataset version")
    expected = set(range(len(dataset.frame)))
    tests = []
    for fold in body["folds"]:
        train, test = fold["train"], fold["test"]
        if (
            len(train) != len(set(train))
            or len(test) != len(set(test))
            or set(train) & set(test)
            or set(train) | set(test) != expected
            or not train
            or not test
        ):
            raise ValueError("Invalid outer partition")
        tests.extend(test)
    if sorted(tests) != sorted(expected):
        raise ValueError("Outer test folds must cover each row exactly once")


def select_f1_threshold(y_positive, probability) -> float:
    """Maximize validation F1; ties favor closest to 0.5, then the larger threshold."""
    y = np.asarray(y_positive)
    p = np.asarray(probability, dtype=float)
    if y.shape != p.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid threshold-selection inputs")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Threshold validation requires both classes")
    candidates = np.unique(np.r_[0.0, 0.5, 1.0, p])
    # Sort once: cumulative counts evaluate all thresholds without O(n**2) rescoring.
    order = np.argsort(p)
    sorted_p, sorted_y = p[order], y[order]
    cumulative = np.r_[0, np.cumsum(sorted_y)]
    cut = np.searchsorted(sorted_p, candidates, side="left")
    tp = cumulative[-1] - cumulative[cut]
    fp = len(p) - cut - tp
    fn = cumulative[cut]
    denominator = 2 * tp + fp + fn
    scores = np.divide(
        2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0
    )
    tied = candidates[np.isclose(scores, scores.max(), rtol=0, atol=1e-12)]
    return float(min(tied, key=lambda t: (abs(t - 0.5), -t)))


def _probabilities(model, features, classes):
    predicted = np.asarray(model.predict_proba(features), dtype=float)
    model_classes = [canonical_json(scalar(c)) for c in model.classes_]
    wanted = [canonical_json(c) for c in classes]
    if set(model_classes) != set(wanted) or predicted.shape != (len(features), len(classes)):
        raise ValueError("Estimator must return the complete fixed class vocabulary")
    predicted = predicted[:, [model_classes.index(c) for c in wanted]]
    if (
        not np.isfinite(predicted).all()
        or ((predicted < 0) | (predicted > 1)).any()
        or not np.allclose(predicted.sum(axis=1), 1, atol=1e-5, rtol=0)
    ):
        raise ValueError("Invalid model probabilities")
    return predicted


def evaluate_cv(
    dataset,
    features: pd.DataFrame,
    estimator_factory,
    *,
    specification: dict,
    model_id: str,
    checkpoint_id: str,
    feature_set: str,
    allow_model_runs: bool = False,
    inner_folds: int = 3,
    phase: str = "experiment_1",
    synchronize=None,
    device: str = "cpu",
    estimator_settings: dict | None = None,
    feature_artifact_ids: list[str] | None = None,
    permutation_columns: list[str] | None = None,
    fold_numbers: list[int] | None = None,
    representation_recipe: dict | None = None,
):
    """Fit only after an explicit opt-in; save inner and outer predictions on PROJECT.

    Factory must return a fresh estimator with fold-local preprocessing each time.
    GPU callers must provide device and synchronization; no silent row subsampling.
    Synthetic tests use fake estimators, never train a prediction model.
    Adapters must provide resolved estimator settings and artifact IDs (an empty
    list for arms without cached features). These enter the immutable run identity.
    """
    if allow_model_runs is not True:
        raise RuntimeError("Model execution is disabled; an explicit opt-in is required")
    validate_splits(dataset, specification)
    if phase not in {
        "experiment_0",
        "experiment_1",
        "experiment_2",
        "experiment_3",
        "experiment_4",
    }:
        raise ValueError("Evaluation requires an experiment phase")
    row_ids = [f"csv:{i}" for i in range(len(dataset.frame))]
    if list(features.index) != row_ids or dataset.metadata.target in features:
        raise ValueError("Features must align exactly by source row ID and exclude the target")
    if not checkpoint_id:
        raise ValueError("Explicit checkpoint/package identity is required")
    if estimator_settings is not None and not isinstance(estimator_settings, dict):
        raise ValueError("Estimator settings must be an explicit resolved configuration")
    if feature_artifact_ids is None:
        feature_artifact_ids = features.attrs.get("feature_artifact_ids")
    if feature_artifact_ids is not None and (
        not isinstance(feature_artifact_ids, (list, tuple))
        or any(not isinstance(v, str) or not v for v in feature_artifact_ids)
        or len(set(feature_artifact_ids)) != len(feature_artifact_ids)
    ):
        raise ValueError("Feature artifact IDs must be distinct nonempty strings")
    if estimator_settings is None or feature_artifact_ids is None:
        raise ValueError(
            "Resolved estimator settings and explicit feature artifact IDs are required"
        )
    feature_identity = feature_input_identity(features)
    permutation_columns = list(permutation_columns or [])
    if len(set(permutation_columns)) != len(permutation_columns) or set(permutation_columns) - set(
        features
    ):
        raise ValueError("Invalid permutation-control columns")

    def partition_inputs(train, test, seed):
        left, right = features.iloc[train].copy(), features.iloc[test].copy()
        if permutation_columns:
            from src.evaluation.features import shuffle_feature_block

            shuffled = shuffle_feature_block(
                left[permutation_columns], right[permutation_columns], seed=seed
            )
            left.loc[:, permutation_columns], right.loc[:, permutation_columns] = [
                p.to_numpy() for p in shuffled
            ]
        return left, right

    # Fail before any fit if configuration metadata cannot be serialized reproducibly.
    canonical_json(estimator_settings)
    canonical_json(representation_recipe)
    target = dataset.frame[dataset.metadata.target].reset_index(drop=True)
    metadata = dataset.metadata
    binary = metadata.task_type == "binary_classification"
    classes = metadata.classes
    positive = classes[-1] if binary else None
    text_ids = text_identities(dataset)
    paths.results_dir("splits", phase=phase).mkdir(parents=True, exist_ok=True)
    write_json(
        paths.results_dir("splits", specification["split_id"] + ".json", phase=phase), specification
    )
    records = []
    if fold_numbers is not None and (
        not fold_numbers
        or len(set(fold_numbers)) != len(fold_numbers)
        or any(type(i) is not int or not 0 <= i < len(specification["folds"]) for i in fold_numbers)
    ):
        raise ValueError("Invalid requested outer-fold subset")
    recipe = {
        "schema": "resumable-cv-v1",
        "dataset_id": metadata.dataset_id,
        "data_identity": dataset.source["data_identity"],
        "task_type": metadata.task_type,
        "task_schema_sha256": metadata.fingerprint,
        "model_id": model_id,
        "checkpoint_id": checkpoint_id,
        "feature_set": feature_set,
        "feature_input_identity": feature_identity,
        "feature_artifact_ids": list(feature_artifact_ids),
        "representation_recipe": representation_recipe,
        "estimator_settings": estimator_settings,
        "split_id": specification["split_id"],
        "inner_folds": inner_folds if binary else None,
        "permutation_columns": permutation_columns,
        "text_overlap_version": OVERLAP_VERSION,
        "protocol_sha256": code_identity(
            [
                "src/evaluation/protocol.py",
                "src/evaluation/metrics.py",
                "src/evaluation/records.py",
                "src/evaluation/features.py",
                "src/evaluation/text_overlap.py",
            ]
        ),
    }
    for number, fold in enumerate(specification["folds"]):
        if fold_numbers is not None and number not in fold_numbers:
            continue
        fold_recipe = {**recipe, "outer_fold": number}
        # Fixed lock stripes bound filesystem entries independently of campaign size.
        lock_slot = int(digest(fold_recipe)[:8], 16) % 4096
        with file_lock(
            paths.results_dir("cv_locks", f"{lock_slot:04d}.lock", phase=phase), timeout=86400
        ):
            reused = completed_fold(fold_recipe, phase)
            if reused is not None:
                records.extend(reused)
                continue
            attempt_id = uuid4().hex
            record_start = len(records)
            with (
                ResourceMonitor(device) as monitor,
                warnings.catch_warnings(record=True) as captured,
            ):
                warnings.simplefilter("always")
                train, test = np.array(fold["train"]), np.array(fold["test"])
                timer = PhaseTimer(device=device, synchronize=synchronize)
                threshold, oof = None, None
                if binary:
                    encoded = np.array([canonical_json(scalar(v)) for v in target.iloc[train]])
                    if pd.Series(encoded).value_counts().min() < inner_folds:
                        raise ValueError("Insufficient examples for binary inner stratified CV")
                    oof = np.full((len(train), 2), np.nan)
                    splitter = StratifiedKFold(
                        inner_folds, shuffle=True, random_state=specification["seed"] + number
                    )
                    for inner_number, (a, b) in enumerate(splitter.split(train, encoded)):
                        inner_train, inner_test = partition_inputs(
                            train[a], train[b], specification["seed"] + 1000 * number + inner_number
                        )
                        with timer.measure("inner_model_loading"):
                            model = estimator_factory()
                        try:
                            with timer.measure("inner_fit"):
                                model.fit(inner_train, target.iloc[train[a]])
                            with timer.measure("inner_predict"):
                                oof[b] = _probabilities(model, inner_test, classes)
                        finally:
                            # Release this checkpoint before constructing the next estimator.
                            # Adapters must not retain it in global caches or factory closures.
                            del model
                    with timer.measure("threshold_selection"):
                        threshold = select_f1_threshold(
                            (encoded == canonical_json(positive)).astype(int), oof[:, -1]
                        )
                outer_train, outer_test = partition_inputs(
                    train, test, specification["seed"] + 1000 * number + 999
                )
                with timer.measure("model_loading"):
                    model = estimator_factory()
                try:
                    with timer.measure("fit"):
                        model.fit(outer_train, target.iloc[train])
                    with timer.measure("predict"):
                        prediction = (
                            np.asarray(model.predict(outer_test), dtype=float)
                            if metadata.task_type == "regression"
                            else _probabilities(model, outer_test, classes)
                        )
                finally:
                    del model
                base = {
                    "dataset_id": metadata.dataset_id,
                    "data_identity": dataset.source["data_identity"],
                    "model_id": model_id,
                    "checkpoint_id": checkpoint_id,
                    "feature_set": feature_set,
                    "feature_columns": list(features.columns),
                    "feature_input_identity": feature_identity,
                    "feature_artifact_ids": list(feature_artifact_ids),
                    "representation_recipe": representation_recipe,
                    "estimator_settings": estimator_settings,
                    "outer_fold": number,
                    "split_id": specification["split_id"],
                    "n_train": len(train),
                    "n_validation": len(train) if binary else 0,
                    "target_scale": "original_dataset_target",
                    "class_order": list(classes),
                    "positive_class": positive,
                    "selected_threshold": threshold,
                    "inner_folds": inner_folds if binary else None,
                    "preprocessing": "inside_estimator_fit_training_partition_only",
                    "permutation_columns": permutation_columns,
                    "single_text_representations_equivalent": len(metadata.text_columns) == 1,
                    "fold_recipe_id": digest(fold_recipe),
                }
                for role, positions, predictions in [
                    ("outer_test", test, prediction),
                    *([("inner_oof_binary_threshold", train, oof)] if binary else []),
                ]:
                    table = pd.DataFrame(
                        {
                            "row_id": [row_ids[i] for i in positions],
                            "y_true": target.iloc[positions].to_numpy(),
                        }
                    )
                    with timer.measure("metric_evaluation"):
                        if metadata.task_type == "regression":
                            scores = regression_metrics(table.y_true, predictions)
                            table["prediction"] = predictions
                        else:
                            scores = classification_metrics(
                                table.y_true,
                                predictions,
                                classes=classes,
                                positive_class=positive,
                                selected_threshold=threshold,
                            )
                            for i in range(len(classes)):
                                table[f"p_c{i:03d}"] = predictions[:, i]
                            if binary:
                                table["decision_default"] = predictions[:, -1] >= 0.5
                                table["decision_selected"] = predictions[:, -1] >= threshold
                        if role == "outer_test":
                            overlap = fold_overlap(text_ids, train, test)
                            for column in overlap:
                                table[column] = overlap[column].to_numpy()
                            scores["text_overlap"] = cohort_scores(
                                metadata,
                                table.y_true,
                                predictions,
                                overlap,
                                selected_threshold=threshold,
                            )
                    identity = {**base, "prediction_role": role}
                    records.append(
                        save_evaluation(
                            scores=scores,
                            predictions=table,
                            timer=timer,
                            metadata=identity,
                            run_id=f"{digest(identity)}_{attempt_id}",
                            phase=phase,
                            resources=monitor.snapshot(),
                            warnings=[
                                {"category": w.category.__name__, "message": str(w.message)[:2000]}
                                for w in captured
                            ],
                        )
                    )
            complete_fold(fold_recipe, records[record_start:], phase)
    return records
