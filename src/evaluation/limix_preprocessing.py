"""Enforce the benchmark's training-only preprocessing in the pinned LimiX V2 API.

The publisher passes a concatenated support/query matrix to each preprocessing
step. Its distribution transformer already fits on support, but column filtering,
category encoding and polynomial scaling also inspect query rows by default.
Keep the publisher's encoders and randomization, restricting those fits to support.
This adapter is validated synthetically; its checkpoint still requires Experiment 0B.
"""

from __future__ import annotations

from types import MethodType

import numpy as np

POLICY = "limix_v2_training_only_preprocessing_v1"


def _support(x, y):
    if y is None or x.ndim != 2 or not 0 < len(y) < len(x):
        raise ValueError("LimiX preprocessing requires explicit support and query rows")
    return x[: len(y)], x[len(y) :]


def _filter_fit_transform(self, x, categorical_features, seed, *, y, **kwargs):
    train, _ = _support(x, y)
    # A varying or partially missing training feature stays available even if
    # every query value is missing. Query values cannot rescue a constant feature.
    valid = ~np.all(train == train[:1], axis=0) & ~np.isnan(train).all(axis=0)
    if not valid.any():
        raise ValueError("LimiX has no nonconstant, observed training features")
    self.valid_features = valid
    self.invalid_indices = ~valid
    self.categorical_idx = [
        index
        for index, original in enumerate(np.flatnonzero(valid))
        if original in categorical_features
    ]
    return self.transform(x)


def _categorical_fit_transform(self, x, categorical_features, seed, *, y, **kwargs):
    train, query = _support(x, y)
    transformed_train, categories = self._jevpfn_original_fit_transform(
        train, categorical_features, seed, y=y, **kwargs
    )
    transformer = self.transformer
    if transformer is None:
        transformed_query = query
    else:
        transformed_query = np.asarray(transformer.transform(query))
        # The publisher shuffles encoded categories after ColumnTransformer.
        # Reuse those training-fitted permutations, leaving unseen categories NaN.
        if self.encoding_strategy.startswith("ordinal") and self.encoding_strategy.endswith(
            "_shuffled"
        ):
            for column, permutation in self.category_mappings.items():
                values = transformed_query[:, column]
                observed = np.isfinite(values)
                transformed_query[observed, column] = permutation[values[observed].astype(int)]
    if transformed_query.shape[1] != transformed_train.shape[1]:
        raise ValueError("LimiX training/query categorical schemas differ")
    return np.concatenate([transformed_train, transformed_query], axis=0), categories


def _polynomial_fit_transform(self, x, categorical_features, seed, *, y, **kwargs):
    train, _ = _support(x, y)
    self.fit(train, categorical_features, seed, **kwargs)
    return self.transform(x, **kwargs)


def _training_categories(self, x):
    return self._jevpfn_original_category_inference(x[: self._jevpfn_train_rows])


class TrainOnlyLimiX:
    """Preserve the public predict signature while fitting preprocessing on support."""

    def __init__(self, predictor):
        if type(predictor).__module__ != "inference.v2_0.predictor":
            raise ValueError("Training-only adapter requires the reviewed LimiX V2 implementation")
        if len(getattr(predictor, "gpu_ids", None) or []) > 1:
            raise ValueError(
                "LimiX worker-pool preprocessing has not been validated for this adapter"
            )
        self.predictor = predictor
        predictor._jevpfn_original_category_inference = predictor.get_categorical_features_indices
        predictor.get_categorical_features_indices = MethodType(_training_categories, predictor)
        self.preprocessing_policy = POLICY

    def predict(self, x_train, y_train, x_test, **kwargs):
        self.predictor._jevpfn_train_rows = len(y_train)
        # Recheck on every call: upstream can rebuild pipelines when configs change.
        for pipeline in self.predictor.preprocess_pipelines:
            for step in pipeline:
                name = type(step).__name__
                if name not in {
                    "FilterValidFeatures",
                    "CategoricalFeatureEncoder",
                    "PolynomialInteractionGenerator",
                }:
                    continue
                if type(step).__module__ != "inference.v2_0.preprocess":
                    raise ValueError(
                        "LimiX preprocessing implementation differs from the reviewed API"
                    )
                if not hasattr(step, "_jevpfn_original_fit_transform"):
                    step._jevpfn_original_fit_transform = step.fit_transform
                replacement = {
                    "FilterValidFeatures": _filter_fit_transform,
                    "CategoricalFeatureEncoder": _categorical_fit_transform,
                    "PolynomialInteractionGenerator": _polynomial_fit_transform,
                }[name]
                step.fit_transform = MethodType(replacement, step)
        return self.predictor.predict(x_train, y_train, x_test, **kwargs)
