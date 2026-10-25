"""Training-only LimiX interface tests without importing publisher snapshots or weights."""

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.evaluation.limix_preprocessing import (
    TrainOnlyLimiX,
    _categorical_fit_transform,
    _filter_fit_transform,
    _polynomial_fit_transform,
)


def test_filter_schema_uses_support_even_when_all_query_values_are_missing():
    def transform(x):
        return x[:, step.valid_features], step.categorical_idx

    step = SimpleNamespace(transform=transform)
    train = np.array([[1, 2, np.nan], [1, 4, np.nan], [1, 3, np.nan]])
    query = np.array([[999, np.nan, 1000], [2000, np.nan, 2000]])
    out, cats = _filter_fit_transform(step, np.vstack([train, query]), [1], 7, y=[0, 1, 0])
    assert out.shape == (5, 1) and cats == [0]
    np.testing.assert_array_equal(step.valid_features, [False, True, False])
    np.testing.assert_array_equal(out[:3, 0], [2, 4, 3])
    assert np.isnan(out[3:, 0]).all()
    changed, _ = _filter_fit_transform(step, np.vstack([train, query * -5]), [1], 7, y=[0, 1, 0])
    np.testing.assert_array_equal(changed[:3], out[:3])


@pytest.mark.parametrize("strategy", ["ordinal", "ordinal_shuffled", "onehot", "none"])
def test_categories_fit_only_support_and_unseen_query_uses_fitted_transform(strategy):
    # A small public-API double: actual sklearn encoders, matching publisher state.
    step = SimpleNamespace(encoding_strategy=strategy, category_mappings=None)

    def fit_transform(train, categories, seed, **kwargs):
        if strategy == "none":
            step.transformer = None
            return train, categories
        encoder = (
            OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            if strategy == "onehot"
            else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
        )
        step.transformer = ColumnTransformer(
            [("encoder", encoder, categories)], remainder="passthrough"
        )
        out = step.transformer.fit_transform(train)
        if strategy == "ordinal_shuffled":
            step.category_mappings = {0: np.array([1, 0])}
            out[:, 0] = step.category_mappings[0][out[:, 0].astype(int)]
        return out, [0]

    step._jevpfn_original_fit_transform = fit_transform
    train = np.array([[10.0, 1.0], [20.0, 2.0], [10.0, 3.0]])
    query = np.array([[30.0, 4.0], [10.0, 5.0]])
    out, _ = _categorical_fit_transform(step, np.vstack([train, query]), [0], 7, y=[0, 1, 0])
    if strategy == "none":
        np.testing.assert_array_equal(out, np.vstack([train, query]))
    else:
        categories = step.transformer.named_transformers_["encoder"].categories_[0]
        np.testing.assert_array_equal(categories, [10.0, 20.0])
        if strategy.startswith("ordinal"):
            assert np.isnan(out[3, 0])
            assert out[4, 0] == out[0, 0]
        else:
            np.testing.assert_array_equal(out[3, :2], [0, 0])
    changed, _ = _categorical_fit_transform(
        step, np.vstack([train, query * 1000]), [0], 7, y=[0, 1, 0]
    )
    np.testing.assert_array_equal(changed[:3], out[:3])


def test_polynomial_normalization_never_fits_query_rows():
    normalizer = StandardScaler(with_mean=False)
    step = SimpleNamespace(
        fit=lambda x, categories, seed, **kwargs: normalizer.fit(x),
        transform=lambda x, **kwargs: (normalizer.transform(x), []),
    )
    train = np.array([[1.0], [2.0], [3.0]])
    first, _ = _polynomial_fit_transform(step, np.vstack([train, [[1000.0]]]), [], 7, y=[0, 1, 0])
    second, _ = _polynomial_fit_transform(step, np.vstack([train, [[-1e9]]]), [], 7, y=[0, 1, 0])
    np.testing.assert_array_equal(first[:3], second[:3])
    np.testing.assert_allclose(normalizer.mean_, [2.0])


def test_unknown_publisher_and_worker_pool_are_rejected():
    with pytest.raises(ValueError, match="reviewed LimiX V2"):
        TrainOnlyLimiX(SimpleNamespace())
    predictor_type = type("Predictor", (), {"__module__": "inference.v2_0.predictor"})
    predictor = predictor_type()
    predictor.gpu_ids = [0, 1]
    with pytest.raises(ValueError, match="worker-pool"):
        TrainOnlyLimiX(predictor)


def test_inferred_categorical_features_do_not_depend_on_query_rows():
    predictor_type = type("Predictor", (), {"__module__": "inference.v2_0.predictor"})
    predictor = predictor_type()
    predictor.preprocess_pipelines = []
    predictor.get_categorical_features_indices = lambda x: [
        i for i in range(x.shape[1]) if len(np.unique(x[:, i])) < 3
    ]
    predictor.predict = lambda x_train, y_train, x_test, **kwargs: (
        predictor.get_categorical_features_indices(np.vstack([x_train, x_test]))
    )
    wrapped = TrainOnlyLimiX(predictor)
    train = np.array([[0.0, 1.0], [1.0, 2.0], [0.0, 3.0]])
    assert wrapped.predict(train, [0, 1, 0], np.array([[99, 99], [88, 88]])) == [0]


def test_wrapper_installs_filter_before_real_predict_interface():
    filter_type = type("FilterValidFeatures", (), {"__module__": "inference.v2_0.preprocess"})
    step = filter_type()
    step.fit_transform = lambda *a, **kw: pytest.fail("unrestricted fit must not run")
    step.transform = lambda x: (x[:, step.valid_features], step.categorical_idx)
    predictor_type = type("Predictor", (), {"__module__": "inference.v2_0.predictor"})
    predictor = predictor_type()
    predictor.preprocess_pipelines = [[step]]
    predictor.get_categorical_features_indices = lambda x: []
    predictor.predict = lambda x_train, y_train, x_test, **kw: step.fit_transform(
        np.vstack([x_train, x_test]), [], 7, y=y_train
    )[0]
    wrapped = TrainOnlyLimiX(predictor)
    out = wrapped.predict(np.array([[0, 1], [0, 2]]), [0, 1], np.array([[7, np.nan]]))
    assert out.shape == (3, 1) and np.isnan(out[-1, 0])
    # Calling again must not wrap the wrapper recursively.
    again = wrapped.predict(np.array([[0, 1], [0, 2]]), [0, 1], np.array([[8, np.nan]]))
    np.testing.assert_array_equal(out, again)
