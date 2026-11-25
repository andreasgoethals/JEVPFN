"""Cross-experiment contracts without weights, API calls or benchmark training."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data import text_embeddings
from src.evaluation.embedding_control import EmbeddingControl
from src.evaluation.features import combine_representations
from src.evaluation.models import KumoBackend
from src.evaluation.widths import feature_width, width_status


def test_joint_pca_fits_only_embedding_coordinates_from_training_rows():
    train = pd.DataFrame(
        {
            "amount": [1, 2, 3, 4],
            "decision": [0.2, 0.3, 0.4, 0.5],
            "a0": [0.0, 2.0, np.nan, np.nan],
            "a1": [2.0, 4.0, np.nan, np.nan],
            "b0": [10.0, np.nan, 20.0, np.nan],
            "b1": [30.0, np.nan, 40.0, np.nan],
        }
    )
    received = []
    model = SimpleNamespace(fit=lambda x, y: received.append(x))
    wrapped = EmbeddingControl(
        model, [["a0", "a1"], ["b0", "b1"]], components=3, scope="joint"
    ).fit(train, [0, 1, 2, 3])
    np.testing.assert_allclose(wrapped.imputation_[0], [1, 3, 15, 35])
    np.testing.assert_allclose(wrapped.pcas[0].mean_, [1, 3, 15, 35])
    assert received[0].amount.equals(train.amount)
    assert received[0].decision.equals(train.decision)
    assert received[0].filter(like="e5_pca").iloc[-1].isna().all()
    heldout = train.copy()
    heldout.loc[0, ["a0", "a1"]] = [1e6, 2e6]
    assert list(wrapped.transform(heldout)) == list(received[0])
    np.testing.assert_allclose(wrapped.imputation_[0], [1, 3, 15, 35])
    broken = train.copy()
    broken.loc[0, "a0"] = np.nan
    with pytest.raises(ValueError, match="Partially"):
        EmbeddingControl(model, [["a0", "a1"], ["b0", "b1"]], scope="joint").fit(broken, [0] * 4)


def test_provider_concatenation_preserves_original_text_and_shared_presence_once():
    base = pd.DataFrame({"number": [1.0, 2.0], "text": ["sample", None]}, index=["csv:0", "csv:1"])
    additions = []
    for name, probability in (("jev", 0.2), ("liquid", 0.8)):
        matrix = base[["number"]].copy()
        matrix["p_positive"] = [probability, np.nan]
        matrix["text_presence__000__missing"] = [0, 1]
        matrix.attrs["feature_artifact_ids"] = [name + "-artifact"]
        additions.append((name, matrix))
    result = combine_representations(base, additions)
    assert result.text.equals(base.text)
    assert result.columns.tolist() == [
        "number",
        "text",
        "jev__p_positive",
        "text_presence__000__missing",
        "liquid__p_positive",
    ]
    assert result.attrs["feature_artifact_ids"] == ["jev-artifact", "liquid-artifact"]
    additions[1][1].index = ["csv:1", "csv:0"]
    with pytest.raises(ValueError, match="row identities"):
        combine_representations(base, additions)


def test_rotten_tomatoes_pca_width_limit_and_joint_alternative():
    metadata = SimpleNamespace(
        text_columns=range(13), non_text_columns=range(2), task_type="regression", classes=[]
    )
    arm = dict(
        feature_set="non_text_plus_frozen_e5",
        model="kumo_tabular",
        provider=None,
        pca_components=64,
        pca_scope="per_column",
    )
    assert feature_width(metadata, arm) == 847
    # Kumo's 500-column recipe default is widened (owner-approved 23 Nov 2026), not excluded.
    assert width_status(metadata, arm) == {
        "predictor_columns": 847, "recipe_column_limit": 847,
        "width_status": "within_known_width_limit", "kumo_max_columns_override": 847,
    }
    # Arms within the default keep their historical width fields (and plan identity).
    assert width_status(metadata, arm | {"pca_components": 32}) == {
        "predictor_columns": 431, "recipe_column_limit": 500,
        "width_status": "within_known_width_limit",
    }
    assert width_status(metadata, arm | {"model": "catboost"})["recipe_column_limit"] is None
    assert feature_width(metadata, arm | {"pca_scope": "joint"}) == 79
    assert feature_width(metadata, arm | {"pca_components": 32}) == 431
    combined = arm | dict(feature_set="non_text_plus_both", providers=["jev", "liquid"])
    assert feature_width(metadata, combined) == 2 + 13 + 2 * 14 * 9
    assert feature_width(metadata, combined | {"view": "mean"}) == 2 + 13 + 2 * 14


class _SelectColumns:
    def __init__(self, max_columns):
        self.max_columns = max_columns


def _kumo(calls, *, selectors=(500,), values=None, column_names=()):
    """KumoBackend with publisher-shaped fakes: default_recipe() holds SelectColumns(500)."""
    from contextlib import nullcontext

    def default_recipe():
        modules = [object(), *(_SelectColumns(n) for n in selectors), object()]
        return SimpleNamespace(features=SimpleNamespace(modules=lambda: iter(modules)))

    class Output:
        def __init__(self, rows):
            self.numerical = self
            self.rows = rows
            self.columns = {"numerical": column_names}

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((self.rows, 3)) if values is None else values[:self.rows]

    def call(**kwargs):
        calls.append(kwargs)
        return Output(len(kwargs["x_query"]))

    backend = object.__new__(KumoBackend)
    backend.categorical_columns = {"category"}
    backend.task = "regression"
    backend.settings = {"device": "cpu", "seed": 1}
    backend.sdm = SimpleNamespace(processing=SimpleNamespace(SelectColumns=_SelectColumns),
                                  Stype=SimpleNamespace(numerical="numerical"))
    backend.torch = SimpleNamespace(
        Generator=lambda device: SimpleNamespace(manual_seed=lambda seed: "generator"),
        inference_mode=nullcontext, autocast=lambda **kw: nullcontext(), float16="float16",
    )
    backend.model = type("Model", (), {"default_recipe": staticmethod(default_recipe),
                                       "__call__": lambda self, **kw: call(**kw)})()
    backend._table = lambda frame, target=False: frame
    return backend


def _wide(columns, *, categories):
    values = pd.DataFrame(np.zeros((51, columns)))
    values["category"] = [f"value_{i % categories}" for i in range(51)]
    return values


def test_kumo_class_predictions_follow_reordered_probabilities_and_regression_averages_quantiles():
    x = pd.DataFrame({"category": ["a", "b", "c"]})
    # The publisher can return the class columns in a different order from np.unique(y).
    scores = np.array([[.1, .2, .7], [.8, .1, .1], [.05, .9, .05]])
    classifier = _kumo([], values=scores, column_names=["7", "2", "5"])
    classifier.task = "multiclass_classification"
    classifier.fit(x, [5, 7, 2])
    np.testing.assert_array_equal(classifier.classes_, [2, 5, 7])
    np.testing.assert_allclose(classifier.predict_proba(x), scores[:, [1, 2, 0]])
    np.testing.assert_array_equal(classifier.predict(x), [5, 7, 2])

    quantiles = np.array([[2., 4., 9.], [-4., 0., 7.], [0., 2., 4.]])
    regressor = _kumo([], values=quantiles).fit(x, [1., 2., 3.])
    np.testing.assert_allclose(regressor.predict(x), [5., 1., 2.])


def test_kumo_within_500_expanded_columns_passes_nothing_new(recwarn):
    calls = []
    model = _kumo(calls)
    # 499 numeric + one categorical with 50 values: 500 columns, no count column added.
    model.fit(_wide(499, categories=50), np.zeros(51))
    model.predict(_wide(499, categories=50))
    assert model.recipe_ is None and model.max_columns_override_ is None
    assert "recipe" not in calls[0]
    assert set(calls[0]) == {"x_context", "y_context", "x_query", "num_estimators", "generator"}
    assert not [w for w in recwarn if "kumo_max_columns_override" in str(w.message)]


def test_kumo_wider_than_500_widens_select_columns_to_every_expanded_column():
    calls = []
    model = _kumo(calls)
    # 499 numeric + one categorical with 51 values: Kumo adds a count column (501 columns).
    with pytest.warns(UserWarning, match="kumo_max_columns_override=501"):
        model.fit(_wide(499, categories=51), np.zeros(51))
    model.predict(_wide(499, categories=51))
    assert model.max_columns_override_ == 501
    selector, = [m for m in calls[0]["recipe"].features.modules() if isinstance(m, _SelectColumns)]
    assert selector.max_columns == 501
    calls.clear()
    with pytest.warns(UserWarning, match="kumo_max_columns_override=557"):
        model.fit(_wide(556, categories=2), np.zeros(51))


@pytest.mark.parametrize("selectors", [(), (500, 500), (400,)])
def test_kumo_override_refuses_an_unexpected_publisher_recipe(selectors):
    model = _kumo([], selectors=selectors)
    with pytest.raises(ValueError, match="SelectColumns"):
        model.fit(_wide(556, categories=2), np.zeros(51))


def test_encoder_namespaces_are_distinct_and_recipes_match_publishers(isolated_output):
    configs = [text_embeddings.encoder_config(i) for i in text_embeddings.MODEL_NAMESPACES.values()]
    locations = [text_embeddings.cache_path("dataset", c) for c in configs]
    assert len(set(locations)) == 3
    assert [c["dimensions"] for c in configs] == [384, 1024, 1024]
    assert [c["pooling"] for c in configs] == ["attention_mask_mean"] * 2 + [
        "last_nonpadding_token"
    ]
    assert [c["prefix"] for c in configs] == ["query: ", "query: ", ""]
    for c in configs:
        assert text_embeddings.encoder_id_for_model(c["model"]) == c["encoder_id"]


def test_all_encoder_caches_transfer_without_namespace_collision(
    isolated_output, tmp_path, monkeypatch
):
    from test_frozen_text import Encoder, fixture_data

    from src.cluster.transfer import export_bundle, import_bundle
    from src.utils import paths

    recipes = [
        text_embeddings.encoder_config(i) | {"dimensions": 3, "batch_size": 2}
        for i in text_embeddings.MODEL_NAMESPACES.values()
    ]
    for cfg in recipes:
        text_embeddings.build(fixture_data(), cfg, Encoder())
    bundle = export_bundle()
    monkeypatch.delenv("VSC_DATA")
    monkeypatch.delenv("JEVPFN_STAGING_ROOT")
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path / "new_checkout")
    report = import_bundle(bundle)
    assert len(report["embedding_caches"]) == 3
    for cfg in recipes:
        assert text_embeddings.load(fixture_data(), cfg).shape == (4, 6)
