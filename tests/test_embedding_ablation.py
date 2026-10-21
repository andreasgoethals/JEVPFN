import copy

import numpy as np

from src.cluster import predictor_contracts, preflight
from src.evaluation.campaign import benchmark_plan
from src.evaluation.models import MAIN_MODELS, TEXT_MODELS
from src.utils.config import load_yaml


def test_pca16_is_optional_and_reuses_all_six_predictors_without_new_extraction():
    main = benchmark_plan()
    assert not any(a.get("control") == "embedding_dimension" for a in main)
    optional = [
        a
        for a in benchmark_plan(include_controls=True)
        if a.get("control") == "embedding_dimension"
    ]
    assert len(optional) == 18 * 6
    assert {a["model"] for a in optional} == set(MAIN_MODELS)
    assert all(a["provider"] is None and a["pca_components"] == 16 for a in optional)
    assert all(
        a["feature_set"] == "non_text_plus_frozen_e5" and a["view"] == "pca_16" for a in optional
    )
    assert load_yaml("feature_creation/text_embeddings")["pca_components_per_column"] == 32
    assert load_yaml("experiment_3/output_ablation")["enabled"] is False


def test_actual_predictor_fixtures_exercise_both_pca_widths_and_combined_features():
    class Model:
        def fit(self, x, y):
            self.values = x.copy()

    for dataset in predictor_contracts.fixtures():
        cases = list(predictor_contracts.feature_cases(dataset, MAIN_MODELS[0]))
        assert len(cases) == 4
        # These are interface fixtures, not claimed real E5 vectors or dataset predictions.
        for name, features, width in cases:
            assert "text" not in features and "target" not in features
            if width is None:
                continue
            factory = predictor_contracts.case_factory(Model, features, width)
            train = features.iloc[:60]
            wrapped = factory().fit(train, dataset.frame.target.iloc[:60])
            assert wrapped.pcas[0].n_components_ == width
            assert len([c for c in wrapped.estimator.values if c.startswith("e5_pca__")]) == width
            assert wrapped.estimator.values.iloc[0].filter(like="e5_pca__").isna().all()
            heldout = wrapped.transform(features.iloc[60:])
            assert list(heldout) == list(wrapped.estimator.values)
            if "both" in name:
                for c in features.filter(like="synthetic_decision__"):
                    np.testing.assert_allclose(heldout[c], features.iloc[60:][c], equal_nan=True)
        # Changing labels cannot alter the constructed feature fixtures.
        changed = copy.deepcopy(dataset)
        changed.frame["target"] = changed.frame["target"].iloc[::-1].to_numpy()
        for before, after in zip(cases, predictor_contracts.feature_cases(changed, MAIN_MODELS[0])):
            assert before[1].equals(after[1])
        for native in TEXT_MODELS:
            cases_native = list(predictor_contracts.feature_cases(dataset, native))
            assert len(cases_native) == 1 and "text" in cases_native[0][1]


def test_0b_embedding_checks_cover_every_selected_dataset_and_report_missing_cache(monkeypatch):
    from src.data import text_embeddings

    entries = [{"id": "good"}, {"id": "missing"}]
    monkeypatch.setattr(preflight, "load_catalog", lambda: {})
    monkeypatch.setattr(preflight, "selected_entries", lambda *a: entries)
    monkeypatch.setattr(preflight, "load_dataset", lambda e, **kw: e)

    def validate(dataset, cfg):
        if dataset["id"] == "missing":
            raise FileNotFoundError()
        return {"rows": 3, "text_columns": ["description"]}

    monkeypatch.setattr(text_embeddings, "validate_dataset", validate)
    result = preflight.embedding_checks({})
    assert [(r["dataset_id"], r["status"]) for r in result] == [
        ("good", "passed"),
        ("missing", "failed"),
    ]
    assert result[1]["error_type"] == "FileNotFoundError"
