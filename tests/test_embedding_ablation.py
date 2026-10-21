import copy

import numpy as np

from src.cluster import predictor_contracts, preflight
from src.evaluation.campaign import benchmark_plan
from src.evaluation.models import MAIN_MODELS, TEXT_MODELS
from src.utils.config import load_yaml


def test_all_encoders_dimensions_and_providers_are_prespecified():
    main = [a for a in benchmark_plan() if a["phase"] == "experiment_2"]
    assert len(main) == 18 * 6 * 3 * 7
    assert {a["model"] for a in main} == set(MAIN_MODELS)
    assert {a["pca_components"] for a in main} == {8, 16, 32, 64, 128, 256}
    assert {a["pca_scope"] for a in main} == {"per_column", "joint"}
    assert len({a["encoder"] for a in main}) == 3
    optional = [
        a
        for a in benchmark_plan(include_controls=True)
        if a["phase"] == "experiment_2" and a["pca_scope"] == "joint"
    ]
    assert len(optional) == 18 * 6 * 3 * 6
    assert all(a["provider"] is None for a in main)
    assert load_yaml("experiment_2/embeddings")["enabled"] is False


def test_actual_predictor_fixtures_exercise_both_pca_widths_and_combined_features():
    class Model:
        def fit(self, x, y):
            self.values = x.copy()

    for dataset in predictor_contracts.fixtures():
        cases = list(predictor_contracts.feature_cases(dataset, MAIN_MODELS[0]))
        assert len(cases) == 10
        # These are interface fixtures, not claimed real E5 vectors or dataset predictions.
        for name, features, width in cases:
            assert "text" not in features and "target" not in features
            if width is None:
                continue
            factory = predictor_contracts.case_factory(Model, features, width)
            train = features.iloc[:450]
            wrapped = factory().fit(train, dataset.frame.target.iloc[:450])
            assert wrapped.pcas[0].n_components_ == width
            assert len([c for c in wrapped.estimator.values if c.startswith("e5_pca__")]) == width
            assert wrapped.estimator.values.iloc[0].filter(like="e5_pca__").isna().all()
            heldout = wrapped.transform(features.iloc[450:])
            assert list(heldout) == list(wrapped.estimator.values)
            if "both" in name:
                for c in features.filter(like="synthetic_decision__"):
                    np.testing.assert_allclose(heldout[c], features.iloc[450:][c], equal_nan=True)
        # Changing labels cannot alter the constructed feature fixtures.
        changed = copy.deepcopy(dataset)
        changed.frame["target"] = changed.frame["target"].iloc[::-1].to_numpy()
        for before, after in zip(cases, predictor_contracts.feature_cases(changed, MAIN_MODELS[0])):
            assert before[1].equals(after[1])
        for native in TEXT_MODELS:
            cases_native = list(predictor_contracts.feature_cases(dataset, native))
            assert len(cases_native) == 4 and "text" in cases_native[0][1]


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
    assert len(result) == 6
    assert {r["encoder_id"] for r in result} == {
        "e5_small_v2",
        "e5_large_v2",
        "qwen3_embedding_0_6b",
    }
    assert [(r["dataset_id"], r["status"]) for r in result] == [("good", "passed")] * 3 + [
        ("missing", "failed")
    ] * 3
    assert result[3]["error_type"] == "FileNotFoundError"
