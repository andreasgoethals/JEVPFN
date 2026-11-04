import copy

from src.cluster import predictor_contracts, preflight
from src.evaluation.campaign import benchmark_plan
from src.evaluation.models import MAIN_MODELS, TEXT_MODELS
from src.utils.config import load_yaml


def test_all_encoders_dimensions_and_providers_are_prespecified():
    main = [a for a in benchmark_plan() if a["phase"] == "experiment_2"]
    assert len(main) == 18 * 6 * 3 * 2
    assert {a["model"] for a in main} == set(MAIN_MODELS)
    assert {(a["pca_scope"], a["pca_components"]) for a in main} == {
        ("per_column", 32), ("joint", 64),
    }
    assert len({a["encoder"] for a in main}) == 3
    from src.evaluation.qualification import profile_plan

    profiles = profile_plan(main)
    assert {a["encoder"] for a in profiles} <= {"e5_large_v2", "qwen3_embedding_0_6b"}
    assert {(a["model"], a["pca_scope"], a["pca_components"]) for a in profiles} == {
        (model, scope, count) for model in MAIN_MODELS
        for scope, count in (("per_column", 32), ("joint", 64))
    }
    assert all(a["provider"] is None for a in main)
    assert load_yaml("experiment_2/embeddings")["enabled"] is False


def test_actual_predictor_fixtures_exercise_both_pca_widths_and_combined_features():
    class Model:
        def fit(self, x, y):
            self.values = x.copy()

    for dataset in predictor_contracts.fixtures():
        cases = list(predictor_contracts.feature_cases(dataset, MAIN_MODELS[0]))
        assert len(cases) == 8
        assert {
            (dataset.metadata.task_type, len(dataset.metadata.classes), name)
            for name, _, _ in cases
        } == {
            case for case in predictor_contracts.required_fixture_cases(MAIN_MODELS[0])
            if case[:2] == (dataset.metadata.task_type, len(dataset.metadata.classes))
        }
        # These are interface fixtures, not claimed real E5 vectors or dataset predictions.
        for _, features, width in cases:
            assert not {"text", "notes", "target"} & set(features)
            if width is None:
                continue
            factory = predictor_contracts.case_factory(Model, features, width)
            train = features.iloc[:144]
            wrapped = factory().fit(train, dataset.frame.target.iloc[:144])
            joint = features.attrs["pca_scope"] == "joint"
            assert [p.n_components_ for p in wrapped.pcas] == [width] * (1 if joint else 2)
            projected = wrapped.estimator.values.filter(like="e5_pca__")
            assert projected.shape[1] == 64
            assert projected.iloc[0].isna().all()
            assert projected.iloc[17].isna().sum() == (0 if joint else 32)
            heldout = wrapped.transform(features.iloc[144:])
            assert list(heldout) == list(wrapped.estimator.values)
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
