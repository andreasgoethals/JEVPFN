"""Group B (20 November 2026): predictor inputs, new arms, B3/B4 analyses. No real model fits
of tabular predictors; heavy backends are mocked."""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from src.data.loaders import LoadedDataset  # noqa: E402
from src.data.metadata import extract_metadata  # noqa: E402
from src.evaluation import data_efficiency, group_b, label_probe  # noqa: E402
from src.evaluation.features import without_text_column_block  # noqa: E402
from src.evaluation.models import (  # noqa: E402
    CATEGORY_DTYPE_MODELS,
    MAIN_MODELS,
    STRING_CATEGORY_MODELS,
    FoldInput,
    LightGBMBackend,
    PredictorAdapter,
    _backend,
)
from src.evaluation.protocol import validate_splits  # noqa: E402


@pytest.fixture
def mixed():
    """Categorical, missing, infinite, date and boolean columns plus one text column."""
    frame = pd.DataFrame({
        "amount": [1.0, np.inf, 3.0, np.nan, -np.inf, 6.0],
        "colour": ["red", None, "blue", "red", "green", "blue"],
        "when": ["2020-01-01", "2021-06-30", None, "2022-02-02", "2023-03-03", "2024-04-04"],
        "flag": [True, False, True, True, False, False],
        "text": ["a", "", "b", "c", None, "d"],
        "target": ["n", "y", "n", "y", "n", "y"],
    })
    official = {"target": "target", "task_type": "cls", "num_classes": 2}
    meta = extract_metadata(frame, official, dataset_id="mixed", dataset_name="Mixed",
                            text_columns=["text"])
    return LoadedDataset(frame, meta, official, {"data_identity": "mixed"}, [])


@pytest.mark.parametrize("model_id", MAIN_MODELS)
def test_backend_receives_intended_types(mixed, model_id):
    """Each structured adapter's backend input: float numerics (inf -> NaN, dates -> seconds),
    categories as pandas `category` (TabPFN, TabICL, LightGBM), typed-JSON strings (linear,
    CatBoost) or float codes (LimiX, Kumo); unseen test categories become missing."""
    seen = {}

    class Backend:
        def fit(self, x, y):
            seen["fit"] = x.copy()
            self.classes_ = np.array([0, 1])

        def predict_proba(self, x):
            seen["predict"] = x.copy()
            return np.full((len(x), 2), 0.5)

        def predict(self, *args, **kwargs):
            seen["predict"] = args[2] if model_id == "limix_2" else args[0]
            return np.tile([0.5, 0.5], (len(seen["predict"]), 1))

    features = mixed.frame.drop(columns=["text", "target"])
    adapter = PredictorAdapter(model_id, mixed.metadata, {}, backend_loader=lambda *a: Backend())
    train, test = features.iloc[:4], features.iloc[4:].copy()
    test.loc[test.index[0], "colour"] = "purple"  # unseen in training
    adapter.fit(train, mixed.frame.target.iloc[:4])
    adapter.predict_proba(test)
    if model_id == "limix_2":
        x = pd.DataFrame(adapter.train_x_, columns=list(features.columns))
        xt = pd.DataFrame(np.asarray(seen["predict"]), columns=list(features.columns))
    else:
        x, xt = seen["fit"], seen["predict"]
    assert np.isnan(x["amount"].iloc[1]) and np.isfinite(x["amount"].iloc[0])
    assert x["when"].dtype == float and x["when"].iloc[0] == pytest.approx(1577836800.0)
    assert np.isnan(x["when"].iloc[2])
    assert x["flag"].dtype == float
    colour = x["colour"]
    if model_id in CATEGORY_DTYPE_MODELS:
        assert isinstance(colour.dtype, pd.CategoricalDtype)
        assert list(colour.cat.categories) == ['"blue"', '"red"']
        assert colour.isna().iloc[1]  # missing stays missing
        assert isinstance(xt["colour"].dtype, pd.CategoricalDtype)
        assert list(xt["colour"].cat.categories) == ['"blue"', '"red"']
        assert xt["colour"].isna().iloc[0]  # unseen category -> missing
    elif model_id in STRING_CATEGORY_MODELS:
        assert colour.tolist()[:2] == ['"red"', "null"]
    else:
        assert colour.dtype == float and np.isnan(colour.iloc[1])
        assert np.isnan(xt["colour"].astype(float).iloc[0])


def test_tabpfn_declares_categorical_indices_and_lightgbm_is_cpu(monkeypatch, tmp_path):
    from src.evaluation import models
    from src.utils.serialization import file_sha256

    checkpoint = tmp_path / "w.ckpt"
    checkpoint.write_bytes(b"never loaded")
    pin = {"path": str(checkpoint), "sha256": file_sha256(checkpoint)}
    calls = {}

    class TabPFN:
        @classmethod
        def create_default_for_version(cls, version, **kwargs):
            calls["tabpfn"] = kwargs
            return cls()

    class LGBM:
        def __init__(self, **kwargs):
            calls["lightgbm"] = kwargs

        def fit(self, x, y):
            calls["lightgbm_columns"] = list(x.columns)
            calls["lightgbm_dtypes"] = list(map(str, x.dtypes))
            self.classes_ = np.unique(y)
            return self

        def predict_proba(self, x):
            return np.full((len(x), len(self.classes_)), 1 / len(self.classes_))

    modules = {
        "tabpfn": SimpleNamespace(TabPFNClassifier=TabPFN),
        "tabpfn.constants": SimpleNamespace(ModelVersion=SimpleNamespace(V3_5="V3_5")),
        "lightgbm": SimpleNamespace(LGBMClassifier=LGBM, LGBMRegressor=LGBM),
    }
    monkeypatch.setattr(models.importlib, "import_module", modules.__getitem__)
    frame = pd.DataFrame({"n": [1.0, 2.0], 'c "quoted": x': ["a", "b"]})
    encoder = FoldInput().fit(frame)
    settings = {"seed": 5, "device": "cpu", "threads": 3, "parameters": {},
                "checkpoints": {"binary_classification": pin}}
    _backend("tabpfn_3_5", "binary_classification", settings, encoder, 2)
    assert calls["tabpfn"]["categorical_features_indices"] == [1]
    backend = _backend("lightgbm", "binary_classification", settings, encoder, 2)
    assert isinstance(backend, LightGBMBackend)
    assert calls["lightgbm"] == {"random_state": 5, "n_jobs": 3, "verbose": -1}
    backend.fit(encoder.transform(frame, categories=True), [0, 1])
    # Positional names avoid LightGBM's JSON-character restriction; dtypes are unchanged.
    assert calls["lightgbm_columns"] == ["f00000", "f00001"]
    assert calls["lightgbm_dtypes"][1] == "category"
    from src.cluster.benchmark_jobs import CPU_ONLY_MODELS
    from src.utils.config import load_yaml

    assert "lightgbm" in CPU_ONLY_MODELS
    runtime = load_yaml("experiment_1/models")["models"]["lightgbm"]["runtime"]
    assert runtime["device"] == "cpu" and runtime["packages"] == {"lightgbm": "4.7.0"}
    assert runtime["parameters"] == {}


def test_lightgbm_adapter_end_to_end_with_mock(mixed):
    class Backend:
        def fit(self, x, y):
            assert isinstance(x["colour"].dtype, pd.CategoricalDtype)
            self.classes_ = np.array([1, 0])

        def predict_proba(self, x):
            return np.tile([0.2, 0.8], (len(x), 1))

    features = mixed.frame.drop(columns=["text", "target"])
    adapter = PredictorAdapter("lightgbm", mixed.metadata, {}, backend_loader=lambda *a: Backend())
    adapter.fit(features, mixed.frame.target)
    p = adapter.predict_proba(features)
    assert np.allclose(p[:, 0], 0.8)  # realigned to the fixed class order


# ------------------------------------------------------------------ label-trained classifier


def _probe_data(task="binary_classification", n=120, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"e{i}" for i in range(6)],
                     index=[f"csv:{i}" for i in range(n)])
    x.iloc[:7, :3] = np.nan  # first text column missing for 7 rows
    if task == "regression":
        y = pd.Series(x.e3.fillna(0) * 2 + rng.normal(size=n))
        classes = []
    elif task == "multiclass_classification":
        y = pd.Series(np.array(["a", "b", "c"])[rng.integers(0, 3, n)])
        classes = ["a", "b", "c"]
    else:
        y = pd.Series(np.where(x.e4 > 0, "y", "n"))
        classes = ["n", "y"]
    return x, y, classes


@pytest.mark.parametrize("task,width", [("binary_classification", 1),
                                        ("multiclass_classification", 3), ("regression", 9)])
def test_probe_width_matches_decision_features_and_missing_text_stays_missing(task, width):
    x, y, classes = _probe_data(task)
    groups = [["e0", "e1", "e2"], ["e3", "e4", "e5"]]
    probe = label_probe.SupervisedProbe(groups, view="per_column", task_type=task, classes=classes, seed=1)
    train = probe.fit_transform(x.iloc[:100], y.iloc[:100])
    assert train.shape[1] == 2 * width
    first = [c for c in train if c.startswith("probe__text_000")]
    assert train.loc[x.index[:7], first].isna().all().all()
    assert train.loc[x.index[7:100], first].notna().all().all()
    test = probe.transform(x.iloc[100:])
    assert list(test.columns) == list(train.columns)
    joint = label_probe.SupervisedProbe(groups, view="joint", task_type=task, classes=classes, seed=1)
    assert joint.fit_transform(x.iloc[:100], y.iloc[:100]).shape[1] == width
    if task == "regression":
        edges = label_probe.regression_bins(y.iloc[:100])
        assert len(edges) == 8
        counts = np.bincount(label_probe.encode_target(y.iloc[:100], task, [], edges), minlength=9)
        assert counts.min() >= 100 // 9 - 1 and counts.max() <= 100 // 9 + 2


def test_probe_cross_fitting_never_uses_a_rows_own_label():
    x, y, classes = _probe_data()
    groups = [["e0", "e1", "e2"], ["e3", "e4", "e5"]]

    def features(labels):
        probe = label_probe.SupervisedProbe(groups, view="joint", task_type="binary_classification",
                                            classes=classes, seed=3, c_grid=(1.0,))
        return probe.fit_transform(x.iloc[:100], labels), probe

    base, probe = features(y.iloc[:100])
    flipped = y.iloc[:100].copy()
    flipped.iloc[50] = "n" if flipped.iloc[50] == "y" else "y"
    changed, _ = features(flipped)
    # Row 50's own cross-fitted feature comes from a model that never saw row 50.
    assert changed.iloc[50].equals(base.iloc[50])
    assert not changed.drop(index=x.index[50]).equals(base.drop(index=x.index[50]))
    with pytest.raises(ValueError, match="cross-fitted"):
        probe.transform(x.iloc[:10])
    # Test rows use the full-training model, independent of other test rows.
    one = probe.transform(x.iloc[100:101])
    many = probe.transform(x.iloc[100:])
    assert np.allclose(one.to_numpy(), many.iloc[:1].to_numpy(), equal_nan=True)


def test_probe_selects_c_by_inner_cv_and_records_it():
    x, y, classes = _probe_data()
    probe = label_probe.SupervisedProbe([["e3", "e4", "e5"]], view="per_column",
                                        task_type="binary_classification", classes=classes, seed=2)
    probe.fit_transform(x.iloc[:100], y.iloc[:100])
    (diagnostic,) = probe.diagnostics_
    assert diagnostic["selected_C"] in label_probe.C_GRID
    assert set(diagnostic["inner_log_loss"]) == {str(c) for c in label_probe.C_GRID}


def test_probe_control_uses_prepared_features_and_never_fits_test_rows(tmp_path):
    x, y, classes = _probe_data()
    groups = [["e0", "e1", "e2"], ["e3", "e4", "e5"]]
    cache = tmp_path / "probe.sqlite3"
    kwargs = dict(view="per_column", task_type="binary_classification", classes=classes, seed=4,
                  prepared_cache=cache, embedding_artifacts=["artifact"])
    label_probe.ProbeControl(None, groups, prepare=True, **kwargs).fit(x.iloc[:100], y.iloc[:100], all_rows=x)
    # A resumed preparation job skips stored partitions (it has no predictor to fit).
    label_probe.ProbeControl(None, groups, prepare=True, **kwargs).fit(x.iloc[:100], y.iloc[:100], all_rows=x)
    fitted = {}

    class Estimator:
        def fit(self, frame, target):
            fitted["columns"] = list(frame.columns)
            self.classes_ = np.array(["n", "y"])

        def predict_proba(self, frame):
            fitted["test_rows"] = list(frame.index)
            return np.full((len(frame), 2), 0.5)

    control = label_probe.ProbeControl(Estimator(), groups, **kwargs).fit(x.iloc[:100], y.iloc[:100])
    assert not any(c.startswith("e") for c in fitted["columns"])
    assert control.output_feature_count_ == 2
    control.predict_proba(x.iloc[100:])
    assert fitted["test_rows"] == list(x.index[100:])
    with pytest.raises(ValueError, match="Missing CPU-prepared"):
        label_probe.ProbeControl(Estimator(), groups, **{**kwargs, "seed": 5}).fit(x.iloc[:100], y.iloc[:100])


# ---------------------------------------------------------------------------- B4


def test_b4_subsamples_are_nested_stratified_and_keep_test_rows():
    rng = np.random.default_rng(0)
    labels = np.array(["a"] * 600 + ["b"] * 300 + ["c"] * 6)
    rng.shuffle(labels)
    rows = len(labels)
    folds = [{"train": [i for i in range(rows) if i % 5 != k], "test": [i for i in range(rows) if i % 5 == k]}
             for k in range(5)]
    parent = {"dataset_id": "d", "data_identity": "x", "task_schema_sha256": "s", "rows": rows,
              "seed": 1, "strategy": "stratified_kfold", "folds": folds}
    from src.utils.serialization import digest

    parent["split_id"] = digest(parent)
    specs = {size: data_efficiency.subsample_specification(parent, labels, size, 0, seed=9,
                                                           classes=["a", "b", "c"])
             for size in (16, 128, 512)}
    for k in range(5):
        small, mid, large = (set(specs[s]["folds"][k]["train"]) for s in (16, 128, 512))
        assert small < mid < large <= set(folds[k]["train"])
        assert specs[16]["folds"][k]["test"] == folds[k]["test"]
        assert {labels[i] for i in small} == {"a", "b", "c"}
        share = np.mean([labels[i] == "a" for i in large])
        assert abs(share - 600 / 906) < 0.05
    assert specs[16]["subsample"]["missing_classes_by_fold"] == [[]] * 5
    other_draw = data_efficiency.subsample_specification(parent, labels, 128, 1, seed=9, classes=["a", "b", "c"])
    assert other_draw["folds"][0]["train"] != specs[128]["folds"][0]["train"]
    assert other_draw["split_id"] != specs[128]["split_id"]

    class Dataset:
        frame = pd.DataFrame(index=range(rows))
        metadata = SimpleNamespace(dataset_id="d", fingerprint="s")
        source = {"data_identity": "x"}

    validate_splits(Dataset(), specs[128])
    regression = data_efficiency.subsample_specification(parent, None, 64, 0, seed=9)
    assert all(len(f["train"]) == 64 for f in regression["folds"])


def test_b4_fixed_dataset_sets_draws_and_grid():
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries
    from src.utils.config import load_yaml

    entries = selected_entries(load_catalog(), load_yaml("experiment_1/main"))
    sizes = data_efficiency.curve_membership(entries)
    extended = [d for d, s in sizes.items() if max(s) == 16384]
    assert sorted(extended) == sorted(data_efficiency.config()["curves"]["extended"]["datasets"])
    assert len(extended) == 7 and len(sizes) == 18
    assert all(s == [128, 256, 512, 1024, 2048] for d, s in sizes.items() if d not in extended)
    assert [data_efficiency.draws_for(s) for s in (128, 512, 1024, 2048, 4096, 16384)] == [5, 5, 3, 3, 1, 1]
    broken = [dict(e, paper_summary={**e["paper_summary"], "rows": 10_000}) if e["id"] == extended[0] else e
              for e in entries]
    with pytest.raises(ValueError, match="membership"):
        data_efficiency.curve_membership(broken)


def test_group_b_plan_counts():
    from collections import Counter

    from src.evaluation.campaign import benchmark_plan

    # Group C phases (representations, rq3) are counted in test_group_c.
    plan = [a for a in benchmark_plan() if a["phase"] not in ("representations", "rq3")]
    counts = Counter(a["phase"] for a in plan)
    # B1: (1 + 17 x 3) x 7 x 18; B2: 3 encoders x 2 PCA + 17 decision+frozen (+ TAR 2 + 6 probes);
    # B3: 17 models x 78 text columns (16 multi-text datasets) x 7; B4: 399 sample cells x 22 x 7.
    assert counts == {"experiment_1": 6552, "experiment_2": (6 + 17) * 7 * 18,
                      "experiment_3": (2 + 6) * 7 * 18, "experiment_4": 54, "rq2": 17 * 78 * 7,
                      "data_efficiency": 399 * 22 * 7}
    unsupported = [a for a in plan if a["width_status"] == "unsupported_width"]
    assert {(a["model"], a["dataset_id"], a["feature_set"]) for a in unsupported} == {
        ("kumo_tabular", "REG_TEXT_ROTTEN_TOMATOES", "non_text_plus_both_plus_frozen_e5")}
    assert not any(a.get("feature_set") == "all_models" for a in plan)


def test_text_column_refit_drops_only_that_column_block():
    columns = ["n", "text_presence__000__missing", "text_presence__001__missing",
               "jev__jev__per_column__text_only__text_000__p_positive",
               "jev__jev__per_column__text_only__text_001__p_positive"]
    frame = pd.DataFrame(np.zeros((2, len(columns))), columns=columns)
    frame.attrs["feature_artifact_ids"] = ["a"]
    out = without_text_column_block(frame, 1)
    assert list(out.columns) == columns[:4]
    assert out.attrs["feature_artifact_ids"] == ["a"]
    with pytest.raises(ValueError, match="No per-column"):
        without_text_column_block(frame, 5)


def test_text_column_importance_pairs_folds():
    rows = []
    for fold in range(5):
        for rep, value in (("decision/per_column/jev", 0.8), ("without_text/jev/0", 0.7),
                           ("without_text/jev/1", 0.79)):
            rows.append(dict(dataset_id="d", task_type="binary_classification", model_id="linear",
                             checkpoint_id="c", split_id="s", outer_fold=fold, representation=rep,
                             value=value))
    out = group_b.text_column_importance(pd.DataFrame(rows))
    assert out.set_index("text_column").importance.round(3).to_dict() == {0: 0.1, 1: 0.01}


# ------------------------------------------------------------------- gains and LODO


def _folds(n_datasets=6):
    rows = []
    for d in range(n_datasets):
        for fold in range(5):
            for model in ("linear", "catboost"):
                base = 0.7 + 0.01 * d
                for rep, delta in (("baseline", 0.0), ("decision/both/m1", 0.02 * d),
                                   ("decision/both/m2", 0.01), ("frozen/e5_small_v2/per_column32", 0.005)):
                    rows.append(dict(dataset_id=f"d{d}", task_type="binary_classification", model_id=model,
                                     checkpoint_id="c", split_id="s", outer_fold=fold, representation=rep,
                                     value=base + delta))
    return pd.DataFrame(rows)


def test_paired_gains_summary_and_median_over_models():
    gains = group_b.dataset_gains(group_b.fold_gains(_folds()))
    one = gains[(gains.dataset_id == "d2") & (gains.model_id == "linear")].set_index("representation")
    assert one.loc["decision/both/m1", "raw_gain"] == pytest.approx(0.04)
    assert one.loc["decision/both/m1", "headroom_share"] == pytest.approx(0.04 / 0.28)
    medians = group_b.median_over_decision_models(gains, roster=["m1", "m2"])
    assert set(medians.representation) == {"decision/both/median"}
    summary = group_b.summarize(gains)
    row = summary[(summary.model_id == "linear") & (summary.representation == "decision/both/m2")].iloc[0]
    assert row.wins == 6 and row.losses == 0 and row.raw_ci_low == pytest.approx(0.01)
    row = summary[(summary.model_id == "linear") & (summary.representation == "decision/both/m1")].iloc[0]
    assert row.ties == 1 and row.wins == 5  # d0 has zero gain


def test_lodo_rule_excludes_the_held_out_dataset_and_scores_policies():
    properties = pd.DataFrame({"dataset_id": [f"d{i}" for i in range(8)],
                               "p1": np.arange(8, dtype=float), "p2": np.arange(8, dtype=float) ** 0.5})
    outcome = pd.DataFrame({"dataset_id": properties.dataset_id, "raw_gain": np.arange(8) * 0.01 - 0.03})
    predictions = group_b.lodo_predictions(outcome, properties, features=["p1", "p2"])
    changed = outcome.copy()
    changed.loc[3, "raw_gain"] = 5.0
    again = group_b.lodo_predictions(changed, properties, features=["p1", "p2"])
    assert again.loc[3, "predicted"] == pytest.approx(predictions.loc[3, "predicted"])
    result = group_b.lodo_evaluation(predictions, n_shuffles=50, outcome=outcome, properties=properties,
                                     features=["p1", "p2"])
    assert result["spearman_rho"] > 0.9 and result["use_accuracy"] >= 0.75
    assert result["oracle_gain"] >= result["realised_gain"] and result["regret_never"] == result["oracle_gain"]
    assert 0 < result["shuffle_p_rho"] <= 1
    with pytest.raises(ValueError, match="at most two"):
        group_b.lodo_predictions(outcome, properties.assign(p3=1.0), features=["p1", "p2", "p3"])


def test_lodo_input_sets_share_one_cohort_and_label_each_claim():
    rng = np.random.default_rng(3)
    ids = [f"d{i}" for i in range(9)]
    properties = pd.DataFrame({"dataset_id": ids, "text": rng.normal(size=9),
                               "pilot": np.arange(9.0), "full": np.arange(9.0)})
    properties.loc[4, "pilot"] = np.nan  # Undefined pilot redundancy drops d4 from every set.
    outcome = pd.DataFrame({"dataset_id": ids, "raw_gain": np.arange(9) * 0.01 - 0.04})
    sets = {"pre_extraction": {"properties": ["text"], "claim": "acquire"},
            "pilot": {"properties": ["text", "pilot"], "claim": "acquire_after_pilot"},
            "post_extraction": {"properties": ["text", "full"], "claim": "use"}}
    predictions, summary = group_b.lodo_input_sets(outcome, properties, input_sets=sets, n_shuffles=20)
    assert list(summary.input_set) == list(sets) and set(summary.claim) == {"acquire", "acquire_after_pilot", "use"}
    assert (summary.datasets == 8).all() and (summary.excluded_datasets == "d4").all()
    assert set(predictions.dataset_id) == set(ids) - {"d4"}
    assert predictions.groupby("input_set").dataset_id.count().eq(8).all()
    # Pilot and full redundancy are identical here, so their rules must agree exactly.
    pilot, full = (predictions[predictions.input_set.eq(s)].predicted.to_numpy() for s in ("pilot", "post_extraction"))
    assert np.allclose(pilot, full)
    assert summary.set_index("input_set").loc["post_extraction", "spearman_rho"] > 0.9
    with pytest.raises(ValueError, match="absent"):
        group_b.lodo_input_sets(outcome, properties.drop(columns="pilot"), input_sets=sets, n_shuffles=0)


def test_pilot_rows_are_fixed_label_free_and_independent_of_content():
    first = group_b.pilot_rows(5000, dataset_id="d", size=200, seed=7)
    assert len(first) == 200 == len(set(first)) and (np.diff(first) > 0).all()
    assert np.array_equal(first, group_b.pilot_rows(5000, dataset_id="d", size=200, seed=7))
    assert not np.array_equal(first, group_b.pilot_rows(5000, dataset_id="e", size=200, seed=7))
    assert len(group_b.pilot_rows(50, dataset_id="d", size=200)) == 50
    rng = np.random.default_rng(1)
    non_text = pd.DataFrame({"x": rng.normal(size=3000)})
    decision = 1 / (1 + np.exp(-3 * non_text.x.to_numpy()))[:, None]
    result = group_b.pilot_redundancy(non_text, decision, dataset_id="d", task_type="binary_classification",
                                      size=200, seed=7)
    assert result["pilot_rows"] == 200
    assert result["pilot_r2"] > 0.8 and result["pilot_top_agreement"] > 0.9
    # Selection depends only on (seed, dataset, position): shuffling contents keeps the rows.
    shuffled = group_b.pilot_redundancy(non_text.sample(frac=1, random_state=0).reset_index(drop=True),
                                        decision, dataset_id="d", task_type="binary_classification", size=200, seed=7)
    assert shuffled["pilot_rows"] == 200 and shuffled["pilot_r2"] < 0.5
    # Sparse text: the pilot size stays 200 and the scored count is reported separately.
    sparse = decision.copy()
    sparse[::2] = np.nan
    partial = group_b.pilot_redundancy(non_text, sparse, dataset_id="d", task_type="binary_classification",
                                       size=200, seed=7)
    assert partial["pilot_rows"] == 200 and 0 < partial["pilot_scored_rows"] < 200


def test_redundancy_is_label_free_and_detects_predictable_outputs():
    rng = np.random.default_rng(1)
    non_text = pd.DataFrame({"x": rng.normal(size=300), "c": rng.choice(["a", "b"], 300)})
    decision = 1 / (1 + np.exp(-3 * non_text.x.to_numpy()))[:, None]
    decision[:10] = np.nan
    folds = [{"train": [i for i in range(300) if i % 5 != k], "test": [i for i in range(300) if i % 5 == k]}
             for k in range(5)]
    predictable = group_b.decision_redundancy(non_text, decision, folds, task_type="binary_classification")
    noise = group_b.decision_redundancy(non_text, rng.random((300, 1)), folds, task_type="binary_classification")
    assert predictable["r2"] > 0.8 > noise["r2"]
    assert predictable["rows"] == 290


def test_b4_size_gains_and_pooled_curve():
    rows = []
    for d in range(4):
        for size in (128, 256):
            for draw in range(3):
                for fold in range(5):
                    for rep, delta in (("baseline", 0), ("decision/both/m", 0.01 * (size == 128) + 0.001 * draw)):
                        rows.append(dict(dataset_id=f"d{d}", task_type="regression", model_id="linear",
                                         representation=rep, training_size=size, draw=draw,
                                         outer_fold=fold, value=0.5 + delta + 0.01 * d))
    sizes = data_efficiency.size_gains(pd.DataFrame(rows))
    assert sizes.draws.eq(3).all()
    pooled = data_efficiency.pooled_curve(sizes)
    assert pooled.set_index("training_size").pooled.round(4).to_dict() == {128: 0.011, 256: 0.001}
    slopes = data_efficiency.gain_slopes(sizes)
    assert slopes.slope_per_doubling.round(3).eq(-0.01).all()


def test_group_b_figures_render_at_a4_width():
    from src.visualize import group_b as figures
    from src.visualize import style

    gains = group_b.dataset_gains(group_b.fold_gains(_folds()))
    summary = pd.concat([group_b.summarize(gains),
                         group_b.summarize(group_b.median_over_decision_models(gains, roster=["m1", "m2"]))])
    produced = [
        *figures.gain_forest(summary, ["decision/both/median", "frozen/e5_small_v2/per_column32"],
                             name="a", what="Test"),
        *figures.win_tie_loss(summary, ["decision/both/median"], name="b", what="Test"),
        *figures.model_by_predictor(gains, name="c", what="Test"),
    ]
    assert len(produced) == 3
    for fig, _, caption in produced:
        assert fig.get_figwidth() == pytest.approx(style.WIDTH_FULL) and caption
        matplotlib.pyplot.close(fig)
    assert list(figures.gain_forest(pd.DataFrame(), [], name="x", what="x")) == []


def test_exports_wait_when_absent(tmp_path):
    from src.evaluation import benchmark_export, study_export

    frames, receipt = benchmark_export.read("B4", target=tmp_path / "absent.zip")
    assert frames == {} and receipt["status"] == study_export.AWAITING
    assert set(benchmark_export.STUDIES) == {"B1", "B2", "B3", "B4", "B5"}
    assert "exp11_data_efficiency" in str(benchmark_export.export_path("B4"))


def test_permutation_attribution_is_disabled_and_row_cohorts_are_pre_treatment():
    from src.evaluation import row_benefit
    from src.utils.config import load_yaml

    assert load_yaml("experiment_1/protocol")["evaluation"]["field_attribution"]["enabled"] is False
    assert row_benefit.POST_TREATMENT_AXES == ()
    reference = pd.DataFrame({"row_id": ["a", "b"], "y_true": ["n", "y"], "text_present": [True, False],
                              "joint_text_seen_in_train": [False, False],
                              "p_c000": [0.6, 0.4], "p_c001": [0.4, 0.6]})
    rows, _ = row_benefit.paired_cohorts(reference, reference.copy(), ["n", "y"], None,
                                         np.array([3.0, 5.0]), np.array([4.0, np.nan]))
    assert {r["axis"] for r in rows} == {"all", "availability", "length", "baseline_entropy"}


def test_regression_metadata_for_probe_arm_widths():
    from src.evaluation.widths import feature_width

    meta = SimpleNamespace(text_columns=range(3), non_text_columns=range(2), classes=(),
                           task_type="regression")
    assert feature_width(meta, {"feature_set": "non_text_plus_supervised_probe",
                                "probe_view": "per_column"}) == 2 + 3 + 27
    assert feature_width(meta, {"feature_set": "non_text_plus_supervised_probe",
                                "probe_view": "joint"}) == 2 + 3 + 9
    assert feature_width(meta, {"feature_set": "non_text_plus_per_column_without_one_text",
                                "provider": "jev", "view": "probabilities"}) == 2 + 3 + 2 * 9


def test_qualified_artifacts_cover_group_b_arms():
    from src.evaluation.qualification import expected_artifacts

    decisions = {("d", "jev"): {"artifact_ids": {"per_column": "pc", "joint": "jt"},
                                "artifact_checksums": {"pc": "1", "jt": "2"}}}
    embeddings = {("d", "e5_small_v2"): {"artifact_id": "emb"},
                  ("d", "e5_small_v2_tar"): {"artifact_ids": ["t0", "t1"]}}
    evidence = (None, decisions, embeddings)

    def ids(**arm):
        return sorted(expected_artifacts(evidence, {"dataset_id": "d", "provider": None, **arm})[0])

    assert ids(phase="experiment_2", feature_set="non_text_plus_both_plus_frozen_e5", provider="jev",
               encoder="e5_small_v2") == ["emb", "jt", "pc"]
    assert ids(phase="rq2", feature_set="non_text_plus_per_column_without_one_text", provider="jev") == ["pc"]
    assert ids(phase="experiment_3", feature_set="non_text_plus_supervised_probe",
               encoder="e5_small_v2") == ["emb"]
    assert ids(phase="experiment_3", feature_set="non_text_plus_tar", encoder="e5_small_v2_tar") == ["t0", "t1"]
    assert ids(phase="data_efficiency", feature_set="non_text_plus_both", provider="jev") == ["jt", "pc"]


def test_headline_accepts_categorical_export_fold_tables():
    """collect_folds (and the Parquet round trip) store labels as `category`; the analyses
    must neither fail on mismatched categories nor create unobserved combinations."""
    from src.evaluation.benchmark_export import RECIPE_COLUMNS

    rows = []
    for d in range(4):
        for fold in range(5):
            for model in ("linear", "catboost"):
                for fs, provider, delta in (("non_text", None, 0.0), ("non_text_plus_both", "m1", 0.02),
                                            ("non_text_plus_both", "m2", 0.01)):
                    for metric, value in (("roc_auc", 0.7 + delta), ("log_loss", 0.5)):
                        rows.append({**dict.fromkeys(RECIPE_COLUMNS), "phase": "experiment_1",
                                     "dataset_id": f"d{d}", "task_type": "binary_classification",
                                     "model_id": model, "checkpoint_id": "c", "feature_set": fs,
                                     "provider": provider, "view": "main", "split_id": "s",
                                     "outer_fold": fold, "metric": metric, "value": value})
    folds = pd.DataFrame(rows)
    for column in ("phase", "dataset_id", "task_type", "model_id", "feature_set", "provider",
                   "metric", "view", "checkpoint_id", "split_id"):
        folds[column] = folds[column].astype("category")
    head = group_b.headline(folds)
    assert set(head.metric) == {"roc_auc"} and len(head) == 4 * 5 * 2 * 3
    gains = group_b.dataset_gains(group_b.fold_gains(head))
    assert len(gains) == 4 * 2 * 2 and gains.raw_gain.notna().all()
    summary = group_b.summarize(gains)
    assert len(summary) == 2 * 2
    medians = group_b.median_over_decision_models(gains, roster=["m1", "m2"])
    assert len(medians) == 4 * 2 and medians.raw_gain.round(3).eq(0.015).all()


def test_model_diagnostics_count_each_distinct_input_once():
    """A1 tables carry both "rows" and "unique_inputs" weightings and both request modes; B3
    uses distinct joint inputs only."""
    shape = pd.DataFrame({"provider": ["m"] * 3, "dataset_id": ["d"] * 3,
                          "representation": ["joint", "joint", "per_column"],
                          "weighting": ["rows", "unique_inputs", "unique_inputs"],
                          "mean_top_probability": [0.9, 0.6, 0.2]})
    consensus = shape.rename(columns={"mean_top_probability": "distance"}).assign(distance=[0.4, 0.1, 0.8])
    stability = pd.DataFrame({"provider": ["m", "m"], "dataset": ["d", "d"], "mode": ["joint", "per_column"],
                              "excess_tv": [0.02, 0.5]})
    out = group_b.model_diagnostics(output_shape=shape, consensus=consensus,
                                    stability=stability).set_index("provider")
    assert out.loc["m", "stability_excess_tv"] == pytest.approx(0.02)
    assert out.loc["m", "mean_top_probability"] == pytest.approx(0.6)
    assert out.loc["m", "consensus_distance"] == pytest.approx(0.1)


def test_probe_crossfit_bins_and_c_never_use_the_scored_rows():
    """Each cross-fit part chooses its regression bins and C from its own training rows only:
    an extreme target of one row cannot change that row's own cross-fitted feature."""
    x, y, _ = _probe_data("regression")
    groups = [["e0", "e1", "e2"], ["e3", "e4", "e5"]]

    def features(target):
        probe = label_probe.SupervisedProbe(groups, view="joint", task_type="regression", classes=[],
                                            seed=3)
        return probe.fit_transform(x.iloc[:100], target), probe

    base, probe = features(y.iloc[:100])
    changed_y = y.iloc[:100].copy()
    changed_y.iloc[50] = changed_y.max() + 100.0  # row 50 is the lowest target: every edge moves
    changed, _ = features(changed_y)
    assert np.allclose(changed.iloc[50].to_numpy(), base.iloc[50].to_numpy())
    (diagnostic,) = probe.diagnostics_
    assert len(diagnostic["crossfit_selected_C"]) == label_probe.CROSSFIT_FOLDS
