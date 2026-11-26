"""B4 field attribution: grouping, deterministic permutations, loss math and isolation."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.evaluation import attribution
from src.evaluation.features import missing_column

DECISION_JOINT = "jev__jev__joint__text_only__all_text__p_positive"
DECISION_FIELD = "liquid__liquid__per_column__text_only__text_001__p_positive"


def _dataset(rows=40, *, binary=True):
    values = np.arange(rows)
    frame = pd.DataFrame({
        "number": values.astype(float),
        "body": [f"text {i}" if i % 7 else None for i in values],
        "title": [f"title {i % 5}" for i in values],
        "target": values % 2 if binary else values * 0.5,
    })
    official = {"target": "target", "task_type": "cls" if binary else "reg"}
    metadata = extract_metadata(frame, official, dataset_id="attr", dataset_name="Attr",
                                text_columns=["body", "title"])
    return LoadedDataset(frame, metadata, official, {"data_identity": "fixed"}, [])


def _features(dataset):
    rng = np.random.default_rng(0)
    index = [f"csv:{i}" for i in range(len(dataset.frame))]
    features = pd.DataFrame({
        "number": dataset.frame.number.to_numpy(),
        missing_column(0): dataset.frame.body.isna().astype(int).to_numpy(),
        missing_column(1): 0,
        # Synthetic, informative decision signal so that permuting it must cost loss.
        DECISION_JOINT: 0.1 + 0.6 * (dataset.frame.target.to_numpy() % 2)
        + 0.1 * rng.uniform(size=len(index)),
        DECISION_FIELD: rng.uniform(size=len(index)),
    }, index=index)
    features.attrs["feature_artifact_ids"] = []
    return features


def test_groups_cover_every_column_once_and_unknown_columns_fail():
    dataset = _dataset()
    groups = {g["group"]: g for g in attribution.feature_groups(_features(dataset), dataset.metadata)}
    assert set(groups) == {"non_text", "text_availability", "decision__jev__joint__all_text",
                           "decision__liquid__per_column__text_001"}
    assert groups["text_availability"]["columns"] == [missing_column(0), missing_column(1)]
    field = groups["decision__liquid__per_column__text_001"]
    assert (field["provider"], field["mode"], field["text_index"], field["text_field"]) == (
        "liquid", "per_column", 1, "title")
    assert groups["decision__jev__joint__all_text"]["text_field"] == "all_text"
    with pytest.raises(ValueError, match="cannot be assigned"):
        attribution.feature_groups(_features(dataset).assign(stray=1), dataset.metadata)
    unknown_field = _features(dataset).rename(columns={DECISION_FIELD: DECISION_FIELD.replace("001", "007")})
    with pytest.raises(ValueError, match="unknown text field"):
        attribution.feature_groups(unknown_field, dataset.metadata)


def test_embedding_groups_carry_their_availability_flag():
    dataset = _dataset()
    features = _features(dataset).drop(columns=[DECISION_JOINT, DECISION_FIELD])
    for i in range(2):
        for j in range(3):
            features[f"e5__{i}__{j}"] = float(j)
    features.attrs["embedding_groups"] = [[f"e5__{i}__{j}" for j in range(3)] for i in range(2)]
    groups = {g["group"]: g for g in attribution.feature_groups(features, dataset.metadata)}
    assert set(groups) == {"non_text", "embedding__text_000", "embedding__text_001"}
    assert groups["embedding__text_000"]["columns"][-1] == missing_column(0)
    assert groups["embedding__text_001"]["text_field"] == "title"


def test_permutations_are_deterministic_and_group_specific():
    first = attribution.group_rng(7, 2, "decision__jev__joint__all_text").permutation(50)
    again = attribution.group_rng(7, 2, "decision__jev__joint__all_text").permutation(50)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, attribution.group_rng(7, 3, "decision__jev__joint__all_text").permutation(50))
    assert not np.array_equal(first, attribution.group_rng(7, 2, "non_text").permutation(50))
    frame = pd.DataFrame({"a": [1, 2, 3], "b": pd.Categorical(["x", "y", "z"]), "c": [4, 5, 6]},
                         index=["csv:0", "csv:1", "csv:2"])
    moved = attribution.permute_columns(frame, ["a", "b"], [2, 0, 1])
    assert moved.index.equals(frame.index) and moved.dtypes.equals(frame.dtypes)
    assert moved.a.tolist() == [3, 1, 2] and moved.b.tolist() == ["z", "x", "y"]
    assert moved.c.equals(frame.c)


def test_loss_increase_matches_hand_computation():
    x = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "z": [5.0, 5.0, 5.0, 5.0]},
                     index=[f"csv:{i}" for i in range(4)])
    model = SimpleNamespace(predict=lambda f: f.a.to_numpy())
    groups = [{"group": "a", "kind": "non_text", "columns": ["a"]},
              {"group": "z", "kind": "non_text", "columns": ["z"]}]
    settings = {"repeats": 2, "max_test_rows": None}
    y = x.a.to_numpy()
    result = attribution.fold_attribution(
        model, x, y, y, groups, predict=lambda m, f: m.predict(f), classes=[], variance=2.0,
        seed=11, fold=0, settings=settings)
    expected = []
    rng = attribution.group_rng(11, 0, "a")
    for _ in range(2):
        order = rng.permutation(4)
        expected.append(float(np.mean((y - y[order]) ** 2 / 2.0)))
    (a, z) = result["groups"]
    assert a["repeat_loss_increases"] == pytest.approx(expected)
    assert a["mean_loss_increase"] == pytest.approx(np.mean(expected))
    assert z["mean_loss_increase"] == 0 and result["unpermuted_repredict_mean_loss_change"] == 0
    # Classification uses the observed-class log loss.
    def probs(_, f):
        return np.column_stack([1 - f.a.to_numpy() / 4 - 0.1, f.a.to_numpy() / 4 + 0.1])

    labels = np.array([0, 0, 1, 1])
    out = attribution.fold_attribution(
        model, x, labels, probs(None, x), groups, predict=probs, classes=[0, 1], variance=None,
        seed=11, fold=0, settings={"repeats": 1, "max_test_rows": None})
    order = attribution.group_rng(11, 0, "a").permutation(4)
    p, q = probs(None, x), probs(None, x.iloc[order].set_axis(x.index))
    manual = -np.log(q[np.arange(4), labels]) + np.log(p[np.arange(4), labels])
    assert out["groups"][0]["mean_loss_increase"] == pytest.approx(manual.mean())
    assert out["loss"] == "log_loss"


def test_row_cap_is_recorded_and_hook_inputs_must_be_covered():
    x = pd.DataFrame({"a": np.arange(10.0)}, index=[f"csv:{i}" for i in range(10)])
    model = SimpleNamespace(predict=lambda f: f.a.to_numpy())
    groups = [{"group": "a", "kind": "non_text", "columns": ["a"]}]
    out = attribution.fold_attribution(
        model, x, x.a.to_numpy(), x.a.to_numpy(), groups, predict=lambda m, f: m.predict(f),
        classes=[], variance=1.0, seed=1, fold=0, settings={"repeats": 1, "max_test_rows": 4})
    assert out["rows_used"] == 4 and out["outer_test_rows"] == 10
    assert out["row_subset"]["max_test_rows"] == 4

    class Wrapper:
        def attribution_inputs(self, frame):
            expanded = frame.assign(v0=1.0, v1=2.0)
            return expanded, [{"group": "embedding__text_000", "kind": "embedding",
                               "columns": ["v0"]}], model

    with pytest.raises(ValueError, match="cover each predictor input"):
        attribution.fold_attribution(
            Wrapper(), x, x.a.to_numpy(), x.a.to_numpy(), groups,
            predict=lambda m, f: m.predict(f), classes=[], variance=1.0, seed=1, fold=0,
            settings={"repeats": 1, "max_test_rows": None})


def test_frozen_embedding_wrapper_accepts_joint_vector_and_flag_permutation():
    from src.evaluation.embedding_control import EmbeddingControl

    dataset = _dataset(30)
    features = _features(dataset).drop(columns=[DECISION_JOINT, DECISION_FIELD])
    missing = dataset.frame.body.isna().to_numpy()
    for j in range(3):
        values = np.sin(np.arange(30.0) * (j + 1))
        values[missing] = np.nan
        features[f"e5__0__{j}"] = values
        features[f"e5__1__{j}"] = np.cos(np.arange(30.0) * (j + 1))
    groups_attr = [[f"e5__{i}__{j}" for j in range(3)] for i in range(2)]
    features.attrs["embedding_groups"] = groups_attr
    groups = attribution.feature_groups(features, dataset.metadata)
    inner = SimpleNamespace(fit=lambda x, y: None, predict=lambda x: x.fillna(0).sum(axis=1).to_numpy())
    model = EmbeddingControl(inner, groups_attr, components=2).fit(features, None)
    prediction = model.predict(features)
    out = attribution.fold_attribution(
        model, features, prediction, prediction, groups, predict=lambda m, f: m.predict(f),
        classes=[], variance=1.0, seed=3, fold=1, settings={"repeats": 2, "max_test_rows": None})
    assert {g["group"] for g in out["groups"]} == {"non_text", "embedding__text_000",
                                                   "embedding__text_001"}


def test_conditional_permutation_stays_within_availability_strata():
    rng = np.random.default_rng(0)
    flags = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 0])
    for _ in range(20):
        order = attribution.stratified_order(rng, flags)
        assert sorted(order) == list(range(10))
        np.testing.assert_array_equal(flags[order], flags)
    # Joint blocks stratify on every flag pattern; no strata reproduces the plain draw.
    pattern = np.column_stack([flags, flags[::-1]])
    order = attribution.stratified_order(np.random.default_rng(1), pattern)
    np.testing.assert_array_equal(pattern[order], pattern)
    np.testing.assert_array_equal(
        attribution.stratified_order(np.random.default_rng(2), np.empty((10, 0))),
        np.random.default_rng(2).permutation(10))


def test_strata_follow_group_kind_and_scheme():
    frame = pd.DataFrame({missing_column(0): [0, 1], missing_column(1): [0, 0], "x": [1, 2]})
    field = {"kind": "decision", "text_index": 1}
    joint = {"kind": "decision", "text_index": None}
    embedding = {"kind": "embedding", "text_index": 0}
    cond = attribution.CONDITIONAL
    assert attribution.strata_columns(field, frame, cond) == [missing_column(1)]
    assert attribution.strata_columns(joint, frame, cond) == [missing_column(0), missing_column(1)]
    assert attribution.strata_columns(embedding, frame, cond) == [missing_column(0)]
    for kind in ("text_availability", "non_text"):
        assert attribution.strata_columns({"kind": kind, "text_index": None}, frame, cond) == []
    assert attribution.strata_columns(field, frame, attribution.UNCONDITIONAL) == []


def test_conditional_attribution_ignores_missingness_signal_unconditional_does_not():
    # The decision column only encodes availability (0 when text is missing, 1 otherwise),
    # which the flags already carry: conditional permutation must give exactly zero.
    n = 40
    flag = (np.arange(n) % 4 == 0).astype(int)
    x = pd.DataFrame({missing_column(0): flag, DECISION_FIELD.replace("001", "000"): 1.0 - flag},
                     index=[f"csv:{i}" for i in range(n)])
    decision = x.columns[1]
    model = SimpleNamespace(predict=lambda f: f[decision].to_numpy())
    groups = [{"group": "decision__liquid__per_column__text_000", "kind": "decision",
               "text_index": 0, "columns": [decision]},
              {"group": "text_availability", "kind": "text_availability", "text_index": None,
               "columns": [missing_column(0)]}]
    y = x[decision].to_numpy()
    common = dict(predict=lambda m, f: m.predict(f), classes=[], variance=1.0, seed=5, fold=0)
    cond = attribution.fold_attribution(model, x, y, y, groups,
                                        settings={"repeats": 2, "max_test_rows": None}, **common)
    plain = attribution.fold_attribution(
        model, x, y, y, groups, settings={"repeats": 2, "max_test_rows": None,
                                          "permutation": "unconditional"}, **common)
    assert cond["permutation"] == "conditional_on_availability"
    first = cond["groups"][0]
    assert first["permutation"] == "conditional_on_availability" and first["n_strata"] == 2
    assert first["strata_columns"] == [missing_column(0)] and first["mean_loss_increase"] == 0
    assert plain["permutation"] == "unconditional"
    assert plain["groups"][0]["mean_loss_increase"] > 0
    # Availability flags themselves are permuted unconditionally in both schemes.
    assert cond["groups"][1]["permutation"] == "unconditional"
    with pytest.raises(ValueError, match="permutation"):
        attribution.fold_attribution(model, x, y, y, groups, settings={
            "repeats": 1, "max_test_rows": None, "permutation": "other"}, **common)


def test_settings_skip_native_text_and_validate():
    on = attribution.settings_for_phase({"enabled": True, "repeats": 3, "max_test_rows": None},
                                        "B2")
    assert on["enabled"] and on["skip_reason"] is None
    assert on["permutation"] == attribution.CONDITIONAL and on["version"].endswith("v2")
    plain = attribution.settings_for_phase({"enabled": True, "permutation": "unconditional"},
                                           "B1")
    assert plain["permutation"] == "unconditional"
    with pytest.raises(ValueError, match="permutation"):
        attribution.settings_for_phase({"permutation": "within_class"}, "B1")
    native = attribution.settings_for_phase({"enabled": True}, "B6")
    assert not native["enabled"] and native["skip_reason"].startswith("native_text_system")
    for bad in ({"repeats": 0}, {"max_test_rows": 1}, {"enabled": "yes"}, {"extra": 1}):
        with pytest.raises(ValueError):
            attribution.settings_for_phase(bad, "B1")


class _Model:
    classes_ = [0, 1]

    def fit(self, x, y):
        self.scale = float(np.std(x["number"])) or 1.0

    def predict_proba(self, x):
        signal = x[DECISION_JOINT].to_numpy() * 0.6 + x["number"].to_numpy() / self.scale * 0.01
        p = np.clip(0.2 + signal, 0.05, 0.95)
        return np.column_stack([1 - p, p])


def _run(dataset, features, phase, settings):
    from src.evaluation import protocol

    return protocol.evaluate_cv(
        dataset, features, _Model, specification=protocol.shared_splits(dataset),
        model_id="fixture", checkpoint_id="pinned", feature_set="non_text_plus_joint",
        allow_model_runs=True, phase=phase, estimator_settings={"seed": 1},
        feature_artifact_ids=[], representation_recipe={"provider": "jev"},
        field_attribution=settings,
    )


def test_attribution_leaves_primary_records_unchanged_and_outside_collection(isolated_output):
    import json

    from src.evaluation import results, store

    dataset = _dataset()
    features = _features(dataset)
    off = _run(dataset, features, "B1", None)
    on_settings = attribution.settings_for_phase({"enabled": True, "repeats": 2}, "B2")
    on = _run(dataset, features, "B2", on_settings)
    assert len(on) == len(off) == 10
    for left, right in zip(off, on, strict=True):
        a, b = json.loads(left.read_text()), json.loads(right.read_text())
        assert a["metadata"] == b["metadata"] and a["scores"] == b["scores"]
        assert a["warnings"] == b["warnings"]
        assert set(a["timing"]["stages"]) == set(b["timing"]["stages"])
        assert "field_attribution" not in b["timing"]["stages"]
        pd.testing.assert_frame_equal(store.read_predictions(left), store.read_predictions(right))
    bodies = list(store.attribution_bodies("B2"))
    assert len(bodies) == 5 and all(b["status"] == "complete" for b in bodies)
    assert {b["outer_test_record"] for b in bodies} <= {r.key for r in on}
    assert "field_attribution" in bodies[0]["timing"]["stages"]
    joint = next(g for g in bodies[0]["groups"] if g["group"] == "decision__jev__joint__all_text")
    assert joint["mean_loss_increase"] > 0 and len(joint["repeat_loss_increases"]) == 2
    # Headline collection reads the same records whether or not attribution exists.
    _, folds_off, _, _ = results.collect(("B1",))
    _, folds_on, _, _ = results.collect(("B2",))
    assert set(folds_off.metric) == set(folds_on.metric)
    # Resume reuses completed folds without new fits or duplicate attribution.
    assert _run(dataset, features, "B2", on_settings) == on
    status, counts = attribution.attribution_status(("B1", "B2"))
    assert counts.set_index(["phase", "status"]).folds.to_dict() == {
        ("B1", "missing"): 5, ("B2", "complete"): 5}
    by_group, by_type, _ = attribution.attribution_summary(
        ("B2",), field_types=pd.DataFrame(
            {"dataset_id": ["attr", "attr"], "field": ["body", "title"],
             "field_type": ["short_text", "categorical"]}))
    assert set(by_group.field_type) == {"non_text", "text_availability", "joint_all_text",
                                         "categorical"}
    assert by_group.folds.eq(5).all() and by_type.n_datasets.eq(1).all()
    assert bodies[0]["permutation"] == bodies[0]["settings"]["permutation"] == "conditional_on_availability"
    assert set(by_group.permutation) == set(by_type.permutation) == {"conditional_on_availability"}


def test_native_text_records_skip_reason(isolated_output):
    from src.evaluation import store

    dataset = _dataset()
    features = _features(dataset)
    _run(dataset, features, "B6",
         attribution.settings_for_phase({"enabled": True}, "B6"))
    bodies = list(store.attribution_bodies("B6"))
    assert len(bodies) == 5 and {b["status"] for b in bodies} == {"skipped"}
    assert bodies[0]["reason"].startswith("native_text_system")


def test_store_merge_carries_attribution(isolated_output, tmp_path):
    from src.evaluation import store

    dataset = _dataset()
    records = _run(dataset, _features(dataset), "B1",
                   attribution.settings_for_phase({"enabled": True, "repeats": 1}, "B1"))
    destination = tmp_path / "merged" / "attr.sqlite3"
    destination.parent.mkdir()
    assert store.merge(records[0].database, destination) == "merged"
    assert store.merge(records[0].database, destination) == "merged"
    with store.readonly(destination) as db:
        assert db.execute("SELECT COUNT(*) FROM field_attribution").fetchone()[0] == 5


def test_unassignable_column_fails_before_any_fit(isolated_output):
    dataset = _dataset()
    calls = []

    class Recording(_Model):
        def fit(self, x, y):
            calls.append(len(x))

    from src.evaluation import protocol

    with pytest.raises(ValueError, match="cannot be assigned"):
        protocol.evaluate_cv(
            dataset, _features(dataset).assign(stray=1.0), Recording,
            specification=protocol.shared_splits(dataset), model_id="fixture",
            checkpoint_id="pinned", feature_set="non_text_plus_joint", allow_model_runs=True,
            phase="B1", estimator_settings={}, feature_artifact_ids=[],
            field_attribution=attribution.settings_for_phase({"enabled": True}, "B1"))
    assert not calls
