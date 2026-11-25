"""Regressions for the independent review of commit 65c183f (Group B/C analyses, 0B identity).

Each test reproduces one reviewed claim on constructed data; no predictor is fitted.
"""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation import data_efficiency, group_b, label_probe
from src.evaluation import strands_adaptation as sa

HEAD = dict(task_type="binary_classification", checkpoint_id="c", split_id="s")


def _head(rows):
    return pd.DataFrame([{**HEAD, **r} for r in rows])


# ------------------------------------------------------------------ claim 1: probe fallback


def test_probe_fallback_is_one_constant_shared_with_non_training_rows():
    """Changing a scored row's own label cannot affect the sparse fallback."""
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(30, 2)), columns=["a", "b"], index=[f"csv:{i}" for i in range(30)])
    x.iloc[8:, 0] = np.nan  # column "a" present in 8 training rows only -> fallback
    y = pd.Series(np.where(np.arange(30) % 3 == 0, "y", "n"))
    probe = label_probe.SupervisedProbe([["a"]], view="per_column", task_type="binary_classification",
                                        classes=["n", "y"], seed=1)
    train = probe.fit_transform(x.iloc[:20], y.iloc[:20])
    assert probe.diagnostics_[0]["fallback"] == "uniform_label_independent"
    present = train.iloc[:8].to_numpy()
    assert np.unique(present).size == 1  # identical for all present training rows
    other = probe.transform(x.iloc[[20]].assign(a=0.0))
    assert np.allclose(other.to_numpy(), present[0])
    y.iloc[0] = "n"
    assert train.equals(probe.fit_transform(x.iloc[:20], y.iloc[:20]))
    assert present[0, 0] == 0.5


# ------------------------------------------------------- claim 2: coherent five-fold pairs


def test_dataset_gain_needs_five_finite_folds():
    rows = []
    for fold in range(5):
        rows.append(dict(dataset_id="d", model_id="m", outer_fold=fold, representation="baseline", value=0.7))
        rows.append(dict(dataset_id="d", model_id="m", outer_fold=fold, representation="decision/both/p",
                         value=np.nan if fold == 2 else 0.8))
    gains = group_b.dataset_gains(group_b.fold_gains(_head(rows)))
    assert gains.empty  # an undefined fold gain leaves the dataset incomplete, not folds=5


def test_dataset_gain_refuses_arms_mixing_split_identities():
    rows = []
    for split in ("s1", "s2"):
        for fold in range(5):
            for rep, value in (("baseline", 0.7), ("decision/both/p", 0.8 if split == "s1" else 0.6)):
                rows.append(dict(dataset_id="d", model_id="m", outer_fold=fold, representation=rep,
                                 value=value, split_id=split))
    with pytest.raises(ValueError, match="one split"):
        group_b.dataset_gains(group_b.fold_gains(_head(rows)))


def test_column_importance_pairs_only_the_same_split():
    rows = []
    for fold in range(5):
        rows.append(dict(dataset_id="d", model_id="m", outer_fold=fold, representation="decision/per_column/p",
                         value=0.8, split_id="s1"))
        rows.append(dict(dataset_id="d", model_id="m", outer_fold=fold, representation="without_text/p/0",
                         value=0.7, split_id="s2"))
    assert group_b.text_column_importance(_head(rows)).empty
    rows = [r | {"split_id": "s1"} for r in rows]
    rows[1]["value"] = np.nan
    assert group_b.text_column_importance(_head(rows)).empty  # four finite folds are not five


def _recipe(artifacts, fold):
    return {"dataset_id": "d", "task_type": "binary_classification", "model_id": "linear",
            "checkpoint_id": "c", "feature_set": "non_text_plus_both", "feature_artifact_ids": artifacts,
            "representation_recipe": {"provider": "p"}, "split_id": "s", "outer_fold": fold}


def _collect(monkeypatch, tmp_path, recipes, **selection):
    from src.evaluation import benchmark_export, records, store

    def completed(recipe, phase):
        path = tmp_path / f"{len(list(tmp_path.iterdir()))}.json"
        path.write_text(json.dumps({"metadata": {"prediction_role": "outer_test"},
                                    "scores": {"metrics": {"roc_auc": 0.8}},
                                    "timing": {"stages": {}}}), encoding="utf-8")
        return [path]

    monkeypatch.setattr(store, "all_recipes", lambda phase: recipes)
    monkeypatch.setattr(records, "completed_fold", completed)
    return benchmark_export.collect_folds(("experiment_1",), **selection)


def test_collect_folds_never_completes_an_arm_from_two_recipes(monkeypatch, tmp_path):
    # Folds 0-2 from one feature artifact and 3-4 from another are not one complete arm.
    mixed = [_recipe(["old"], f) for f in range(3)] + [_recipe(["new"], f) for f in range(3, 5)]
    assert _collect(monkeypatch, tmp_path, mixed).empty
    both = [_recipe(["old"], f) for f in range(5)] + [_recipe(["new"], f) for f in range(5)]
    with pytest.raises(ValueError, match="several complete recipes"):
        _collect(monkeypatch, tmp_path, both)
    one = _collect(monkeypatch, tmp_path, [_recipe(["new"], f) for f in range(5)])
    assert len(one) == 5 * 3 and one.recipe_id.nunique() == 1


def test_scoped_folds_match_the_recorded_version_selector_and_artifacts(monkeypatch, tmp_path):
    from src.evaluation import benchmark_export as export
    from src.evaluation.design import decision_versions
    from src.utils.config import load_yaml

    main = load_yaml("experiment_1/main")
    registered, = decision_versions(["kev"])
    selected = {k: registered[k] for k in ("provider", "model", "version")}
    selected.update(question_bank_id="bank", question_variant="main", settings_sha256="settings")
    main["feature_selectors"]["kev"] = selected
    main["dataset_feature_artifacts"]["d"] = {"kev": {"per_column": "per", "joint": "joint"}}
    monkeypatch.setattr(export, "load_yaml", lambda name: main)

    def folds(selector, artifacts, *, dataset="d", feature_set="non_text_plus_both", provider="kev"):
        return [_recipe(artifacts, fold) | {
            "dataset_id": dataset, "feature_set": feature_set,
            "representation_recipe": {"provider": provider, "selector": selector}
        } for fold in range(5)]

    valid = folds(selected, ["per", "joint"])
    hybrid = folds(selected, ["per", "joint", "frozen"], feature_set="non_text_plus_both_plus_frozen_e5")
    joint = folds(selected, ["joint"], feature_set="non_text_plus_joint")
    stale = folds({**selected, "version": "previous"}, ["per", "joint"])
    legacy = folds(None, ["per", "joint"])
    wrong = [
        *folds({**selected, "model": "another-model"}, ["per", "joint"]),
        *folds({**selected, "question_variant": "other"}, ["per", "joint"]),
        *folds({**selected, "settings_sha256": "other"}, ["per", "joint"]),
        *folds({**selected, "question_bank_id": "other"}, ["per", "joint"]),
        *folds(selected, ["joint", "per"]),
        *folds(selected, ["per", "wrong", "joint"], feature_set="non_text_plus_both_plus_frozen_e5"),
        *folds(selected, ["per", "joint"], provider=None),
        *folds(None, ["p1"], provider="jev"), *folds(None, ["p2"], provider="jev"),
    ]
    scoped = _collect(monkeypatch, tmp_path, [*valid, *hybrid, *joint, *stale, *legacy, *wrong],
                      metrics=["roc_auc"], cohort="open_weight")
    assert len(scoped) == 15
    assert set(scoped.recipe_id) == {export.arm_recipe_id(a[0]) for a in (valid, hybrid, joint)}
    assert _collect(monkeypatch, tmp_path, [*valid[:4], *stale, *legacy], cohort="open_weight").empty
    assert len(_collect(monkeypatch, tmp_path, legacy, metrics=["roc_auc"])) == 5

    main["feature_selectors"]["kev"] = None
    first = folds(selected, ["older-publication"], dataset="other", feature_set="non_text_plus_joint")
    second = folds(selected, ["another-publication"], dataset="other", feature_set="non_text_plus_joint")
    assert len(_collect(monkeypatch, tmp_path, [*first, *second[:4]], metrics=["roc_auc"],
                        cohort="open_weight")) == 5
    with pytest.raises(ValueError, match="several complete recipes"):
        _collect(monkeypatch, tmp_path, [*first, *second], metrics=["roc_auc"], cohort="open_weight")


# ----------------------------------------------------------- claim 3: fixed decision roster


def _gains(providers, datasets=("d0", "d1")):
    rows = []
    for d in datasets:
        for fold in range(5):
            rows.append(dict(dataset_id=d, model_id="m", outer_fold=fold, representation="baseline", value=0.7))
            for p in providers:
                rows.append(dict(dataset_id=d, model_id="m", outer_fold=fold,
                                 representation=f"decision/both/{p}", value=0.8))
    return group_b.dataset_gains(group_b.fold_gains(_head(rows)))


def test_median_requires_the_configured_roster_and_names_the_missing_models():
    gains = _gains(["jev"])
    assert group_b.median_over_decision_models(gains, roster=["jev", "kev"]).empty
    gaps = group_b.roster_gaps(gains, roster=["jev", "kev"])
    assert set(gaps.missing_decision_models) == {"kev"} and len(gaps) == 2
    full = group_b.median_over_decision_models(_gains(["jev", "kev"]), roster=["jev", "kev"])
    assert full.decision_models.eq(2).all() and len(full) == 2
    with pytest.raises(TypeError):
        group_b.median_over_decision_models(gains)  # never inferred from what happens to exist


def test_b2_contrasts_wait_when_only_frozen_arms_exist():
    from src.evaluation import benchmark_export

    rows = []
    for fold in range(5):
        rows.append(dict(dataset_id="d", model_id="m", outer_fold=fold, representation="baseline", value=0.7))
        rows.append(dict(dataset_id="d", model_id="m", outer_fold=fold,
                         representation="frozen/e5_small_v2/per_column32", value=0.75))
    gains = group_b.dataset_gains(group_b.fold_gains(_head(rows)))
    out = benchmark_export.contrasts(gains, roster=["jev"])
    assert out.empty and "difference" in out.columns


# --------------------------------------------------------------- claim 4: 0B identity


def test_group_b_and_c_scientific_configs_enter_the_0b_identity(monkeypatch):
    from src.evaluation import acceptance

    original = acceptance.load_yaml
    cache = {}

    def load(name):
        if name not in cache:
            cache[name] = copy.deepcopy(original(name))
        return cache[name]

    monkeypatch.setattr(acceptance, "load_yaml", load)
    before = acceptance.evaluation_identity()
    load("rq3/adaptation")["training_enabled"] = True  # execution switch: identity unchanged
    load("rq3/adaptation")["status"] = "pilot_reviewed"
    assert acceptance.evaluation_identity() == before
    load("rq3/adaptation")["regression_bins"] = 3
    assert acceptance.evaluation_identity() != before
    load("rq3/adaptation")["regression_bins"] = 9
    load("data_efficiency/data_efficiency")["training_sizes"] = [128]
    assert acceptance.evaluation_identity() != before


# ------------------------------------------------------------------- claim 5: LODO null


def test_lodo_shuffle_p_is_undefined_when_the_observed_rank_correlation_is():
    properties = pd.DataFrame({"dataset_id": [f"d{i}" for i in range(6)], "p1": np.arange(6.0)})
    outcome = pd.DataFrame({"dataset_id": properties.dataset_id, "raw_gain": 0.01})  # equal gains
    predictions = group_b.lodo_predictions(outcome, properties, features=["p1"])
    result = group_b.lodo_evaluation(predictions, n_shuffles=20, outcome=outcome, properties=properties,
                                     features=["p1"])
    assert np.isnan(result["spearman_rho"]) and np.isnan(result["shuffle_p_rho"])
    assert result["status"] == "rank_correlation_undefined"


# --------------------------------------------------------------- claim 6: C2 fold bundle


def test_fold_assembly_requires_finite_probabilities_on_rows_with_text():
    n, train, test = 40, np.arange(30), np.arange(30, 40)
    parts = sa.crossfit_parts(train, dataset_id="D", split_id="s", fold=0, seed=1)
    seen = [sa.fit_partition(train, parts, k)[0] for k in range(3)]
    nonempty = np.ones(n, dtype=bool)
    empty = [np.full((n, 2), np.nan) for _ in range(3)]
    with pytest.raises(ValueError, match="finite"):
        sa.assemble_fold(n, train, test, parts, empty, seen, nonempty=nonempty)
    good = []
    for k in range(3):
        values = np.full((n, 2), np.nan)
        values[train[parts == k]] = [0.3, 0.7]
        values[test] = [0.4, 0.6]
        good.append(values)
    out = sa.assemble_fold(n, train, test, parts, good, seen, nonempty=nonempty)
    assert np.isfinite(out).all()
    unnormalised = [v * 2 for v in good]
    with pytest.raises(ValueError, match="normalised"):
        sa.assemble_fold(n, train, test, parts, unnormalised, seen, nonempty=nonempty)
    no_text = nonempty.copy()
    no_text[5] = False  # a row without joint text must stay missing
    with pytest.raises(ValueError, match="missing"):
        sa.assemble_fold(n, train, test, parts, good, seen, nonempty=no_text)


def test_republishing_a_sealed_fold_keeps_its_artifact_identity(tmp_path):
    (tmp_path / "probabilities.npy").write_bytes(b"x")
    recipe = {"schema": "s", "fold": 0}
    reports = [{"part": 0, "best_validation_loss": 0.1, "seconds": 1.0, "peak_gpu_bytes": 1, "history": []}]
    first = sa.seal_bundle(tmp_path, recipe, ["probabilities.npy"], reports)
    manifest = (tmp_path / "manifest.json").read_bytes()
    assert sa.sealed_bundle(tmp_path, recipe) == first
    assert sa.seal_bundle(tmp_path, recipe, ["probabilities.npy"], reports) == first
    assert (tmp_path / "manifest.json").read_bytes() == manifest
    assert sa.sealed_bundle(tmp_path, {**recipe, "fold": 1}) is None


# ---------------------------------------------------------------------- claim 7: B4


def _sizes():
    rows = []
    for d in range(4):
        sizes = (128, 256, 4096) if d == 3 else (128, 256)
        for size in sizes:
            for draw in range(3):
                for fold in range(5):
                    base = 0.5 + 0.1 * d
                    for rep, delta in (("baseline", 0), ("decision/both/m", 0.01 + 0.002 * draw * (d + 1))):
                        rows.append(dict(dataset_id=f"d{d}", task_type="regression", model_id="linear",
                                         representation=rep, training_size=size, draw=draw,
                                         checkpoint_id="ck", split_id=f"split-{size}-{draw}",
                                         outer_fold=fold, value=base + delta))
    return data_efficiency.size_gains(pd.DataFrame(rows), cfg={"draws": [{"max_size": 4096, "draws": 3}]})


def test_headroom_share_pooling_uses_its_own_draw_variance():
    sizes = _sizes()
    assert not np.allclose(sizes.draw_variance, sizes.headroom_share_draw_variance)
    from src.evaluation.moderator_statistics import dersimonian_laird

    one = sizes[sizes.training_size.eq(128)]
    expected = dersimonian_laird(one.headroom_share.to_numpy(),
                                 (one.headroom_share_draw_variance / one.draws).to_numpy())
    pooled = data_efficiency.pooled_curve(sizes, outcome="headroom_share")
    assert pooled.set_index("training_size").loc[128, "pooled"] == pytest.approx(expected["pooled"])


def test_b4_curves_keep_their_fixed_dataset_cohorts():
    cohorts = {"main": {"datasets": ["d0", "d1", "d2", "d3"], "sizes": [128, 256]},
               "extended": {"datasets": ["d3"], "sizes": [128, 256, 4096]}}
    tagged = data_efficiency.cohort_gains(_sizes(), cohorts)
    pooled = data_efficiency.pooled_curve(tagged).set_index(["curve", "training_size"])
    assert pooled.loc[("main", 128), "datasets"] == 4 and ("main", 4096) not in pooled.index
    assert pooled.loc[("extended", 128), "datasets"] == 1
    assert pooled.loc[("extended", 4096), "datasets"] == 1


def test_paid_extension_adds_only_new_arms_to_the_open_benchmark():
    from src.evaluation.campaign import benchmark_plan
    from src.evaluation.design import decision_providers

    def identities(cohort):
        return {tuple(sorted(arm.items())) for arm in benchmark_plan(cohort=cohort)}

    # Include every setting in each identity, including sample draws and width overrides.
    open_arms = identities("open_weight")
    paid = benchmark_plan(cohort="paid")
    assert {arm["provider"] for arm in paid} == set(decision_providers("paid"))
    paid_arms = {tuple(sorted(arm.items())) for arm in paid}
    assert open_arms.isdisjoint(paid_arms)
    assert open_arms | paid_arms == identities("all")


# ----------------------------------------------------------- claim 8: reader identities


@pytest.mark.parametrize("module,study", [("benchmark_export", "B1"), ("group_c", "C1")])
def test_group_b_and_c_readers_reject_exports_of_another_configuration(tmp_path, module, study):
    import importlib

    from src.evaluation import study_export

    reader = importlib.import_module(f"src.evaluation.{module}")
    tables = reader.tables_for(study) if module == "benchmark_export" else reader.TABLES[study]
    target = study_export.write_export(tmp_path / "x.zip", {t: pd.DataFrame({"a": [1]}) for t in tables},
                                       schema=reader.SCHEMA, identity="older-configuration", origin="test")
    with pytest.raises(ValueError, match="configuration differs"):
        reader.read(study, target)
