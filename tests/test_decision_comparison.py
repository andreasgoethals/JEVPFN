"""Numerical and provenance checks; synthetic values never enter project feature stores."""

import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.evaluation.decision_comparison import (
    Comparison,
    GroupFeatures,
    analyze_group,
    compare_saved_features,
    dataset_balanced,
    js_divergence,
    normalize_group,
    payload_hashes,
    save_comparison,
)
from src.jev.cache import ResponseCache
from src.jev.features import FeatureBlock, save_block
from src.jev.requests import IntendedRequest
from src.utils import paths
from src.utils.config import load_config
from src.utils.serialization import canonical_json, digest
from src.visualize import decision_comparison as plots
from src.visualize import style


def options(providers=("jev", "liquid", "strands")):
    result = copy.deepcopy(load_config("feature_comparison")["comparison"])
    result["providers"] = list(providers)
    result["scatter_rows_per_group"] = 4
    return result


def group(provider, p, *, task="binary_classification", hashes=None):
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = np.column_stack([1 - p, p])
    labels = () if task == "regression" else tuple(range(p.shape[1]))
    hashes = hashes or [f"input{i}" for i in range(len(p))]
    return GroupFeatures(
        provider,
        np.array([f"csv:{i}" for i in range(len(p))]),
        p,
        np.array(hashes, dtype=object),
        labels,
        task,
        "same-recipe",
    )


CONTEXT = {
    "dataset_id": "fixture",
    "dataset": "Fixture",
    "representation": "joint",
    "group": "all_text",
    "text_column": "All text",
}


def test_agreement_counts_and_duplicate_weighting_are_exact():
    hashes = ["repeat", "repeat", "repeat", "other"]
    a, b = (
        group("jev", [0.9, 0.9, 0.9, 0.9], hashes=hashes),
        group("liquid", [0.9, 0.9, 0.9, 0.1], hashes=hashes),
    )
    result = Comparison()
    analyze_group([a, b], CONTEXT, options(("jev", "liquid")), result)
    pairs = result.table("pairs").set_index("weighting")
    assert pairs.loc["rows", "agree_count"] == 3
    assert pairs.loc["rows", "agreement"] == 0.75
    assert pairs.loc["unique_inputs", "agree_count"] == 1
    assert pairs.loc["unique_inputs", "agreement"] == 0.5
    assert pairs.loc["unique_inputs", "total_variation"] == pytest.approx(0.4)
    # With two models both are equally far from one another; do not invent a winner.
    assert result.table("consensus").farthest_share.eq(0.5).all()


def test_regression_uses_probability_expectation_and_direction_not_class_argmax():
    a = np.eye(9)[[4, 6, 1]]
    b = np.eye(9)[[4, 7, 1]]
    c = np.eye(9)[[5, 6, 3]]
    result = Comparison()
    analyze_group(
        [group(n, p, task="regression") for n, p in zip(options()["providers"], [a, b, c])],
        CONTEXT,
        options(),
        result,
    )
    pairs = result.table("pairs")
    pair = pairs[
        (pairs.model_a == "jev") & (pairs.model_b == "liquid") & (pairs.weighting == "rows")
    ].iloc[0]
    assert pair.agreement == 1  # same signs, but unequal direction magnitudes
    assert pair.scalar_mae == pytest.approx(1 / 3)
    assert pair.within_tolerance_count == 2
    votes = result.table("votes")
    assert votes[votes.weighting.eq("rows")]["count"].sum() == 3
    assert np.isfinite(result.table("consensus").distance_to_others).all()


def test_multiclass_vectors_and_ties_and_js():
    assert js_divergence(np.array([[1.0, 0]]), np.array([[0.0, 1]]))[0] == 1
    p = np.array([[0.5, 0.5, 0], [0.1, 0.2, 0.7]])
    result = Comparison()
    analyze_group(
        [group(n, p, task="multiclass_classification") for n in options()["providers"]],
        CONTEXT,
        options(),
        result,
    )
    assert result.table("pairs").agreement.eq(1).all()
    assert result.table("pairs").js_divergence.eq(0).all()
    assert result.table("consensus").tied_top_probability_count.eq(1).all()
    assert result.table("consensus").farthest_share.eq(1 / 3).all()


def test_row_order_is_aligned_but_input_mismatch_and_missing_policy_fail():
    a, b = group("jev", [0.1, 0.9]), group("liquid", [0.1, 0.9])
    b.row_ids = b.row_ids[::-1]
    b.probabilities = b.probabilities[::-1]
    b.input_hashes = b.input_hashes[::-1]
    result = Comparison()
    analyze_group([a, b], CONTEXT, options(), result)
    assert result.table("pairs").agreement.eq(1).all()
    assert result.table("consensus").empty  # configured third model missing
    b.input_hashes[0] = "truncated-for-only-this-provider"
    with pytest.raises(ValueError, match="different text/question"):
        analyze_group([a, b], CONTEXT, options(), Comparison())
    b.probabilities[:] = np.nan
    with pytest.raises(ValueError, match="missing"):
        analyze_group([a, b], CONTEXT, options(), Comparison())


def test_consensus_excludes_self_and_equal_dataset_weighting():
    result = Comparison()
    analyze_group(
        [group("jev", [0.5]), group("liquid", [0.5]), group("strands", [1.0])],
        CONTEXT,
        options(),
        result,
    )
    rows = result.table("consensus").query("weighting == 'rows'").set_index("provider")
    assert rows.loc["strands", "distance_to_others"] == 0.5
    assert rows.loc["jev", "distance_to_others"] == 0.25
    assert rows.loc["strands", "farthest_share"] == 1
    table = pd.DataFrame(
        {"dataset_id": ["big", "big", "small"], "provider": ["jev"] * 3, "score": [0, 1, 1]}
    )
    assert dataset_balanced(table, ["provider"], ["score"]).iloc[0].score == 0.75


def block_fixture():
    identity = {
        "dataset_id": "fixture",
        "data_identity": "frozen",
        "task_schema_sha256": "schema",
        "question": {"positive_class": 1},
        "provider": "jev",
        "model": "fixture-model",
        "version": "1",
        "representation": "joint",
    }
    column = "jev__joint__text_only__all_text__p_positive"
    values = pd.DataFrame({"row_id": ["csv:0", "csv:1", "csv:2"], column: [0.7, 0.7, np.nan]})
    refs = pd.DataFrame(
        {
            "row_id": values.row_id,
            "group": ["all_text"] * 3,
            "request_key": ["key", "key", None],
            "status": ["verified_response"] * 2 + ["missing_text"],
        }
    )
    return FeatureBlock(
        values,
        refs,
        {
            "identity": identity,
            "class_labels": [0, 1],
            "text_columns": ["text"],
            "feature_columns": [column],
            "artifact_id": digest(identity),
        },
    )


def test_normalization_validates_probabilities_and_reuse():
    block = block_fixture()
    normalized = normalize_group(block, "all_text", {"key": "same"})
    assert normalized.probabilities[0].tolist() == pytest.approx([0.3, 0.7])
    assert np.isnan(normalized.probabilities[2]).all()
    block.values.iloc[1, 1] = 0.9
    with pytest.raises(ValueError, match="inconsistent"):
        normalize_group(block, "all_text", {"key": "same"})
    block.values.iloc[1, 1] = 1.1
    with pytest.raises(ValueError, match="Invalid"):
        normalize_group(block, "all_text", {"key": "same"})


def test_semantic_hashes_come_from_verified_raw_requests(tmp_path):
    path = tmp_path / "fixture.sqlite3"
    block = block_fixture()
    request = IntendedRequest(
        canonical_json({"state": {"text": {"text": "hi"}}}),
        canonical_json({"provider": "jev", "model": "fixture-model", "version": "1"}),
        canonical_json({"dataset_id": "fixture"}),
    )
    cache = ResponseCache(path, "fixture")
    cache.put(request, {"synthetic_test_only": True}, kind="verified_response")
    block.requests.request_key = block.requests.request_key.replace("key", request.cache_key)
    assert payload_hashes(path, block) == {
        request.cache_key: digest({"state": {"text": {"text": "hi"}}})
    }
    block.manifest["identity"]["version"] = "wrong"
    with pytest.raises(ValueError, match="settings"):
        payload_hashes(path, block)


def test_empty_real_cache_is_read_only_and_report_has_no_fake_results(tmp_path, monkeypatch):
    cfg = load_config("feature_comparison")
    monkeypatch.setattr(
        paths, "dataset_cache_path", lambda dataset: tmp_path / "protected" / f"{dataset}.sqlite3"
    )
    result = compare_saved_features(cfg, progress=None)
    assert len(result.table("inventory")) == 18 * 6 * 2
    assert result.table("inventory").status.eq("not_created").all()
    assert result.table("pairs").empty
    assert not (tmp_path / "protected").exists()
    monkeypatch.setattr(paths, "notebook_artifacts", lambda name, **kw: tmp_path / "results")
    folder = save_comparison(result, cfg, "test")
    assert (folder / "manifest.json").is_file()


def test_real_sqlite_contract_and_ambiguous_versions(isolated_output, monkeypatch):
    import src.evaluation.decision_comparison as comparison

    cfg = load_config("feature_comparison")
    cfg["dataset_selection"]["exclude"] = []
    cfg["comparison"] = options(("jev", "liquid"))
    cfg["comparison"]["representations"] = ["joint"]
    monkeypatch.setattr(
        comparison, "load_catalog", lambda _: {"datasets": [{"id": "fixture", "name": "Fixture"}]}
    )
    path = paths.dataset_cache_path("fixture")
    cache = ResponseCache(path, "fixture")
    artifact_ids = {}
    for provider, version in [("jev", "1"), ("liquid", "1")]:
        block = block_fixture()
        identity = block.manifest["identity"]
        identity.update(provider=provider, version=version)
        request = IntendedRequest(
            canonical_json({"state": {"text": {"text": "same"}}}),
            canonical_json({"provider": provider, "model": identity["model"], "version": version}),
            canonical_json({"dataset_id": "fixture"}),
        )
        cache.put(request, {"synthetic_test_only": True}, kind="verified_response")
        block.requests.request_key = block.requests.request_key.replace("key", request.cache_key)
        block.manifest.update(
            artifact_id=digest(identity),
            status="complete",
            namespace="verified_response",
            row_count=len(block.values),
            row_ids_sha256=digest(block.values.row_id.tolist()),
            response_store=str(path),
        )
        save_block(block, path)
        artifact_ids[provider] = block.manifest["artifact_id"]
    result = compare_saved_features(cfg, progress=None)
    assert result.table("pairs").agreement.eq(1).all()
    assert result.table("coverage").missing_text_rows.eq(1).all()
    # A changed code version is a different artifact; never silently use the newest.
    block.manifest["identity"]["code_sha256"] = "another-code-version"
    block.manifest["artifact_id"] = digest(block.manifest["identity"])
    save_block(block, path)
    with pytest.raises(ValueError, match="Multiple versions"):
        compare_saved_features(cfg, progress=None)
    cfg["comparison"]["artifacts"] = {"fixture": {"liquid": {"joint": artifact_ids["liquid"]}}}
    assert not compare_saved_features(cfg, progress=None).table("pairs").empty


def test_missing_or_stale_context_screen_is_explicit(tmp_path, monkeypatch):
    import json

    from src.jev.context import saved_screen_tables

    monkeypatch.setattr(paths, "manifests_dir", lambda _: tmp_path)
    summary, scenarios, origins = saved_screen_tables()
    assert summary["status"] == "not_screened" and scenarios.empty and origins.empty
    (tmp_path / "context_screen.json").write_text(json.dumps({"datasets": []}), encoding="utf-8")
    summary, scenarios, origins = saved_screen_tables()
    assert summary["status"] == "historical_inputs_changed" and scenarios.empty and origins.empty


@pytest.mark.parametrize(
    "task", ["binary_classification", "multiclass_classification", "regression"]
)
def test_every_plot_path_accepts_realistic_synthetic_results(task):
    style.apply()
    result = Comparison()
    rng = np.random.default_rng(3)
    k = 9 if task == "regression" else 3 if task == "multiclass_classification" else 2
    for mode in ["per_column", "joint"]:
        groups = [
            group(name, rng.dirichlet(np.ones(k), 12), task=task) for name in options()["providers"]
        ]
        analyze_group(groups, {**CONTEXT, "representation": mode}, options(), result)
        for name in options()["providers"]:
            result.add(
                "inventory",
                {**CONTEXT, "representation": mode, "provider": name, "status": "available"},
            )
    count = 0
    for function in [
        plots.coverage_figures,
        plots.pair_figures,
        plots.consensus_figures,
        plots.distribution_figures,
        plots.example_figures,
    ]:
        for fig, name, caption in function(result, options()):
            fig.canvas.draw()
            assert name and caption
            plt.close(fig)
            count += 1
    assert count >= 15
