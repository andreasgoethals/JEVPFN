"""Sourced decision-model properties and their optional use as Exp 8 model-level covariates."""

import copy
import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation import model_properties as mp
from src.evaluation import moderators as m
from src.utils.config import load_yaml


@pytest.fixture(scope="module")
def raw():
    return load_yaml(mp.CONFIG)


@pytest.fixture(scope="module")
def extractors():
    return load_yaml("feature_creation/extractors")


def test_all_registered_providers_present_in_extractor_order(raw, extractors):
    mp.validate(raw, extractors)
    assert list(raw["providers"]) == list(extractors["candidates"])


def test_missing_values_are_never_guessed(raw):
    for provider, row in raw["providers"].items():
        for name, cell in row.items():
            if cell["value"] is None:
                assert cell["verified"] is False and cell["source"] is None, (provider, name)
            else:
                assert cell["source"], (provider, name)


def test_revision_and_context_match_extractors(raw, extractors):
    for provider, row in raw["providers"].items():
        candidate = extractors["candidates"][provider]
        assert row["context_tokens"]["value"] == candidate["context_tokens"]
        assert row["revision"]["value"] in {candidate.get("revision"), candidate.get("model")}


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r["providers"].pop("sieve"), "match extractors"),
    (lambda r: r["providers"]["kev"]["context_tokens"].update(value=1234), "differs"),
    (lambda r: r["providers"]["kev"]["architecture_family"].update(value="transformer"), "architecture"),
    (lambda r: r["providers"]["kev"]["readout_type"].update(value="magic"), "readout"),
    (lambda r: r["providers"]["kev"]["hidden_size"].update(value=None, verified=True), "unverified"),
    (lambda r: r["providers"]["kev"]["license"].update(source=None), "source"),
    (lambda r: r["providers"]["kev"]["parameters_billion"].update(value="27B"), "positive number"),
    (lambda r: r.update(schema="other"), "schema"),
])
def test_validation_rejects_inconsistent_tables(raw, extractors, mutate, message):
    broken = copy.deepcopy(raw)
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        mp.validate(broken, extractors)


def test_covariates_use_verified_values_only(raw):
    table = mp.covariates(raw)
    assert list(table.columns) == ["provider", *mp.COVARIATES]
    for provider, row in raw["providers"].items():
        params = row["parameters_billion"]
        value = table.set_index("provider").loc[provider, "parameters_billion"]
        if params["verified"] and params["value"] is not None:
            assert value == pytest.approx(params["value"])
            assert table.set_index("provider").loc[provider, "log10_parameters"] == pytest.approx(
                math.log10(params["value"] * 1e9))
        else:
            assert np.isnan(value)
    shown = mp.display_table(raw)
    assert len(shown) == 17 and "not verified" in shown.to_string()


def _gains():
    providers = ["a", "b", "c", "d", "e", "f"]
    rows = []
    for rank, provider in enumerate(providers):
        for dataset in ("d1", "d2", "d3"):
            rows.append({"model_id": "catboost", "checkpoint_id": "c", "family": "decision",
                         "aggregation": "per_decision_model", "feature_set": "non_text_plus_joint",
                         "representation": f"decision/joint/{provider}", "provider": provider,
                         "dataset_id": dataset, "gain": 0.1 * rank, "baseline": 0.7})
    # A dataset seen by only one provider never enters the comparison.
    rows.append({**rows[0], "dataset_id": "d4", "gain": 100.0})
    return pd.DataFrame(rows)


def test_model_level_associations_relate_size_and_architecture_to_gain():
    covariates = pd.DataFrame({
        "provider": ["a", "b", "c", "d", "e", "f"],
        "log10_parameters": np.log10([1e9, 2e9, 4e9, 9e9, 27e9, 31e9]),
        "architecture_family": ["decoder_lm_head"] * 3 + ["bidirectional_encoder"] * 3,
        "readout_type": ["pointer_head", None, "pointer_head", "api", "api", "api"],
    })
    table = m.model_level_associations(_gains(), covariates, n_boot=100, seed=1)
    size = table[table.covariate.eq("log10_parameters")].iloc[0]
    assert size.value == pytest.approx(1.0) and size.status == "descriptive" and size.n_providers == 6
    family = table[table.covariate.eq("architecture_family")].set_index("level").value
    assert family["decoder_lm_head"] == pytest.approx(0.1)
    assert family["bidirectional_encoder"] == pytest.approx(0.4)
    readout = table[table.covariate.eq("readout_type")].set_index("level")
    assert readout.loc["pointer_head", "n_providers"] == 2
    few = m.model_level_associations(_gains(), covariates.iloc[:3], n_boot=0)
    assert few.set_index("covariate").loc["log10_parameters", "status"] == "too_few_providers"
    assert m.model_level_associations(_gains().iloc[0:0], covariates).empty


def test_model_covariates_are_optional():
    table, status = m.model_covariates()
    assert status == "available" and len(table) == 17
    empty, pending = m.model_covariates({"schema": "missing"})
    assert empty.empty and pending.startswith("pending")
