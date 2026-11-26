"""A later provider must not change published membership, pins, or shared fit recipes."""

from collections import Counter
from copy import deepcopy

import pytest

from src.evaluation import benchmark_export, campaign, design
from src.utils.config import load_yaml
from src.utils.serialization import digest


@pytest.mark.parametrize("first_delivery", ["self_hosted", "hosted"])
def test_successive_releases_add_only_new_fits_and_preserve_edition_identities(monkeypatch, first_delivery):
    main = load_yaml("B1")
    catalog = load_yaml("extractors")
    earlier = benchmark_export.cohort_record("open_weight", main=main)
    first_jev = benchmark_export.cohort_record("with_jev", main=main)
    main["candidate_extractors"].extend(["added_a", "added_b"])
    first_pin = ({"model": "example/open-a", "revision": "a" * 40} if first_delivery == "self_hosted"
                 else {"model": "api-a-2026-01-01", "version": "2026-01-01"})
    catalog["candidates"]["added_a"] = {**first_pin, "delivery": first_delivery}
    catalog["candidates"]["added_b"] = {"model": "api-b-2026-02-01", "version": "2026-02-01", "delivery": "hosted"}
    main["decision_extensions"].update({
        "first_addition": {"parent": "open_weight", "add": ["added_a"]},
        "second_addition": {"parent": "first_addition", "add": ["added_b"]},
    })
    monkeypatch.setattr(design, "load_yaml", lambda name: catalog if name == "extractors"
                        else main if name == "B1" else load_yaml(name))
    def recipes(name):
        return Counter(digest(arm) for arm in campaign.benchmark_plan(main=main, cohort=name))

    initial = recipes("open_weight")
    first = recipes("first_addition")
    first_new = recipes("first_addition_added")
    second = recipes("second_addition")
    second_new = recipes("second_addition_added")
    assert len(initial) == 79_539 and len(first_new) == len(second_new) == 3_843
    assert first == initial + first_new and second == first + second_new
    assert not (initial & first_new or first & second_new)
    assert "jev" not in design.decision_providers("second_addition", main)
    assert design.decision_providers("second_addition_added", main) == ("added_b",)
    expected_paid = ("jev", "added_b") if first_delivery == "self_hosted" else ("jev", "added_a", "added_b")
    assert design.decision_providers("paid", main) == expected_paid
    first_record = benchmark_export.cohort_record("first_addition", main=main)["decision_model_versions"][-1]
    assert first_record["version"] == first_pin.get("revision", first_pin.get("version"))
    assert first_record["delivery"] == first_delivery
    assert benchmark_export.cohort_record("open_weight", main=main) == earlier
    assert benchmark_export.cohort_record("with_jev", main=main) == first_jev
    old_key = benchmark_export.cohort_key("second_addition", main=main)
    first_key = benchmark_export.cohort_key("first_addition", main=main)
    catalog["candidates"]["added_b"]["version"] = "2026-03-01"
    assert benchmark_export.cohort_key("second_addition", main=main) != old_key
    assert benchmark_export.cohort_key("first_addition", main=main) == first_key
    catalog["candidates"]["added_b"]["version"] = "api-b:latest"
    with pytest.raises(ValueError, match="exact registered model and version"):
        benchmark_export.cohort_record("second_addition", main=main)


@pytest.mark.parametrize("field", ["base_model_revision", "runtime_revision", "tokenizer_revision"])
def test_an_open_model_needs_fixed_component_revisions(monkeypatch, field):
    cfg = {"model": "example/open", "revision": "a" * 40, "delivery": "self_hosted", field: "main"}
    monkeypatch.setattr(design, "load_yaml", lambda _: {"candidates": {"extra": cfg}})
    with pytest.raises(ValueError, match=field):
        design.decision_versions(["extra"])
    cfg[field] = "b" * 40
    assert design.decision_versions(["extra"])[0][field] == cfg[field]
    if field == "base_model_revision":
        cfg["base_model"] = "example/base"
        del cfg[field]
        with pytest.raises(ValueError, match=field):
            design.decision_versions(["extra"])


@pytest.mark.parametrize("change", [
    {"parent": "with_jev", "add": ["jev"]},
    {"parent": "open_weight", "add": ["unregistered"]},
    {"parent": "all", "add": ["jev"]},
])
def test_an_extension_cannot_redefine_its_parent_or_add_unregistered_work(change):
    main = deepcopy(load_yaml("B1"))
    main["decision_extensions"]["later"] = change
    with pytest.raises(ValueError):
        design.decision_scopes(main)
