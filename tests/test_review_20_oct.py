import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from src.cluster.semantic_contracts import real_robustness_probes, robustness_summary
from src.jev.cache import ResponseCache
from src.jev.requests import IntendedRequest
from src.jev.self_hosted import cached_batches
from src.utils.serialization import canonical_json, digest


def test_bad_batch_is_isolated_and_valid_neighbour_reused(tmp_path):
    settings = dict(
        provider="intern",
        model="fixed",
        version="pin",
        api_mapping_version="intern-self-hosted-v1",
        request_configuration={},
    )
    good = IntendedRequest(
        canonical_json(
            {
                "state": {"text": {"t": "good"}},
                "question": {"kind": "binary_probability", "text": "Supported?"},
            }
        ),
        canonical_json(settings),
        canonical_json({"dataset_id": "fixture"}),
    )
    bad = replace(good, payload_json=good.payload_json.replace("good", "bad"))
    response = dict(
        provider="intern",
        model="fixed",
        version="pin",
        settings_sha256=digest(settings),
        answers={"semantic": {"type": "noul", "noul": 0.6}},
    )
    client = Mock()
    client.evaluate_many.side_effect = RuntimeError("Batch failed")

    def single(request):
        if request == bad:
            raise ValueError("Only this input fails")
        return response

    client.evaluate.side_effect = single
    cache = ResponseCache(tmp_path / "fixture.sqlite3", "fixture")
    assert list(cached_batches(cache, client, [good, bad], 2, continue_on_error=True)) == [
        "completed",
        "failed_recorded",
    ]
    assert cache.failure_count(good) == 0 and cache.failure_count(bad) == 1
    client.evaluate.reset_mock()
    assert list(
        cached_batches(cache, client, [good, bad], 2, continue_on_error=True, retry_failed=True)
    ) == ["cache_hit", "failed_recorded"]
    assert client.evaluate.call_args.args[0] == bad
    with pytest.raises(OSError, match="disk"):
        fresh = ResponseCache(tmp_path / "other.sqlite3", "fixture")
        fresh.put = Mock(side_effect=OSError("disk"))
        list(cached_batches(fresh, client, [good], 2, continue_on_error=True))


def test_real_diagnostics_use_frozen_inputs_and_align_reversed_options(monkeypatch):
    from src.jev import inputs as module

    records = []
    for kind in ("binary_probability", "choice", "ordered_choice"):
        question = {
            "kind": kind,
            "text": "For the dataset row described by the supplied text, which value of target is best supported?",
        }
        if kind == "choice":
            question.update(allowed_answers=["a", "b", "c"], answer_descriptions=["A", "B", "C"])
        payload = {"state": {"text": {"t": "ordinary ambiguous text"}}, "question": question}
        records.append(
            dict(input_set_id=kind, input_no=0, payload=payload, payload_sha256=digest(payload))
        )
    original = canonical_json(records)

    def iterator(path, *, input_set_id, **kwargs):
        yield next(r for r in records if r["input_set_id"] == input_set_id)

    monkeypatch.setattr(module, "iter_prepared_inputs", iterator)
    sample = [{"dataset_id": r["input_set_id"], "input_set_id": r["input_set_id"]} for r in records]
    probes = list(real_robustness_probes({}, sample))
    assert len(probes) == 13
    assert canonical_json(records) == original
    reverse = next(r for n, r, _ in probes if n.endswith("__option_order"))
    assert json.loads(reverse.payload_json)["question"]["answer_descriptions"] == ["C", "B", "A"]
    order = json.loads(reverse.provenance_json)["canonical_probability_order"]
    summary = robustness_summary(
        [
            {"probe": "x__base", "features": {"p_a": 0.8, "p_b": 0.1, "p_c": 0.1}},
            {
                "probe": "x__option_order",
                "features": {"p_a": 0.1, "p_b": 0.1, "p_c": 0.8},
                "canonical_probability_order": order,
            },
        ]
    )
    assert summary["comparisons"][1]["total_variation_from_base"] == 0
    with pytest.raises(ValueError, match="no observations"):
        robustness_summary([])
    with pytest.raises(ValueError, match="Invalid diagnostic"):
        robustness_summary([{"probe": "x__base", "features": {"p_a": float("nan")}}])


def test_library_check_refuses_uncommitted_folder_and_pin_drift(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from src.utils import check_libraries

    def git(command, **kwargs):
        if command[1] == "ls-tree":
            name = command[-1]
            return SimpleNamespace(
                stdout="" if name == "dm-library" else f"160000 commit pinned\t{name}", returncode=0
            )
        return SimpleNamespace(
            stdout="pinned" if "tfm-library" in command[2] else "different", returncode=0
        )

    monkeypatch.setattr(check_libraries.subprocess, "run", git)
    result = check_libraries.check(tmp_path)
    assert result["status"] == "failed"
    assert [r["status"] for r in result["libraries"]] == [
        "passed",
        "missing_committed_gitlink",
        "missing_or_different_checkout",
    ]
