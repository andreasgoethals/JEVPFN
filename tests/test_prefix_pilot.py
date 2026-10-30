import copy
from types import SimpleNamespace

import pytest

from src.cluster.gpu_contracts import synthetic_datasets
from src.jev import hosted, prefix_pilot
from src.jev.features import assemble_features, feature_inventory, load_block
from src.jev.inputs import prepare_dataset_inputs
from src.jev.question_bank import build_question_bank, save_question_bank
from src.utils import paths
from src.utils.config import load_yaml


def test_large_selection_batches_without_missing_requests(monkeypatch, tmp_path):
    calls = []

    def reader(path, *, input_set_id, input_numbers):
        calls.append(input_numbers)
        return iter(input_numbers)

    monkeypatch.setattr(prefix_pilot, "iter_prepared_inputs", reader)
    assert list(prefix_pilot.selected_records(tmp_path, "set", set(range(2151)))) == list(range(2151))
    assert list(map(len, calls)) == [1000, 1000, 151]


@pytest.mark.parametrize("rows", [0, -1, 101, True, 1.5])
def test_prefix_cannot_become_full_extraction(tmp_path, rows):
    with pytest.raises(ValueError, match="100"):
        prefix_pilot.prefix_numbers(tmp_path / "absent", "set", rows)


def test_prefix_materialization_missingness_resume_and_full_data_rejection(isolated_output, monkeypatch):
    dataset = next(synthetic_datasets())
    # Keep the original task vocabulary, while adding a genuinely empty text row.
    dataset.frame.loc[0, list(dataset.metadata.text_columns)] = ""
    bank = build_question_bank([dataset.metadata], load_yaml("question_design"))
    save_question_bank(bank)
    prepared = prepare_dataset_inputs(dataset, bank, variant="accepted")
    identifier = dataset.metadata.dataset_id
    monkeypatch.setattr(paths, "on_vsc", lambda: False)
    monkeypatch.setattr(hosted, "selected_inputs", lambda: [
        {"dataset_id": identifier, "input_set_id": prepared["input_set_id"]}
    ])
    monkeypatch.setattr("src.data.loaders.load_catalog", lambda: {"datasets": [{"id": identifier}]})
    monkeypatch.setattr("src.data.loaders.load_dataset", lambda *a, **kw: dataset)
    monkeypatch.setattr(hosted, "RateGate", lambda _: SimpleNamespace(acquire=lambda: None))
    cfg = load_yaml("feature_creation")
    cfg["live_extraction"].update(
        enabled=True, campaign_spending_limit_usd=1,
        usd_per_eur=1, exchange_rate_evidence="synthetic",
        dataset_budget_allocations_usd={identifier: 1}, workers=1,
        requests_per_second=1, verified_account_requests_per_second=80,
        input_tokens_per_second=100000, max_billed_input_tokens_per_attempt=64000,
        context_envelope={"verified": True, "model": cfg["jev"]["model"],
            "mode": "provider_enforced", "max_request_utf8_bytes": 32 * 1024 * 1024,
            "max_billed_input_tokens": 64000, "per_question_context_tokens": 32000,
            "evidence": "synthetic"},
    )
    calls = []

    def evaluate(request):
        calls.append(request.cache_key)
        return {"model": cfg["jev"]["model"], "answers": {"semantic": {"type": "noul", "noul": .4}},
                "usage": {"input_tokens": 10}}

    policy = load_yaml("experiment_0/jev_prefix_pilot")
    first = hosted.run(cfg, execute=True, prefix_policy=policy,
                       transport_factory=lambda *a, **kw: SimpleNamespace(evaluate=evaluate))
    assert first["status"] == "completed_prefix_pilot_requires_review"
    count = len(calls)
    second = hosted.run(cfg, execute=True, prefix_policy=policy,
                        transport_factory=lambda *a, **kw: SimpleNamespace(evaluate=evaluate))
    assert len(calls) == count
    assert set(second["datasets"][0]["counts"]) == {"cache_hit"}
    for item in first["datasets"][0]["features"]:
        block = load_block(paths.dataset_cache_path(identifier), item["artifact_id"])
        assert block.manifest["purpose"] == "validation_only"
        assert block.values.iloc[0].drop("row_id").isna().all()
        with pytest.raises(ValueError):
            assemble_features(dataset, [block])
    assert len(calls) == len(set(calls))
    assert feature_inventory(paths.dataset_cache_path(identifier)).empty
    changed = copy.deepcopy(cfg)
    changed["live_extraction"]["workers"] = 2
    with pytest.raises(ValueError, match="one worker"):
        hosted.run(changed, execute=True, prefix_policy=policy)


def test_budget_stop_happens_before_dispatch_and_cached_responses_remain_readable(monkeypatch):
    requests = [object(), object()]
    cache = SimpleNamespace(path="unused", get=lambda req, **kw: "paid" if req is requests[0] else None)
    monkeypatch.setattr(prefix_pilot, "ledger_total", lambda _: .998)
    limits = SimpleNamespace(max_billed_input_tokens=64000, input_price_per_million=.042)
    stopped = []
    assert list(hosted.budgeted_requests(iter(requests), cache, 0, 1, limits, stopped)) == requests[:1]
    assert stopped == [True]
