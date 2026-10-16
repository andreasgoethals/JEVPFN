from types import SimpleNamespace

import pytest

from src.jev.extraction import campaign_limits
from src.jev.setup_pilot import account_check, pilot_overlay
from src.utils.config import _merge, load_yaml


def configured():
    base = load_yaml("feature_creation")
    overlay = pilot_overlay(
        base,
        usd_per_eur=1.1,
        exchange_evidence="synthetic exchange evidence",
        rps=1,
        tps=1000,
        max_bytes=10000,
        max_tokens=32000,
        context_evidence="synthetic provider contract",
        account_evidence="synthetic rates",
    )
    return _merge(base, overlay)


def test_pilot_budget_is_in_euros_and_never_enables_full_extraction():
    cfg = configured()
    assert cfg["live_extraction"]["campaign_spending_limit_usd"] == 5.5
    assert cfg["live_extraction"]["full_feature_creation_enabled"] is False
    assert cfg["live_extraction"]["pilot_accepted"] is False
    limits = campaign_limits(cfg)
    assert len(limits) == 18
    assert sum(v.dataset_budget_usd for v in limits.values()) <= 5.5
    cfg["live_extraction"]["campaign_spending_limit_usd"] = 5.51
    with pytest.raises(ValueError, match="approved euro budget"):
        campaign_limits(cfg)


def test_currency_must_be_verified_even_with_a_usd_ceiling():
    cfg = configured()
    cfg["live_extraction"]["usd_per_eur"] = None
    with pytest.raises(ValueError, match="EUR-to-USD"):
        campaign_limits(cfg)


def test_account_check_never_dispatches_an_inference(isolated_output, monkeypatch):
    pytest.importorskip("typesafe_sdk")  # Hosted Jev runs locally; the VSC CPU extra omits its SDK.
    monkeypatch.setattr("src.jev.setup_pilot.paths.on_vsc", lambda: False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-test-key")
    calls = []

    class Client:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.models = SimpleNamespace(
                list=lambda: SimpleNamespace(models=[SimpleNamespace(name="jev-latest")])
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    report = account_check(Client)
    assert report["authenticated"] and not report["inference_performed"]
    assert not report["pinned_version_access_verified"]  # Alias listing is not a real model call.
    assert "synthetic-test-key" not in str(report)
    assert calls[0]["base_url"] == "https://api.typesafe.ai"
