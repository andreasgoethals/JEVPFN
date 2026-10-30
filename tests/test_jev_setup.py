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
    assert cfg["live_extraction"]["requests_per_second"] == 0.2
    assert cfg["live_extraction"]["verified_account_requests_per_second"] == 1
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


@pytest.mark.parametrize("fails", [False, True])
def test_interactive_account_key_is_never_printed_or_left_in_environment(
    monkeypatch, capsys, fails
):
    import os

    from src.jev import setup_pilot

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["setup_pilot", "--check-account"])
    monkeypatch.setattr(setup_pilot.getpass, "getpass", lambda _: "synthetic-hidden-key")

    def check():
        assert os.environ["TYPESAFE_API_KEY"] == "synthetic-hidden-key"
        if fails:
            raise ValueError("synthetic-hidden-key must never be printed")
        return {"authenticated": True}

    monkeypatch.setattr(setup_pilot, "account_check", check)
    if fails:
        with pytest.raises(SystemExit):
            setup_pilot.main()
    else:
        setup_pilot.main()
    assert "TYPESAFE_API_KEY" not in os.environ
    assert "synthetic-hidden-key" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "text,expected",
    [
        ("$0.042", 0.042),
        ("USD 0.042", 0.042),
        ("0.042 USD", 0.042),
        ("100,000", 100000),
        (".042", 0.042),
    ],
)
def test_setup_accepts_console_numeric_formats(text, expected):
    from src.jev.setup_pilot import positive_number

    assert positive_number(text, currency=True) == expected


@pytest.mark.parametrize("text", ["", "NaN", "inf", "-5", "0", "0,042", "1,00", "80 RPS", "1e999"])
def test_setup_rejects_ambiguous_or_nonpositive_numbers(text):
    from src.jev.setup_pilot import positive_number

    with pytest.raises(ValueError):
        positive_number(text, currency=True)


def test_prompt_retries_before_accepting_a_value(monkeypatch, capsys):
    from src.jev.setup_pilot import prompt_number

    answers = iter(["wrong", "$0.042"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert prompt_number("Price", currency=True) == 0.042
    assert "decimal point" in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert prompt_number("Tokens", integer=True, default=100000) == 100000


def test_setup_integer_bounds_cannot_be_fractional():
    from src.jev.setup_pilot import positive_number

    assert positive_number("64,000", integer=True) == 64000
    with pytest.raises(ValueError):
        positive_number("64000.5", integer=True)


def test_configure_uses_public_rejection_contract_and_recorded_account_limits(
    tmp_path, monkeypatch
):
    import yaml

    from src.jev import setup_pilot
    from src.jev.hosted import ReviewedContextEnvelope

    target = tmp_path / "jev.local.yaml"
    monkeypatch.setattr("sys.argv", ["setup_pilot", "--configure"])
    original_path = setup_pilot.paths.config_path
    monkeypatch.setattr(
        setup_pilot.paths,
        "config_path",
        lambda name: target if name == "feature_creation/jev.local" else original_path(name),
    )
    answers = iter(["$0.042", "1.1", "synthetic payment conversion", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    setup_pilot.main()
    overlay = yaml.safe_load(target.read_text())
    cfg = _merge(load_yaml("feature_creation"), overlay)
    live = cfg["live_extraction"]
    assert live["verified_account_requests_per_second"] == 80
    assert live["verified_account_input_tokens_per_second"] == 100000
    assert live["requests_per_second"] == 0.2
    assert live["full_feature_creation_enabled"] is False
    assert live["context_envelope"]["mode"] == "provider_enforced"
    validator = ReviewedContextEnvelope(cfg)
    monkeypatch.setattr(
        "src.jev.hosted.api_preview", lambda _: {"model": cfg["jev"]["model"], "state": "hello"}
    )
    validator(None)
    # Small HTTP body is not falsely reported as a locally counted token guarantee.
    assert live["context_envelope"]["per_question_context_tokens"] == 32000
    monkeypatch.setattr("src.jev.hosted.canonical_json", lambda _: "x" * (32 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="body-size"):
        validator(None)
    live["context_envelope"]["max_request_utf8_bytes"] = 64000
    with pytest.raises(ValueError, match="documented limits"):
        ReviewedContextEnvelope(cfg)


def test_setup_interruption_writes_no_partial_configuration(tmp_path, monkeypatch):
    from src.jev import setup_pilot

    target = tmp_path / "jev.local.yaml"
    monkeypatch.setattr("sys.argv", ["setup_pilot", "--configure"])
    original_path = setup_pilot.paths.config_path
    monkeypatch.setattr(
        setup_pilot.paths,
        "config_path",
        lambda name: target if name == "feature_creation/jev.local" else original_path(name),
    )

    def interrupt(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr("builtins.input", interrupt)
    with pytest.raises(KeyboardInterrupt):
        setup_pilot.main()
    assert not target.exists()
