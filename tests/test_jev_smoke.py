from types import SimpleNamespace

import pytest

from src.jev import smoke
from src.utils import paths


class Client:
    def __init__(self, fail=False, wrong_model=False):
        self.calls, self.closed, self.fail, self.wrong_model = [], False, fail, wrong_model

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        if self.fail:
            raise TimeoutError("No automatic retry may follow this uncertain outcome")
        question = questions["semantic"]
        kind = question["type"]
        if kind == "noul":
            answer = {"type": kind, "noul": 0.9}
        elif kind == "choice":
            answer = {
                "type": kind,
                "choice": "c001",
                "probabilities": {"c000": 0.05, "c001": 0.9, "c002": 0.05},
                "confidence": 0.9,
            }
        else:
            answer = {
                "type": kind,
                "score": 8,
                "legend": dict(zip(map(str, range(9)), question["criteria"])),
                "probabilities": {str(i): float(i == 8) for i in range(9)},
                "confidence": 1,
            }
        raw = {
            "model": "jev-latest" if self.wrong_model else smoke.MODEL,
            "answers": {"semantic": answer},
            "usage": {"input_tokens": 120},
        }
        return SimpleNamespace(raw_http_response=SimpleNamespace(json=lambda: raw))

    def close(self):
        self.closed = True


@pytest.fixture
def local(isolated_output, monkeypatch):
    monkeypatch.setattr(paths, "on_vsc", lambda: False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-key")
    return isolated_output


def test_three_real_contract_shapes_resume_without_another_call(local):
    client, delays = Client(), []
    report = smoke.run(execute=True, client_factory=lambda: client, sleep=delays.append)
    assert len(client.calls) == 3 and client.closed and delays == [5, 5]
    assert report["pinned_version_access_verified"]
    assert report["total_cost_usd"] == pytest.approx(360 * 0.042 / 1e6)
    assert report["maximum_new_call_cost_usd"] < 0.01
    assert not report["account_limits_verified"]
    smoke.run(execute=True, client_factory=lambda: pytest.fail("Must reuse saved responses"))
    assert len(client.calls) == 3


def test_model_upgrade_preserves_legacy_paid_ledger_and_cleanup_can_find_new_ledger(local):
    import json

    directory = paths.jev_cache_path("jev_cache/_control/jev_smoke")
    directory.mkdir(parents=True, exist_ok=True)
    legacy = {"model": "jev-1.13.0", "request_identity": "historical", "attempts": []}
    old_state = directory / "state.json"
    old_state.write_text(json.dumps(legacy), encoding="utf-8")
    before = old_state.read_bytes()
    report = smoke.run(execute=True, client_factory=Client, sleep=lambda _: None)
    assert report["model"] == "jev-1.14.0"
    smoke.preserve_smoke_state()
    assert old_state.read_bytes() == before
    assert (directory / "models/jev-1.14.0/state.json").is_file()


def test_explicit_frozen_round_preserves_old_ledger_and_reuses_its_own_responses(local):
    original = Client()
    smoke.run(execute=True, client_factory=lambda: original, sleep=lambda _: None)
    round_client = Client()
    result = smoke.run(execute=True, round_id="a" * 64, client_factory=lambda: round_client, sleep=lambda _: None)
    assert len(round_client.calls) == 3 and result["round_id"] == "a" * 64
    for round_id in (None, "a" * 64):
        smoke.run(execute=True, round_id=round_id, client_factory=lambda: pytest.fail("Saved responses must survive"))
    with pytest.raises(ValueError, match="SHA-256"):
        smoke.run(execute=True, round_id="../outside")


def test_robustness_has_separate_durable_budget_and_reuses_after_output_reset(local):
    client = Client()
    report = smoke.run(
        execute=True, robustness=True, client_factory=lambda: client, sleep=lambda _: None
    )
    assert len(client.calls) == 15
    assert report["maximum_new_call_cost_usd"] == pytest.approx(0.04032)
    assert report["prompt_robustness"]["structurally_valid"]
    assert report["status"] == "jev_robustness_requires_review"
    assert report["prompt_robustness"]["comparisons"][1]["total_variation_from_base"] == 0
    smoke.run(
        execute=True,
        robustness=True,
        client_factory=lambda: pytest.fail("No repayment on resume"),
        sleep=lambda _: None,
    )


@pytest.mark.parametrize("wrong_model", [False, True])
def test_timeout_or_version_drift_stops_without_retry(local, wrong_model):
    client = Client(fail=not wrong_model, wrong_model=wrong_model)
    with pytest.raises((TimeoutError, ValueError)):
        smoke.run(execute=True, client_factory=lambda: client, sleep=lambda _: None)
    assert len(client.calls) == 1 and client.closed
    with pytest.raises((RuntimeError, ValueError)):
        smoke.run(execute=True, client_factory=lambda: pytest.fail("No automatic repayment"))


def test_no_implicit_paid_execution_or_vsc_calls(local, monkeypatch):
    with pytest.raises(RuntimeError):
        smoke.run(client_factory=lambda: pytest.fail("Must not connect"))
    monkeypatch.setattr(paths, "on_vsc", lambda: True)
    with pytest.raises(RuntimeError):
        smoke.run(execute=True, client_factory=lambda: pytest.fail("Must not connect"))


def test_client_setup_failure_has_no_attempt_and_can_resume_without_stale_error(local):
    import json

    def constructor_failure():
        raise ValueError("Synthetic client initialization failure; no request")

    with pytest.raises(ValueError):
        smoke.run(execute=True, client_factory=constructor_failure)
    path = paths.manifests_dir("experiment_0") / "before_extraction/jev_smoke.json"
    assert json.loads(path.read_text())["attempts"] == []
    report = smoke.run(execute=True, client_factory=Client, sleep=lambda _: None)
    assert report["status"] == "jev_smoke_passed_requires_dataset_pilot"
    assert "error_type" not in report and "http_status" not in report


def test_explicit_authentication_retry_keeps_history_and_reservation(local):
    class TypeSafeAuthenticationError(Exception):
        status = 401

    class Rejected(Client):
        def system_one(self, state, questions):
            raise TypeSafeAuthenticationError("Synthetic invalid key")

    with pytest.raises(TypeSafeAuthenticationError):
        smoke.run(execute=True, client_factory=Rejected)
    with pytest.raises(RuntimeError, match="no saved response"):
        smoke.run(execute=True, client_factory=lambda: pytest.fail("Requires explicit retry"))
    # Ordinary resume preserves the original definitive failure on the attempted request.
    report = smoke.run(
        execute=True, retry_auth_rejected=True, client_factory=Client, sleep=lambda _: None
    )
    assert len(report["attempts"]) == 3
    assert len(report["authentication_rejections"]) == 1
    assert report["historical_reserved_usd"] > 0
    assert report["conservative_total_with_prior_reservations_usd"] > report["total_cost_usd"]
    smoke.run(
        execute=True,
        retry_auth_rejected=True,
        client_factory=lambda: pytest.fail("Reuse saved responses"),
    )


def test_authentication_retry_never_overrides_an_uncertain_timeout(local):
    with pytest.raises(TimeoutError):
        smoke.run(execute=True, client_factory=lambda: Client(fail=True))
    with pytest.raises(ValueError, match="HTTP 401"):
        smoke.run(
            execute=True, retry_auth_rejected=True, client_factory=lambda: pytest.fail("No call")
        )


def test_output_cleanup_cannot_erase_paid_attempts_or_repeat_calls(local):
    import shutil

    report = smoke.run(execute=True, client_factory=Client, sleep=lambda _: None)
    shutil.rmtree(paths.outputs_dir())
    shutil.rmtree(paths.large_outputs_dir())
    after = smoke.run(execute=True, client_factory=lambda: pytest.fail("No duplicate paid calls"))
    assert report["request_identity"] == after["request_identity"]
    assert after["total_cost_usd"] == report["total_cost_usd"]
    assert (paths.manifests_dir("experiment_0") / "before_extraction/jev_smoke.json").exists()


def test_legacy_unresolved_attempt_survives_output_cleanup(local):
    from src.utils.serialization import write_json

    legacy = paths.manifests_dir("experiment_0") / "before_extraction/jev_smoke.json"
    write_json(
        legacy,
        {
            "request_identity": "fixed",
            "attempts": [{"probe": "binary_True", "reserved_usd": 0.002688}],
        },
    )
    directory, state = smoke.preserve_smoke_state()
    assert state.exists() and not (directory / "binary_True.json").exists()
    import json

    assert json.loads(state.read_text())["attempts"][0]["reserved_usd"] == 0.002688


def test_legacy_response_corruption_blocks_migration(local):
    from src.utils.serialization import write_json

    legacy = paths.manifests_dir("experiment_0") / "before_extraction/jev_smoke.json"
    write_json(
        legacy,
        {
            "request_identity": "fixed",
            "attempts": [{"probe": "binary_True", "response_sha256": "incorrect"}],
        },
    )
    write_json(
        paths.results_dir("jev_smoke", phase="experiment_0") / "binary_True.json", {"answer": 1}
    )
    with pytest.raises(ValueError, match="checksum"):
        smoke.preserve_smoke_state()


def test_divergent_output_ledger_cannot_silently_replace_durable_spending(local):
    from src.utils.serialization import write_json

    directory, state = smoke.preserve_smoke_state()
    write_json(state, {"request_identity": "fixed", "attempts": []})
    legacy = paths.manifests_dir("experiment_0") / "before_extraction/jev_smoke.json"
    write_json(legacy, {"request_identity": "fixed", "attempts": [{"probe": "binary_True"}]})
    with pytest.raises(ValueError, match="ledgers differ"):
        smoke.preserve_smoke_state()


def test_rounded_score_is_revalidated_offline_without_rebilling(local, monkeypatch):
    from src.jev.provider import score_consistency

    class Rounded(Client):
        def system_one(self, state, questions):
            wrapped = super().system_one(state, questions)
            raw = wrapped.raw_http_response.json()
            if raw["answers"]["semantic"]["type"] == "score":
                raw["answers"]["semantic"]["score"] = 7.99
            return SimpleNamespace(raw_http_response=SimpleNamespace(json=lambda: raw))

    report = smoke.run(execute=True, client_factory=Rounded, sleep=lambda _: None, robustness=True)
    assert report["attempts"][-1]["score_validation"]["rounding_envelope_used"]
    monkeypatch.setattr(
        smoke, "api_key", lambda: pytest.fail("Offline revalidation must not read credentials")
    )
    after = smoke.run(
        revalidate_only=True, robustness=True, client_factory=lambda: pytest.fail("No network")
    )
    assert after["total_cost_usd"] == report["total_cost_usd"]
    assert len(after["attempts"]) == 15
    probabilities = [0.0] * 8 + [1.0]
    for provider, score in [("intern", 7.99), ("jev", 7.8), ("jev", 7.991)]:
        with pytest.raises(ValueError, match="inconsistent"):
            score_consistency(score, probabilities, provider)


def test_offline_revalidation_cannot_fill_missing_responses(local, monkeypatch):
    monkeypatch.setattr(smoke, "api_key", lambda: pytest.fail("No credential access"))
    with pytest.raises(RuntimeError, match="offline revalidation"):
        smoke.run(revalidate_only=True, client_factory=lambda: pytest.fail("No network"))
