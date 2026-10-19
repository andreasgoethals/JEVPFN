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
