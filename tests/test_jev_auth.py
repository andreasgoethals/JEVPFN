import json

import pytest

from src.jev import auth, smoke


@pytest.mark.parametrize("value", ["", " \t ", "synthetic key", "synthetic\x16key", "synthetic–key"])
def test_invalid_key_input_is_rejected_without_exposing_its_value(monkeypatch, value):
    monkeypatch.setenv("TYPESAFE_API_KEY", value)
    with pytest.raises(auth.CredentialFormatError) as caught:
        auth.api_key()
    if value.strip():
        assert value not in str(caught.value)
    assert "No request was sent" in auth.error_hint(caught.value)


def test_key_normalization_is_only_at_the_edges(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "  synthetic-test-key\r\n")
    assert auth.api_key() == "synthetic-test-key"
    assert auth.error_hint(ValueError("sensitive SDK text")) is None


def test_smoke_paste_error_explains_failure_without_creating_a_client(
    isolated_output, monkeypatch, capsys
):
    monkeypatch.setattr(smoke.paths, "on_vsc", lambda: False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(smoke.getpass, "getpass", lambda _: "synthetic\x16paste")
    monkeypatch.setattr("sys.argv", ["smoke", "--execute"])
    with pytest.raises(SystemExit):
        smoke.main()
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["error_type"] == "CredentialFormatError"
    assert "No request was sent" in report["hint"]
    assert "synthetic" not in output
    assert not (smoke.paths.results_dir("jev_smoke", phase="experiment_0")).exists()
