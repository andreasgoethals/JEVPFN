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


def test_local_file_is_reloaded_without_prompt_and_environment_takes_precedence(monkeypatch):
    monkeypatch.setattr(auth.paths, "on_vsc", lambda: False)
    auth.LOCAL_CREDENTIAL_PATH.write_text(json.dumps({"api_key": "synthetic-file-key"}), encoding="utf-8-sig")
    assert auth.credentials_available()
    assert auth.api_key() == "synthetic-file-key"
    auth.LOCAL_CREDENTIAL_PATH.write_text(json.dumps({"api_key": "synthetic-replacement"}))
    assert auth.api_key() == "synthetic-replacement"
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-environment-key")
    assert auth.api_key() == "synthetic-environment-key"
    monkeypatch.setenv("TYPESAFE_API_KEY", "bad value")
    with pytest.raises(auth.CredentialFormatError):
        auth.api_key()  # An explicit invalid override must not silently use another key.


@pytest.mark.parametrize("text", ["malformed synthetic-secret", "[]", "{}", '{"api_key": 7}'])
def test_malformed_local_credentials_never_echo_their_content(monkeypatch, text):
    monkeypatch.setattr(auth.paths, "on_vsc", lambda: False)
    auth.LOCAL_CREDENTIAL_PATH.write_text(text)
    with pytest.raises(auth.CredentialFormatError) as caught:
        auth.api_key()
    assert "synthetic-secret" not in str(caught.value)


def test_vsc_never_reads_the_laptop_credential_file(monkeypatch):
    auth.LOCAL_CREDENTIAL_PATH.write_text("unreadable credential syntax")
    monkeypatch.setattr(auth.paths, "on_vsc", lambda: True)
    assert not auth.credentials_available()
    with pytest.raises(auth.CredentialFormatError, match="empty"):
        auth.api_key()


def test_account_command_uses_local_file_without_prompt_or_environment_mutation(monkeypatch, capsys):
    import os

    from src.jev import setup_pilot

    monkeypatch.setattr(auth.paths, "on_vsc", lambda: False)
    auth.LOCAL_CREDENTIAL_PATH.write_text(json.dumps({"api_key": "synthetic-file-key"}))
    monkeypatch.setattr(setup_pilot.getpass, "getpass", lambda _: pytest.fail("No prompt expected"))
    monkeypatch.setattr("sys.argv", ["setup_pilot", "--check-account"])

    def check():
        assert auth.api_key() == "synthetic-file-key"
        return {"authenticated": True, "inference_performed": False}

    monkeypatch.setattr(setup_pilot, "account_check", check)
    setup_pilot.main()
    output = capsys.readouterr().out
    assert json.loads(output)["authenticated"]
    assert "synthetic-file-key" not in output
    assert "TYPESAFE_API_KEY" not in os.environ


def test_http_401_has_an_authentication_hint_without_exception_text():
    error = type("AuthError", (Exception,), {"status": 401})("sensitive exception text")
    hint = auth.error_hint(error)
    assert "rejected" in hint and "--check-account" in hint
    assert "sensitive exception text" not in hint
