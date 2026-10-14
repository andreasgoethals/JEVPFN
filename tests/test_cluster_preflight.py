"""Mock network contracts; this suite never reaches a real API or submits an allocation."""

import urllib.error
from unittest.mock import Mock

import pytest

from src.cluster import preflight


def test_experiment_zero_requires_compute_allocation(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("VSC_DATA", raising=False)
    with pytest.raises(RuntimeError, match="VSC compute allocation"):
        preflight.run_preflight()


def test_probe_sends_only_unauthenticated_head(monkeypatch):
    monkeypatch.setattr(preflight.socket, "getaddrinfo", Mock(return_value=[]))
    opener = Mock()
    opener.open.side_effect = urllib.error.HTTPError(
        "https://api.typesafe.ai/v1/systemone", 405, "method", {}, None
    )
    monkeypatch.setattr(preflight.urllib.request, "build_opener", Mock(return_value=opener))
    result = preflight.network_probe()
    assert result["reachable"] is True
    request = opener.open.call_args.args[0]
    assert request.get_method() == "HEAD"
    assert request.data is None
    assert not request.has_header("Authorization")
    assert result["inference_sent"] is False


def test_network_failure_not_misreported_as_success(monkeypatch):
    monkeypatch.setattr(
        preflight.socket, "getaddrinfo", Mock(side_effect=OSError("synthetic DNS failure"))
    )
    result = preflight.network_probe()
    assert result["reachable"] is False
    assert result["error_type"] == "OSError"


def test_vsc_suite_excludes_only_local_notebook_tests(isolated_output, monkeypatch):
    from types import SimpleNamespace

    runner = Mock(return_value=SimpleNamespace(returncode=0, stdout="passed", stderr=""))
    monkeypatch.setattr(preflight.subprocess, "run", runner)
    report = preflight.contract_tests("fixture")
    assert runner.call_args.args[0][-2:] == ["-m", "not notebook"]
    assert report["excluded_markers"] == ["notebook"]


def test_local_suite_retains_notebook_checks(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(preflight.paths, "logs_dir", lambda _: tmp_path)
    runner = Mock(return_value=SimpleNamespace(returncode=0, stdout="passed", stderr=""))
    monkeypatch.setattr(preflight.subprocess, "run", runner)
    report = preflight.contract_tests("fixture")
    assert "not notebook" not in runner.call_args.args[0]
    assert report["excluded_markers"] == []


def test_failed_contract_details_appear_in_main_job_log(tmp_path, monkeypatch, capsys):
    log = tmp_path / "contracts.log"
    log.write_text("ModuleNotFoundError: No module named 'IPython'\n3 failed", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["preflight", "--local", "--tests"])
    monkeypatch.setattr(preflight.paths, "manifests_dir", lambda _: tmp_path)
    monkeypatch.setattr(
        preflight,
        "run_preflight",
        lambda **kwargs: {
            "run_id": "fixture",
            "failures": ["Offline contract tests failed"],
            "contract_tests": {"status": "failed", "log": str(log)},
        },
    )
    assert preflight.main() == 1
    output = capsys.readouterr().out
    assert "ModuleNotFoundError" in output and "3 failed" in output
