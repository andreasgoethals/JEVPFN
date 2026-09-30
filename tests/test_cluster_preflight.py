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
