"""Pinned offline task checkpoints; downloads are always synthetic in this suite."""

import copy
import os

import httpx
import pytest

from src.cluster import predictor_assets
from src.evaluation.tabstar_assets import asset_manifest, manifest_path, verified_settings
from src.utils import paths
from src.utils.config import load_yaml


@pytest.mark.parametrize("failures", [1, 2])
def test_download_resumes_transport_failures_in_same_pinned_cache(tmp_path, monkeypatch, failures):
    calls, waits = [], []
    monkeypatch.setattr(predictor_assets.time, "sleep", waits.append)

    def download(**kwargs):
        calls.append(kwargs)
        if len(calls) <= failures:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response")
        return tmp_path

    kwargs = {"repo_id": "publisher/model", "revision": "fixed-sha", "cache_dir": tmp_path}
    assert predictor_assets._download_snapshot(download, **kwargs) == tmp_path
    assert calls == [kwargs] * (failures + 1)
    assert waits == [5, 10][:failures]


def test_failed_staging_is_bounded_and_never_writes_acceptance(isolated_output, monkeypatch):
    monkeypatch.setattr(predictor_assets, "check_if_present", lambda: {"frozen": False})
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    waits, calls = [], []
    monkeypatch.setattr(predictor_assets.time, "sleep", waits.append)

    def download(**kwargs):
        calls.append(kwargs)
        raise httpx.RemoteProtocolError("Connection reset")

    with pytest.raises(httpx.RemoteProtocolError, match="Connection reset"):
        predictor_assets.stage_tabstar(execute=True, download=download)
    assert len(calls) == 3 and calls == [calls[0]] * 3
    assert waits == [5, 10]
    assert not manifest_path().exists()


@pytest.mark.parametrize("status", [401, 403, 404])
def test_http_rejections_are_not_download_retries(tmp_path, monkeypatch, status):
    calls, waits = [], []
    monkeypatch.setattr(predictor_assets.time, "sleep", waits.append)
    request = httpx.Request("GET", "https://example.test/model")
    error = httpx.HTTPStatusError("Rejected", request=request,
                                 response=httpx.Response(status, request=request))

    def download(**kwargs):
        calls.append(kwargs)
        raise error

    with pytest.raises(httpx.HTTPStatusError) as caught:
        predictor_assets._download_snapshot(
            download, repo_id="publisher/model", revision="fixed-sha", cache_dir=tmp_path)
    assert caught.value is error
    assert len(calls) == 1 and not waits


@pytest.fixture
def staged(isolated_output, monkeypatch):
    monkeypatch.setattr(predictor_assets, "check_if_present", lambda: {"frozen": False})
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    for name in ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    calls = []

    def download(*, repo_id, revision, cache_dir):
        calls.append((repo_id, revision))
        destination = cache_dir / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text("{}", encoding="utf-8")
        (destination / "model.safetensors").write_bytes(b"fixture weights; never executed")
        return destination

    body = predictor_assets.stage_tabstar(execute=True, download=download)
    return body, calls, download


def test_offline_aliases_resolve_distinct_cls_and_regression_snapshots(staged):
    body, calls, download = staged
    settings = load_yaml("experiment_1/models")["models"]["tabstar"]["runtime"]
    assert len(calls) == 3 and len(body["assets"]) == 9
    assert manifest_path().is_file()
    assert predictor_assets.stage_tabstar(execute=True, download=download) == body
    assert len(calls) == 3
    for task in ("binary_classification", "multiclass_classification", "regression"):
        repository, assets = verified_settings(settings, task)
        kind = "regression" if task == "regression" else "classification"
        assert repository == settings["repositories"][kind]["model"]
        assert len(assets) == 9
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_corruption_and_external_cache_never_fall_back_online(staged, monkeypatch, tmp_path):
    body, _, _ = staged
    settings = load_yaml("experiment_1/models")["models"]["tabstar"]["runtime"]
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "external"))
    with pytest.raises(ValueError, match="active native"):
        verified_settings(settings, "regression")
    monkeypatch.delenv("HF_HUB_CACHE")
    (paths.checkpoints_dir() / body["assets"][0]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verified_settings(settings, "regression")


def test_changed_model_pin_invalidates_asset_manifest(staged):
    settings = copy.deepcopy(load_yaml("experiment_1/models")["models"]["tabstar"]["runtime"])
    settings["repositories"]["classification"]["revision"] = "other"
    with pytest.raises(ValueError, match="configured v1.1"):
        asset_manifest(settings)


def test_staging_requires_explicit_execution_unfrozen_cpu(isolated_output, monkeypatch):
    with pytest.raises(RuntimeError, match="--execute"):
        predictor_assets.stage_tabstar()
    monkeypatch.setattr(predictor_assets, "check_if_present", lambda: {"frozen": True})
    with pytest.raises(ValueError, match="before freezing"):
        predictor_assets.stage_tabstar(execute=True)
    monkeypatch.setattr(predictor_assets, "check_if_present", lambda: {"frozen": False})
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    with pytest.raises(ValueError, match="on CPU"):
        predictor_assets.stage_tabstar(execute=True)


def test_new_generation_has_own_container_path_and_fixed_runtime(isolated_output):
    from src.cluster.containers import container_dir
    config = load_yaml("experiment_1/models")["models"]["tabstar"]
    assert config["model_generation"] == "v1.1"
    assert config["runtime"]["packages"] == {"tabstar": "1.2.2", "transformers": "4.57.6"}
    assert container_dir("tabstar", profile="blackwell-cu130").name == "tabstar-v1p1"


def test_container_install_does_not_mix_tabstar_with_v5_tokenizer_dependencies(tmp_path, monkeypatch):
    import sys

    source = (paths.REPO_ROOT / "scripts/build_decision_container.sh").read_text(encoding="utf-8")
    start = source.index("import sys,shlex\n")
    code = source[start:source.index("\nPY\n", start)]
    monkeypatch.setattr(sys, "argv", ["build", str(tmp_path), str(paths.REPO_ROOT), "tabstar",
                                     str(tmp_path / "project"), "https://download.pytorch.org/whl/cu130",
                                     "blackwell-cu130"])
    exec(compile(code, "container-post-install-fixture", "exec"), {})
    generated = (tmp_path / "post_install.sh").read_text(encoding="utf-8")
    assert ".[dev,tabstar]" in generated
    assert "token-audit" not in generated and "token_audit" not in generated
    assert "verify_runtime" in generated and "cu130" in generated
