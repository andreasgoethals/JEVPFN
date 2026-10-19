"""Offline fault tests for the real GPU harness. No weights, allocations or network."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.cluster import gpu_contracts as gc
from src.cluster.token_audit import audit_provider
from src.jev.inputs import prepare_dataset_inputs
from src.jev.provider import QUESTION_ID, semantic_question
from src.jev.question_bank import build_question_bank, save_question_bank
from src.jev.requests import require_nonempty_text
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import digest, file_sha256


class FakeClient:
    def __init__(self, provider):
        self.settings = {
            "provider": provider,
            "model": "fixture",
            "version": "pinned",
            "api_mapping_version": f"{provider}-self-hosted-v1",
            "request_configuration": {},
            "template_version": "fixture",
        }
        self.counter = self
        self.limit = 64
        self.forwards = 0

    def __call__(self, payload):
        require_nonempty_text(payload)
        return max(1, sum(len(str(t)) for t in payload["state"]["text"].values()) // 4)

    def evaluate(self, request):
        payload = json.loads(request.payload_json)
        if self(payload) > self.limit:
            raise ValueError("Context overflow")
        self.forwards += 1
        q = semantic_question(payload["question"])
        if q["type"] == "noul":
            answer = {"type": "noul", "noul": 0.7}
        else:
            keys = list(q["criteria"]) if q["type"] == "choice" else list(map(str, range(9)))
            probabilities = dict.fromkeys(keys, 1 / len(keys))
            answer = {
                "type": q["type"],
                "probabilities": probabilities,
                "confidence": 1 / len(keys),
            }
            if q["type"] == "choice":
                answer["choice"] = keys[0]
            else:
                answer.update(score=4, legend=dict(zip(keys, q["criteria"])))
        return {
            "provider": self.settings["provider"],
            "model": "fixture",
            "version": "pinned",
            "settings_sha256": digest(self.settings),
            "answers": {QUESTION_ID: answer},
        }


@pytest.mark.parametrize("provider", ["cloudflare", "intern"])
def test_real_harness_reopens_features_without_extra_forwards(isolated_output, provider):
    client = gc.MeasuredClient(FakeClient(provider), 100)
    directory = paths.results_dir("fixture_contracts", phase="experiment_0")
    directory.mkdir(parents=True)
    for dataset in gc.synthetic_datasets():
        report = gc.exercise_feature_pipeline(client, dataset, directory)
        assert report["unique_requests"] == 10
        assert report["empty_features_missing"] and report["cache_reopen_and_reuse"]
        assert len(report["artifacts"]) == 2
    assert client.calls == client.client.forwards == 30
    assert len(list((paths.manifests_dir("experiment_0") / "features").glob("*.json"))) == 6
    assert not paths.manifests_dir("feature_creation").exists()
    assert (
        not paths.jev_cache_path().exists()
    )  # Synthetic values never become real dataset features.


@pytest.mark.parametrize("provider", ["cloudflare", "intern"])
def test_boundary_inputs_cannot_reach_forward(isolated_output, provider):
    directory = paths.results_dir("fixture_boundaries", phase="experiment_0")
    directory.mkdir(parents=True)
    client = gc.MeasuredClient(FakeClient(provider), 1)
    assert all(gc.exercise_boundaries(client, directory).values())
    assert client.calls == client.client.forwards == 1
    assert all(gc.exercise_boundaries(client, directory).values())  # Cached wide vector.
    assert client.calls == 1


def test_hard_compute_ceiling_counts_failed_attempts():
    client = Mock()
    client.evaluate.side_effect = ValueError("runtime failed")
    metered = gc.MeasuredClient(client, 1)
    with pytest.raises(ValueError, match="runtime"):
        metered.evaluate(None)
    with pytest.raises(RuntimeError, match="ceiling"):
        metered.evaluate(None)
    assert client.evaluate.call_count == 1


def test_real_input_selection_and_reuse(isolated_output):
    dataset = next(gc.synthetic_datasets())
    bank = build_question_bank([dataset.metadata], load_yaml("question_design"))
    save_question_bank(bank)
    prepared = prepare_dataset_inputs(dataset, bank, variant="accepted")
    inputs = [
        {
            "dataset_id": dataset.metadata.dataset_id,
            "path": str(prepared["path"]),
            "input_set_id": prepared["input_set_id"],
            "source_rows": len(dataset.frame),
            "input_file_sha256": file_sha256(Path(prepared["path"])),
            "unique_inputs": prepared["unique_inputs"],
        }
    ]
    counter = SimpleNamespace(
        identity={"counter_id": "fixture"},
        limit=100,
        count_many=lambda rows: [len(str(r)) for r in rows],
    )
    report = audit_provider("cloudflare", inputs, counter=counter)
    sample = gc.real_sample(report, Path(report["counts_database"]))
    client = gc.MeasuredClient(FakeClient("cloudflare"), 100)
    log = paths.logs_dir("experiment_0") / "fixture.log"
    first = gc.exercise_real_inputs(client, sample, log)
    second = gc.exercise_real_inputs(client, sample, log)
    assert first[0]["new"] == second[0]["reused"] == len(sample[0][1])
    assert second[0]["new"] == 0 and first[0]["reopened_cache_hit"]
    assert client.calls == len(sample[0][1])
    assert file_sha256(Path(prepared["path"])) == inputs[0]["input_file_sha256"]


def test_no_gpu_loading_without_explicit_allocation(monkeypatch):
    loader = Mock()
    monkeypatch.setattr(gc, "SelfHostedClient", loader)
    with pytest.raises(RuntimeError, match="allocation"):
        gc.run("intern", execute=True)
    with pytest.raises(RuntimeError, match="execute"):
        gc.run("cloudflare")
    loader.assert_not_called()


def test_missing_token_report_stops_before_model_loading(isolated_output, monkeypatch):
    paths.staging_root().mkdir(parents=True)
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    limit = load_yaml("extractors")["candidates"]["cloudflare"]["context_tokens"]
    monkeypatch.setattr(gc, "ExactCounter", lambda _: SimpleNamespace(limit=limit))
    loader = Mock()
    monkeypatch.setattr(gc, "SelfHostedClient", loader)
    with pytest.raises(ValueError, match="experiment_0_tokens.slurm"):
        gc.run("cloudflare", execute=True)
    loader.assert_not_called()
    reports = list((paths.manifests_dir("experiment_0") / "before_extraction/gpu_contracts").glob("*.json"))
    report = json.loads(reports[0].read_text())
    assert report["status"] == "failed" and report["synthetic"] == []
    assert "Do not rebuild the container" in report["error"]


@pytest.mark.parametrize("all_datasets", [True, False])
def test_login_input_check_checks_coverage_without_gpu(monkeypatch, capsys, all_datasets):
    monkeypatch.setattr(gc, "ExactCounter", Mock())
    monkeypatch.setattr(gc, "token_evidence", lambda *a: (
        {"datasets": [{"dataset_id": "fixture"}], "audit_id": "checked"}, Path("counts.sqlite3")
    ))
    monkeypatch.setattr("src.cluster.token_audit.selected_inputs", lambda: [
        {"dataset_id": "fixture" if all_datasets else "another"}
    ])
    loader = Mock()
    monkeypatch.setattr(gc, "SelfHostedClient", loader)
    monkeypatch.setattr("sys.argv", ["gpu_contracts", "--provider", "cloudflare", "--check-inputs"])
    if all_datasets:
        gc.main()
        result = json.loads(capsys.readouterr().out)
        assert result["status"] == "token_input_evidence_passed"
        assert result["inference_executed"] is False and result["gpu_execution_verified"] is False
    else:
        with pytest.raises(ValueError, match="every selected dataset"):
            gc.main()
    loader.assert_not_called()


def test_checkpoint_load_failure_leaves_diagnostic_manifest(isolated_output, monkeypatch):
    paths.staging_root().mkdir(parents=True)
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    monkeypatch.setattr(gc, "ExactCounter", Mock())
    monkeypatch.setattr(
        gc, "token_evidence", lambda *a: ({"datasets": [], "audit_id": "test"}, None)
    )
    monkeypatch.setattr(gc, "real_sample", lambda *a: [])
    monkeypatch.setattr("src.cluster.token_audit.selected_inputs", lambda: [])
    monkeypatch.setattr(gc, "SelfHostedClient", Mock(side_effect=RuntimeError("CUDA mismatch")))
    with pytest.raises(RuntimeError, match="CUDA mismatch"):
        gc.run("intern", execute=True)
    files = list(
        (paths.manifests_dir("experiment_0") / "before_extraction/gpu_contracts").glob("*.json")
    )
    report = json.loads(files[0].read_text())
    assert report["status"] == "failed" and report["error_type"] == "RuntimeError"
    assert not report["full_extraction_authorized"] and not report["full_experiment_0_passed"]


def test_success_report_exposes_acceptance_identity(isolated_output, monkeypatch):
    paths.staging_root().mkdir(parents=True)
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    monkeypatch.setattr(gc, "inference_environment", lambda: None)
    monkeypatch.setattr(gc, "ExactCounter", Mock())
    monkeypatch.setattr(
        gc, "token_evidence", lambda *a: ({"datasets": [], "audit_id": "audit"}, None)
    )
    monkeypatch.setattr("src.cluster.token_audit.selected_inputs", lambda: [])
    monkeypatch.setattr(gc, "real_sample", lambda *a: [])
    monkeypatch.setattr(gc, "synthetic_datasets", lambda: [])
    monkeypatch.setattr(gc, "exercise_boundaries", lambda *a: {})
    monkeypatch.setattr(gc, "exercise_semantics", lambda *a: {"passed": True})
    monkeypatch.setattr(gc, "exercise_real_inputs", lambda *a: [])
    cuda = Mock()
    cuda.get_device_name.return_value = "fixture GPU"
    cuda.get_device_properties.return_value = SimpleNamespace(total_memory=123)
    cuda.max_memory_allocated.return_value = 0
    cuda.max_memory_reserved.return_value = 0
    client = SimpleNamespace(
        settings={"provider": "intern", "revision": "fixed"},
        load_seconds=0,
        torch=SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="fixture")),
    )
    monkeypatch.setattr(gc, "SelfHostedClient", lambda *a, **kw: client)
    report = gc.run("intern", execute=True)
    assert report["settings_sha256"] == digest(client.settings)
    assert report["token_audit_id"] == "audit"
    assert report["status"] == "gpu_contracts_passed_requires_review"
