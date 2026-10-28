"""The approved roster, CPU preparation, native rendering and resumable setup graph.

Tiny contracts run everywhere. Set JEVPFN_TEST_TOKENIZERS to separately staged,
fingerprinted publisher assets to exercise actual tokenizers without weights.
"""

import json
import os
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.cluster import setup_extractors
from src.jev import checkpoint_hashes, registry
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import digest


def test_eleven_pins_licenses_and_no_download_on_import():
    candidates = load_yaml("extractors")["candidates"]
    assert len(candidates) == 12 and len(registry.SELF_HOSTED) == 10
    for provider in registry.ADDED:
        cfg = candidates[provider]
        assert cfg["delivery"] == "self_hosted" and not cfg["enabled"]
        assert cfg["license"] == "Apache-2.0"
        assert re.fullmatch("[0-9a-f]{40}", cfg["revision"])
        assert cfg["max_options"] >= 30
        assert all(
            cfg["revision"] in url
            or cfg.get("base_model_revision", "absent") in url
            or cfg.get("runtime_revision", "absent") in url
            for url in registry.asset_urls(provider).values()
        )


def test_cpu_graph_before_gpu_with_bounded_builds():
    steps = setup_extractors.plan(list(registry.SELF_HOSTED))
    previous = set()
    for step in steps:
        assert set(step["dependencies"]) <= previous
        previous.add(step["name"])
        assert "a100" not in str(step).lower()
    for provider in registry.SELF_HOSTED:
        by_name = {s["name"]: s for s in steps}
        assert provider + "_stage" in by_name[provider + "_tokens_prepare"]["dependencies"]
        assert len(by_name[provider + "_tokens"]["dependencies"]) == 10
        for name in by_name[provider + "_tokens"]["dependencies"]:
            assert by_name[name]["dependencies"] == [provider + "_tokens_prepare"]
        contracts = by_name[provider + "_contracts"]
        assert {"cpu", "storage", "gpfs_storage" if provider == "cloudflare" else provider + "_gpfs_storage", provider + "_scratch"} <= set(
            contracts["dependencies"]
        )
        assert {provider + "_tokens", provider + "_build"} <= set(
            by_name[provider + "_scratch"]["dependencies"]
        )
        assert "--partition=gpu_b200" in contracts["flags"]
        assert contracts["cluster"] == "mindwell"
        assert contracts["args"][2].endswith("/jevpfn-work/" + provider)
        assert by_name[provider + "_build"]["args"][3] == "blackwell-cu130"
        assert all(
            by_name[provider + suffix]["cluster"] == "wice"
            for suffix in ("_stage", "_tokens", "_build", "_scratch")
        )
        if provider in setup_extractors.LARGE:
            assert "--mem=112000M" in contracts["flags"]
            assert "--cpus-per-task=14" in contracts["flags"]
        assert by_name[provider + "_pilot"]["dependencies"] == [provider + "_contracts"]
    assert by_name["gpfs_storage"]["cluster"] == "mindwell"
    assert by_name["gpfs_storage"]["dependencies"] == ["cloudflare_scratch"]
    assert "embedding_stage" in by_name["cloudflare_scratch"]["dependencies"]
    encoders = [s for s in steps if s["name"].startswith("embeddings_")]
    assert len(encoders) == 3
    assert all({"cloudflare_scratch", "gpfs_storage"} <= set(s["dependencies"]) for s in encoders)
    reused = setup_extractors.plan(["intern", "h2o"], reuse=["intern"])
    assert "intern_build" not in {s["name"] for s in reused}
    assert (
        "intern_stage" in next(s for s in reused if s["name"] == "intern_scratch")["dependencies"]
    )


def _submission_fixture(monkeypatch, tmp_path):
    from src.cluster import containers

    monkeypatch.setattr(paths, "manifests_dir", lambda *a: tmp_path / "manifests")
    monkeypatch.setattr(paths, "logs_dir", lambda *a: tmp_path / "logs")
    monkeypatch.setattr(paths, "on_vsc", lambda: True)
    monkeypatch.setattr(setup_extractors, "source_snapshot", lambda: {"sha": "unchanged"})
    monkeypatch.setattr(setup_extractors, "check_if_present", lambda: {"frozen": False})
    monkeypatch.setattr(setup_extractors, "require_idle_vsc", lambda: None)
    monkeypatch.setattr(containers, "container_dir", lambda p, **k: tmp_path / "containers" / p)
    monkeypatch.setattr(containers, "verify", lambda p, **k: {"runtime_profile": "blackwell-cu130"})
    return tmp_path / "manifests/container_setup/open_models_mindwell_b200.json"


def test_submission_resume_never_duplicates_and_preserves_original_plan(monkeypatch, tmp_path):
    receipt = _submission_fixture(monkeypatch, tmp_path)
    commands = []

    def submit(command, **kwargs):
        if command[0] == "sacct":
            return SimpleNamespace(stdout="")
        commands.append(command)
        return SimpleNamespace(stdout=f"{100 + len(commands)};wice\n")

    monkeypatch.setattr(setup_extractors.subprocess, "run", submit)
    report = setup_extractors.run(["h2o"], execute=True)
    assert report["status"] == "waiting_for_job"
    n = len(commands)
    # The image now exists, but resume must use the original submitted graph.
    marker = tmp_path / "containers/h2o/_JEVPFN_CONTAINER.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    assert setup_extractors.run(["h2o"], execute=True)["steps"] == report["steps"]
    assert len(commands) == n
    assert json.loads(receipt.read_text())["status"] == report["status"]
    assert all("--clusters=wice" in c for c in commands)
    assert not any(arg.startswith("--dependency=") for c in commands for arg in c)
    assert {s["name"] for s in report["steps"] if s.get("job_id")} == {
        "cpu",
        "storage",
        "h2o_stage",
    }


def test_cross_cluster_steps_wait_for_successful_cpu_stage_and_do_not_duplicate(
    monkeypatch, tmp_path
):
    _submission_fixture(monkeypatch, tmp_path)
    submitted, completed = [], set()

    def scheduler(command, **kwargs):
        if command[0] == "sacct":
            job = command[command.index("-j") + 1]
            return SimpleNamespace(stdout=f"{job}|COMPLETED|0:0\n" if job in completed else "")
        job = str(100 + len(submitted))
        cluster = next(a.split("=", 1)[1] for a in command if a.startswith("--clusters="))
        submitted.append((job, command))
        return SimpleNamespace(stdout=f"{job};{cluster}\n")

    monkeypatch.setattr(setup_extractors.subprocess, "run", scheduler)
    for _ in range(len(setup_extractors.plan(["h2o"])) + 1):
        report = setup_extractors.run(["h2o"], execute=True)
        by_name = {s["name"]: s for s in report["steps"]}
        for step in report["steps"]:
            if step.get("job_id"):
                assert all(by_name[d]["job_id"] in completed for d in step["dependencies"])
        completed.update(j for j, _ in submitted)
        if report["status"] == "setup_completed_requires_review":
            break
    assert report["status"] == "setup_completed_requires_review"
    assert len(submitted) == len(report["steps"])
    assert sum("--clusters=mindwell" in command for _, command in submitted) == 3
    assert sum("--partition=gpu_b200" in command for _, command in submitted) == 2
    assert all(not any(a.startswith("--dependency=") for a in command) for _, command in submitted)


def test_failed_prerequisite_never_dispatches_gpu(monkeypatch, tmp_path):
    receipt = _submission_fixture(monkeypatch, tmp_path)
    submissions = []

    def scheduler(command, **kwargs):
        if command[0] == "sacct":
            job = command[command.index("-j") + 1]
            return SimpleNamespace(stdout=f"{job}|FAILED|1:0\n")
        submissions.append(command)
        return SimpleNamespace(stdout=f"{100 + len(submissions)};wice\n")

    monkeypatch.setattr(setup_extractors.subprocess, "run", scheduler)
    setup_extractors.run(["h2o"], execute=True)
    before = len(submissions)
    report = setup_extractors.run(["h2o"], execute=True)
    assert len(submissions) == before
    assert report["status"] == "setup_completed_with_failures"
    assert len(report["summary"]["failures"]) == 3
    assert json.loads(receipt.read_text())["summary"] == report["summary"]


def test_ambiguous_submission_requires_review_not_automatic_retry(monkeypatch, tmp_path):
    receipt = _submission_fixture(monkeypatch, tmp_path)
    commands = []

    def submit(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="scheduler accepted but response is ambiguous")

    monkeypatch.setattr(setup_extractors.subprocess, "run", submit)
    with pytest.raises(ValueError, match="Ambiguous"):
        setup_extractors.run(["h2o"], execute=True)
    assert json.loads(receipt.read_text())["steps"][0]["status"] == "starting"
    with pytest.raises(ValueError, match="Earlier setup stopped"):
        setup_extractors.run(["h2o"], execute=True)
    assert len(commands) == 1
    assert json.loads(receipt.read_text())["steps"][0]["submission_diagnostics"]["stdout"] == (
        "scheduler accepted but response is ambiguous"
    )


@pytest.mark.parametrize("timed_out", [False, True])
def test_submission_error_is_visible_and_keeps_ambiguous_receipt(
    monkeypatch, tmp_path, capsys, timed_out
):
    import subprocess

    receipt = _submission_fixture(monkeypatch, tmp_path)
    commands = []

    def fail(command, **kwargs):
        commands.append(command)
        if timed_out:
            raise subprocess.TimeoutExpired(command, 60, stderr="Controller not reachable")
        raise subprocess.CalledProcessError(1, command, stderr="Controller not reachable")

    monkeypatch.setattr(setup_extractors.subprocess, "run", fail)
    with pytest.raises((subprocess.CalledProcessError, subprocess.TimeoutExpired)):
        setup_extractors.run(["h2o"], execute=True)
    body = json.loads(receipt.read_text())
    assert body["steps"][0]["status"] == "starting"
    assert body["steps"][0]["submission_diagnostics"]["stderr"] == "Controller not reachable"
    assert "Controller not reachable" in capsys.readouterr().err
    assert len(commands) == 1


def test_restart_retains_history_reuses_sealed_assets_and_rechecks_source(monkeypatch, tmp_path):
    from src.cluster import experiment_zero

    receipt = _submission_fixture(monkeypatch, tmp_path)
    submits = []

    def submit(command, **kwargs):
        submits.append(command)
        return SimpleNamespace(stdout=f"{100 + len(submits)};wice\n")

    monkeypatch.setattr(setup_extractors.subprocess, "run", submit)
    monkeypatch.setattr(experiment_zero, "scheduler_state", lambda *a, **kw: None)
    old = setup_extractors.run(["h2o"], execute=True)
    old["status"] = "stopped_requires_review"
    receipt.write_text(json.dumps(old))
    old_bytes = receipt.read_bytes()
    monkeypatch.setattr(setup_extractors, "source_snapshot", lambda: {"sha": "fixed-source"})
    marker = tmp_path / "containers/h2o/_JEVPFN_CONTAINER.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    preview = setup_extractors.run(["h2o"], restart=True)
    assert preview["reused_containers"] == ["h2o"]
    assert receipt.read_bytes() == old_bytes
    with pytest.raises(ValueError, match="accounting"):
        setup_extractors.run(["h2o"], restart=True, execute=True)
    assert receipt.read_bytes() == old_bytes and len(submits) == 3
    monkeypatch.setattr(experiment_zero, "scheduler_state", lambda *a, **kw: ("FAILED", "1:0"))
    new = setup_extractors.run(["h2o"], restart=True, execute=True)
    assert new["prior_attempts"] == [old]
    assert new["identity"] != old["identity"]
    assert "h2o_build" not in {s["name"] for s in new["steps"]}
    assert len(submits) == 6  # Fresh cheap checks; cached data/model bytes remain intact.
    monkeypatch.setattr(experiment_zero, "scheduler_state", lambda *a, **kw: None)
    resumed = setup_extractors.run(["h2o"], execute=True)
    assert resumed["prior_attempts"] == [old] and len(submits) == 6


def test_restart_refuses_ambiguous_submission_and_active_project_jobs(monkeypatch, tmp_path):
    receipt = _submission_fixture(monkeypatch, tmp_path)
    receipt.parent.mkdir(parents=True)
    old = {"providers": ["h2o"], "steps": [{"status": "starting", "name": "cpu"}]}
    receipt.write_text(json.dumps(old))
    before = receipt.read_bytes()
    with pytest.raises(ValueError, match="Unrecorded"):
        setup_extractors.run(["h2o"], restart=True, execute=True)
    monkeypatch.setattr(setup_extractors, "require_idle_vsc", lambda: (_ for _ in ()).throw(RuntimeError("active jobs")))
    with pytest.raises(RuntimeError, match="active jobs"):
        setup_extractors.run(["h2o"], restart=True, execute=True)
    assert receipt.read_bytes() == before


def test_restart_watcher_restarts_only_once(monkeypatch):
    calls = []

    def run(*a, **kw):
        calls.append(kw)
        return {"status": "waiting_for_job" if len(calls) == 1 else "setup_completed_requires_review"}

    monkeypatch.setattr(setup_extractors, "run", run)
    monkeypatch.setattr(setup_extractors.time, "sleep", lambda _: None)
    monkeypatch.setattr("sys.argv", ["setup", "--all", "--execute", "--restart", "--watch"])
    setup_extractors.main()
    assert [c["restart"] for c in calls] == [True, False]


def test_hash_memo_invalidates_same_size_replacement_and_inventory(
    isolated_output, tmp_path, monkeypatch
):
    folder = tmp_path / "weights"
    folder.mkdir()
    file = folder / "weights.safetensors"
    file.write_bytes(b"old")
    before = checkpoint_hashes.hashes(folder)
    original_hash = checkpoint_hashes.file_sha256
    if os.name != "nt":
        monkeypatch.setattr(
            checkpoint_hashes, "file_sha256", lambda p: pytest.fail("unchanged bytes rehashed")
        )
    assert checkpoint_hashes.hashes(folder) == before
    monkeypatch.setattr(checkpoint_hashes, "file_sha256", original_hash)
    old = file.stat()
    file.write_bytes(b"new")
    os.utime(file, ns=(old.st_atime_ns, old.st_mtime_ns))
    after = checkpoint_hashes.hashes(folder)
    assert before != after
    file.unlink()
    (folder / "extra.json").write_text("{}")
    assert set(checkpoint_hashes.hashes(folder)) == {"extra.json"}


@pytest.mark.parametrize("provider", registry.ADDED)
def test_compressed_token_cache_reuse_and_corruption(provider):
    counter = object.__new__(registry.ExtendedCounter)
    counter._prepared = lru_cache(maxsize=64)(counter._prepared)
    counter.cache_db = sqlite3.connect(":memory:")
    counter.cache_writable = True
    counter.attach_cache(counter.cache_db, writable=True)
    payload = {"state": {"text": {"column": "hello"}}, "question": {"kind": "binary_probability"}}
    prepared = {"ids": [1, 2, 3], "keys": ["false", "true"]}
    counter._prepare = lambda p: prepared
    assert counter._prepared(json.dumps(payload)) == prepared
    counter.cache_db.commit()
    counter._prepare = lambda p: pytest.fail("cached tokens were recomputed")
    counter._prepared.cache_clear()
    assert counter._prepared(json.dumps(payload)) == prepared
    counter.cache_db.execute("UPDATE prepared_tokens SET prepared_sha256=?", ("0" * 64,))
    counter._prepared.cache_clear()
    with pytest.raises(ValueError, match="checksum"):
        counter._prepared(json.dumps(payload))
    counter.cache_db.close()


def test_missing_prepared_tokens_fail_before_gpu_loading():
    db = sqlite3.connect(":memory:")
    try:
        db.execute("CREATE TABLE counts(payload_sha256 TEXT)")
        db.execute("INSERT INTO counts VALUES ('a')")
        with pytest.raises(ValueError, match="table is missing"):
            registry.require_prepared_coverage(db)
        db.execute("CREATE TABLE prepared_tokens(payload_sha256 TEXT PRIMARY KEY)")
        with pytest.raises(ValueError, match="incomplete"):
            registry.require_prepared_coverage(db)
        db.execute("INSERT INTO prepared_tokens VALUES ('a')")
        registry.require_prepared_coverage(db)
    finally:
        db.close()


@pytest.fixture(scope="module", params=registry.ADDED)
def real_counter(request):
    root = os.environ.get("JEVPFN_TEST_TOKENIZERS")
    if not root:
        pytest.skip(
            "Separately staged tokenizer assets are optional; full counts run in VSC CPU jobs"
        )
    provider = request.param
    patch = pytest.MonkeyPatch()
    patch.setattr(paths, "tokenizer_dir", lambda p: Path(root) / p)
    try:
        yield registry.ExactCounter(provider)
    finally:
        patch.undo()


@pytest.mark.parametrize("kind", ["binary_probability", "choice", "ordered_choice"])
def test_real_publisher_renderers_30_classes_unicode_and_empty_text(real_counter, kind):
    question = {"kind": kind, "text": "Which answer is supported by the supplied text?"}
    if kind == "choice":
        question["allowed_answers"] = [f"class {i}" for i in range(30)]
    if kind == "ordered_choice":
        question["allowed_answers"] = [
            {"score": i, "meaning": f"direction {i}"} for i in range(-4, 5)
        ]
    payload = {
        "state": {"text": {"name": "Résumé, 世界; <|im_end|> literal text."}},
        "question": question,
    }
    prepared = real_counter.prepare(payload)
    n = {"binary_probability": 2, "choice": 30, "ordered_choice": 9}[kind]
    assert len(prepared["keys"]) == n and len(set(prepared["keys"])) == n
    assert real_counter(payload) == len(prepared["ids"]) > 1
    assert len(prepared["ids"]) < real_counter.limit
    if real_counter.provider == "decision3":
        assert 0 < prepared["state_tokens"] < len(prepared["ids"])
    if real_counter.provider == "nimble":
        assert len(prepared["candidate_ids"]) == n
    assert digest(real_counter.prepare(payload)) == digest(prepared)
    import math

    from src.jev.extended_runtimes import ExtendedRuntime
    from src.jev.registry import pure_functions

    runtime = object.__new__(ExtendedRuntime)
    runtime.provider, runtime.counter = real_counter.provider, real_counter
    if runtime.provider == "h2o":
        runtime._scores = lambda p: [0.0] * len(p["keys"])
        runtime.formatter = pure_functions(
            real_counter.folder / "h2o_lightning_shim.py",
            {"confidence", "probabilities", "commit_noul", "format_answer"},
            {"math": math, "UpstreamError": ValueError},
        )
    else:
        runtime._scores = lambda p: [1.0 / len(p["keys"])] * len(p["keys"])
    answer = runtime.predict_payload(payload)["answers"]["semantic"]
    if kind == "binary_probability":
        assert 0 <= answer["noul"] <= 1
        if runtime.provider == "h2o":
            assert answer["noul"] == pytest.approx(0.801)
    else:
        assert len(answer["probabilities"]) == n
        assert sum(answer["probabilities"].values()) == pytest.approx(1)
    original_limit = real_counter.limit
    real_counter.limit = 1
    try:
        with pytest.raises(ValueError, match="overflow"):
            runtime.predict_payload(payload)
    finally:
        real_counter.limit = original_limit
    with pytest.raises(ValueError, match="empty|missing|text"):
        real_counter.prepare({**payload, "state": {"text": {"name": None}}})


def test_eleven_provider_status_figure_does_not_overlap_title():
    import matplotlib.pyplot as plt

    from src.evaluation.decision_comparison import Comparison
    from src.visualize import style
    from src.visualize.decision_comparison import coverage_figures

    style.apply()
    options = load_yaml("feature_comparison")["comparison"]
    result = Comparison()
    for provider in options["providers"]:
        for mode in options["representations"]:
            result.add(
                "inventory",
                dict(
                    provider=provider, representation=mode, dataset="fixture", status="not_created"
                ),
            )
    fig, _, _ = next(coverage_figures(result, options))
    try:
        fig.canvas.draw()
        ax = fig.axes[0]
        renderer = fig.canvas.get_renderer()
        assert ax.tables[0].get_window_extent(renderer).y1 < ax.title.get_window_extent(renderer).y0
    finally:
        plt.close(fig)
