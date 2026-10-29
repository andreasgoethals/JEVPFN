import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_cache_merge import cache, request, response
from test_scratch_stage import marker
from test_scratch_stage import roots as roots_fixture

from src.cluster import job_graph
from src.cluster import scratch_capacity as cap
from src.cluster import scratch_stage as stage
from src.cluster.workspace_graph import add_releases
from src.utils import paths
from src.utils.config import load_yaml

roots = roots_fixture


def copied_assets(durable, work):
    marker(durable, work)
    for relative in (
        "data/raw/x.csv",
        "data/decision_inputs/x.sqlite3",
        "checkpoints/model.bin",
        "containers/image.sif",
    ):
        for root in (durable, work):
            p = root / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"copied inputs")


def test_release_preserves_mutable_queues_and_publishes_features(roots):
    durable, work = roots
    copied_assets(durable, work)
    remote = cache(work / "data/jev_cache/fixture.sqlite3", request("intern"))
    with remote._connect() as db:
        db.execute("CREATE TABLE extraction_jobs (request_key TEXT PRIMARY KEY,status TEXT)")
        db.execute("INSERT INTO extraction_jobs VALUES ('uncertain','attempting')")
    preview = cap.release("intern")
    assert preview["removable_bytes"] > 0 and (work / "checkpoints/model.bin").exists()
    result = cap.release("intern", apply=True)
    assert result["applied"] and not (work / "checkpoints").exists()
    assert (durable / "checkpoints/model.bin").exists()
    assert remote.get(request("intern"), kind="verified_response") == response()
    with remote._connect() as db:
        assert db.execute("SELECT status FROM extraction_jobs").fetchone() == ("attempting",)
    assert (
        cache(durable / "data/jev_cache/fixture.sqlite3").get(
            request("intern"), kind="verified_response"
        )
        == response()
    )
    assert json.loads((work / stage.MARKER).read_text())["status"] == "released_requires_restage"
    assert cap.release("intern", apply=True)["removable_bytes"] == 0


def test_failed_publication_never_removes_inputs(roots):
    durable, work = roots
    copied_assets(durable, work)
    cache(durable / "data/jev_cache/fixture.sqlite3", value=0.7)
    cache(work / "data/jev_cache/fixture.sqlite3", value=0.9)
    with pytest.raises(ValueError):
        cap.release("intern", apply=True)
    assert (work / "containers/image.sif").exists()


def test_partial_stage_can_release_copies_but_unknown_workspace_cannot(roots):
    durable, work = roots
    copied_assets(durable, work)
    p = work / stage.MARKER
    d = json.loads(p.read_text())
    d["status"] = "staging"
    p.write_text(json.dumps(d))
    assert cap.release("intern", apply=True)["applied"]
    p.unlink()
    with pytest.raises(ValueError, match="No workspace identity"):
        cap.release("intern", apply=True)


def test_missing_authoritative_inputs_prevent_release(roots):
    durable, work = roots
    marker(durable, work)
    (work / "checkpoints").mkdir()
    (work / "checkpoints/only-copy").write_text("keep")
    with pytest.raises(ValueError, match="Missing durable"):
        cap.release("intern", apply=True)
    assert (work / "checkpoints/only-copy").exists()


def test_release_uses_verified_unmasked_control_image_not_its_executing_bin(roots, monkeypatch):
    from src.cluster import control_runtime as control
    from src.utils.serialization import write_json

    durable, work = roots
    copied_assets(durable, work)
    source = durable / "containers/control-v1"
    unmasked = durable / "host-view"
    source.mkdir()
    (unmasked / "bin").mkdir(parents=True)
    (unmasked / "deps.sqfs").write_bytes(b"dependencies")
    (unmasked / "bin/_debug_exec").write_bytes(b"host-only wrapper")
    body = dict(provider="control", runtime_profile="cpu-control-v1", files=control.image_files(unmasked))
    body["container_id"] = control.digest(body)
    write_json(source / control.MARKER, body)
    native = work / "containers/control-v1/bin/_debug_exec"
    native.parent.mkdir(parents=True)
    native.write_bytes(b"host-only wrapper")
    monkeypatch.setenv("JEVPFN_CONTROL_IMAGE", str(source))
    monkeypatch.setattr(control, "HOST_VIEW", unmasked)
    assert cap.release("intern")["status"] == "preview"
    (unmasked / "deps.sqfs").write_bytes(b"corrupt authoritative image")
    with pytest.raises(ValueError, match="sealed identity"):
        cap.release("intern", apply=True)
    assert native.is_file()


def test_release_refuses_running_workspace_lease(roots):
    durable, work = roots
    copied_assets(durable, work)
    with stage.workspace_lease(work), pytest.raises(RuntimeError, match="in use"):
        cap.release("intern", apply=True)
    assert (work / "data/raw/x.csv").exists()


@pytest.mark.parametrize("owner,blocked", [("intern", True), ("liquid", False), (None, True)])
def test_queued_consumers_are_protected_and_other_runtime_does_not_block(
    monkeypatch, owner, blocked
):
    monkeypatch.setattr(paths, "on_vsc", lambda: True)

    def output(argv, **kwargs):
        if argv[0] == "squeue":
            return (
                "123|jevpfn-exp0-contracts|/data/JEVPFN\n" if "--clusters=mindwell" in argv else ""
            )
        return (
            "SubmitLine=sbatch scripts/slurm/test.slurm " + (owner or "") + " WorkDir=/data/JEVPFN"
        )

    monkeypatch.setattr(cap.subprocess, "check_output", output)
    if blocked:
        with pytest.raises(RuntimeError, match="123"):
            cap.require_runtime_idle("intern")
    else:
        cap.require_runtime_idle("intern")


def test_capacity_limits_count_resident_sets_not_just_active_copies(roots, monkeypatch):
    durable, work = roots
    marker(durable, work)
    for runtime in ("liquid", "strands"):
        (work.parent.parent / runtime / "JEVPFN/checkpoints").mkdir(parents=True)
    with pytest.raises(OSError, match="slots"):
        cap.check_capacity(work, {}, durable / "containers/blackwell-cu130/intern")


@pytest.mark.parametrize("budget", ["scratch_working_set_gib", "scratch_allocation_gib"])
def test_byte_budget_is_not_confused_with_filesystem_free_space(roots, monkeypatch, budget):
    durable, work = roots
    marker(durable, work)
    settings = load_yaml("experiment_0/scheduling")
    settings[budget] = 1
    monkeypatch.setattr(cap, "load_yaml", lambda name: settings)
    monkeypatch.setattr(cap.shutil, "disk_usage", lambda p: SimpleNamespace(free=10**15))
    with pytest.raises(OSError, match="budget"):
        cap.check_capacity(work, {}, durable / "containers/blackwell-cu130/intern")


def test_predictors_do_not_copy_any_decision_token_database(roots, monkeypatch):
    from src.cluster import containers

    durable, _ = roots
    monkeypatch.setattr(paths, "REPO_ROOT", Path(stage.__file__).resolve().parents[2])
    p = durable / "data/decision_inputs/fixture.sqlite3"
    p.parent.mkdir(parents=True)
    with sqlite3.connect(p) as db:
        db.execute("CREATE TABLE x (id)")
    tokens = durable / "output_JEVPFN" / paths.phase_folder("experiment_0") / "results/token_audit"
    tokens.mkdir(parents=True)
    (tokens / "decision3_large.sqlite3").write_bytes(b"must never be copied")
    image = durable / "containers/blackwell-cu130/linear"
    image.mkdir(parents=True)
    (image / containers.MARKER).write_text("{}")
    monkeypatch.setattr(containers, "relocate", lambda *a: None)
    result = stage.stage("linear")
    assert not (Path(result["working_root"]) / tokens.relative_to(durable)).exists()


def test_graph_bounds_residency_through_queue_wait_and_releases_after_failure():
    steps = []
    for runtime in ("a", "b", "c"):
        steps += [
            dict(name=runtime + "_scratch", dependencies=[], cluster="wice"),
            dict(
                name=runtime + "_contracts", dependencies=[runtime + "_scratch"], cluster="mindwell"
            ),
            dict(
                name=runtime + "_pilot", dependencies=[runtime + "_contracts"], cluster="mindwell"
            ),
        ]
    steps = add_releases(steps)
    cfg = load_yaml("experiment_0/scheduling")
    cfg["test_pool_limits"]["transfer"] = (
        2  # Exercise residency independently of transfer throttling.
    )
    states, jobs = {}, {}

    def submit(s):
        s.update(job_id=str(len(jobs) + 1), status="submitted")
        jobs[s["job_id"]] = s["name"]

    def tick():
        return job_graph.advance(
            steps,
            settings=cfg,
            state=lambda j, **k: states.get(j),
            submit=submit,
            save=lambda: None,
        )

    tick()
    assert set(jobs.values()) == {"a_scratch", "b_scratch"}
    states.update({j: ("COMPLETED", "0:0") for j in jobs})
    tick()
    tick()  # Queued consumers retain both slots.
    assert "c_scratch" not in jobs.values() and "a_release" not in jobs.values()
    for _ in range(15):
        states.update(
            {
                j: ("FAILED", "1:0") if n == "a_contracts" else ("COMPLETED", "0:0")
                for j, n in jobs.items()
            }
        )
        result = tick()
        if result["all_terminal"]:
            break
    assert result["all_terminal"]
    assert {s["name"] for s in result["failures"]} == {"a_contracts"}
    assert next(s for s in steps if s["name"] == "c_pilot")["status"] == "completed"


def test_known_lustre_cpu_job_does_not_block_runtime_release(monkeypatch):
    monkeypatch.setattr(paths, "on_vsc", lambda: True)
    monkeypatch.setattr(
        cap.subprocess,
        "check_output",
        lambda argv, **kw: (
            "321|jevpfn-exp0-cpu|/data/JEVPFN\n"
            if argv[0] == "squeue"
            else "Command=/data/JEVPFN/scripts/slurm/experiment_0_cpu.slurm SubmitLine=sbatch script"
        ),
    )
    cap.require_runtime_idle("intern")


def test_publication_failure_blocks_new_staging_and_preserves_other_branches():
    steps = add_releases(
        [
            dict(name="a_scratch", dependencies=[], cluster="wice", job_id="1", status="completed"),
            dict(name="b_scratch", dependencies=[], cluster="wice"),
        ]
    )
    steps[-2].update(status="failed", job_id="2")
    result = job_graph.advance(
        steps,
        settings=load_yaml("experiment_0/scheduling"),
        state=lambda *a, **k: None,
        submit=lambda s: s.update(status="submitted", job_id="3"),
        save=lambda: None,
    )
    assert next(s for s in steps if s["name"] == "b_scratch")["status"] == "blocked"
    assert result["failures"][0]["name"] == "a_release"


def test_cloudflare_release_waits_for_all_encoder_consumers():
    from src.cluster.experiment_zero import steps

    graph = steps("vsc")
    release = next(s for s in graph if s["name"] == "cloudflare_release")
    assert {s["name"] for s in graph if s["name"].startswith("embeddings_")} <= set(
        release["terminal_dependencies"]
    )
    assert "cloudflare_pilot" in release["terminal_dependencies"]


def test_missing_file_inside_existing_durable_tree_prevents_release(roots):
    durable, work = roots
    copied_assets(durable, work)
    (durable / "checkpoints/model.bin").unlink()
    with pytest.raises(ValueError, match="Missing durable input file"):
        cap.release("intern", apply=True)
    assert (work / "checkpoints/model.bin").exists()


def test_filesystem_capacity_guard_is_independent_of_quota_budget(roots, monkeypatch):
    durable, work = roots
    marker(durable, work)
    monkeypatch.setattr(cap.shutil, "disk_usage", lambda p: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="filesystem"):
        cap.check_capacity(work, {}, durable / "containers/blackwell-cu130/intern")


@pytest.mark.parametrize("filename", ["__init__.py", "modeling_decision.py", "module.pyc"])
def test_generated_hf_module_cache_needs_no_durable_copy(roots, filename):
    durable, work = roots
    copied_assets(durable, work)
    generated = work / "checkpoints/huggingface/modules/transformers_modules/decision3" / filename
    generated.parent.mkdir(parents=True)
    generated.write_bytes(b"generated import cache")
    assert cap.release("intern")["status"] == "preview"
    assert generated.exists()
    assert cap.release("intern", apply=True)["applied"]
    assert not generated.exists()
    assert (durable / "checkpoints/model.bin").exists()


@pytest.mark.parametrize(
    "relative",
    [
        "checkpoints/huggingface/modules/model.safetensors",
        "checkpoints/decision_models/intern/modeling_decision.py",
        "data/raw/huggingface/modules/__init__.py",
    ],
)
def test_hf_cache_exception_does_not_bypass_original_asset_checks(roots, relative):
    durable, work = roots
    copied_assets(durable, work)
    original = work / relative
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_text("only scratch copy")
    with pytest.raises(ValueError, match="Missing durable input file"):
        cap.release("intern", apply=True)
    assert original.exists()
    assert (work / "checkpoints/model.bin").exists()
