"""No Slurm/network/CUDA: exercise failures, dependencies and pool starvation."""

from copy import deepcopy

import pytest

from src.cluster import diagnostics, job_graph, setup_extractors
from src.jev.registry import SELF_HOSTED
from src.utils.config import load_yaml


def test_all_provider_graph_finishes_independent_branches_after_multiple_failures():
    steps = setup_extractors.plan(list(SELF_HOSTED))
    settings = load_yaml("experiment_0/scheduling")
    jobs, states = {}, {}
    failures = {"liquid_stage", "sii_build", "cloudflare_contracts"}

    def submit(step):
        assert set(job_graph.dependencies(step)) <= {s["name"] for s in steps if s.get("status") == "completed"}
        job = str(len(jobs) + 1)
        jobs[job] = step["name"]
        step.update(job_id=job, status="submitted")

    for _ in range(80):
        report = job_graph.advance(steps, settings=settings, state=lambda j, **k: states.get(j),
                                   submit=submit, save=lambda: None)
        for pool, limit in settings["test_pool_limits"].items():
            active = [s for s in steps if s.get("job_id") and s.get("status") not in job_graph.TERMINAL and job_graph.pool(s) == pool]
            assert len(active) <= limit
        if report["all_terminal"]:
            break
        # Every step gets a terminal outcome; failed steps are never resubmitted.
        states.update({j: ("FAILED", "1:0") if name in failures else ("COMPLETED", "0:0") for j, name in jobs.items()})
    assert report["all_terminal"]
    assert {s["name"] for s in report["failures"]} == failures
    by_name = {s["name"]: s for s in steps}
    assert by_name["strands_pilot"]["status"] == "completed"
    assert by_name["liquid_pilot"]["status"] == "blocked"
    assert by_name["sii_pilot"]["status"] == "blocked"
    assert all(by_name["embeddings_" + e]["status"] == "completed"
               for e in load_yaml("feature_creation/embedding_models")["models"])
    assert len(jobs) == len(set(jobs.values()))


def test_long_cpu_audits_do_not_starve_ready_gpu_or_builds():
    steps = [dict(name=f"m{i}_tokens", dependencies=[], cluster="wice", status="submitted", job_id=str(i)) for i in range(10)]
    steps += [dict(name="ready_contracts", dependencies=[], cluster="mindwell"),
              dict(name="ready_build", dependencies=[], cluster="wice")]
    submitted = []
    job_graph.advance(steps, settings=load_yaml("experiment_0/scheduling"), state=lambda *a, **k: None,
                      submit=lambda s: submitted.append(s["name"]), save=lambda: None)
    assert submitted == ["ready_contracts", "ready_build"]


@pytest.mark.parametrize("bad", ["unknown", "cycle", "duplicate", "limit"])
def test_invalid_graph_rejected_before_accounting_or_submitting(bad):
    steps = [dict(name="a", dependencies=[]), dict(name="b", dependencies=["a"])]
    cfg = deepcopy(load_yaml("experiment_0/scheduling"))
    if bad == "unknown":
        steps[0]["dependencies"] = ["absent"]
    elif bad == "cycle":
        steps[0]["dependencies"] = ["b"]
    elif bad == "duplicate":
        steps[1]["name"] = "a"
    else:
        cfg["test_pool_limits"]["gpu"] = 0
    with pytest.raises(ValueError):
        job_graph.advance(steps, settings=cfg, state=lambda *a, **k: pytest.fail(),
                          submit=lambda s: pytest.fail(), save=lambda: None)


def test_gpu_diagnostics_collect_then_fail_but_poisoned_cuda_stops():
    report, snapshots = {}, []
    def save():
        snapshots.append(deepcopy(report))
    assert diagnostics.check(report, "semantics", lambda: {"passed": False}, save=save) is None
    assert diagnostics.check(report, "next", lambda: {"passed": True}, save=save)["passed"]
    with pytest.raises(RuntimeError, match="CUDA error"):
        diagnostics.check(report, "device", lambda: (_ for _ in ()).throw(RuntimeError("CUDA error: device-side assert")), save=save)
    assert [s["status"] for s in report["checks"]] == ["failed", "passed", "failed"]
    assert report["checks"][-1]["runtime_unusable"]
    assert len(snapshots) == 3 and all(c["seconds"] >= 0 for c in report["checks"])
