"""Submit all ready feature jobs without changing a frozen inference checkout.

This operational companion can be deployed as a checksummed standalone file while
the original checkout has consumers. It imports that checkout's unchanged plan,
freezer, acceptance gates, scheduler queries and transactional graph. The admission
policy is recorded separately; existing job IDs and the original grid stay intact.
Native workspace/byte limits and the single bulk-transfer writer remain enforced.
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

# Standalone copies run from the protected DATA checkout through its CPU image.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path.cwd()))

from src.cluster.experiment_zero import scheduler_state
from src.cluster.feature_campaign import capacity_rejection, plan
from src.cluster.freeze import check_if_present, runtime_pins
from src.cluster.job_graph import SubmissionDeferred, advance, pool
from src.cluster.maintenance import require_idle_vsc
from src.jev.catalog import SELF_HOSTED
from src.utils import paths
from src.utils.acceptance import reviewed_acceptance
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import digest, file_sha256, write_json


def admission_settings(base, steps):
    """A finite plan is the ceiling; Slurm governs actual account admission."""
    sizes = Counter(pool(step) for step in steps)
    limits = dict(base["test_pool_limits"])
    for provider in SELF_HOSTED:
        limits[provider] = max(1, sizes[provider])
    limits["gpu"] = max(1, sizes["gpu"])
    return {**base, "max_active_jobs": len(steps), "test_pool_limits": limits}


def partition_encoders(steps, selected):
    """Split only explicitly unsubmitted encoder work; preserve its exact dataset set."""
    datasets = [s["command"][-1] for s in steps if s.get("pool") == SELF_HOSTED[0]]
    if len(datasets) != 18 or len(set(datasets)) != 18:
        raise ValueError("Encoder partitions require the frozen eighteen datasets")
    eligible = {s["name"] for s in steps if s["name"].startswith("embeddings_")}
    if set(selected) - eligible:
        raise ValueError("Unknown encoder partition group")
    replacements = {name: [name + "_" + dataset for dataset in datasets] for name in selected}
    expanded = []
    for step in copy.deepcopy(steps):
        if step["name"] not in replacements:
            expanded.append(step)
            continue
        if step.get("job_id") or step.get("status", "pending") != "pending":
            raise ValueError("Never partition an already submitted or terminal encoder job")
        for dataset, name in zip(datasets, replacements[step["name"]], strict=True):
            child = copy.deepcopy(step)
            child.update(name=name, encoder_group=step["name"], dataset=dataset)
            child["command"].append(dataset)  # Existing Slurm script's fourth argument.
            exports = [i for i, arg in enumerate(child["command"]) if arg.startswith("--export=")]
            assignment = "JEVPFN_PUBLISH_DATASET=" + dataset
            if exports:
                child["command"][exports[0]] += "," + assignment
            else:
                child["command"].insert(1, "--export=ALL," + assignment)
            expanded.append(child)
    for step in expanded:
        for key in ("depends_on", "dependencies", "terminal_dependencies"):
            if key in step:
                step[key] = [name for old in step[key] for name in replacements.get(old, [old])]
    return expanded


def same_grid(saved, expected):
    return len(saved) == len(expected) and all(
        all(old.get(k) == new.get(k) for k in (
            "name", "cluster", "command", "depends_on", "terminal_dependencies",
            "pool", "workspace_stage", "workspace_release",
        )) for old, new in zip(saved, expected, strict=True)
    )


def dispatch_order(steps):
    """Release finished work first; admit encoders at the next genuine storage slot."""
    def priority(step):
        if step.get("workspace_release"):
            return 0
        if step["name"].startswith("embeddings_"):
            return 1
        if step["name"] == "cloudflare_scratch":
            return 2
        if pool(step) in SELF_HOSTED:
            return 3
        return 4

    return sorted(steps, key=priority)  # Same step objects; receipt order/plan stays unchanged.


def submit_step(step, *, save, policy_sha256):
    """Persist intent first. Only a definite refusal may return to pending."""
    step.update(status="starting", submission_policy_sha256=policy_sha256,
                submission_started_utc=datetime.now(UTC).isoformat())
    save()
    try:
        result = subprocess.run(step["command"], cwd=paths.REPO_ROOT, capture_output=True,
                                text=True, timeout=60, check=True)
    except (subprocess.SubprocessError, OSError) as exc:
        diagnostic = {
            "error_type": type(exc).__name__,
            **{k: str(getattr(exc, k, None) or "")[-4096:] for k in ("stdout", "stderr")},
        }
        step["submission_diagnostics"] = diagnostic
        if isinstance(exc, subprocess.CalledProcessError) and capacity_rejection(
            diagnostic["stdout"], diagnostic["stderr"]
        ):
            step["status"] = "pending"
            save()
            raise SubmissionDeferred("Account submission ceiling; retry next poll") from exc
        save()
        raise
    step["submission_diagnostics"] = {k: getattr(result, k) for k in ("stdout", "stderr")}
    save()
    value = result.stdout.strip()
    if not re.fullmatch(r"[0-9]+(?:;" + step["cluster"] + r")?", value):
        raise ValueError("Ambiguous sbatch receipt; retain intent and inspect accounting")
    step.update(job_id=value.split(";")[0], status="submitted")
    save()
    print(f"Submitted {step['name']}: {step['job_id']} ({step['cluster']})", flush=True)


def recover_campaign(*, previous_identity, previous_freeze, expected_source, reviewed_by):
    """Rebind an idle expanded grid after a reviewed operational-only source repair."""
    require_idle_vsc()
    frozen = check_if_present()
    if not reviewed_by or frozen.get("source_sha256") != expected_source or not frozen.get("frozen"):
        raise ValueError("Recovery requires a named reviewer and the exact new frozen source")
    old_freeze = json.loads(Path(previous_freeze).read_text())["frozen_receipt"]
    from src.cluster.freeze import source_snapshot

    current = source_snapshot()
    changed = {k for k in old_freeze["files"].keys() | current["files"].keys()
               if old_freeze["files"].get(k) != current["files"].get(k)}
    allowed = {
        "AGENTS.md",  # Previously reviewed operational companion policy, no scientific inputs.
        "src/cluster/feature_admission.py", "src/cluster/scratch_stage.py",
        "src/cluster/scratch_capacity.py", "scripts/slurm/experiment_0_scratch_release.slurm",
        "tests/test_feature_admission.py", "tests/test_scratch_stage.py",
        "tests/test_scratch_capacity.py",
        "scripts/slurm/feature_recovery_test.slurm",
    }
    if changed - allowed or old_freeze["runtime_pins"] != runtime_pins():
        raise ValueError("Recovery cannot authorize changed science/configuration/runtime pins")
    receipt = paths.manifests_dir("feature_creation") / "open_weight_campaign.json"
    report = json.loads(receipt.read_text())
    if report["identity"] != previous_identity or not report.get("parallel_admission"):
        raise ValueError("Recovery requires the exact prior expanded campaign")
    if report["parallel_admission"]["policy"]["source_sha256"] != old_freeze["source_sha256"]:
        raise ValueError("Previous freeze does not belong to the campaign")
    original = plan()
    selected = report["parallel_admission"]["policy"]["split_encoders"]
    expanded = partition_encoders(original, selected)
    if not same_grid(report["origin_steps"], original) or not same_grid(report["steps"], expanded):
        raise ValueError("Recovery cannot change the original grid or encoder partitions")
    # Review ALL accepted IDs; queue absence alone never establishes completion.
    for old in report["steps"]:
        if old.get("status") == "starting" and not old.get("job_id"):
            raise ValueError("Reconcile ambiguous submission before recovery")
        if old.get("job_id"):
            result = scheduler_state(old["job_id"], cluster=old["cluster"])
            if result is None:
                raise ValueError("Recovery requires terminal accounting for every known ID")
            old.update(scheduler_state=result[0], exit_code=result[1],
                       status="completed" if result == ("COMPLETED", "0:0") else "failed")
    history = receipt.parent / "recovery_history" / f"before_source_recovery_{digest(report)[:16]}.json"
    write_json(history, report)
    for step in expanded:
        old = next(s for s in report["steps"] if s["name"] == step["name"])
        if old.get("status") == "completed":
            step.update(old)
        elif old.get("job_id") or old.get("previous_attempt"):
            step["previous_attempt"] = old
    base = load_yaml("experiment_0/scheduling")
    legacy = {**base, "test_pool_limits": {**base["test_pool_limits"], **dict.fromkeys(SELF_HOSTED, 2)}}
    identity = digest(dict(frozen=frozen, runtime_pins=runtime_pins(), steps=original, settings=legacy))
    policy = {**report["parallel_admission"]["policy"], "campaign_identity": identity,
              "source_sha256": expected_source, "dispatcher_sha256": file_sha256(Path(__file__)),
              "reviewed_by": reviewed_by, "settings": admission_settings(base, expanded)}
    policy_sha = digest(policy)
    execution = digest(dict(origin_identity=identity, policy_sha256=policy_sha,
                            plan=partition_encoders(original, selected)))
    write_json(receipt, {
        "identity": execution, "origin_identity": identity, "origin_steps": original,
        "steps": expanded, "status": "recovered_requires_dispatch",
        "parallel_admission": {"sha256": policy_sha, "policy": policy,
                               "approved_utc": datetime.now(UTC).isoformat(), "previous_receipt": str(history)},
        "recovery": {"reviewed_by": reviewed_by, "previous_receipt": str(history),
                     "previous_freeze": str(previous_freeze), "operational_files_changed": sorted(changed),
                     "created_utc": datetime.now(UTC).isoformat()},
    })
    return identity


def run(*, execute=False, expected_campaign=None, expected_source=None, reviewed_by=None):
    original_steps = plan()
    base = load_yaml("experiment_0/scheduling")
    legacy = {**base, "test_pool_limits": {**base["test_pool_limits"], **dict.fromkeys(SELF_HOSTED, 2)}}
    all_encoders = [s["name"] for s in original_steps if s["name"].startswith("embeddings_")]
    settings = admission_settings(base, partition_encoders(original_steps, all_encoders))
    if not execute:
        return {"status": "preview", "decision_jobs_per_prepared_runtime": 18,
                "encoder_jobs": 54, "max_resident_workspaces": settings["max_resident_workspaces"],
                "scratch_working_set_gib": settings["scratch_working_set_gib"],
                "embedding_stage_priority": "next_free_workspace", "total_steps": settings["max_active_jobs"]}
    if not paths.on_vsc() or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run the lightweight dispatcher on a VSC login node")
    if not all((expected_campaign, expected_source, reviewed_by)):
        raise ValueError("Explicit campaign, frozen source and named admission reviewer required")
    frozen = check_if_present()
    if frozen.get("source_sha256") != expected_source or not frozen.get("frozen"):
        raise ValueError("Dispatcher does not match the frozen inference source")
    identity = digest(dict(frozen=frozen, runtime_pins=runtime_pins(), steps=original_steps, settings=legacy))
    if identity != expected_campaign:
        raise ValueError("Frozen campaign plan/configuration/runtime differs from admission approval")
    cfg = load_yaml("experiment_0/self_hosted")
    if cfg["full_feature_creation_enabled"] is not True:
        raise ValueError("Full feature creation is disabled")
    for provider in SELF_HOSTED:
        reviewed_acceptance(cfg["accepted_before_extraction_manifests"].get(provider),
                            stage="before_extraction")
    from src.data.text_embeddings import encoder_config

    for encoder in load_yaml("feature_creation/embedding_models")["models"]:
        if encoder_config(encoder)["enabled"] is not True:
            raise ValueError("Encoder configuration is disabled")
    receipt = paths.manifests_dir("feature_creation") / "open_weight_campaign.json"
    report = json.loads(receipt.read_text())
    previous = report.get("parallel_admission")
    origin = report.get("origin_steps", report["steps"])
    if report.get("origin_identity", report["identity"]) != identity:
        raise ValueError("Existing campaign identity differs; never create a second grid")
    if not same_grid(origin, original_steps):
        raise ValueError("Receipt plan/dependencies differ from the frozen grid")
    if any(s.get("status") == "starting" and not s.get("job_id") for s in report["steps"]):
        raise ValueError("Interrupted submission has no job ID; inspect Slurm before retrying")
    selected = previous["policy"]["split_encoders"] if previous else [
        s["name"] for s in origin if s["name"].startswith("embeddings_")
        and not s.get("job_id") and s.get("status", "pending") == "pending"
    ]
    expanded = partition_encoders(origin, selected)
    settings = admission_settings(base, expanded)
    policy = {"schema": "jevpfn.feature-admission.v1", "campaign_identity": identity,
              "source_sha256": expected_source, "dispatcher_sha256": file_sha256(Path(__file__)),
              "reviewed_by": reviewed_by, "settings": settings, "split_encoders": selected,
              "priority": "release_then_embeddings_then_cloudflare_then_decision_jobs"}
    policy_sha256 = digest(policy)
    if previous is not None and previous.get("sha256") != policy_sha256:
        raise ValueError("Admission policy changed; review the existing dispatcher before replacement")
    execution_identity = digest(dict(origin_identity=identity, policy_sha256=policy_sha256,
                                     plan=partition_encoders(original_steps, selected)))
    if previous and (report["identity"] != execution_identity or not same_grid(report["steps"], expanded)):
        raise ValueError("Partitioned execution grid differs from its recorded approval")

    def save():
        write_json(receipt, report)

    if previous is None:
        history = receipt.parent / "recovery_history" / f"before_parallel_{digest(report)[:16]}.json"
        write_json(history, report)
        report.update(origin_identity=identity, origin_steps=copy.deepcopy(origin),
                      identity=execution_identity, steps=expanded)
        report["parallel_admission"] = {"sha256": policy_sha256, "policy": policy,
                                        "approved_utc": datetime.now(UTC).isoformat(),
                                        "previous_receipt": str(history)}
        save()
    try:
        result = advance(dispatch_order(report["steps"]), settings=settings,
                         state=scheduler_state, save=save,
                         submit=lambda step: submit_step(step, save=save, policy_sha256=policy_sha256))
    except BaseException as exc:
        report.update(status="stopped_requires_review", dispatcher_error={
            "error_type": type(exc).__name__, "message": str(exc)[-4096:]})
        save()
        raise
    report["summary"] = result
    report["status"] = ("completed_with_failures" if result["failures"] else
                        "completed_requires_feature_validation") if result["all_terminal"] else "running"
    report["admission_checked_utc"] = datetime.now(UTC).isoformat()
    save()
    return {"status": report["status"], **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--expected-campaign")
    parser.add_argument("--expected-source")
    parser.add_argument("--reviewed-by")
    parser.add_argument("--recover-from", help="Exact prior expanded identity; requires an idle checkout")
    parser.add_argument("--previous-freeze", type=Path, help="Archived completed-revision receipt")
    args = parser.parse_args()
    if args.watch and not args.execute:
        parser.error("--watch requires --execute")
    if args.recover_from and not (args.execute and args.previous_freeze and args.reviewed_by):
        parser.error("--recover-from requires --execute, --previous-freeze and --reviewed-by")
    if not args.execute:
        print(json.dumps(run()))
        return 0
    root = paths.manifests_dir("feature_creation")
    with file_lock(root / ".parallel-controller.lock", timeout=0):
        if args.recover_from:
            with file_lock(root / ".campaign.lock", timeout=45):
                args.expected_campaign = recover_campaign(
                    previous_identity=args.recover_from, previous_freeze=args.previous_freeze,
                    expected_source=args.expected_source, reviewed_by=args.reviewed_by)
        while True:
            with file_lock(root / ".campaign.lock", timeout=45):
                report = run(execute=True, expected_campaign=args.expected_campaign,
                             expected_source=args.expected_source, reviewed_by=args.reviewed_by)
            print(json.dumps(report), flush=True)
            if not args.watch or report["status"] != "running":
                return int(report["status"] == "completed_with_failures")
            time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
