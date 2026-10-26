"""Preview, then explicitly sign matching open-provider Experiment 0A evidence.

This records an owner's review; it never submits jobs, enables extraction, or
claims that a passing finite test suite proves the absence of all bugs.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from src.cluster.token_audit import selected_inputs
from src.jev.registry import SELF_HOSTED
from src.utils import paths
from src.utils.serialization import digest, file_sha256, write_json

PROVIDERS = SELF_HOSTED


def choose(files, predicate):
    matches = []
    for path in files:
        body = json.loads(path.read_text(encoding="utf-8"))
        if predicate(body):
            matches.append((path, body))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one matching successful report, found {len(matches)}; review the output campaign"
        )
    return matches[0]


def evidence_for(provider, *, campaign=None):
    from src.cluster.evidence import current_identity

    identity = current_identity()
    jobs = {}
    campaign_path = Path(campaign) if campaign else None
    if campaign_path:
        receipt = json.loads(campaign_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "vsc_checks_completed_requires_review":
            raise ValueError("Select a completed 0A_VSC campaign receipt")
        from src.cluster.experiment_zero import steps
        from src.cluster.freeze import check_if_present, runtime_pins

        expected = digest(
            {"source": check_if_present(), "runtime_pins": runtime_pins(), "steps": steps("vsc")}
        )
        if receipt.get("stage_identity") != expected:
            raise ValueError("Selected campaign does not match current frozen source/runtime")
        jobs = {
            s["name"]: str(s["job_id"]) for s in receipt["steps"] if s.get("status") == "completed"
        }

    def matching(report, step):
        return report.get("validation_identity") == identity and (
            not campaign_path or str(report.get("job_id")) == jobs.get(step)
        )

    root = paths.manifests_dir("experiment_0")
    before = root / "before_extraction"
    current = {
        i["dataset_id"]: (i["input_set_id"], i["input_file_sha256"]) for i in selected_inputs()
    }
    token_file, token = choose(
        before.glob(f"tokens_{provider}_*.json"),
        lambda r: (
            r.get("status") == "counted_all_inputs"
            and r.get("overflow_unique_inputs") == 0
            and {
                d["dataset_id"]: (d["input_set_id"], d["input_file_sha256"]) for d in r["datasets"]
            }
            == current
        ),
    )
    gpu_file, gpu = choose(
        (before / "gpu_contracts").glob(f"{provider}_*.json"),
        lambda r: (
            r.get("status") == "gpu_contracts_passed_requires_review"
            and matching(r, provider + "_contracts")
            and r.get("token_audit_id") == token["audit_id"]
            and r.get("runtime_token_count_parity") is True
            and r.get("semantic_diagnostics", {}).get("passed") is True
            and r.get("prompt_robustness", {}).get("structurally_valid") is True
            and r.get("real_prompt_robustness", {}).get("structurally_valid") is True
            and r.get("batch_parity", {}).get("passed") is True
            and {d["dataset_id"] for d in r["real_datasets"]} == set(current)
        ),
    )
    if digest(gpu["settings"]) != gpu["settings_sha256"]:
        raise ValueError("GPU runtime identity is corrupt")
    pilot_file, pilot = choose(
        before.glob(f"{provider}_*.json"),
        lambda r: (
            r.get("status") == "completed_requires_review"
            and matching(r, provider + "_pilot")
            and r.get("mode") == "pilot"
            and r.get("token_audit_id") == token["audit_id"]
            and r.get("settings_sha256") == gpu["settings_sha256"]
            and {d["dataset_id"] for d in r["datasets"]} == set(current)
            and all(d.get("failed_requests") == 0 for d in r["datasets"])
        ),
    )
    if digest(pilot["settings"]) != gpu["settings_sha256"]:
        raise ValueError("Pilot runtime identity is corrupt")
    cpu_file, _ = choose(
        root.glob("preflight_*.json"),
        lambda r: (
            r.get("status") == "preflight_passed"
            and matching(r, "cpu")
            and r.get("execution_site") == "vsc_compute_node"
            and r.get("stage") == "before_extraction"
            and r.get("contract_tests", {}).get("status") == "passed"
            and {d["dataset_id"] for d in r["datasets"]} == set(current)
        ),
    )
    # Both filesystems participate in B200 extraction: native GPFS for active
    # writes and Lustre for durable publication. A pass on one cannot certify the other.
    from src.cluster.scheduling import gpu_profile

    storage_files = []
    backends = [("lustre", "storage")]
    if gpu_profile().get("storage_mode") == "scratch_mirror":
        backends.append(("gpfs", "gpfs_storage" if provider == "cloudflare" else provider + "_gpfs_storage"))
    for backend, step in backends:
        storage_file, _ = choose(
            (before / "storage_contracts").glob("*.json"),
            lambda r, backend=backend, step=step: (
                r.get("status") == "storage_contracts_passed_requires_quota_and_backup_review"
                and matching(r, step)
                and r.get("storage_backend") == backend
                and (backend != "gpfs" or r.get("runtime") == provider)
                and r.get("snapshot_restore") is True
            ),
        )
        storage_files.append(storage_file)
    question_file = root / "question_audit.json"
    question = json.loads(question_file.read_text(encoding="utf-8"))
    if (
        question.get("status") != "mechanical_checks_passed_requires_semantic_review"
        or {d["dataset_id"]: (d["input_set_id"], d["file_sha256"]) for d in question["datasets"]}
        != current
    ):
        raise ValueError("Exhaustive question audit has not passed")
    return {
        "stage": "before_extraction",
        "provider": provider,
        "settings_sha256": gpu["settings_sha256"],
        "token_audit_id": token["audit_id"],
        "input_sets": {k: v[0] for k, v in current.items()},
        "validation_identity": identity,
        "evidence": [
            {"path": str(p.resolve()), "sha256": file_sha256(p)}
            for p in (
                cpu_file,
                *storage_files,
                question_file,
                token_file,
                gpu_file,
                pilot_file,
                *((campaign_path,) if campaign_path else ()),
            )
        ],
    }


def approve(
    provider,
    *,
    reviewer,
    quota_verified=False,
    backup_verified=False,
    throughput_reviewed=False,
    apply=False,
    campaign=None,
    robustness_reviewed=False,
):
    report = evidence_for(provider, campaign=campaign)
    report.update(
        accepted=False,
        reviewed_by=reviewer,
        quota_verified=quota_verified,
        backup_verified=backup_verified,
        throughput_reviewed=throughput_reviewed,
        robustness_reviewed=robustness_reviewed,
    )
    if apply:
        if not reviewer or not all(
            (quota_verified, backup_verified, throughput_reviewed, robustness_reviewed)
        ):
            raise ValueError(
                "Approval requires a named reviewer and explicit quota, backup, throughput and robustness review"
            )
        report.update(accepted=True, reviewed_utc=datetime.now(UTC).isoformat())
        destination = (
            paths.manifests_dir("experiment_0") / "before_extraction" / f"accepted_{provider}.json"
        )
        if destination.exists():
            raise ValueError("Acceptance already exists; do not silently replace a signed review")
        write_json(destination, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=PROVIDERS, required=True)
    parser.add_argument("--reviewed-by")
    parser.add_argument("--quota-verified", action="store_true")
    parser.add_argument("--backup-verified", action="store_true")
    parser.add_argument("--throughput-reviewed", action="store_true")
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--robustness-reviewed", action="store_true")
    parser.add_argument(
        "--campaign",
        type=Path,
        help="Exact completed 0A_VSC receipt; disambiguates successful reruns",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            approve(
                args.provider,
                reviewer=args.reviewed_by,
                quota_verified=args.quota_verified,
                backup_verified=args.backup_verified,
                throughput_reviewed=args.throughput_reviewed,
                apply=args.approve,
                campaign=args.campaign,
                robustness_reviewed=args.robustness_reviewed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
