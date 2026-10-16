"""Preview, then explicitly sign matching open-provider Experiment 0A evidence.

This records an owner's review; it never submits jobs, enables extraction, or
claims that a passing finite test suite proves the absence of all bugs.
"""

import argparse
import json
from datetime import UTC, datetime

from src.cluster.token_audit import selected_inputs
from src.utils import paths
from src.utils.serialization import digest, file_sha256, write_json

PROVIDERS = ("cloudflare", "intern", "liquid", "strands", "perplexity")


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


def evidence_for(provider):
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
            and r.get("token_audit_id") == token["audit_id"]
            and r.get("runtime_token_count_parity") is True
            and r.get("semantic_diagnostics", {}).get("passed") is True
            and {d["dataset_id"] for d in r["real_datasets"]} == set(current)
        ),
    )
    if digest(gpu["settings"]) != gpu["settings_sha256"]:
        raise ValueError("GPU runtime identity is corrupt")
    pilot_file, pilot = choose(
        before.glob(f"{provider}_*.json"),
        lambda r: (
            r.get("status") == "completed_requires_review"
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
            and r.get("execution_site") == "vsc_compute_node"
            and r.get("stage") == "before_extraction"
            and r.get("contract_tests", {}).get("status") == "passed"
            and {d["dataset_id"] for d in r["datasets"]} == set(current)
        ),
    )
    storage_file, _ = choose(
        (before / "storage_contracts").glob("*.json"),
        lambda r: (
            r.get("status") == "storage_contracts_passed_requires_quota_and_backup_review"
            and r.get("snapshot_restore") is True
        ),
    )
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
        "evidence": [
            {"path": str(p.resolve()), "sha256": file_sha256(p)}
            for p in (cpu_file, storage_file, question_file, token_file, gpu_file, pilot_file)
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
):
    report = evidence_for(provider)
    report.update(
        accepted=False,
        reviewed_by=reviewer,
        quota_verified=quota_verified,
        backup_verified=backup_verified,
        throughput_reviewed=throughput_reviewed,
    )
    if apply:
        if not reviewer or not all((quota_verified, backup_verified, throughput_reviewed)):
            raise ValueError(
                "Approval requires a named reviewer and explicit quota, backup and throughput review"
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
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
