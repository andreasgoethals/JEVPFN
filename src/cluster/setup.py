"""Create/probe the two JEVPFN storage trees from a VSC login shell; no jobs or installs.

python -m src.cluster.setup --cluster wice
The allocation parent must already exist. This never creates a missing mount, changes Git,
downloads weights, deletes data or claims to verify a quota/backup policy.
"""

import argparse
import json
import os
from datetime import UTC, datetime

from src.cluster.preflight import storage_policy_failures
from src.utils import paths
from src.utils.serialization import write_json


def setup_storage(cluster):
    if not paths.on_vsc():
        raise RuntimeError("Run storage setup from the VSC Open OnDemand login shell")
    failures = storage_policy_failures(cluster)
    if failures:
        raise ValueError("; ".join(failures))
    directories = {
        "raw_datasets": paths.raw_dir(),
        "prepared_questions": paths.decision_inputs_dir(),
        "permanent_features": paths.jev_cache_path(),
        "checkpoints": paths.checkpoints_dir(),
        "token_counts": paths.token_audit_dir(),
        "small_manifests": paths.manifests_dir("experiment_0") / "before_extraction",
        "small_logs": paths.logs_dir("experiment_0"),
        "feature_creation_logs": paths.logs_dir("feature_creation"),
    }
    access_verified = {}
    for name, directory in directories.items():
        if name not in {"small_manifests", "small_logs", "feature_creation_logs"}:
            paths.require_project_output(directory)
        access = "directory_read" if name == "raw_datasets" else "write"
        try:
            if name == "raw_datasets":
                # Existing inputs may be immutable. Only create the destination on
                # first setup; file readability/hashes are checked by CPU preflight.
                if not directory.exists():
                    directory.mkdir(parents=True)
                with os.scandir(directory) as entries:
                    next(entries, None)
            else:
                paths.resolve_writable(directory)
        except PermissionError as exc:
            raise PermissionError(
                f"Storage setup cannot verify {name} ({access}) at {directory}: {exc}"
            ) from exc
        access_verified[name] = access
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "cluster": cluster,
        "paths": {k: str(v) for k, v in directories.items()},
        "access_verified": access_verified,
        "raw_dataset_contents_verified": False,
        "quota_verified": False,
        "backup_verified": False,
        "jobs_submitted": False,
    }
    write_json(
        paths.manifests_dir("experiment_0") / "before_extraction" / "storage_setup.json", report
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", choices=["wice", "mindwell"], default="wice")
    args = parser.parse_args()
    print(json.dumps(setup_storage(args.cluster), indent=2))


if __name__ == "__main__":
    main()
