"""Offline local/compute checks: python -m src.cluster.preflight --local --verify-data.

The network probe sends one credential-free HEAD request, never an inference POST.
Passing it establishes HTTPS reachability, not authentication or permission for a bulk job.
Omit --local inside a VSC compute allocation. --tests also runs deterministic contracts.
Neither a local check nor a compute preflight establishes full experiment-0 acceptance.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from src.data.loaders import load_catalog, load_dataset
from src.data.selection import selected_entries
from src.jev.features import load_block
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import write_json


def network_probe(timeout: float = 15) -> dict:
    host = "api.typesafe.ai"
    result = {"host": host, "method": "HEAD", "credentials_sent": False, "inference_sent": False}
    try:
        socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        result["dns"] = "ok"
        request = urllib.request.Request(f"https://{host}/v1/systemone", method="HEAD")
        # Do not follow redirects: a response from another host cannot establish API reachability.
        opener = urllib.request.build_opener(_NoRedirect())
        try:
            with opener.open(request, timeout=timeout) as response:
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        result.update(
            https_status=status, reachable=status in {200, 204, 400, 401, 403, 404, 405, 422, 429}
        )
        result["interpretation"] = (
            "DNS/TLS/HTTP response only; verify account authentication and site policy separately."
        )
    except (OSError, urllib.error.URLError) as exc:
        result.update(reachable=False, error_type=type(exc).__name__)
    return result


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def storage_policy_failures(cluster: str | None = None) -> list[str]:
    """Check tier separation/native storage without writing to protected inputs."""
    if not paths.on_vsc():
        return []
    failures = []
    project = paths.staging_root().resolve()
    personal = paths.data_root().resolve()
    if project == personal or project.is_relative_to(personal):
        failures.append("Large files must use PROJECT outside personal DATA.")
    if not project.is_dir():
        failures.append(
            "The project allocation parent must already exist; verify its mount/access."
        )
    # Site paths identify the native filesystem when available. Never infer that a
    # cross-mounted allocation is suitable merely because it is readable/writable.
    location = str(project).lower()
    if cluster == "mindwell" and "lustre" in location:
        failures.append("Mindwell requires native GPFS project storage, not cross-mounted Lustre.")
    if cluster == "wice" and "gpfs" in location:
        failures.append("wICE requires native Lustre project storage, not cross-mounted GPFS.")
    native_variable = {"wice": "VSC_PROJECT_LUSTRE1", "mindwell": "VSC_PROJECT_GPFS1"}.get(cluster)
    if native_variable and os.environ.get(native_variable):
        native = Path(os.environ[native_variable]).resolve()
        if not project.is_relative_to(native):
            failures.append(f"Project storage is outside the site's {native_variable} root.")
    return failures


def dataset_checks(cfg: dict) -> list[dict]:
    """Load selected cached data, verify hashes and construct CV partitions without fitting."""
    from src.evaluation.protocol import shared_splits, validate_splits

    results = []
    for entry in selected_entries(load_catalog(), cfg):
        item = {"dataset_id": entry["id"], "status": "failed"}
        try:
            dataset = load_dataset(entry, allow_download=False)
            splits = shared_splits(dataset)
            validate_splits(dataset, splits)
            if dataset.metadata.task_type == "binary_classification":
                target = dataset.frame[dataset.metadata.target]
                for fold in splits["folds"]:
                    if target.iloc[fold["train"]].value_counts().min() < 3:
                        raise ValueError("Insufficient binary inner-fold class coverage")
            item.update(
                status="passed",
                rows=len(dataset.frame),
                text_columns=len(dataset.metadata.text_columns),
                task_type=dataset.metadata.task_type,
                data_identity=dataset.source["data_identity"],
                task_schema_sha256=dataset.metadata.fingerprint,
                split_id=splits["split_id"],
                outer_folds=len(splits["folds"]),
                issues=dataset.issues,
            )
        except Exception as exc:
            # Report the dataset failure and continue, without dumping raw records.
            item["error_type"] = type(exc).__name__
            item["error"] = str(exc)
        results.append(item)
    return results


def _validate_artifacts(cfg: dict, report: dict) -> None:
    catalog = {entry["id"]: entry for entry in selected_entries(load_catalog(), cfg)}
    for reference in cfg.get("feature_artifacts", []):
        item = {"reference": reference, "status": "failed"}
        try:
            if not isinstance(reference, dict) or set(reference) != {"dataset_id", "artifact_id"}:
                raise ValueError("Feature references require dataset_id and explicit artifact_id")
            dataset_id = reference["dataset_id"]
            if dataset_id not in catalog:
                raise ValueError("Feature artifact dataset is excluded or unknown")
            block = load_block(paths.dataset_cache_path(dataset_id), reference["artifact_id"])
            dataset = load_dataset(catalog[dataset_id], allow_download=False)
            identity = block.manifest["identity"]
            if (
                identity["dataset_id"] != dataset_id
                or identity["data_identity"] != dataset.source["data_identity"]
                or identity["task_schema_sha256"] != dataset.metadata.fingerprint
                or set(block.values.row_id) != {f"csv:{i}" for i in range(len(dataset.frame))}
            ):
                raise ValueError("Feature rows/source identity differ from the current raw dataset")
            item.update(
                status="passed", artifact_id=block.manifest["artifact_id"], rows=len(block.values)
            )
        except Exception as exc:
            item.update(error_type=type(exc).__name__, error=str(exc))
            report["failures"].append(f"Feature artifact validation failed: {type(exc).__name__}")
        report["feature_artifacts"].append(item)


def contract_tests(run_id: str) -> dict:
    """Exercise synthetic tests only, preserving the exact test process outcome in DATA logs."""
    command = [sys.executable, "-m", "pytest", "-q"]
    if paths.on_vsc():
        command += ["-m", "not notebook"]
    result = subprocess.run(
        command,
        cwd=paths.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    log = paths.logs_dir("experiment_0") / f"contracts_{run_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as handle:
        handle.write(result.stdout + result.stderr)
    return {
        "status": "passed" if result.returncode == 0 else "failed",
        "exit_code": result.returncode,
        "log": str(log),
        "excluded_markers": ["notebook"] if paths.on_vsc() else [],
        "scope": "Synthetic contracts; no evidence of real provider/model/cluster compatibility.",
    }


def run_preflight(
    *,
    network: bool = False,
    gpu: bool = False,
    local: bool = False,
    verify_data: bool = False,
    tests: bool = False,
    stage: str = "before_extraction",
    predictor_imports: bool = False,
    config: str = "experiment_0/debug",
) -> dict:
    if local and (paths.on_vsc() or os.environ.get("SLURM_JOB_ID")):
        raise RuntimeError("--local must run off VSC; use a VSC compute allocation there.")
    if not local and (not paths.on_vsc() or not os.environ.get("SLURM_JOB_ID")):
        raise RuntimeError("Experiment 0 must run inside a VSC compute allocation.")
    cfg = load_yaml(config)
    if stage not in cfg["stages"]:
        raise ValueError("Unknown Experiment 0 stage")
    os.environ["JEVPFN_PHASE"] = "experiment_0"
    report = {
        "phase": "experiment_0_preflight",
        "stage": stage,
        "run_id": uuid4().hex,
        "execution_site": "local" if local else "vsc_compute_node",
        "created_at": datetime.now(UTC).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
        "node": platform.node(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "paths": paths.describe(),
        "packages": {},
        "failures": [],
        "full_experiment_0_passed": False,
        "pending_acceptance": cfg["acceptance_checks"],
        "feature_artifacts": [],
        "datasets": [],
    }
    if sys.version_info[:2] != (3, 12):
        report["failures"].append("Use Python 3.12 on both local and VSC environments.")
    required_packages = ("numpy", "pandas", "pyarrow", "pytest", "PyYAML", "scikit-learn")
    for name in (*required_packages, "tabpfn", "torch", "typesafe-sdk"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
            if name in required_packages:
                report["failures"].append(f"Missing offline preflight dependency: {name}")
    report["failures"].extend(storage_policy_failures(report["cluster"]))
    report["stage_pending_acceptance"] = [
        c for c in cfg["acceptance_checks"] if c["id"] in cfg["stages"][stage]["acceptance_ids"]
    ]
    for label, directory in {
        "small_outputs": paths.outputs_dir(),
        "large_outputs": paths.large_outputs_dir(),
    }.items():
        try:
            if label == "large_outputs" and not local and report["failures"]:
                # Do not create a directory under a missing/misconfigured project mount.
                continue
            paths.resolve_writable(directory)
        except OSError as exc:
            report["failures"].append(f"{label}: storage unavailable ({type(exc).__name__})")
    report["raw_directory_exists"] = paths.raw_dir().is_dir()
    report["feature_cache_exists"] = paths.jev_cache_path().is_dir()
    report["storage_capacity"] = {}
    for label, root in {"DATA": paths.data_root(), "PROJECT": paths.staging_root()}.items():
        if root.is_dir():
            try:
                report["storage_capacity"][label] = {
                    "filesystem_free_bytes": shutil.disk_usage(root).free,
                    "quota_verified": False,
                    "backup_verified": False,
                }
            except OSError as exc:
                report["failures"].append(f"{label}: capacity unavailable ({type(exc).__name__})")
    if not local and paths.outputs_dir().resolve() == paths.large_outputs_dir().resolve():
        report["failures"].append("Personal DATA and project output storage must differ on VSC.")
    if network:
        report["network"] = network_probe()
        if not report["network"]["reachable"]:
            report["failures"].append("API HTTPS reachability was not established from this node.")
    if gpu:
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            if predictor_imports:
                from tabpfn.constants import ModelVersion

                if ModelVersion.V3_5.value != "v3.5" or ModelVersion.V3.value != "v3":
                    raise RuntimeError("TabPFN-3/3.5 is unavailable")
            result = torch.ones((2, 2), device="cuda") @ torch.ones((2, 2), device="cuda")
            torch.cuda.synchronize()
            if result.sum().item() != 8:
                raise RuntimeError("CUDA arithmetic check failed")
            report["gpu"] = {
                "name": torch.cuda.get_device_name(),
                "torch_cuda": torch.version.cuda,
                "capability": list(torch.cuda.get_device_capability()),
                "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "status": "tensor_smoke_passed",
                "model_weights_loaded": False,
            }
        except (ImportError, RuntimeError, AttributeError, OSError) as exc:
            report["failures"].append(f"GPU/package preflight: {type(exc).__name__}")
    if verify_data:
        report["datasets"] = dataset_checks(cfg)
        failed = sum(item["status"] != "passed" for item in report["datasets"])
        if failed:
            report["failures"].append(
                f"{failed} selected dataset(s) failed offline data/CV checks."
            )
    _validate_artifacts(cfg, report)
    if stage == "after_extraction" and not cfg["feature_artifacts"]:
        report["failures"].append(
            "Experiment 0B requires explicit complete feature artifact selections."
        )
    report["feature_validation"] = (
        "see_per_artifact_results_not_full_campaign_acceptance"
        if cfg["feature_artifacts"]
        else "not_required_before_extraction"
        if stage == "before_extraction"
        else "pending_complete_features"
    )
    if tests:
        report["contract_tests"] = contract_tests(report["run_id"])
        if report["contract_tests"]["status"] != "passed":
            report["failures"].append("Offline contract tests failed; inspect the saved log.")
    report["performed_checks"] = {
        "package_metadata_and_output_writes": True,
        "selected_dataset_hashes_schema_and_cv_feasibility": verify_data,
        "synthetic_contract_suite": tests,
        "unauthenticated_https_head": network,
        "cuda_tensor_and_tabpfn_import": gpu and predictor_imports,
        "cuda_tensor": gpu,
        "model_weights_loaded": False,
        "live_decision_inference": False,
        "predictive_models_fitted": False,
    }
    report["status"] = "failed" if report["failures"] else "preflight_passed"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--local", action="store_true", help="Offline preparation on a local host")
    parser.add_argument(
        "--verify-data", action="store_true", help="Read cached data; never download"
    )
    parser.add_argument("--tests", action="store_true", help="Run the deterministic pytest suite")
    parser.add_argument(
        "--stage", choices=["before_extraction", "after_extraction"], default="before_extraction"
    )
    parser.add_argument(
        "--predictor-imports",
        action="store_true",
        help="Also import TabPFN in its own model environment",
    )
    args = parser.parse_args()
    report = run_preflight(
        network=args.network,
        gpu=args.gpu,
        local=args.local,
        verify_data=args.verify_data,
        tests=args.tests,
        stage=args.stage,
        predictor_imports=args.predictor_imports,
    )
    write_json(paths.manifests_dir("experiment_0") / f"preflight_{report['run_id']}.json", report)
    print(json.dumps(report, indent=2))
    tests = report.get("contract_tests", {})
    if tests.get("status") == "failed":
        print(f"\nContract tests failed. Full log: {tests['log']}", flush=True)
        print(
            "\n".join(Path(tests["log"]).read_text(encoding="utf-8").splitlines()[-80:]), flush=True
        )
    return int(bool(report["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
