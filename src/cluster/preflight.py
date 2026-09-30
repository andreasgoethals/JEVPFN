"""Run inside a VSC allocation: python -m src.cluster.preflight [--network] [--gpu].

The network probe sends one credential-free HEAD request, never an inference POST.
Passing it establishes HTTPS reachability, not authentication or permission for a bulk job.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime

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


def run_preflight(
    *, network: bool = False, gpu: bool = False, config: str = "experiment_0/debug"
) -> dict:
    if not paths.on_vsc() or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Experiment 0 must run inside a VSC compute allocation.")
    cfg = load_yaml(config)
    os.environ["JEVPFN_PHASE"] = "experiment_0"
    report = {
        "phase": "experiment_0_preflight",
        "created_at": datetime.now(UTC).isoformat(),
        "job_id": os.environ["SLURM_JOB_ID"],
        "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
        "node": platform.node(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "paths": paths.describe(),
        "packages": {},
        "failures": [],
        "full_experiment_0_passed": False,
    }
    if sys.version_info[:2] != (3, 12):
        report["failures"].append("Use Python 3.12 on both local and VSC environments.")
    for name in ("numpy", "pandas", "pyarrow", "pytest", "tabpfn", "torch", "typesafe-sdk"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    for label, directory in {
        "small_outputs": paths.outputs_dir(),
        "large_outputs": paths.large_outputs_dir(),
        "raw_data": paths.raw_dir(),
        "feature_cache": paths.jev_cache_path().parent,
    }.items():
        try:
            paths.resolve_writable(directory)
        except OSError as exc:
            report["failures"].append(f"{label}: storage unavailable ({type(exc).__name__})")
    if paths.outputs_dir().resolve() == paths.large_outputs_dir().resolve():
        report["failures"].append("Personal DATA and project output storage must differ on VSC.")
    if network:
        report["network"] = network_probe()
        if not report["network"]["reachable"]:
            report["failures"].append("API HTTPS reachability was not established from this node.")
    if gpu:
        try:
            import torch
            from tabpfn.constants import ModelVersion

            if ModelVersion.V3_5.value != "v3.5" or not torch.cuda.is_available():
                raise RuntimeError("TabPFN-3.5 or CUDA is unavailable")
            result = torch.ones((2, 2), device="cuda") @ torch.ones((2, 2), device="cuda")
            torch.cuda.synchronize()
            if result.sum().item() != 8:
                raise RuntimeError("CUDA arithmetic check failed")
            report["gpu"] = {
                "name": torch.cuda.get_device_name(),
                "torch_cuda": torch.version.cuda,
                "status": "tensor_smoke_passed",
                "model_weights_loaded": False,
            }
        except (ImportError, RuntimeError, AttributeError) as exc:
            report["failures"].append(f"GPU/package preflight: {type(exc).__name__}")
    report["feature_artifacts"] = []
    for relative in cfg["feature_artifacts"]:
        block = load_block(paths.jev_cache_path(relative))
        report["feature_artifacts"].append(
            {"artifact_id": block.manifest["artifact_id"], "rows": len(block.values)}
        )
    report["feature_validation"] = (
        "validated_listed_artifacts"
        if cfg["feature_artifacts"]
        else "pending_complete_local_features"
    )
    report["status"] = "failed" if report["failures"] else "preflight_passed"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    report = run_preflight(network=args.network, gpu=args.gpu)
    write_json(paths.manifests_dir("experiment_0") / f"preflight_{report['job_id']}.json", report)
    print(json.dumps(report, indent=2))
    return int(bool(report["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
