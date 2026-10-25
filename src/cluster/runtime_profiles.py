"""Hardware-specific runtime identity and a bounded Blackwell kernel check.

CUDA wheel selection is explicit; this module never installs dependencies. The
GPU check establishes basic kernel execution, not model/checkpoint compatibility.
"""

import argparse
import json
import os
import re
import subprocess
from urllib.parse import unquote
from urllib.request import urlopen

LEGACY = "legacy-cu126"
BLACKWELL = "blackwell-cu130"
PROFILES = (LEGACY, BLACKWELL)


def check_wheels(provider, profile=None):
    """Resolve exact wheel presence before an image build, without downloading wheels."""
    profile = selected(profile)
    suffix = "cu130" if profile == BLACKWELL else "cu126"
    versions = (
        {"torch": "2.14.0", "torchvision": "0.29.0"}
        if provider == "perplexity"
        else {"torch": "2.12.1", "torchvision": "0.27.1"}
    )
    for package, version in versions.items():
        url = f"https://download.pytorch.org/whl/{suffix}/{package}/"
        with urlopen(url, timeout=30) as response:
            listing = unquote(response.read().decode("utf-8"))
        candidates = [
            link
            for link in listing.split('href="')[1:]
            if f"{package}-{version}+{suffix}-cp312-cp312-" in link.split('"')[0]
            and "linux" in link.split('"')[0]
            and "x86_64.whl" in link.split('"')[0]
        ]
        if not candidates:
            raise RuntimeError(
                f"No official Python 3.12 Linux x86_64 wheel for {package}=={version}+{suffix}. "
                "The reviewed runtime pin needs an explicit compatible release; no CPU wheel, "
                "older CUDA build or dependency downgrade will be substituted."
            )
    return {
        "runtime_profile": profile,
        "exact_wheels_available": versions,
        "inference_verified": False,
    }


def selected(profile=None):
    profile = (
        profile
        or os.environ.get("JEVPFN_RUNTIME_PROFILE")
        or (BLACKWELL if os.environ.get("SLURM_CLUSTER_NAME") == "mindwell" else LEGACY)
    )
    if profile not in PROFILES:
        raise ValueError(f"Unknown runtime profile: {profile}")
    if os.environ.get("SLURM_CLUSTER_NAME") == "mindwell" and profile != BLACKWELL:
        raise ValueError("Mindwell requires the separately built blackwell-cu130 runtime")
    return profile


def validate_build(runtime, profile=None):
    """Reject an image built against an older CUDA runtime before GPU submission."""
    profile = selected(profile)
    if profile == BLACKWELL and runtime.get("torch_cuda") != "13.0":
        raise ValueError("blackwell-cu130 requires a verified Torch CUDA 13.0 build")


def driver_version():
    """Read the host driver exposed to this allocation, without changing it."""
    text = subprocess.check_output(
        [
            os.environ.get("JEVPFN_NVIDIA_SMI") or "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=15,
    )
    versions = {line.strip() for line in text.splitlines() if line.strip()}
    if len(versions) != 1 or not re.fullmatch(r"\d+(?:\.\d+)+", next(iter(versions), "")):
        raise RuntimeError("Cannot identify one NVIDIA driver version for the allocated host")
    return versions.pop()


def gpu_check():
    import torch

    profile = selected()
    runtime = {
        "runtime_profile": profile,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    validate_build(runtime, profile)
    if not torch.cuda.is_available():
        raise RuntimeError("The selected runtime cannot access its allocated CUDA device")
    if profile == BLACKWELL:
        runtime["driver_version"] = driver_version()
        if int(runtime["driver_version"].split(".", 1)[0]) < 580:
            raise RuntimeError("The pinned CUDA 13.0 runtime requires NVIDIA driver 580 or newer")
    capability = tuple(torch.cuda.get_device_capability())
    name = torch.cuda.get_device_name()
    if profile == BLACKWELL and (capability[0] != 10 or "B200" not in name):
        raise RuntimeError(f"Expected Mindwell B200; got {name} with capability {capability}")
    if profile == BLACKWELL and not {"sm_100", "sm_100a"}.intersection(torch.cuda.get_arch_list()):
        raise RuntimeError("Torch wheel has no native sm_100 Blackwell kernels")
    # Exercise BF16, reduction, matmul, and host export before loading expensive weights.
    x = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
    value = (x @ x).float().mean()
    torch.cuda.synchronize()
    if value.item() != 32.0:
        raise RuntimeError("Blackwell matrix kernel returned an incorrect result")
    return {
        **runtime,
        "device": name,
        "capability": list(capability),
        "basic_kernel_check": "passed",
        "model_inference_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-gpu", action="store_true")
    parser.add_argument("--check-wheels", action="store_true")
    parser.add_argument("--provider", default="liquid")
    args = parser.parse_args()
    if args.check_gpu and args.check_wheels:
        parser.error("Select one runtime check")
    result = (
        gpu_check()
        if args.check_gpu
        else (check_wheels(args.provider) if args.check_wheels else {"runtime_profile": selected()})
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
