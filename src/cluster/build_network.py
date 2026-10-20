"""Explicit installation-only connectivity probe; no credentials or model inference."""

from urllib.error import HTTPError
from urllib.request import Request, urlopen

ENDPOINTS = (
    "https://pypi.org/simple/",
    "https://download.pytorch.org/whl/cu126/",
    "https://conda.anaconda.org/conda-forge/linux-64/repodata.json",
)


def check():
    for url in ENDPOINTS:
        try:
            with urlopen(Request(url, method="HEAD"), timeout=20) as response:
                status = response.status
        except HTTPError as exc:
            raise RuntimeError(
                f"Package index refused the build-node preflight: {url} ({exc.code})"
            ) from exc
        if not 200 <= status < 400:
            raise RuntimeError(f"Package index unavailable: {url}")
        print(f"Build-node package index reachable: {url}", flush=True)
    print("Connectivity only; exact package availability is checked by installation.", flush=True)


if __name__ == "__main__":
    check()
