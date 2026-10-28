"""Record isolated environment versions without importing a model or running inference."""

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from src.jev.catalog import SELF_HOSTED
from src.utils import paths
from src.utils.serialization import write_json


def record(provider):
    expected = paths.checkpoints_dir().parent / "environments" / provider
    if Path(sys.prefix).resolve() != expected.resolve():
        raise ValueError("Runtime manifest must describe the selected PROJECT environment")
    report = {
        "provider": provider,
        "python": platform.python_version(),
        "prefix": str(expected),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "gpu_verified": False,
        "checkpoint_inference_verified": False,
    }
    # Keep installed-version evidence beside the durable environment across output resets.
    write_json(expected / "JEVPFN_RUNTIME.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "provider", choices=SELF_HOSTED
    )
    args = parser.parse_args()
    print(json.dumps(record(args.provider), indent=2))


if __name__ == "__main__":
    main()
