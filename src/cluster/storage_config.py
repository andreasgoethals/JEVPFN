"""Private deployment authority; stdlib-only bootstrap before activating an image."""

import json
import os
import shlex
from pathlib import Path, PurePosixPath


def configured_root():
    for key in ("JEVPFN_DURABLE_ROOT", "JEVPFN_PROJECT_ROOT", "VSC_PROJECT_ROOT"):
        if value := os.environ.get(key):
            return validate_root(Path(value))
    config = Path(os.environ.get("JEVPFN_STORAGE_CONFIG") or
                  Path(__file__).resolve().parents[2] / ".vsc-storage.local.json")
    if config.is_file():
        root = Path(json.loads(config.read_text(encoding="utf-8"))["durable_root"])
        return validate_root(root)
    return None


def validate_root(root):
    if not (root.is_absolute() or PurePosixPath(root.as_posix()).is_absolute()) or root.name == "JEVPFN" or ".." in root.parts or any(
        c in str(root) for c in ("\n", "\r", ",")
    ):
        raise ValueError("Durable root must name an absolute allocated shared PROJECT root")
    return root


def main():
    root = configured_root()
    if root is None:
        raise ValueError("Configure .vsc-storage.local.json or an explicit durable PROJECT root")
    print("export JEVPFN_DURABLE_ROOT=" + shlex.quote(str(root)))
    print("export JEVPFN_PROJECT_ROOT=" + shlex.quote(str(root)))


if __name__ == "__main__":
    main()
