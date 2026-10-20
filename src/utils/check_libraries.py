"""Read-only check that each reference library is committed and checked out at its pin."""

import argparse
import json
import subprocess

from src.utils import paths

LIBRARIES = ("tfm-library", "dm-library", "vsc-library")


def check(root=None):
    root = paths.REPO_ROOT if root is None else root
    results = []
    for name in LIBRARIES:
        tracked = subprocess.run(
            ["git", "ls-tree", "HEAD", "--", name],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not tracked.startswith("160000 commit "):
            results.append({"library": name, "status": "missing_committed_gitlink"})
            continue
        pin = tracked.split()[2]
        checkout = subprocess.run(
            ["git", "-C", str(root / name), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        actual = checkout.stdout.strip() if checkout.returncode == 0 else None
        results.append(
            {
                "library": name,
                "pin": pin,
                "checkout": actual,
                "status": "passed" if actual == pin else "missing_or_different_checkout",
            }
        )
    return {
        "status": "passed" if all(r["status"] == "passed" for r in results) else "failed",
        "libraries": results,
    }


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = check()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
