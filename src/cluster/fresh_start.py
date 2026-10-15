"""Archive old outputs on their existing storage tier and create a fresh test workspace.

Default: preview. --apply --writers-stopped performs reversible directory renames.
Never deletes data, prepared inputs, cached features, weights, environments or archives.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from uuid import uuid4

from src.cluster.maintenance import reject_links, require_idle_vsc
from src.utils import paths
from src.utils.locking import file_lock
from src.utils.serialization import write_json


def archive_directory(source, destination):
    """Retry brief Windows sharing locks without replacing an existing archive."""
    for attempt in range(7):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Archive destination already exists: {destination}")
        try:
            source.rename(destination)  # Same tier/filesystem; never recursive copy/delete.
            return
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(min(0.1 * 2**attempt, 1.0))


def output_roots():
    if not paths.on_vsc():
        raise RuntimeError("This fresh-start command is for the VSC login shell")
    roots = {"DATA": paths.outputs_dir(), "PROJECT": paths.large_outputs_dir()}
    if roots["DATA"].resolve() == roots["PROJECT"].resolve():
        raise ValueError("DATA and PROJECT outputs must be separate")
    for tier, root in roots.items():
        # Resolve the site's /data mount alias first, then reject user-created links.
        parent = root.parent.resolve(strict=True)
        root = parent / root.name
        if root.name != paths.OUTPUT_DIR_NAME or parent.name != paths.PROJECT_NAME:
            raise ValueError("Only the exact JEVPFN output folder can be archived")
        reject_links(root)
        if root.exists() and not root.is_dir():
            raise ValueError("Output root is not a directory")
        if tier == "PROJECT":
            paths.require_project_output(root)
        roots[tier] = root
    return roots


def run(*, apply=False, writers_stopped=False):
    roots = output_roots()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    report = {
        "run_id": run_id,
        "status": "preview",
        "action": "archive_outputs_without_deletion",
        "moves": [
            {
                "tier": tier,
                "source": str(root),
                "archive": str(root.parent / "archived_output_JEVPFN" / run_id),
                "exists": root.exists(),
            }
            for tier, root in roots.items()
        ],
        "preserved_in_place": [
            "data/raw",
            "data/decision_inputs",
            "data/jev_cache",
            "checkpoints",
            "environments",
        ],
    }
    if not apply:
        return report
    if not writers_stopped:
        raise ValueError("Stop manual writers as well as jobs; then pass --writers-stopped")
    require_idle_vsc()
    journal = roots["DATA"].parent / "archived_output_JEVPFN" / f"{run_id}.json"
    for move in report["moves"]:
        reject_links(move["archive"])
    with file_lock(roots["DATA"].parent / ".maintenance.lock", timeout=1):
        report["status"] = "archiving"
        write_json(journal, report)
        try:
            for move in report["moves"]:
                source = roots[move["tier"]]
                if source.exists():
                    destination = source.parent / "archived_output_JEVPFN" / run_id
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    archive_directory(source, destination)
                    move["archived"] = True
                    write_json(journal, report)
                source.mkdir(exist_ok=True)
            paths.logs_dir("experiment_0").mkdir(parents=True, exist_ok=True)
            report["status"] = "fresh_outputs_ready"
        except BaseException as exc:
            report.update(status="interrupted_archives_preserved", error_type=type(exc).__name__)
            raise
        finally:
            write_json(journal, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--writers-stopped", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(apply=args.apply, writers_stopped=args.writers_stopped), indent=2))


if __name__ == "__main__":
    main()
