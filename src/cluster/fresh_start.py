"""Reset generated outputs on their existing storage tier for a fresh test workspace.

Default: preview. --delete permanently removes output instead of archiving it.
--include-archives also removes earlier output archives; it requires --delete.
Changes require --apply --writers-stopped, with no active JEVPFN jobs on VSC.
Use --local for the local notebook output tree; otherwise both VSC tiers are required.
Never removes data, prepared inputs, permanent features, weights or environments.
"""

import argparse
import json
import os
import shutil
import stat
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
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


def output_roots(*, local=False):
    if local:
        if paths.on_vsc():
            raise RuntimeError("--local is for the local computer, not a VSC shell")
        roots = {"LOCAL": paths.notebook_outputs_dir()}
    else:
        if not paths.on_vsc():
            raise RuntimeError(
                "Use --local on the local computer; otherwise use the VSC login shell"
            )
        roots = {"DATA": paths.outputs_dir(), "PROJECT": paths.large_outputs_dir()}
        if roots["DATA"].resolve() == roots["PROJECT"].resolve():
            raise ValueError("DATA and PROJECT outputs must be separate")
    for tier, root in roots.items():
        # Resolve the site's /data mount alias first, then reject user-created links.
        parent = root.parent.resolve(strict=True)
        root = parent / root.name
        if root.name != paths.OUTPUT_DIR_NAME or (not local and parent.name != paths.PROJECT_NAME):
            raise ValueError("Only the exact JEVPFN output folder can be reset")
        reject_links(root)
        if root.exists() and not root.is_dir():
            raise ValueError("Output root is not a directory")
        if tier == "PROJECT":
            paths.require_project_output(root)
        roots[tier] = root
    return roots


def validate_deletion_tree(root):
    """Inspect every entry before deletion; never traverse linked or mounted subtrees."""
    reject_links(root)
    if not root.exists():
        return
    if not root.is_dir():
        raise ValueError(f"Output deletion target is not a directory: {root}")
    if root.is_mount():
        raise ValueError(f"Refusing to delete a mounted output directory: {root}")

    def fail(error):
        raise error

    for directory, folders, files in os.walk(root, followlinks=False, onerror=fail):
        for name in (*folders, *files):
            entry = Path(directory) / name
            reject_links(entry)
            if entry.is_mount():
                raise ValueError(f"Refusing to delete a mounted output entry: {entry}")


def delete_directory(root):
    """Remove a validated output tree, handling Windows read-only files/directories.

    Only retry unlink/rmdir within this tree. Do not change ACLs, chmod parent
    folders, follow links or suppress persistent errors. The caller has already
    validated every output tier and holds the maintenance lock.
    """
    root = Path(root).absolute()

    def on_error(function, filename, error):
        entry = Path(filename).absolute()
        if not entry.is_relative_to(root):
            raise ValueError(f"Deletion error outside the selected output tree: {entry}") from error
        if (
            function not in (os.unlink, os.rmdir)
            or not isinstance(error, PermissionError)
            or getattr(error, "winerror", None) not in {5, 32, 33}
        ):
            raise error
        for attempt in range(7):
            reject_links(entry)
            if entry.is_mount():
                raise ValueError(f"Refusing to delete a mounted output entry: {entry}")
            try:
                info = entry.stat(follow_symlinks=False)
                if getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_READONLY:
                    # Windows chmod only changes the read-only attribute. Preserve
                    # the existing mode; never broaden permissions on VSC/Linux.
                    entry.chmod(info.st_mode | stat.S_IWRITE)
                function(filename)
                return
            except FileNotFoundError:
                return  # A completed removal needs no retry.
            except PermissionError as exc:
                if attempt == 6:
                    exc.add_note(
                        f"Output cleanup could not remove {entry}. Close programs using this "
                        "path and retry the same cleanup command before running tests."
                    )
                    raise
                time.sleep(min(0.1 * 2**attempt, 1.0))

    if sys.version_info >= (3, 12):
        shutil.rmtree(root, onexc=on_error)
    else:
        # Preserve the template's Python 3.11 support without 3.12 deprecation warnings.
        shutil.rmtree(root, onerror=lambda fn, name, info: on_error(fn, name, info[1]))


def run(*, apply=False, writers_stopped=False, local=False, delete=False, include_archives=False):
    if include_archives and not delete:
        raise ValueError("--include-archives requires --delete")
    roots = output_roots(local=local)
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
            "data/text_embeddings",
            "checkpoints",
            "environments",
            "containers",
        ],
    }
    if delete:
        report["action"] = "permanently_delete_outputs"
        del report["moves"]
        deletion_targets = [
            (tier, target)
            for tier, root in roots.items()
            for target in (
                (root, root.parent / "archived_output_JEVPFN") if include_archives else (root,)
            )
        ]
        report["deletions"] = [
            {"tier": tier, "path": str(target), "exists": target.exists()}
            for tier, target in deletion_targets
        ]
        # Preview checks the same boundaries as execution, without writing anything.
        if not apply:
            for _, target in deletion_targets:
                validate_deletion_tree(target)
    if not apply:
        return report
    if not writers_stopped:
        raise ValueError("Stop manual writers as well as jobs; then pass --writers-stopped")
    if not local:
        require_idle_vsc()
    journal_root = roots["LOCAL" if local else "DATA"].parent
    if delete:
        with file_lock(journal_root / ".maintenance.lock", timeout=1):
            # Validate every tier before deleting any of them. Paths are exact output
            # siblings of the validated project roots, never user-supplied subpaths.
            for _, target in deletion_targets:
                validate_deletion_tree(target)
            for (_, target), entry in zip(deletion_targets, report["deletions"], strict=True):
                if target.exists():
                    delete_directory(target)
                entry["deleted"] = True
            for root in roots.values():
                root.mkdir(exist_ok=True)
            if not local:
                paths.logs_dir("experiment_0").mkdir(parents=True, exist_ok=True)
        report["status"] = "fresh_outputs_ready"
        return report
    journal = journal_root / "archived_output_JEVPFN" / f"{run_id}.json"
    for move in report["moves"]:
        reject_links(move["archive"])
    with file_lock(journal_root / ".maintenance.lock", timeout=1):
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
            if not local:
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
    parser.add_argument("--delete", action="store_true", help="permanently delete old output")
    parser.add_argument(
        "--include-archives", action="store_true", help="also delete earlier output archives"
    )
    parser.add_argument("--local", action="store_true", help="reset the local notebook output tree")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                apply=args.apply,
                writers_stopped=args.writers_stopped,
                local=args.local,
                delete=args.delete,
                include_archives=args.include_archives,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
