"""Preview or delete generated output only, locally or on VSC DATA/PROJECT.

Examples (all preview unless --apply --writers-stopped is supplied):
    python -m src.utils.output_cleanup
    python -m src.utils.output_cleanup --phase experiment_0 --kind logs
    python -m src.utils.output_cleanup --tier project --kind results --older-than-days 7
    python -m src.utils.output_cleanup --include-archives --apply --writers-stopped

Run on the machine being cleaned. No SSH, data deletion, model runs or job submission.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import stat
import time
from pathlib import Path

from src.cluster.fresh_start import output_roots, validate_deletion_tree
from src.cluster.maintenance import reject_links, require_idle_vsc
from src.utils import paths
from src.utils.locking import file_lock
from src.utils.serialization import write_json

KINDS = ("logs", "manifests", "results", "figures", "reports", "transfers")


def selected_roots(tier="auto"):
    local = not paths.on_vsc()
    if local and tier not in {"auto", "local"}:
        raise ValueError("DATA/PROJECT selections require a VSC login shell; use --tier local")
    if not local and tier == "local":
        raise ValueError("Run --tier local on the laptop, not on VSC")
    roots = output_roots(local=local)
    return {k: v for k, v in roots.items() if tier in {"auto", "both", "local"} or k.lower() == tier}


def classify(relative):
    parts = relative.parts
    name = relative.name
    if name in {"All Results.md", "all_results.md", "Captions.md"}:
        return "reports"
    if parts[0] in KINDS:
        return parts[0]
    return parts[1] if len(parts) > 1 and parts[1] in KINDS else "other"


def phase_of(relative):
    for part in relative.parts[:-1]:
        if part[:1].isdigit() and ". " in part:
            return paths.phase_name(part)
    return None


def plan(*, tier="auto", phases=(), kinds=(), pattern="*", older_than_days=None, include_archives=False):
    if older_than_days is not None and older_than_days < 0:
        raise ValueError("Age must be nonnegative")
    if include_archives and (phases or kinds or pattern != "*" or older_than_days is not None):
        raise ValueError("Archive removal is available only with an unfiltered full output cleanup")
    selected = {paths.phase_name(p) for p in phases}
    if set(kinds) - set(KINDS):
        raise ValueError("Unknown output category")
    roots = selected_roots(tier)
    targets = [(label, root) for label, root in roots.items()]
    if include_archives:
        targets += [(label, root.parent / "archived_output_JEVPFN") for label, root in roots.items()]
    cutoff = None if older_than_days is None else time.time() - older_than_days * 86400
    files = []
    for label, root in targets:
        validate_deletion_tree(root)  # Validate every tier before applying any change.
        for file in sorted(root.rglob("*")) if root.exists() else ():
            if not file.is_file():
                continue
            relative = file.relative_to(root)
            if selected and phase_of(relative) not in selected:
                continue
            if kinds and classify(relative) not in kinds:
                continue
            if not fnmatch.fnmatchcase(relative.as_posix(), pattern):
                continue
            info = file.stat()
            if cutoff is not None and info.st_mtime >= cutoff:
                continue
            files.append({"tier": label, "path": str(file), "bytes": info.st_size})
    known = {entry["path"] for entry in files}
    for entry in list(files):
        file = Path(entry["path"])
        for label, root in targets:
            if file.suffix == ".pdf" and file.is_relative_to(root / "figures"):
                notebook = file.parent.relative_to(root / "figures")
                report_file = root / "results" / "notebooks" / notebook / "_report.txt"
                if report_file.is_file() and str(report_file) not in known:
                    files.append({"tier": label, "path": str(report_file), "bytes": report_file.stat().st_size,
                                  "reason": "invalidate notebook report describing removed figures"})
                    known.add(str(report_file))
    # Combined documents would describe deleted figures/reports. Invalidate them;
    # the next notebook run or --summaries-only reconstructs them from what remains.
    if any(classify(Path(f["path"]).relative_to(root)) in {"figures", "reports"}
           for f in files for _, root in targets if Path(f["path"]).is_relative_to(root)):
        known = {f["path"] for f in files}
        for label, root in targets:
            for file in root.rglob("*.md") if root.exists() else ():
                if file.name in {"All Results.md", "all_results.md", "Captions.md"} and str(file) not in known:
                    files.append({"tier": label, "path": str(file), "bytes": file.stat().st_size,
                                  "reason": "invalidate aggregate after partial cleanup"})
    return roots, targets, files


def figure_manifest_updates(targets, files):
    """Remove captions only for selected PDFs, preserving the other figure entries."""
    deleted = {Path(entry["path"]) for entry in files}
    updates = {}
    for _, root in targets:
        for pdf in deleted:
            if not pdf.is_relative_to(root / "figures") or pdf.suffix != ".pdf":
                continue
            notebook = pdf.parent.relative_to(root / "figures")
            manifest = root / "manifests" / "notebooks" / notebook / "_figures.json"
            if manifest in deleted or not manifest.exists():
                continue
            entries = updates.get(manifest)
            if entries is None:
                entries = json.loads(manifest.read_text(encoding="utf-8"))
            updates[manifest] = [entry for entry in entries if entry["stem"] != pdf.stem]
    return updates


def remove_file(path):
    """Bounded Windows sharing/read-only recovery; never alter POSIX ACLs."""
    for attempt in range(7):
        reject_links(path)
        try:
            if os.name == "nt":
                info = path.stat(follow_symlinks=False)
                if info.st_file_attributes & stat.FILE_ATTRIBUTE_READONLY:
                    path.chmod(info.st_mode | stat.S_IWRITE)
            path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(min(0.1 * 2**attempt, 1))


def run(*, apply=False, writers_stopped=False, **selection):
    roots, targets, files = plan(**selection)
    updates = figure_manifest_updates(targets, files)
    report = {"status": "preview", "files": files, "file_count": len(files),
              "bytes": sum(f["bytes"] for f in files),
              "caption_manifests_updated": [str(p) for p in updates],
              "preserved": ["data (including paid-call records)", "checkpoints", "environments", "containers", "secrets"]}
    if not apply:
        return report
    if not writers_stopped:
        raise ValueError("Stop notebook/manual writers, then pass --writers-stopped")
    if paths.on_vsc():
        require_idle_vsc()
    # The lock is outside output, so even a full wipe cannot unlink a held lock.
    lock_root = next(iter(roots.values())).parent
    with file_lock(lock_root / ".maintenance.lock", timeout=1):
        roots, targets, files = plan(**selection)
        updates = figure_manifest_updates(targets, files)
        if not paths.on_vsc():
            from src.jev.smoke import preserve_smoke_state

            preserve_smoke_state()
        for entry in files:
            file = Path(entry["path"])
            if not any(file.is_relative_to(root) for _, root in targets):
                raise ValueError("Cleanup target escaped output")
            remove_file(file)
        for manifest, entries in updates.items():
            write_json(manifest, entries)
        # Remove only empty directories. Never recurse through a link or mount.
        for _, root in targets:
            for directory, _, _ in os.walk(root, topdown=False, followlinks=False):
                folder = Path(directory)
                reject_links(folder)
                if folder != root and not any(folder.iterdir()):
                    if os.name == "nt":
                        folder.chmod(folder.stat().st_mode | stat.S_IWRITE)
                    folder.rmdir()
        if paths.on_vsc():
            paths.logs_dir("experiment_0").mkdir(parents=True, exist_ok=True)
    report.update(status="deleted", files=files, file_count=len(files), bytes=sum(f["bytes"] for f in files),
                  caption_manifests_updated=[str(p) for p in updates])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", choices=("auto", "local", "data", "project", "both"), default="auto")
    parser.add_argument("--phase", action="append", default=[], help="Repeatable logical or numbered phase name")
    parser.add_argument("--kind", action="append", choices=KINDS, default=[])
    parser.add_argument("--match", default="*", help="Glob matched against output-relative paths")
    parser.add_argument("--older-than-days", type=float)
    parser.add_argument("--include-archives", action="store_true")
    parser.add_argument("--apply", "--clean", dest="apply", action="store_true")
    parser.add_argument("--writers-stopped", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(tier=args.tier, phases=args.phase, kinds=args.kind, pattern=args.match,
                         older_than_days=args.older_than_days, include_archives=args.include_archives,
                         apply=args.apply, writers_stopped=args.writers_stopped), indent=2))


if __name__ == "__main__":
    main()
