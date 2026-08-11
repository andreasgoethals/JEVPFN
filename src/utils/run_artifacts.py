"""List and delete what a previous run left behind. Lists by default; deletes when asked.

NOT `rm -rf`: a run's output spans **two storage tiers** with different quotas, so "delete the
last run" is not one command — and the version someone types by hand at 23:00 is the version
that eventually removes `data/raw`.

PROTECTED unconditionally, by construction rather than by a flag: `data/raw/` (irreplaceable),
`checkpoints/` (downloaded, or a training run to rebuild), `tfm-library/` (not ours), and the
repository's own directories. No combination of arguments reaches them — a protection you can
switch off is a protection that gets switched off.

CATEGORIES run cheap-to-rebuild first, and only the cheap ones go by default: `results` and
`processed` cost real compute, and "clean up the logs" must not lose a week of runs.
"""

from __future__ import annotations

import contextlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.paths import (
    REPO_ROOT,
    checkpoints_dir,
    figures_dir,
    library_dir,
    logs_dir,
    manifests_dir,
    outputs_dir,
    processed_dir,
    raw_dir,
    results_dir,
)

#: Every removable category, cheap to rebuild first.
CATEGORIES = ("figures", "logs", "manifests", "runs", "results", "processed")

#: Removed by default: regenerable in minutes by re-running the notebooks.
CHEAP = ("figures", "logs", "manifests", "runs")

#: Must be named explicitly. `results` is the record of finished compute; `processed` is a
#: cache that can take a long time to rebuild from raw.
EXPENSIVE = ("results", "processed")

#: Tracked so an empty directory survives a clone. Not run output, so neither counted nor
#: deleted — removing them would leave a fresh clone with no output tree.
KEEP_FILES = frozenset({".gitkeep", ".gitignore"})


@dataclass
class Artifact:
    """One removable location."""

    category: str
    path: Path
    n_files: int
    bytes: int

    @property
    def mb(self) -> float:
        return self.bytes / 1e6

    def describe(self) -> str:
        size = f"{self.mb:8.1f} MB" if self.mb < 1000 else f"{self.mb / 1000:8.2f} GB"
        return f"  {self.category:<11} {size}  {self.n_files:>6} files  {self.path}"


def protected_paths() -> list[Path]:
    """Paths that must never be removed, whatever the arguments say."""
    return [
        raw_dir(),
        REPO_ROOT / "data" / "raw",       # both tiers: the resolver may point elsewhere
        checkpoints_dir(),
        REPO_ROOT / "checkpoints",
        library_dir(),
        REPO_ROOT / "src",
        REPO_ROOT / "config",
        REPO_ROOT / "tests",
        REPO_ROOT / "docs",
        REPO_ROOT / "notebooks",
        REPO_ROOT / "scripts",
    ]


def is_protected(path: Path) -> bool:
    """True if `path` is, contains, or lives inside a protected path.

    Both directions: passing a PARENT of something protected counts too, because deleting
    `data/` to get at `data/processed` would take `data/raw` with it.
    """
    try:
        resolved = path.resolve()
    except OSError:
        return True  # cannot resolve it -> refuse to touch it
    for guard in protected_paths():
        try:
            g = guard.resolve()
        except OSError:
            continue
        if resolved == g or g in resolved.parents or resolved in g.parents:
            return True
    return False


def _measure(path: Path) -> tuple[int, int]:
    """(file count, total bytes) under a path, ignoring structure markers."""
    if not path.exists():
        return 0, 0
    if path.is_file():
        return (0, 0) if path.name in KEEP_FILES else (1, path.stat().st_size)
    n = total = 0
    for p in path.rglob("*"):
        if p.is_file() and p.name not in KEEP_FILES:
            n += 1
            # A file vanishing mid-walk (a concurrent run) is not worth failing over.
            with contextlib.suppress(OSError):
                total += p.stat().st_size
    return n, total


def _candidates() -> list[tuple[str, Path]]:
    """Every location a run may have written, by category.

    Figure folders are enumerated per notebook rather than taking `output/figures` whole, so the
    shared `CAPTIONS.md` beside them is not swept up with them.
    """
    found: list[tuple[str, Path]] = []

    fig_root = figures_dir()
    if fig_root.is_dir():
        found += [("figures", child) for child in sorted(fig_root.iterdir()) if child.is_dir()]

    found += [
        ("logs", logs_dir()),
        ("manifests", manifests_dir()),
        ("results", results_dir()),
        ("processed", processed_dir()),
    ]

    runs_root = outputs_dir() / "runs"
    if runs_root.is_dir():
        found += [("runs", child) for child in sorted(runs_root.iterdir()) if child.is_dir()]

    return found


def find_artifacts() -> list[Artifact]:
    """Everything a previous run left behind, measured but not touched."""
    artifacts = []
    for category, path in _candidates():
        if is_protected(path):
            continue
        n, total = _measure(path)
        if n:
            artifacts.append(Artifact(category=category, path=path, n_files=n, bytes=total))
    return artifacts


def summarise(artifacts: list[Artifact]) -> str:
    """A human-readable listing, grouped by category, cheap first."""
    if not artifacts:
        return "Nothing found — the tree is already clean."

    by_cat: dict[str, list[Artifact]] = {}
    for a in artifacts:
        by_cat.setdefault(a.category, []).append(a)

    lines = ["Artifacts from previous runs:", ""]
    for category in CATEGORIES:
        items = by_cat.get(category)
        if not items:
            continue
        cat_bytes = sum(i.bytes for i in items)
        tag = "   [EXPENSIVE to rebuild — name it explicitly]" if category in EXPENSIVE else ""
        lines.append(f"{category.upper()}  ({cat_bytes / 1e6:.1f} MB){tag}")
        lines += [i.describe() for i in items]
        lines.append("")

    total = sum(a.bytes for a in artifacts)
    lines += [
        f"TOTAL: {total / 1e9:.2f} GB across {len(artifacts)} locations",
        "",
        "NEVER removed by this tool, whatever you pass it:",
        "  data/raw/  checkpoints/  tfm-library/  and the repository's own directories.",
    ]
    return "\n".join(lines)


def clean(
    categories: tuple[str, ...] = CHEAP,
    *,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Remove the named categories. **Dry run by default.**

    Default rather than opt-in because the failure modes are not symmetric: a listing you meant
    as a deletion costs one more command; a deletion you meant as a listing costs the run.

    Deletes a directory's CONTENTS, not the directory, so tracked `.gitkeep` markers survive —
    `rmtree` on `output/logs` removes a directory git expects to exist.
    """
    unknown = [c for c in categories if c not in CATEGORIES]
    if unknown:
        raise ValueError(f"unknown categories {unknown}; choose from {CATEGORIES}")

    artifacts = [a for a in find_artifacts() if a.category in categories]
    removed: list[str] = []
    failed: list[str] = []

    for a in artifacts:
        if is_protected(a.path):
            continue  # belt and braces; find_artifacts already filtered these
        if dry_run:
            removed.append(str(a.path))
            continue
        try:
            if a.path.is_dir():
                for child in a.path.iterdir():
                    if child.name in KEEP_FILES:
                        continue
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
            else:
                a.path.unlink()
            removed.append(str(a.path))
        except OSError as exc:
            failed.append(f"{a.path}: {exc}")

    freed = sum(a.bytes for a in artifacts)
    return {
        "dry_run": dry_run,
        "categories": list(categories),
        "removed": removed,
        "failed": failed,
        "freed_bytes": freed,
        "freed_gb": round(freed / 1e9, 3),
    }
