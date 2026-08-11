"""Every path in the project. Two VSC tiers, one resolver, relative to the repository root.

THE ONLY MODULE THAT BUILDS A PATH — everything else asks this one. A path assembled at a call
site with `"output/" + name` is correct on a laptop and wrong on the cluster, and the failure
shows up as a full quota or an empty results directory hours into a job.

        project storage  /lustre1/project/stg_00211/<Project>/  big files, backed up, LOW INODES
    personal data    $VSC_DATA/<Project>/                   repo + output/, backed up, 75 GiB
    scratch          $VSC_SCRATCH/                          purged after 30 days of no ACCESS

Both tiers are backed up. They differ in size and in convenience: `$VSC_DATA` is only 75 GiB but
can be browsed directly, while project storage is large but has to be pulled down locally first
(PowerShell, `scp`/`rsync`) before you can look at anything in it.

`output/results/` is therefore the one part of `output/` on project storage: per-row predictions
reach gigabytes. Everything else stays where you can read it without a download, and project
storage wants few big files rather than thousands of small ones anyway.

OFF-CLUSTER EVERY TIER COLLAPSES INTO THE REPO. Pretending `/lustre1` exists on a laptop would
mean two code paths, and the one that only runs on the cluster is the one that breaks.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Filled in by `_template/init_project.py`. The per-project folder name on BOTH shared tiers.
PROJECT_NAME = "{{PROJECT_NAME}}"

#: Fallback for project storage when the site variable is not set. The literal path is a
#: last resort, not the primary source — VSC has moved it before.
STAGING_FALLBACK = "/lustre1/project/stg_00211"

#: `parents[2]` because this file is `<root>/src/utils/paths.py`. From __file__, not the working
#: directory, so a script, a test and a notebook agree wherever they were launched from.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Overrides for project storage, in priority order. The supported way to put big files on an
#: external drive locally, and how the tests exercise the staging branch without a cluster.
STAGING_ENV_VARS = ("{{PROJECT_UPPER}}_STAGING_ROOT", "VSC_STAGING_ROOT")


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


# ---------------------------------------------------------------------------
# Which world are we in?
# ---------------------------------------------------------------------------


def on_vsc() -> bool:
    """True on a VSC node. `$VSC_DATA` is set by the site and by nothing else."""
    return bool(os.environ.get("VSC_DATA"))


def staging_override() -> Path | None:
    """An explicitly requested project-storage root, or None.

    Separate from `staging_root()` because the big-file helpers short-circuit to the repo when
    off-cluster, which would make an override silently do nothing on a laptop.
    """
    for var in STAGING_ENV_VARS:
        p = _env_path(var)
        if p:
            return p
    return None


def _use_staging() -> bool:
    """Should big files go to project storage rather than into the repo?"""
    return on_vsc() or staging_override() is not None


# ---------------------------------------------------------------------------
# The three roots.
# ---------------------------------------------------------------------------


def staging_root() -> Path:
    """Project storage — the big, unbacked-up tier."""
    override = staging_override()
    if override:
        return override
    lustre = _env_path("VSC_PROJECT_LUSTRE1")
    if lustre:
        return lustre / "stg_00211"
    if on_vsc():
        return Path(STAGING_FALLBACK)
    return REPO_ROOT


def data_root() -> Path:
    """Personal data — the small, backed-up tier. The repo lives here on the cluster."""
    return _env_path("VSC_DATA") or REPO_ROOT


def scratch_root() -> Path:
    """Working scratch. Purged after 30 days without access — never a result."""
    return _env_path("VSC_SCRATCH") or REPO_ROOT


def _under(root: Path, *parts: str) -> Path:
    """Join under `root`, inserting the project name only when `root` is SHARED.

    The shared tiers need a `<Project>/` component; the repo root already *is* the project, so
    adding it there would give `<Project>/<Project>/output`.
    """
    if root == REPO_ROOT:
        return root.joinpath(*parts)
    return root.joinpath(PROJECT_NAME, *parts)


# ---------------------------------------------------------------------------
# output/ — the single root for everything the code generates.
# ---------------------------------------------------------------------------


def outputs_dir() -> Path:
    """THE root for generated files; nothing generated is written outside it.

    Locally `<repo>/output/`, on the cluster `$VSC_DATA/<Project>/output/`. One root means "what
    did this run produce?" and "what can I delete?" have one answer each.
    """
    if on_vsc():
        return _under(data_root(), "output")
    return REPO_ROOT / "output"


def results_dir(*parts: str) -> Path:
    """Fine-grained results: one row per prediction, per-fold scores, anything large.

    THE ONE PART OF `output/` ON PROJECT STORAGE. A single sweep of per-row predictions would
    fill $VSC_DATA's 75 GiB, and then every job that writes a log also fails.
    """
    if _use_staging():
        return _under(staging_root(), "output", "results", *parts)
    return outputs_dir().joinpath("results", *parts)


def logs_dir() -> Path:
    """Timestamped run logs. Small, many files -> `$VSC_DATA`, not the inode-poor tier."""
    return outputs_dir() / "logs"


def manifests_dir() -> Path:
    """Per-run manifests: the small CSV/JSON record of what a run did."""
    return outputs_dir() / "manifests"


def figures_dir(notebook: str | None = None) -> Path:
    """`output/figures/`, or one notebook's own folder — a notebook clears its own before drawing
    and must not be able to reach another's."""
    root = outputs_dir() / "figures"
    return root / notebook if notebook else root


def captions_path() -> Path:
    """The ONE shared captions file for every figure in the project."""
    return figures_dir() / "CAPTIONS.md"


def all_results_path() -> Path:
    """Every notebook's printed text summary, concatenated in notebook order."""
    return outputs_dir() / "All_Results.md"


# ---------------------------------------------------------------------------
# Inputs, weights and the repo's own directories.
# ---------------------------------------------------------------------------


def raw_dir(*parts: str) -> Path:
    """`data/raw/` — never modified, never committed, never deleted by the cleaner."""
    if _use_staging():
        return _under(staging_root(), "data", "raw", *parts)
    return REPO_ROOT.joinpath("data", "raw", *parts)


def processed_dir(*parts: str) -> Path:
    """`data/processed/` — a cache. Regenerable, so it is the first thing to delete."""
    if _use_staging():
        return _under(staging_root(), "data", "processed", *parts)
    return REPO_ROOT.joinpath("data", "processed", *parts)


def data_search_paths(*parts: str) -> list[Path]:
    """Every root that might hold this input, **repo first** — so a laptop with the data checked
    out works unconfigured, and the same code finds it on the cluster."""
    roots = [REPO_ROOT.joinpath("data", *parts)]
    if _use_staging():
        staged = _under(staging_root(), "data", *parts)
        if staged not in roots:
            roots.append(staged)
    return roots


def find_input(*parts: str) -> Path | None:
    """The first existing candidate from `data_search_paths`, or None."""
    for candidate in data_search_paths(*parts):
        if candidate.exists():
            return candidate
    return None


def checkpoints_dir(*parts: str) -> Path:
    """Model weights. Big -> project storage. Never deleted by the cleaner: downloaded from
    upstream, or a training run to reproduce."""
    if _use_staging():
        return _under(staging_root(), "checkpoints", *parts)
    return REPO_ROOT.joinpath("checkpoints", *parts)


def config_path(name: str) -> Path:
    """`config/<name>.yaml`. Always in the repo — configs are code, not data."""
    stem = name[:-5] if name.endswith(".yaml") else name
    return REPO_ROOT / "config" / f"{stem}.yaml"


def notebooks_dir() -> Path:
    return REPO_ROOT / "notebooks"


def library_dir() -> Path:
    """The read-only literature submodule. READ from it; never write inside it."""
    return REPO_ROOT / "tfm-library"


def repo_dir_on_cluster() -> Path:
    """Where the code is checked out on the cluster: `$VSC_DATA/<Project>` — backed up."""
    return _under(data_root())


# ---------------------------------------------------------------------------
# Making a path usable.
# ---------------------------------------------------------------------------


def ensure(path: Path) -> Path:
    """`mkdir -p` the directory and return it. For a file path, its parent."""
    target = path.parent if path.suffix else path
    target.mkdir(parents=True, exist_ok=True)
    return path


def resolve_writable(preferred: Path, fallback: Path | None = None) -> Path:
    """`preferred` if we can genuinely write there, else `fallback`, loudly.

    Probes with a real create-and-delete: `mkdir(exist_ok=True)` is not enough, because a
    directory on a shared tier can exist and still be unwritable. A completed run in the wrong
    place beats a job that died at hour six with nothing to show.
    """
    fallback = fallback or (data_root() / PROJECT_NAME / "fallback")
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return preferred
    except OSError as exc:
        print(
            f"WARNING: cannot write to {preferred} ({exc}).\n"
            f"         Falling back to {fallback}. Move the output to project storage "
            f"afterwards, or $VSC_DATA will fill up (75 GiB quota).",
            flush=True,
        )
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def touch_tree(path: Path) -> None:
    """Refresh access times against scratch's 30-day purge. `mv` and `rsync -a` do NOT count as an
    access, so freshly staged data can be purged almost immediately. Copy, then call this."""
    for p in path.rglob("*"):
        if p.is_file():
            p.touch()


def describe() -> dict[str, str]:
    """Every resolved root, for logging at job start — so a run's output can still be found six
    months later on a tier that has since been reorganised."""
    return {
        "project": PROJECT_NAME,
        "on_vsc": str(on_vsc()),
        "repo_root": str(REPO_ROOT),
        "staging_root": str(staging_root()),
        "data_root": str(data_root()),
        "scratch_root": str(scratch_root()),
        "outputs_dir": str(outputs_dir()),
        "results_dir": str(results_dir()),
        "checkpoints_dir": str(checkpoints_dir()),
    }
