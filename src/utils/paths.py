"""Where files go. Two VSC storage tiers, one resolver, everything relative to the repo.

THE ONLY MODULE THAT BUILDS A PATH. Every other module, script and notebook asks this one.
WHY: a path assembled at the call site with `"output/" + name` is correct on a laptop and
wrong on the cluster, and the failure shows up as a full quota or an empty results
directory hours into a job. One module means one place to be right, and one place to fix.

    | tier                | path                                       | holds                                                            | backed up | quota                    |
    |---------------------|--------------------------------------------|------------------------------------------------------------------|-----------|--------------------------|
    | **project storage** | `/lustre1/project/stg_00211/<Project>/`    | big files: datasets, checkpoints, caches, **`output/results/`**   | no        | large, low inode budget  |
    | **personal data**   | `$VSC_DATA/<Project>/`                     | the repo, and the rest of `output/` (figures, logs, manifests)    | **yes**   | 75 GiB                   |
    | scratch             | `$VSC_SCRATCH/`                            | working scratch only                                             | no        | purged after 30 days     |

Three consequences that shaped this module:

* `output/results/` — per-row predictions and per-fold scores — is the one part of
  `output/` on project storage. It is the part that reaches gigabytes, and `$VSC_DATA` is
  75 GiB. Locally both tiers collapse to the repo, so `output/results/` is just a
  subdirectory and the split is invisible.
* Project storage has a **low inode budget**: few big files, not thousands of small ones.
  So logs and per-step metrics go to `$VSC_DATA` even though they belong to a run whose
  results are on staging.
* Scratch's purge is on **access** time, and `mv`/timestamp-preserving `rsync` do not
  count as an access. Copy, then call `touch_tree()`.

OFF-CLUSTER EVERYTHING COLLAPSES INTO THE REPO. There is no `/lustre1` on a laptop, so
pretending there is would mean two code paths, and the one that only runs on the cluster
is the one that breaks. Same functions, same call sites, different roots.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Filled in by `_template/init_project.py`. The per-project folder name on BOTH shared tiers.
PROJECT_NAME = "{{PROJECT_NAME}}"

#: Fallback for project storage when the site variable is not set. The literal path is a
#: last resort, not the primary source — VSC has moved it before.
STAGING_FALLBACK = "/lustre1/project/stg_00211"

#: The repository root. `parents[2]` because this file is `<root>/src/utils/paths.py`.
#: Resolved from __file__ rather than from the working directory, so a script, a test and
#: a notebook all agree on where the root is regardless of where they were launched.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Environment variables that override project storage, in priority order. An override is
#: the supported way to put big files on an external drive locally, and it is how the
#: tests exercise the staging branch without a cluster.
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

    Checked separately from `staging_root()` because the big-file helpers short-circuit to
    the repo whenever we are off-cluster — which would make an override silently do
    nothing on a laptop, the one place it is most useful.
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

    `$VSC_DATA` and project storage are shared across every project, so a path there needs
    a `<Project>/` component. The repo root already *is* the project, and adding it there
    would give `<Project>/<Project>/output`.
    """
    if root == REPO_ROOT:
        return root.joinpath(*parts)
    return root.joinpath(PROJECT_NAME, *parts)


# ---------------------------------------------------------------------------
# output/ — the single root for everything the code generates.
# ---------------------------------------------------------------------------


def outputs_dir() -> Path:
    """THE root for generated files. Nothing generated is written outside it.

    Locally `<repo>/output/`; on the cluster `$VSC_DATA/<Project>/output/`, the backed-up
    tier. One root means "what did this run produce?" and "what can I delete?" have one
    answer each — which is the whole reason the rule exists.
    """
    if on_vsc():
        return _under(data_root(), "output")
    return REPO_ROOT / "output"


def results_dir(*parts: str) -> Path:
    """Fine-grained results: one row per prediction, per-fold scores, anything large.

    THE ONE PART OF `output/` ON PROJECT STORAGE. Per-row predictions across every dataset
    and model run to gigabytes, and `$VSC_DATA` is 75 GiB — a single sweep would fill it
    and then every job that writes a log also fails. Off-cluster this is plain
    `output/results/`, so the split is invisible locally.
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
    """`output/figures/`, or one notebook's own folder inside it.

    One folder per notebook, because a notebook clears its OWN figures before drawing and
    must not be able to reach another notebook's.
    """
    root = outputs_dir() / "figures"
    return root / notebook if notebook else root


def captions_path() -> Path:
    """The ONE shared captions file for every figure in the project."""
    return figures_dir() / "CAPTIONS.md"


def all_results_path() -> Path:
    """Every notebook's printed text summary, concatenated in notebook order."""
    return outputs_dir() / "All_Results.md"


def run_dir(run_id: str) -> Path:
    """Per-run scratch inside `output/`: the resolved config, metrics, a partial state."""
    return outputs_dir() / "runs" / run_id


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
    """Every root that might hold this input, **repo first**.

    Repo first so a laptop with the data checked out works with no configuration; project
    storage second so the same code finds it on the cluster. Reading searches; writing
    picks one (`processed_dir`), which on the cluster is project storage.
    """
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
    """Model weights. Big -> project storage. Never deleted by the cleaner: they are
    either downloaded from upstream or cost a training run to reproduce."""
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

    Probes with a real create-and-delete. `mkdir(exist_ok=True)` is NOT enough: a
    directory on a shared tier can exist and still be unwritable by this user, and that is
    exactly the case this guards against. A completed run in the wrong place beats a job
    that died at hour six with nothing to show.
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
    """Refresh access times so `$VSC_SCRATCH`'s 30-day purge does not eat the files.

    `mv` and timestamp-preserving `rsync` do NOT count as an access, so data staged to
    scratch can be purged almost immediately after it lands. Copy, then call this.
    """
    for p in path.rglob("*"):
        if p.is_file():
            p.touch()


def describe() -> dict[str, str]:
    """Every resolved root, for logging at job start.

    A run that records where it wrote is a run whose output can be found six months later
    on a tier that has since been reorganised.
    """
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
