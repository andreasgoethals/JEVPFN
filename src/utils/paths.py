"""Every path in the project. Two VSC tiers, one resolver, relative to the repository root.

THE ONLY MODULE THAT BUILDS A PATH — everything else asks this one. A path assembled at a call
site with `"output_JEVPFN/" + name` is correct on a laptop and wrong on the cluster, and the failure
shows up as a full quota or an empty results directory hours into a job.

        project storage  /lustre1/project/stg_00211/<Project>/  big files; confirm quota/backup
    personal data    $VSC_DATA/<Project>/                   repo + output_JEVPFN/, backed up, 75 GiB
    scratch          $VSC_SCRATCH/                          purged after 30 days of no ACCESS

The VSC documentation lists personal DATA backups, but does not establish the backup policy
of this project's staging allocation. Verify it before storing paid responses. Use Lustre for
wICE I/O; select native GPFS through the staging override before Mindwell bulk I/O.

Experiment tables and predictions use project storage. Notebooks always run locally; their
figures/reports ignore VSC experiment storage variables. Only compact summaries,
experiment manifests and bounded operational logs use personal DATA. Portable artifacts may
carry a small manifest beside their data as well as a DATA summary/index.

OFF-CLUSTER EVERY TIER COLLAPSES INTO THE REPO. Pretending `/lustre1` exists on a laptop would
mean two code paths, and the one that only runs on the cluster is the one that breaks.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

_NOTEBOOK_SCRATCH = None

#: The per-project folder name on BOTH shared tiers.
PROJECT_NAME = "JEVPFN"

#: Fallback for project storage when the site variable is not set. The literal path is a
#: last resort, not the primary source — VSC has moved it before.
STAGING_FALLBACK = "/lustre1/project/stg_00211"

#: `parents[2]` because this file is `<root>/src/utils/paths.py`. From __file__, not the working
#: directory, so a script, a test and a notebook agree wherever they were launched from.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Overrides for project storage, in priority order. The supported way to put big files on an
#: external drive locally, and how the tests exercise the staging branch without a cluster.
STAGING_ENV_VARS = ("JEVPFN_STAGING_ROOT", "VSC_STAGING_ROOT")


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
    """Project storage — verify its quota and backup policy with the allocation owner."""
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
    if root == REPO_ROOT or root.name == PROJECT_NAME:
        return root.joinpath(*parts)
    return root.joinpath(PROJECT_NAME, *parts)


# ---------------------------------------------------------------------------
# output_JEVPFN/ — the single root for everything the code generates.
# ---------------------------------------------------------------------------


OUTPUT_DIR_NAME = "output_JEVPFN"
PHASES = (
    "exploration",
    "question_preparation",
    "feature_creation",
    "experiment_0",
    "experiment_1",
    "experiment_2",
    "experiment_3",
)


def phase_name(phase: str | None = None) -> str:
    value = phase or os.environ.get("JEVPFN_PHASE", "exploration")
    numbered = re.fullmatch(r"([0-9]+)\. (.+)", value)
    if numbered:
        original = value
        value = numbered[2]
        if phase_folder(value) != original:
            raise ValueError(f"Incorrect phase number: {original}")
    if value not in PHASES and not re.fullmatch(r"experiment_[0-9]+", value):
        raise ValueError(f"Unknown research phase: {value}")
    return value


def phase_folder(phase: str) -> str:
    """Stable logical phase names; numbered folders shared by config/notebooks/output."""
    name = phase_name(phase)
    index = (
        int(name.removeprefix("experiment_")) + 3
        if name.startswith("experiment_")
        else ("exploration", "question_preparation", "feature_creation").index(name)
    )
    return f"{index}. {name}"


def notebook_phase(notebook: str) -> str:
    parts = notebook.split("/")
    if any(not p or p in {".", ".."} or "\\" in p or ":" in p for p in parts) or len(parts) > 2:
        raise ValueError("Notebook must be a name or phase/name, without path traversal.")
    if len(parts) == 2:
        return phase_name(parts[0])
    if notebook == "03_feature_creation":
        return "feature_creation"
    match = re.match(r"(experiment_[0-9]+)_", notebook)
    if match:
        return match.group(1)
    return "exploration"


def notebook_stem(notebook: str) -> str:
    notebook_phase(notebook)
    return notebook.split("/")[-1]


def outputs_dir() -> Path:
    """Small outputs on personal DATA; locally all outputs share this root."""
    return _under(data_root(), OUTPUT_DIR_NAME)


def large_outputs_dir() -> Path:
    """Large tables on project storage; never silently fall back to personal DATA."""
    return _under(staging_root(), OUTPUT_DIR_NAME) if _use_staging() else outputs_dir()


def phase_dir(phase: str | None = None) -> Path:
    selected = phase_name(phase)
    if selected == "exploration":
        return notebook_cache_dir("exploration")
    return outputs_dir() / phase_folder(selected)


def results_dir(*parts: str, phase: str | None = None) -> Path:
    selected = phase_name(phase)
    if selected in {"exploration", "feature_creation"}:
        return notebook_cache_dir(selected, "results", *parts)
    return large_outputs_dir().joinpath(phase_folder(selected), "results", *parts)


def notebook_outputs_dir() -> Path:
    """Notebooks always run locally, regardless of experiment storage variables."""
    return (_env_path("JEVPFN_LOCAL_STORAGE_ROOT") or REPO_ROOT) / OUTPUT_DIR_NAME


def notebook_cache_dir(*parts: str) -> Path:
    """Disposable audit/budget intermediates; shared by kernels in a runner invocation."""
    global _NOTEBOOK_SCRATCH
    root = _env_path("JEVPFN_NOTEBOOK_SCRATCH")
    if root is None:
        if _NOTEBOOK_SCRATCH is None:
            _NOTEBOOK_SCRATCH = tempfile.TemporaryDirectory(prefix="jevpfn-notebooks-")
        root = Path(_NOTEBOOK_SCRATCH.name)
    return root.joinpath(*parts)


def logs_dir(phase: str | None = None) -> Path:
    return phase_dir(phase) / "logs"


def manifests_dir(phase: str | None = None) -> Path:
    return phase_dir(phase) / "manifests"


def figures_dir(notebook: str | None = None, *, phase: str | None = None) -> Path:
    """All figures share one root, then phase and notebook subdirectories."""
    root = notebook_outputs_dir() / "figures"
    if notebook is not None:
        inferred_phase = notebook_phase(notebook)  # Also validate a plain notebook name.
        root = root / phase_folder(phase or inferred_phase)
    elif phase is not None:
        root = root / phase_folder(phase)
    return root / notebook_stem(notebook) if notebook else root


def reports_dir(notebook: str | None = None, *, phase: str | None = None) -> Path:
    root = figures_dir(notebook, phase=phase)
    return root / "_report.txt" if notebook else root


def captions_path(phase: str | None = None) -> Path:
    root = figures_dir()
    if phase and phase_name(phase).startswith("experiment_"):
        root /= phase_folder(phase)
    return root / "Captions.md"


def all_results_path(phase: str | None = None) -> Path:
    root = notebook_outputs_dir()
    if phase and phase_name(phase).startswith("experiment_"):
        root /= phase_folder(phase)
    return root / "All Results.md"


def output_component(value: str) -> str:
    """A readable artifact/run identifier, never an absolute path or traversal."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("Output identifier must be a single safe path component")
    return value


def require_project_output(folder: Path) -> Path:
    """Prevent explicit large-output overrides from filling DATA on VSC."""
    folder = Path(folder).resolve()
    if on_vsc() and folder.is_relative_to(data_root().resolve()):
        raise ValueError(
            "Large outputs must use project storage outside personal DATA, including when the PROJECT root is misconfigured"
        )
    if _use_staging() and not folder.is_relative_to(_under(staging_root()).resolve()):
        raise ValueError("Large outputs must remain on configured project storage")
    return folder


def prediction_dir(run_id: str, *, phase: str) -> Path:
    return results_dir("predictions", output_component(run_id), phase=phase)


def feature_artifact_dir(dataset_id: str, representation: str, artifact_id: str) -> Path:
    parts = [output_component(v) for v in (dataset_id, representation, artifact_id)]
    return jev_cache_path("/".join(["jev_cache", "features", *parts]))


def dataset_cache_path(dataset_id: str) -> Path:
    """One durable file per dataset, containing responses and all feature versions."""
    return jev_cache_path(f"jev_cache/{output_component(dataset_id)}.sqlite3")


def decision_inputs_dir() -> Path:
    return jev_cache_path("decision_inputs")


def dataset_question_path(dataset_id: str) -> Path:
    """One dataset file holds versioned questions, unique texts and all source-row links."""
    return decision_inputs_dir() / f"{output_component(dataset_id)}.sqlite3"


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


def tokenizer_dir(provider: str) -> Path:
    return checkpoints_dir("tokenizers", output_component(provider))


def token_audit_dir() -> Path:
    return results_dir("token_audit", phase="experiment_0")


def config_path(name: str) -> Path:
    """Resolve logical config IDs and explicit numbered paths without copying configs."""
    stem = name[:-5] if name.endswith(".yaml") else name
    prerequisites = {
        "audit": "exploration",
        "datasets": "exploration",
        "template_example": "exploration",
        "question_design": "question_preparation",
        "dataset_semantics": "question_preparation",
        "question_preparation": "question_preparation",
        "feature_creation": "feature_creation",
        "decision_pilot": "feature_creation",
        "extractors": "feature_creation",
        "feature_comparison": "feature_creation",
    }
    if stem in prerequisites:
        stem = f"{phase_folder(prerequisites[stem])}/{stem}"
    elif "/" in stem and stem.split("/")[0] in PHASES or re.match(r"experiment_[0-9]+/", stem):
        prefix, rest = stem.split("/", 1)
        stem = f"{phase_folder(prefix)}/{rest}"
    root = (REPO_ROOT / "config").resolve()
    path = (root / f"{stem}.yaml").resolve()
    if not path.is_relative_to(root):
        raise ValueError("Configuration must remain under config/.")
    return path


def jev_cache_path(relative: str = "jev_cache") -> Path:
    """Paid responses survive both ordinary output cleanup and --processed cleanup."""
    root = _under(staging_root(), "data") if _use_staging() else REPO_ROOT / "data"
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or any(
        path.is_relative_to((root / name).resolve()) for name in ("processed", "raw")
    ):
        raise ValueError("Jev cache must stay under data/, outside data/raw/ and data/processed/.")
    return path


def source_snapshot_path(commit: str, filename: str) -> Path:
    return REPO_ROOT / "src" / "data" / "upstream" / commit / filename


def dataset_file(directory: str, filename: str) -> Path:
    """Raw files live directly in each numbered folder; versions remain in metadata."""
    if Path(directory).name != directory or directory in {".", ".."} or "\\" in directory:
        raise ValueError("Dataset directory must be a single local folder name.")
    return raw_dir(directory, filename)


def manifest_path(name: str, *, phase: str | None = None) -> Path:
    return manifests_dir(phase) / f"{name}.json"


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
    # Numbered phase directories ("0. exploration") are not files despite their dot.
    target = path.parent if path.suffix and not re.fullmatch(r"[0-9]+\. .+", path.name) else path
    target.mkdir(parents=True, exist_ok=True)
    return path


def resolve_writable(preferred: Path, fallback: Path | None = None) -> Path:
    """Probe a required storage directory; never redirect large files to personal DATA.

    The legacy fallback argument is accepted for caller compatibility but is never used.
    A storage error must be fixed before a run starts.
    """
    import tempfile

    preferred.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=preferred) as probe:
        probe.write(b"ok")
    return preferred


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
        "raw_dir": str(raw_dir()),
        "jev_cache": str(jev_cache_path()),
        "phase": phase_name(),
        "checkpoints_dir": str(checkpoints_dir()),
    }
