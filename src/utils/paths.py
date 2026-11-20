"""Every path in the project. Two VSC tiers, one resolver, relative to the repository root.

THE ONLY MODULE THAT BUILDS A PATH — everything else asks this one. A path assembled at a call
site with `"output_JEVPFN/" + name` is correct on a laptop and wrong on the cluster, and the failure
shows up as a full quota or an empty results directory hours into a job.

        project storage  $PROJECT_ROOT/<Project>/  big files; confirm quota/backup
    personal data    $VSC_DATA/<Project>/                   repo + output_JEVPFN/, backed up, 75 GiB
    scratch          $VSC_SCRATCH/                          purged after 30 days of no ACCESS

The VSC documentation lists personal DATA backups, but does not establish the backup policy
of this project's staging allocation. Verify it before storing paid responses. Use Lustre for
wICE I/O and allocated GPFS for Mindwell. Durable authority and mutable scratch
working overrides are distinct; suitable native immutable assets can be shared.

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

from src.utils import structure

_NOTEBOOK_SCRATCH = None

#: The per-project folder name on BOTH shared tiers.
PROJECT_NAME = "JEVPFN"

#: `parents[2]` because this file is `<root>/src/utils/paths.py`. From __file__, not the working
#: directory, so a script, a test and a notebook agree wherever they were launched from.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Explicit native working sets take precedence over the durable PROJECT aliases.
#: B200 publication/staging temporarily sets STAGING_ROOT; inheriting PROJECT_ROOT must
#: never redirect its bulk I/O back to cross-mounted Lustre. Every value is a shared root.
STAGING_ENV_VARS = ("JEVPFN_STAGING_ROOT", "JEVPFN_PROJECT_ROOT", "VSC_PROJECT_ROOT", "VSC_STAGING_ROOT")


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
    # A working override is distinct from durable authority. The selected
    # durable root takes priority over older PROJECT aliases, as activation does.
    if p := _env_path("JEVPFN_STAGING_ROOT"):
        return p
    if (on_vsc() or os.environ.get("JEVPFN_DURABLE_ROOT")) and (p := durable_project_override()):
        return p
    for var in STAGING_ENV_VARS[1:]:
        p = _env_path(var)
        if p:
            return p
    return None


def durable_project_override() -> Path | None:
    """Persistent allocation override, independent of a native scratch working set."""
    from src.cluster.storage_config import configured_root

    return configured_root()


def native_gpfs_project(root: Path | None) -> bool:
    """Whether the durable allocation can serve native Mindwell readers."""
    return root is not None and root.as_posix().startswith("/gpfs") and "/project/" in root.as_posix()


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
    if on_vsc():
        raise ValueError("Select the allocated durable PROJECT root explicitly; no filesystem fallback")
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
#: Internal phase identities, unchanged by the 18-11-2026 layout (hash/recipe/receipt keys).
PHASES = structure.PHASES

# Physical folders come from config/structure.yaml; phase identities never change.
#: Phase -> config folder (home folder of the phase).
CONFIG_GROUPS = {phase: structure.registry().home(phase).split("/")[0] for phase in PHASES}
#: Notebook ID (current and previous) -> internal phase.
NOTEBOOK_PHASES = {
    **{n.id: n.phase for n in structure.registry().notebooks},
    **{legacy: n.phase for n in structure.registry().notebooks for legacy in n.legacy},
}
#: Retired pre-research notebook stems, still accepted as `<phase folder>/<stem>`.
_RETIRED_NOTEBOOK_STEMS = {
    "01_data_exploration": "exploration", "02_jev_input_design": "exploration",
    "01_prepare_questions": "question_preparation", "03_feature_creation": "feature_creation",
    "04_decision_feature_comparison": "feature_creation",
    "01_before_feature_creation": "experiment_0", "02_after_feature_creation": "experiment_0",
    "01_paired_representation_effects": "experiment_1", "01_frozen_embeddings": "experiment_2",
    "01_target_aware_embeddings": "experiment_3", "01_native_text_systems": "experiment_4",
    "01_representation_comparison": "synthesis",
}


#: Folders without a per-experiment report index (`<folder>/manifests/reports.json`).
PREPARATION_FOLDERS = ("00_data", "02_feature_creation")


def layout():
    """The physical layout registry (config/structure.yaml)."""
    return structure.registry()


def phase_name(phase: str | None = None) -> str:
    """Validate an internal phase; previous numbered folders and owning folders are aliases."""
    value = phase or os.environ.get("JEVPFN_PHASE", "exploration")
    if value in PHASES:
        return value
    numbered = re.fullmatch(r"([0-9]+)\. (.+)", value)
    if numbered:
        if structure.legacy_phase(value) is None:
            raise ValueError(f"Incorrect phase number: {value}")
        return numbered[2]
    owners = {e.owner_phase for e in layout().entries if value in {e.folder, *e.previous_folders}}
    if len(owners) == 1:
        return owners.pop()
    raise ValueError(f"Unknown research phase: {value}")


def phase_folder(phase: str) -> str:
    """The home folder of an internal phase, e.g. experiment_1 -> exp04_decision_features."""
    return layout().home(phase_name(phase))


def routed_folder(phase: str, *parts: str) -> str:
    """The folder of `<phase>/<parts>`; registry routes split one phase across folders."""
    return layout().folder_for(phase_name(phase), parts)


def research_group(phase: str) -> str:
    """Research question of an internal phase (human grouping; not a folder)."""
    phase = phase_name(phase)
    if phase in {"rq1", "rq2", "rq3"}:
        return phase
    if phase in {"data_efficiency", "representations"}:  # B4 (exp11) and C1 (exp09): RQ2 studies
        return "rq2"
    return "rq2" if phase.startswith("experiment_") and phase != "experiment_0" or phase in {"synthesis", "regression_readout"} else "general"


def result_relative(phase: str, *parts: str) -> Path:
    """Results folder of a phase relative to an output root, routed by its first parts."""
    return Path(routed_folder(phase, "results", *parts)) / "results"


def notebook_id(notebook: str) -> str:
    """Canonical ID (the folder) of a registered notebook; other names are returned as given."""
    _validate_notebook_name(notebook)
    found = layout().notebook(notebook)
    return found.id if found else notebook


def _validate_notebook_name(notebook: str) -> list[str]:
    parts = notebook.split("/")
    if any(not p or p in {".", ".."} or "\\" in p or ":" in p for p in parts) or len(parts) > 2:
        raise ValueError("Notebook must be a name or folder/name, without path traversal.")
    return parts


def notebook_phase(notebook: str) -> str:
    if notebook in {"rq1_behavior", "rq2_prediction", "rq3_adaptation"}:
        return notebook.split("_")[0]
    parts = _validate_notebook_name(notebook)
    found = layout().notebook(notebook)
    if found:
        return found.phase
    if len(parts) == 2:
        if parts[-1] in _RETIRED_NOTEBOOK_STEMS and parts[0] in {"general", "rq1", "rq2", "rq3"}:
            return _RETIRED_NOTEBOOK_STEMS[parts[-1]]
        if parts[0] in {"general", "rq2"}:
            raise ValueError("Unknown notebook in research layout")
        if parts[0] in {"rq1", "rq3"}:
            return parts[0]
        return phase_name(parts[0])
    if notebook == "03_feature_creation":
        return "feature_creation"
    match = re.match(r"(experiment_[0-9]+)_", notebook)
    if match:
        return phase_name(match.group(1))
    return "exploration"


def notebook_folder(notebook: str) -> str:
    """Physical folder of a notebook's figures and records."""
    found = layout().notebook(notebook)
    if found:
        return found.folder
    parts = notebook.split("/")
    if len(parts) == 2 and parts[0] in layout().folders:
        return parts[0]
    return phase_folder(notebook_phase(notebook))


def notebook_stem(notebook: str) -> str:
    notebook_phase(notebook)
    found = layout().notebook(notebook)
    return found.name if found else notebook.split("/")[-1]


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


def phase_path(phase: str, *parts: str) -> Path:
    """A DATA-tier path inside a phase tree, honouring registry routes (e.g. 0B manifests)."""
    selected = phase_name(phase)
    if selected == "exploration":
        return notebook_cache_dir("exploration", *parts)
    if not parts:
        return phase_dir(selected)
    folder = routed_folder(selected, *parts)
    if folder != phase_folder(selected):
        return outputs_dir() / folder / Path(*parts)
    # Unrouted: exactly the phase's own manifests/logs helpers.
    kind, *rest = parts
    base = {"manifests": manifests_dir, "logs": logs_dir}.get(kind)
    return base(selected).joinpath(*rest) if base else phase_dir(selected).joinpath(*parts)


def results_dir(*parts: str, phase: str | None = None) -> Path:
    selected = phase_name(phase)
    if selected in {"exploration", "feature_creation"}:
        return notebook_cache_dir(selected, "results", *parts)
    if selected in {"rq1", "rq2", "rq3", "representations"} and on_vsc():
        durable = durable_project_override()
        if durable is None:
            raise ValueError("Research study results require an allocated durable PROJECT root")
        return _under(durable, OUTPUT_DIR_NAME, str(result_relative(selected, *parts)), *parts)
    return large_outputs_dir().joinpath(result_relative(selected, *parts), *parts)


def tar_results_dir(*parts: str) -> Path:
    """Immutable fold vectors and sealed PCA are shared from native GPFS PROJECT."""
    if on_vsc():
        root = durable_project_override()
        if root is None or not root.as_posix().startswith("/gpfs") or "/project/" not in root.as_posix():
            raise ValueError("TAR requires an allocated native GPFS PROJECT root")
        return _under(root, OUTPUT_DIR_NAME, str(result_relative("experiment_3", *parts)), *parts)
    return results_dir(*parts, phase="experiment_3")


def group_c_dir(phase: str, *parts: str) -> Path:
    """Group C stores (C1 Strands hidden states, C2 adapted probabilities) on native GPFS PROJECT.

    Large immutable vectors and fold bundles are shared from PROJECT exactly like TAR bundles;
    there is no DATA fallback on VSC.
    """
    if phase_name(phase) not in {"representations", "rq3"}:
        raise ValueError("Group C stores belong to the representations or rq3 phase")
    if on_vsc():
        root = durable_project_override()
        if root is None or not root.as_posix().startswith("/gpfs") or "/project/" not in root.as_posix():
            raise ValueError("Group C stores require an allocated native GPFS PROJECT root")
        return _under(root, OUTPUT_DIR_NAME, str(result_relative(phase, *parts)), *parts)
    return results_dir(*parts, phase=phase)


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


def _record_folder(notebook: str | None, phase: str | None) -> str | None:
    if notebook is not None:
        return notebook_folder(notebook) if phase is None else phase_folder(phase)
    return phase_folder(phase).split("/")[0] if phase is not None else None


def _record_parts(notebook: str | None, phase: str | None) -> tuple[str, tuple[str, ...]]:
    """(experiment folder, sub-path): `()` for a registered (flat) notebook, else `(<stem>,)`."""
    folder = _record_folder(notebook, phase)
    if folder is None:
        raise ValueError("Notebook records need a notebook or a phase (experiment folder first)")
    if notebook is None or (phase is None and layout().notebook(notebook) is not None):
        return folder, ()  # One notebook per folder: its records are the folder's own.
    return folder, (notebook_stem(notebook),)


def _unlinked(folder: Path) -> Path:
    root = notebook_outputs_dir()
    for entry in (folder, *folder.parents):
        if entry.is_symlink() or getattr(entry, "is_junction", lambda: False)():
            raise ValueError(f"Linked notebook output is not allowed: {entry}")
        if entry == root:
            break
    return folder


def figures_dir(notebook: str | None = None, *, phase: str | None = None) -> Path:
    """The ONE figures folder, or `figures/<group>/` of a notebook: its PDFs and `_figures.json`.

    The group comes from `structure.figure_group` (today: the notebook ID == its folder).
    """
    root = notebook_outputs_dir() / structure.FIGURES
    if notebook is None and phase is None:
        return root
    folder, sub = _record_parts(notebook, phase)
    # FigureSaver refuses linked folders before clearing or writing (`_safe_folder`).
    return (root / structure.figure_group(folder)).joinpath(*sub)


def notebook_artifacts(notebook: str | None = None, *, kind: str = "manifests", phase: str | None = None) -> Path:
    """`<folder>/<kind>/notebook/`: local notebook records, apart from the experiment's own."""
    if kind not in structure.NOTEBOOK_RECORD_KINDS:
        raise ValueError("Notebook artifact kind must be manifests, logs or results")
    folder, sub = _record_parts(notebook, phase)
    return _unlinked((notebook_outputs_dir() / folder / kind / structure.NOTEBOOK_RECORDS).joinpath(*sub))


def reports_dir(notebook: str | None = None, *, phase: str | None = None) -> Path:
    root = notebook_artifacts(notebook, kind="results", phase=phase)
    return root / "_report.txt" if notebook else root


def report_folder(selection: str | None) -> str | None:
    """The experiment folder (0A/0B, exp01...) with a report index; None otherwise.

    Preparation folders (data, feature creation) only appear in the project-wide documents.
    """
    if selection is None:
        return None
    folder = selection if selection in layout().folders else phase_folder(selection).split("/")[0]
    return None if folder in PREPARATION_FOLDERS else folder


def report_index_path(selection: str) -> Path | None:
    """`<folder>/manifests/reports.json`: an experiment folder's pointers into the project
    documents (no per-folder report copies); None for preparation folders."""
    folder = report_folder(selection)
    return notebook_outputs_dir() / folder / "manifests" / "reports.json" if folder else None


def captions_path() -> Path:
    """The one captions file: `figures/CAPTIONS.md`."""
    return figures_dir() / structure.CAPTIONS


def all_results_path() -> Path:
    """The one results report: `All_Results.md` at the output root."""
    return notebook_outputs_dir() / structure.ALL_RESULTS


def archive_dir(output_root: Path | None = None) -> Path:
    """`archive/`: previous-layout records, beside (never inside) an output root.

    Locally and on VSC DATA this is the checkout's git-ignored `archive/`; migrating the
    PROJECT output root archives beside it on PROJECT, so a move is never a cross-tier copy.
    """
    root = Path(output_root) if output_root is not None else notebook_outputs_dir()
    return root.parent / structure.ARCHIVE_DIR_NAME


def recorded_output_path(value) -> Path:
    """A path recorded in a previous layout (18/19-11-2026 migrations), located in the current one.

    Archived records resolve into `archive_dir()` beside the recorded output root."""
    return structure.recorded_output_path(value, OUTPUT_DIR_NAME)


def output_component(value: str) -> str:
    """A readable artifact/run identifier, never an absolute path or traversal."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("Output identifier must be a single safe path component")
    return value


def _resolved_storage_path(value: Path) -> Path:
    r"""Resolve links, then normalize Windows extended drive/UNC aliases.

    Windows resolve() can retain its internal \\?\ prefix when another writer
    creates a missing parent concurrently. Compare the same canonical spelling
    for the target and both storage roots; no path component or boundary is dropped.
    """
    resolved = Path(value).resolve()
    if os.name == "nt":
        spelling = str(resolved)
        if spelling.startswith("\\\\?\\UNC\\"):
            return Path("\\\\" + spelling[8:])
        if spelling.startswith("\\\\?\\") and len(spelling) > 5 and spelling[5] == ":":
            return Path(spelling[4:])
    return resolved


def require_project_output(folder: Path) -> Path:
    """Prevent explicit large-output overrides from filling DATA on VSC."""
    folder = _resolved_storage_path(folder)
    if on_vsc() and folder.is_relative_to(_resolved_storage_path(data_root())):
        raise ValueError(
            "Large outputs must use project storage outside personal DATA, including when the PROJECT root is misconfigured"
        )
    if _use_staging() and not folder.is_relative_to(_resolved_storage_path(_under(staging_root()))):
        raise ValueError("Large outputs must remain on configured project storage")
    return folder


def prediction_dir(run_id: str, *, phase: str) -> Path:
    return results_dir("predictions", output_component(run_id), phase=phase)


def dataset_cache_path(dataset_id: str) -> Path:
    """One durable file per dataset, containing responses and all feature versions."""
    return jev_cache_path(f"jev_cache/{output_component(dataset_id)}.sqlite3")


def decision_inputs_dir() -> Path:
    if os.environ.get("JEVPFN_SHARED_IMMUTABLES") == "1" and durable_project_override():
        return _under(durable_project_override(), "data", "decision_inputs")
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
        root = (durable_project_override() if os.environ.get("JEVPFN_SHARED_IMMUTABLES") == "1"
                else None) or staging_root()
        return _under(root, "data", "raw", *parts)
    return REPO_ROOT.joinpath("data", "raw", *parts)


def processed_dir(*parts: str) -> Path:
    """Legacy processed-data location, protected from output cleanup; notebooks do not use it."""
    if _use_staging():
        return _under(staging_root(), "data", "processed", *parts)
    return REPO_ROOT.joinpath("data", "processed", *parts)


def data_search_paths(*parts: str) -> list[Path]:
    """Every root that might hold this input, **repo first** — so a laptop with the data checked
    out works unconfigured, and the same code finds it on the cluster."""
    roots = [REPO_ROOT.joinpath("data", *parts)]
    if _use_staging():
        staged = _under((durable_project_override() if os.environ.get("JEVPFN_SHARED_IMMUTABLES") == "1" else None) or staging_root(), "data", *parts)
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
        root = (durable_project_override() if os.environ.get("JEVPFN_SHARED_IMMUTABLES") == "1"
                else None) or staging_root()
        return _under(root, "checkpoints", *parts)
    return REPO_ROOT.joinpath("checkpoints", *parts)


def tokenizer_dir(provider: str) -> Path:
    return checkpoints_dir("tokenizers", output_component(provider))


def token_audit_dir() -> Path:
    if os.environ.get("JEVPFN_SHARED_IMMUTABLES") == "1" and durable_project_override():
        return _under(durable_project_override(), OUTPUT_DIR_NAME, phase_folder("experiment_0"),
                      "results", "token_audit")
    return results_dir("token_audit", phase="experiment_0")


def config_path(name: str) -> Path:
    """Resolve a logical config key through the layout registry.

    Current keys (`exp04_decision_features/main`), internal-phase keys (`experiment_1/main`),
    registered stems (`main`) and previous folders (`rq2/4. experiment_1/main`) all resolve to
    the same file. Previous locations are read only when the current file is absent, so an
    archived checkout or fixture keeps working.
    """
    stem = name[:-5] if name.endswith(".yaml") else name
    if re.match(r"experiment_[0-9]+/", stem):
        phase_name(stem.split("/")[0])  # Retired experiment numbers have no configuration.
    root = (REPO_ROOT / "config").resolve()
    current = structure.config_relative(stem)
    candidates = [current, stem, *_legacy_config_locations(stem, current)]
    path = next(
        (root / f"{c}.yaml" for c in candidates if (root / f"{c}.yaml").is_file()),
        root / f"{current}.yaml",
    ).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Configuration must remain under config/.")
    return path


def config_file(filename: str) -> Path:
    """A registered non-YAML config file (e.g. `question_versions.csv`)."""
    folder = structure.registry().config_index().get(filename)
    if folder is None or Path(filename).name != filename:
        raise ValueError(f"Unregistered config file: {filename}")
    return REPO_ROOT / "config" / folder / filename


def _legacy_config_locations(stem: str, current: str) -> list[str]:
    folder, _, rest = current.partition("/")
    if not rest:
        return []
    phases = [p for p in PHASES if layout().home(p).split("/")[0] == folder]
    found = []
    for phase in phases:
        legacy = structure.LEGACY_PHASE_FOLDERS.get(phase)
        if legacy:
            found += [f"general/{legacy}/{rest}", f"rq2/{legacy}/{rest}", f"{legacy}/{rest}"]
    found += [f"{group}/{rest}" for group in ("rq1", "rq3")]
    found += [f"{old}/{rest}" for old, new in layout().previous_folders.items() if new == folder]
    return found


def jev_cache_path(relative: str = "jev_cache") -> Path:
    """Permanent responses survive every output-cleanup operation."""
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
