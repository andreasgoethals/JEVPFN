"""The one registry of physical folder names: `config/structure.yaml`.

Internal phase IDs stay the keys of every hash, recipe, receipt and manifest. This module
only decides WHERE a phase's config, notebooks and outputs live, and how previous layouts
(numbered `general/rq1/rq2/rq3`; then `<folder>/<name>` notebooks and `exp09_adaptation`
before 19-11-2026; then `{logs,manifests,results}/notebooks/<folder>/` records, a short-lived
`<folder>/figures/` with per-folder reports, and `output_JEVPFN/archives/`) translate into the
current one:

    figures/<group>/          one folder per notebook (`figure_group`): PDFs and `_figures.json`
    figures/CAPTIONS.md       every caption;  All_Results.md   every notebook's report
    <folder>/{logs,manifests,results}/[notebook/]   everything else, per experiment folder
    ../archive/               previous-layout records, beside (not inside) the output root

`paths` builds every path from it; `layout_migration` moves existing trees with the same translation, so code and
data always agree after a migration.

Import-light on purpose: `paths` imports this module, so it must not import `paths`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path, PurePosixPath

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "config" / "structure.yaml"

#: Internal phase identities. Never renamed; folders are resolved from the registry.
PHASES = (
    "exploration",
    "question_preparation",
    "feature_creation",
    "experiment_0",
    "experiment_1",
    "experiment_2",
    "experiment_3",
    "experiment_4",
    "synthesis",
    "rq1",
    "rq2",
    "rq3",
    "regression_readout",
    "data_efficiency",
    "representations",  # Group C, C1 (exp09_representations, 20-11-2026)
)

#: The previous physical folder of each numbered phase (frozen history, not configuration).
LEGACY_PHASE_FOLDERS = {
    "exploration": "0. exploration",
    "question_preparation": "1. question_preparation",
    "feature_creation": "2. feature_creation",
    "experiment_0": "3. experiment_0",
    "experiment_1": "4. experiment_1",
    "experiment_2": "5. experiment_2",
    "experiment_3": "6. experiment_3",
    "experiment_4": "7. experiment_4",
    "synthesis": "8. synthesis",
}
#: Phases whose scientific results lived under `rq2/<legacy folder>/` before 18-11-2026.
LEGACY_RQ2_RESULT_PHASES = ("experiment_1", "experiment_2", "experiment_3", "experiment_4", "synthesis")
LEGACY_GROUPS = ("general", "rq1", "rq2", "rq3")
#: Previous-layout records live in `archive/`, a sibling of the output root (git-ignored at
#: the repository root; on VSC beside the migrated DATA or PROJECT output root, so moving there
#: is a same-filesystem rename). Paths below are relative to the output root.
ARCHIVE_DIR_NAME = "archive"
ARCHIVE_ROOT = PurePosixPath("..", ARCHIVE_DIR_NAME)
#: The archive's former location inside the output root (until 19-11-2026).
PREVIOUS_ARCHIVES = "archives"
#: Notebook records of retired notebook IDs move here, preserving their relative path.
ARCHIVE = ARCHIVE_ROOT / "layout_20261118"
#: Records of notebooks merged into one flat notebook on 19-11-2026 (cannot share a folder).
MERGED_ARCHIVE = ARCHIVE_ROOT / "layout_20261119"
#: Retired per-folder report copies (`<folder>/All_Results.md`, `[figures/]<folder>/CAPTIONS.md`).
REPORTS_ARCHIVE = ARCHIVE_ROOT / "folder_reports_20261119"
SCHEMA = "jevpfn.structure.v2"
NOTEBOOK_RECORD_KINDS = ("logs", "manifests", "results")
#: Sub-folder of `<folder>/{logs,manifests,results}/` holding the folder notebook's records,
#: apart from the experiment's own manifests/logs/results.
NOTEBOOK_RECORDS = "notebook"
#: The one top-level figures folder: `figures/<group>/` per notebook, plus CAPTIONS.md.
FIGURES = "figures"
#: Notebook records kept beside the PDFs in `figures/<group>/`.
FIGURE_RECORDS = ("_figures.json", "_stdout.txt", "_failure.txt")
#: The two generated project documents: `All_Results.md` and `figures/CAPTIONS.md`.
ALL_RESULTS = "All_Results.md"
CAPTIONS = "CAPTIONS.md"
#: Report names of earlier layouts, including retired per-folder copies.
REPORT_NAMES = frozenset({ALL_RESULTS, CAPTIONS, "All Results.md", "all_results.md", "Captions.md"})
PLACEMENTS = ("main", "appendix", "undecided")
#: Generated per-folder index (run_notebooks); it stays in the experiment folder that wrote it.
FOLDER_REPORTS = {("manifests", "reports.json")}


def figure_group(folder: str) -> str:
    """THE figure grouping: `figures/<group>/` of the notebook in registry folder `folder`.

    Today one group per notebook, named after it (notebook ID == folder). Change the grouping
    here only; `paths.figures_dir` and the layout migration (`translate_output`) both use it.
    """
    return folder


def figure_folder(group: str, reg: Registry | None = None) -> str | None:
    """The registry folder whose notebook draws into `figures/<group>/` (inverse of the above)."""
    reg = reg or registry()
    found = [folder for folder in reg.folders if figure_group(folder) == group]
    return found[0] if len(found) == 1 else None


@dataclass(frozen=True)
class Notebook:
    """The one notebook of a folder: `notebooks/<folder>.ipynb`, ID and name = folder."""

    id: str
    folder: str
    name: str
    phase: str
    legacy: tuple[str, ...]
    merged: tuple[str, ...] = ()

    @property
    def aliases(self) -> tuple[str, ...]:
        return (*self.legacy, *self.merged)


@dataclass(frozen=True)
class Planned:
    """A reserved experiment number; documented only, never given a folder."""

    id: str
    folder: str
    title: str
    status: str


@dataclass(frozen=True)
class Entry:
    id: str
    folder: str
    title: str
    phases: tuple[str, ...]
    routes: dict = field(default_factory=dict)
    nested: dict = field(default_factory=dict)
    notebooks: tuple[Notebook, ...] = ()
    configs: tuple[str, ...] = ()
    previous_folders: tuple[str, ...] = ()

    @property
    def owner_phase(self) -> str:
        """The internal phase that owns this folder's output tree."""
        return self.phases[0] if self.phases else next(iter(self.routes))


@dataclass(frozen=True)
class Registry:
    entries: tuple[Entry, ...]
    paper: dict
    planned: tuple[Planned, ...] = ()

    # -- lookups -----------------------------------------------------------------------
    @property
    def folders(self) -> tuple[str, ...]:
        return tuple(e.folder for e in self.entries)

    def entry(self, key: str) -> Entry:
        for e in self.entries:
            if key in {e.id, e.folder, *e.previous_folders}:
                return e
        raise KeyError(f"Unknown layout entry: {key}")

    @property
    def previous_folders(self) -> dict[str, str]:
        """Earlier folder name -> current folder (e.g. exp09_adaptation -> exp10_adaptation)."""
        return {old: e.folder for e in self.entries for old in e.previous_folders}

    def home(self, phase: str) -> str:
        """The default folder of an internal phase (may be nested for a retired phase)."""
        for e in self.entries:
            if phase in e.phases:
                return e.folder
            if phase in e.nested:
                return f"{e.folder}/{e.nested[phase]}"
        raise ValueError(f"Phase has no registered folder: {phase}")

    def routes(self) -> list[tuple[str, tuple[str, ...], str]]:
        found = [
            (phase, PurePosixPath(prefix).parts, e.folder)
            for e in self.entries
            for phase, prefixes in e.routes.items()
            for prefix in prefixes
        ]
        return sorted(found, key=lambda r: -len(r[1]))

    def folder_for(self, phase: str, parts=()) -> str:
        """The folder holding `<phase>/<parts>`: a matching route, else the phase's home."""
        parts = tuple(str(p) for p in parts)
        for routed_phase, prefix, folder in self.routes():
            if routed_phase == phase and parts[: len(prefix)] == prefix:
                return folder
        return self.home(phase)

    def owner_of(self, folder_parts: tuple[str, ...]) -> tuple[str, int] | None:
        """(phase, consumed parts) for a current physical folder prefix, longest first."""
        candidates = []
        for e in self.entries:
            for phase, sub in e.nested.items():
                candidates.append(((e.folder, sub), phase))
            candidates.append(((e.folder,), e.owner_phase))
        for prefix, phase in sorted(candidates, key=lambda c: -len(c[0])):
            if folder_parts[: len(prefix)] == prefix:
                return phase, len(prefix)
        return None

    @property
    def notebooks(self) -> tuple[Notebook, ...]:
        return tuple(n for e in self.entries for n in e.notebooks)

    def notebook(self, notebook_id: str) -> Notebook | None:
        """A current notebook by folder ID, previous (legacy or merged) ID or unique old stem."""
        for n in self.notebooks:
            if notebook_id in {n.id, *n.aliases}:
                return n
        if notebook_id in self.previous_folders:
            found = self.entry(notebook_id).notebooks
            return found[0] if found else None
        stems = [n for n in self.notebooks if any(x.split("/")[-1] == notebook_id for x in n.aliases)]
        return stems[0] if len(stems) == 1 else None

    def previous_notebook(self, identity: str) -> Notebook | None:
        """The current notebook whose legacy or merged IDs contain exactly `identity`."""
        return next((n for n in self.notebooks if identity in n.aliases), None)

    def config_index(self) -> dict[str, str]:
        """Registered config file stem (or data filename) -> folder."""
        return {name: e.folder for e in self.entries for name in e.configs}


def _load(path: Path) -> Registry:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA:
        raise ValueError("Unknown layout registry schema")
    entries = []
    for item in raw["entries"]:
        folder = item["folder"]
        if "notebooks" in item:
            raise ValueError("One flat notebook per folder: use `notebook:`, not `notebooks:`")
        spec = item.get("notebook")
        notebooks = (Notebook(folder, folder, folder, spec["phase"], tuple(spec.get("legacy", ())),
                              tuple(spec.get("merged", ()))),) if spec else ()
        entries.append(Entry(
            id=item["id"], folder=folder, title=item["title"],
            phases=tuple(item.get("phases", ())),
            routes={k: tuple(v) for k, v in (item.get("routes") or {}).items()},
            nested=dict(item.get("nested") or {}),
            notebooks=notebooks, configs=tuple(item.get("configs", ())),
            previous_folders=tuple(item.get("previous_folders", ())),
        ))
    planned = tuple(Planned(p["id"], p["folder"], p["title"], p["status"])
                    for p in raw.get("planned") or ())
    registry = Registry(tuple(entries), dict(raw.get("paper") or {}), planned)
    validate(registry)
    return registry


def validate(registry: Registry) -> None:
    ids = [e.id for e in registry.entries] + [p.id for p in registry.planned]
    folders = ([e.folder for e in registry.entries] + [p.folder for p in registry.planned]
               + [old for e in registry.entries for old in e.previous_folders])
    if len(set(ids)) != len(ids) or len(set(folders)) != len(folders):
        raise ValueError("Layout IDs and folders (current, planned and previous) must be unique")
    for folder in folders:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]*", folder):
            raise ValueError(f"Unsafe layout folder name: {folder}")
    homes = [p for e in registry.entries for p in (*e.phases, *e.nested)]
    if sorted(homes) != sorted(PHASES):
        raise ValueError("Every internal phase needs exactly one home folder")
    for e in registry.entries:
        if not e.phases and not e.routes:
            raise ValueError(f"Layout entry {e.id} owns no phase or route")
        if set(e.routes) - set(PHASES) or any(n.phase not in PHASES for n in e.notebooks):
            raise ValueError(f"Layout entry {e.id} names an unknown phase")
    configs = [c for e in registry.entries for c in e.configs]
    if len(set(configs)) != len(configs):
        raise ValueError("Registered config names must be unique")
    names = [n.name for n in registry.notebooks]
    legacy = [x for n in registry.notebooks for x in n.aliases]
    if len(set(names)) != len(names) or len(set(legacy)) != len(legacy):
        raise ValueError("Notebook names and previous notebook IDs must be unique")
    if set(legacy) & set(names):
        raise ValueError("A previous notebook ID cannot equal a current notebook ID")
    if set(registry.paper) - set(ids):
        raise ValueError("Paper mapping names an unknown layout entry")
    for value in registry.paper.values():
        if value.get("placement", "undecided") not in PLACEMENTS:
            raise ValueError("Paper placement must be main, appendix or undecided")


@lru_cache(maxsize=4)
def _cached(path: str, mtime: float) -> Registry:
    return _load(Path(path))


def registry(path: Path | None = None) -> Registry:
    path = Path(path or REGISTRY)
    return _cached(str(path), path.stat().st_mtime)


def paper_mapping(path: Path | None = None) -> dict[str, dict]:
    """Paper placement per entry; anything not decided reads as `undecided`."""
    reg = registry(path)
    return {
        e.id: {"placement": reg.paper.get(e.id, {}).get("placement", "undecided"),
               }
        for e in reg.entries
    }


# ---------------------------------------------------------------------------
# Previous layout -> current layout.
# ---------------------------------------------------------------------------


def legacy_phase(name: str) -> str | None:
    """Phase of a previous numbered folder ("3. experiment_0"), or None."""
    for phase, folder in LEGACY_PHASE_FOLDERS.items():
        if name == folder:
            return phase
    return None


def _legacy_output_prefixes(reg: Registry | None = None) -> list[tuple[tuple[str, ...], str]]:
    reg = reg or registry()
    prefixes = [((folder,), phase) for phase, folder in LEGACY_PHASE_FOLDERS.items()]
    prefixes += [((old,), reg.entry(old).owner_phase) for old in reg.previous_folders]
    prefixes += [(("rq2", LEGACY_PHASE_FOLDERS[p]), p) for p in LEGACY_RQ2_RESULT_PHASES]
    prefixes += [(("rq2", "regression_readout"), "regression_readout"),
                 (("regression_readout",), "regression_readout"),
                 (("rq1",), "rq1"), (("rq2",), "rq2"), (("rq3",), "rq3")]
    return sorted(prefixes, key=lambda p: -len(p[0]))


def logical_output(parts, reg: Registry | None = None) -> tuple[str, tuple[str, ...]] | None:
    """(internal phase, sub-path) of a phase tree path, in either layout; None otherwise."""
    reg = reg or registry()
    parts = tuple(parts)
    for prefix, phase in _legacy_output_prefixes(reg):
        if parts[: len(prefix)] == prefix:
            return phase, parts[len(prefix):]
    owned = reg.owner_of(parts)
    if owned:
        phase, consumed = owned
        return phase, parts[consumed:]
    return None


def _notebook_record(parts, reg: Registry) -> tuple[str, ...] | None:
    """figures/<folder>/... and {logs,manifests,results}/notebooks/<folder>/...

    Current records are flat (`figures/<folder>/a.pdf`). A previous notebook ID
    (`<folder>/<name>` or `rq2/01_decisions`) moves its records into its folder; records of
    notebooks merged into one are archived; earlier folder names move to the current name.
    """
    if parts[:1] == ("figures",):
        head, rest = parts[:1], parts[1:]
    elif len(parts) >= 2 and parts[0] in NOTEBOOK_RECORD_KINDS and parts[1] == "notebooks":
        head, rest = parts[:2], parts[2:]
    else:
        return None
    if not rest:
        return parts
    group = rest[0]
    if len(rest) == 1:
        return None  # A folder directory: its records translate individually.
    identity = f"{group}/{rest[1]}"
    previous = reg.previous_notebook(identity)
    if previous is not None and identity in previous.merged:
        return (*MERGED_ARCHIVE.parts, *parts)
    if previous is not None:
        return (*head, previous.folder, *rest[2:])
    if group in reg.folders:
        return parts  # A current flat record.
    if group in reg.previous_folders:
        return (*head, reg.previous_folders[group], *rest[1:])
    known_group = group in LEGACY_GROUPS or legacy_phase(group) is not None
    return (*ARCHIVE.parts, *parts) if known_group else None  # Records of a retired notebook.


def _folder_record(parts, reg: Registry) -> tuple[str, ...]:
    """Kind-first notebook record -> current location.

    `figures/<folder>/[<stem>/]x` -> `figures/<group>/[<stem>/]x`;
    `{logs,manifests,results}/notebooks/<folder>/[<stem>/]x` -> `<folder>/<kind>/notebook/...`,
    except `_figures.json`/`_stdout.txt`/`_failure.txt`, which sit beside the PDFs.
    Archived and unknown-folder records are returned unchanged.
    """
    if parts[:1] == (FIGURES,):
        rest = parts[1:]
        if not rest or rest[0] not in reg.folders:
            return parts
        return (FIGURES, figure_group(rest[0]), *rest[1:])
    if len(parts) >= 3 and parts[0] in NOTEBOOK_RECORD_KINDS and parts[1] == "notebooks":
        folder, rest = parts[2], parts[3:]
        if folder not in reg.folders:
            return parts
        if rest and rest[-1] in FIGURE_RECORDS:
            return (FIGURES, figure_group(folder), *rest)
        return (folder, parts[0], NOTEBOOK_RECORDS, *rest)
    return parts


def _folder_report(parts, reg: Registry) -> bool:
    """A retired per-folder report copy, e.g. `exp02_robustness/All_Results.md`,
    `figures/exp02_robustness/Captions.md` or `3. experiment_0/All Results.md`."""
    if len(parts) < 2 or parts[-1] not in REPORT_NAMES:
        return False
    return parts[0] == FIGURES or logical_output(parts, reg) is not None


def translate_output(relative, reg: Registry | None = None) -> PurePosixPath | None:
    """Current location of a path relative to an output root; None when not a layout path.

    Accepts previous-layout and current-layout paths, so a path recorded before or after the
    migration (including a relative `..` reference across a routed folder) resolves the same.
    Archived records translate to `../archive/...`, outside the output root.
    """
    reg = reg or registry()
    parts = PurePosixPath(*Path(relative).parts).parts if str(relative) not in {"", "."} else ()
    if not parts:
        return None
    if parts[0] == PREVIOUS_ARCHIVES:  # `output_JEVPFN/archives/<x>` -> `archive/<x>`.
        return PurePosixPath(*ARCHIVE_ROOT.parts, *parts[1:]) if len(parts) > 1 else None
    if parts in {(ALL_RESULTS,), (".reports.lock",), (FIGURES, CAPTIONS)}:
        return PurePosixPath(*parts)
    if parts in {("All Results.md",), ("all_results.md",)}:
        return PurePosixPath(ALL_RESULTS)
    if parts in {(CAPTIONS,), ("Captions.md",), (FIGURES, "Captions.md")}:
        return PurePosixPath(FIGURES, CAPTIONS)
    if _folder_report(parts, reg):
        return PurePosixPath(*REPORTS_ARCHIVE.parts, *parts)
    record = _notebook_record(parts, reg)
    if record is not None or parts[0] in {FIGURES, *NOTEBOOK_RECORD_KINDS}:
        return PurePosixPath(*_folder_record(record, reg)) if record is not None else None
    logical = logical_output(parts, reg)
    if logical is None:
        return None
    phase, sub = logical
    folder = reg.previous_folders.get(parts[0], parts[0])
    if parts[1:2] == (FIGURES,) and folder in reg.folders:
        # `<folder>/figures/...` (briefly used on 19-11-2026) -> `figures/<group>/...`.
        return PurePosixPath(FIGURES, figure_group(folder), *parts[2:])
    if parts[0] in reg.folders and (sub in FOLDER_REPORTS or _own_notebook_record(parts, sub)):
        return PurePosixPath(*parts)  # Each current folder owns its index and notebook records.
    return PurePosixPath(reg.folder_for(phase, sub), *sub)


def _own_notebook_record(parts, sub) -> bool:
    """`<folder>/<kind>/notebook/...` (possibly in a nested phase)."""
    return len(sub) >= 2 and sub[0] in NOTEBOOK_RECORD_KINDS and sub[1] == NOTEBOOK_RECORDS


def recorded_output_path(value, output_dir_name: str = "output_JEVPFN") -> Path:
    """Locate a path recorded before the layout migration, e.g. in reviewed acceptances.

    Unchanged when it exists or names no output tree. The caller still verifies checksums.
    """
    path = Path(value)
    if path.exists():
        return path
    parts = Path(os.path.normpath(path)).parts
    if output_dir_name not in parts:
        return path
    index = len(parts) - 1 - parts[::-1].index(output_dir_name)
    moved = translate_output(PurePosixPath(*parts[index + 1:])) if parts[index + 1:] else None
    if moved is None:
        return path
    # Archived records resolve to `archive/` beside the output root (`..` is normalised).
    candidate = Path(os.path.normpath(Path(*parts[: index + 1], *moved.parts)))
    return candidate if candidate.exists() else path


# ---------------------------------------------------------------------------
# Config files.
# ---------------------------------------------------------------------------


def config_relative(name: str, reg: Registry | None = None) -> str:
    """Logical config key -> path under config/, without suffix for YAML files.

    Accepts current keys (`exp04_decision_features/main`), registered stems (`main`), internal
    phase keys (`experiment_1/main`, `feature_creation/jev.local`), and previous folders
    (`4. experiment_1/main`, `rq2/4. experiment_1/main`, `general/3. experiment_0/debug`,
    `rq1/behavior`). Unregistered bare names stay at the config root (e.g. `structure`).
    """
    reg = reg or registry()
    index = reg.config_index()
    parts = name.split("/")
    if len(parts) == 1:
        return f"{index[name]}/{name}" if name in index else name
    if parts[0] in {"general", "rq2"} and len(parts) >= 3:
        parts = parts[1:]
    prefix, rest = parts[0], "/".join(parts[1:])
    if prefix in reg.folders:
        return "/".join(parts)
    if prefix in reg.previous_folders:
        return f"{reg.previous_folders[prefix]}/{rest}"
    phase = prefix if prefix in PHASES else legacy_phase(prefix)
    if rest in index and (phase or prefix in LEGACY_GROUPS):
        return f"{index[rest]}/{rest}"
    if phase:
        return f"{reg.home(phase)}/{rest}"
    return "/".join(parts)


def config_aliases(old_files) -> dict[str, str]:
    """Map previous `config/...` source paths to their registry paths (rename detection)."""
    moved = {}
    for name in old_files:
        if not name.startswith("config/") or name == "config/structure.yaml":
            continue
        relative = name.removeprefix("config/")
        stem, suffix = (relative[:-5], ".yaml") if relative.endswith(".yaml") else (relative, "")
        target = "config/" + config_relative(stem) + suffix
        if target != name:
            moved[name] = target
    return moved


#: Physical output folders before and after 18-11-2026, for literal source rewrites.
def legacy_literals(reg: Registry | None = None) -> dict[str, str]:
    reg = reg or registry()
    pairs = {f"output_JEVPFN/rq2/{LEGACY_PHASE_FOLDERS[p]}/": f"output_JEVPFN/{reg.home(p)}/"
             for p in LEGACY_RQ2_RESULT_PHASES}
    pairs.update({f"output_JEVPFN/{folder}/": f"output_JEVPFN/{reg.home(phase)}/"
                  for phase, folder in LEGACY_PHASE_FOLDERS.items()})
    pairs.update({f"output_JEVPFN/{group}/logs/": f"output_JEVPFN/{reg.home(group)}/logs/"
                  for group in ("rq1", "rq2", "rq3")})
    pairs.update({f"output_JEVPFN/{old}/": f"output_JEVPFN/{new}/"
                  for old, new in reg.previous_folders.items()})
    return pairs


def rewrite_legacy_literals(text: str, reg: Registry | None = None) -> str:
    for old, new in sorted(legacy_literals(reg).items(), key=lambda kv: -len(kv[0])):
        text = text.replace(old, new)
    return text


def layout_only_changes(old_files: dict, current_files: dict, read_old, read_current) -> set[str]:
    """Source paths whose difference is only this layout move.

    A config moved byte-for-byte to its registry path, or a file whose previous content with
    previous output folders rewritten equals its current content. Anything else stays changed.
    """
    explained = set()
    for old, new in config_aliases(old_files).items():
        if old not in current_files and old_files.get(old) == current_files.get(new) is not None:
            explained |= {old, new}
    for name in old_files.keys() & current_files.keys():
        if old_files[name] == current_files[name] or name in explained:
            continue
        if not name.endswith((".slurm", ".sh", ".py", ".yaml")):
            continue
        before = read_old(name)
        if before is None:
            continue
        before = before.replace("\r\n", "\n")
        current = read_current(name).replace("\r\n", "\n")
        if before != current and rewrite_legacy_literals(before) == current:
            explained.add(name)
    return explained
