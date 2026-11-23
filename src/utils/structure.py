"""The one registry of physical folder names: `config/structure.yaml`.

Internal phase IDs stay the keys of every hash, recipe, receipt and manifest. This module
only decides WHERE a phase's config, notebooks and outputs live, and how previous layouts
(numbered `general/rq1/rq2/rq3`; then `<folder>/<name>` notebooks and `exp09_adaptation`
before 19-11-2026; then `{logs,manifests,results}/notebooks/<folder>/` records, a short-lived
`<folder>/figures/` with per-folder reports, and `output_JEVPFN/archives/`; then the flat
one-folder-per-experiment layout of 18-23 November 2026) translate into the current one
(schema v3, 23-11-2026: letter groups A/B/C, Group B4 and C renumbered):

    <folder>/ or <group>/<folder>/{logs,manifests,results}/[notebook/]   per registry entry
    figures/<notebook group>/<name>/   one folder per notebook: PDFs and `_figures.json`
    figures/CAPTIONS.md       every caption;  All_Results.md   every notebook's report
    ../archive/               previous-layout records, beside (not inside) the output root

Notebooks live at `notebooks/<notebook group>/<name>.ipynb` (`general` for ungrouped entries);
their ID is that relative path without `.ipynb`. Config files of a grouped entry live flat in
`config/<group>/`, those of a general entry in `config/<folder>/`. Bare folder names stay
globally unique, so a flat path recorded before 23-11-2026 (`exp11_data_efficiency/...`)
still names exactly one entry.

`paths` builds every path from it; `layout_migration` moves existing trees with the same
translation, so code and data always agree after a migration.

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
    "data_efficiency",  # Group B, B4 (B/exp09_data_efficiency; exp11_data_efficiency until 23-11-2026)
    "representations",  # Group C, C1 (C/exp10_representations; exp09_representations until 23-11-2026)
)

#: Letter groups of the research experiments (output `<group>/<folder>/`, config `config/<group>/`).
GROUPS = ("A", "B", "C")
#: Notebook group of the ungrouped (general) entries: `notebooks/general/`.
GENERAL = "general"

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
SCHEMA = "jevpfn.structure.v3"
NOTEBOOK_RECORD_KINDS = ("logs", "manifests", "results")
#: Sub-folder of `<folder>/{logs,manifests,results}/` holding the folder notebook's records,
#: apart from the experiment's own manifests/logs/results.
NOTEBOOK_RECORDS = "notebook"
#: The one top-level figures folder: `figures/<notebook group>/<name>/` per notebook, plus CAPTIONS.md.
FIGURES = "figures"
#: Notebook records kept beside the PDFs in `figures/<notebook group>/<name>/`.
FIGURE_RECORDS = ("_figures.json", "_stdout.txt", "_failure.txt")
#: The two generated project documents: `All_Results.md` and `figures/CAPTIONS.md`.
ALL_RESULTS = "All_Results.md"
CAPTIONS = "CAPTIONS.md"
#: Report names of earlier layouts, including retired per-folder copies.
REPORT_NAMES = frozenset({ALL_RESULTS, CAPTIONS, "All Results.md", "all_results.md", "Captions.md"})
PLACEMENTS = ("main", "appendix", "undecided")
#: Generated per-folder index (run_notebooks); it stays in the experiment folder that wrote it.
FOLDER_REPORTS = {("manifests", "reports.json")}


def figure_group(key: str, reg: Registry | None = None) -> str:
    """THE figure grouping: the `figures/<group>/` of a registry folder or notebook.

    `key` is an entry path (`B/exp04_decision_features`, optionally with a nested phase
    sub-folder), a current or previous folder name, an entry ID or a notebook ID/alias. The
    group is the folder notebook's ID (`<notebook group>/<name>`); a folder without a notebook
    groups as `<notebook group>/<folder>`. Unknown keys are returned unchanged.
    `paths.figures_dir` and the layout migration (`translate_output`) both use it.
    """
    reg = reg or registry()
    hit = reg.locate(key)
    if hit is not None:
        entry, rest = hit
        return "/".join((entry.figure_group, *rest))
    found = reg.notebook(key)
    return found.id if found is not None else key


def figure_folder(group: str, reg: Registry | None = None) -> str | None:
    """The registry path whose notebook draws into `figures/<group>/` (inverse of the above)."""
    reg = reg or registry()
    found = [e.path for e in reg.entries if e.figure_group == group]
    return found[0] if len(found) == 1 else None


@dataclass(frozen=True)
class Notebook:
    """The one notebook of an entry: `notebooks/<group>/<name>.ipynb`, ID `<group>/<name>`.

    `folder` is the physical record folder (the entry path). `previous` holds the flat IDs of
    the 18-23 November 2026 layout (the folder name and its earlier names); `legacy` and
    `merged` hold older IDs. All of them stay accepted aliases.
    """

    id: str
    folder: str
    name: str
    phase: str
    legacy: tuple[str, ...]
    merged: tuple[str, ...] = ()
    group: str = GENERAL
    previous: tuple[str, ...] = ()

    @property
    def aliases(self) -> tuple[str, ...]:
        return (*self.previous, *self.legacy, *self.merged)


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
    group: str | None = None
    notebook_only: bool = False

    @property
    def path(self) -> str:
        """Physical folder relative to an output root: `<group>/<folder>` or `<folder>`."""
        return f"{self.group}/{self.folder}" if self.group else self.folder

    @property
    def parts(self) -> tuple[str, ...]:
        return tuple(self.path.split("/"))

    @property
    def config_dir(self) -> str:
        """`config/<group>/` (flat per letter group) or `config/<folder>/` for a general entry."""
        return self.group or self.folder

    @property
    def notebook_group(self) -> str:
        return self.group or GENERAL

    @property
    def flat_names(self) -> tuple[str, ...]:
        """Top-level output names of previous layouts that now mean this entry's path."""
        return ((self.folder,) if self.group else ()) + self.previous_folders

    @property
    def figure_group(self) -> str:
        return self.notebooks[0].id if self.notebooks else f"{self.notebook_group}/{self.folder}"

    @property
    def owner_phase(self) -> str:
        """The internal phase owning this folder's tree (read-only phase for notebook-only)."""
        if self.phases:
            return self.phases[0]
        if self.routes:
            return next(iter(self.routes))
        return self.notebooks[0].phase


@dataclass(frozen=True)
class Registry:
    entries: tuple[Entry, ...]
    paper: dict
    planned: tuple[Planned, ...] = ()
    notebook_groups: tuple[str, ...] = (GENERAL, *GROUPS)

    # -- lookups -----------------------------------------------------------------------
    @property
    def folders(self) -> tuple[str, ...]:
        """Physical entry paths in registry order (`00_data`, ..., `B/exp04_decision_features`)."""
        return tuple(e.path for e in self.entries)

    @property
    def top_level(self) -> tuple[str, ...]:
        """Top-level output directories holding registry folders (general folders, A, B, C)."""
        return tuple(dict.fromkeys(e.parts[0] for e in self.entries))

    @property
    def config_dirs(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(e.config_dir for e in self.entries if e.configs))

    def entry(self, key: str) -> Entry:
        for e in self.entries:
            if key in {e.id, e.folder, e.path, *e.previous_folders}:
                return e
        raise KeyError(f"Unknown layout entry: {key}")

    def locate(self, key: str) -> tuple[Entry, tuple[str, ...]] | None:
        """(entry, remaining parts) of an entry path, folder or ID, optionally with sub-folders."""
        parts = tuple(p for p in str(key).split("/") if p)
        if not parts:
            return None
        for e in sorted(self.entries, key=lambda x: -len(x.parts)):
            if parts[: len(e.parts)] == e.parts:
                return e, parts[len(e.parts):]
        for e in self.entries:
            if parts[0] in {e.id, e.folder, *e.previous_folders}:
                return e, parts[1:]
        return None

    @property
    def previous_folders(self) -> dict[str, str]:
        """Declared earlier folder name -> current path (exp11_data_efficiency -> B/exp09_data_efficiency)."""
        return {old: e.path for e in self.entries for old in e.previous_folders}

    @property
    def flat_paths(self) -> dict[str, str]:
        """Every previous top-level folder name (flat layout and earlier renames) -> current path."""
        return {old: e.path for e in self.entries for old in e.flat_names}

    def flat_entry(self, name: str) -> Entry | None:
        """The entry a current or previous bare folder name denotes, or None."""
        return next((e for e in self.entries if name in {e.folder, *e.previous_folders}), None)

    def home_entry(self, phase: str) -> Entry:
        for e in self.entries:
            if phase in e.phases or phase in e.nested:
                return e
        raise ValueError(f"Phase has no registered folder: {phase}")

    def home(self, phase: str) -> str:
        """The default folder path of an internal phase (may be nested for a retired phase)."""
        e = self.home_entry(phase)
        return e.path if phase in e.phases else f"{e.path}/{e.nested[phase]}"

    def routes(self) -> list[tuple[str, tuple[str, ...], str]]:
        found = [
            (phase, PurePosixPath(prefix).parts, e.path)
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

    def entry_at(self, folder_parts) -> tuple[Entry, str | None, int] | None:
        """(entry, nested phase or None, consumed parts) for a current physical path prefix."""
        folder_parts = tuple(folder_parts)
        candidates = []
        for e in self.entries:
            for phase, sub in e.nested.items():
                candidates.append(((*e.parts, sub), e, phase))
            candidates.append((e.parts, e, None))
        for prefix, e, nested in sorted(candidates, key=lambda c: -len(c[0])):
            if folder_parts[: len(prefix)] == prefix:
                return e, nested, len(prefix)
        return None

    def owner_of(self, folder_parts) -> tuple[str, int] | None:
        """(phase, consumed parts) for a current physical folder prefix, longest first."""
        hit = self.entry_at(folder_parts)
        if hit is None:
            return None
        entry, nested, consumed = hit
        return nested or entry.owner_phase, consumed

    @property
    def notebooks(self) -> tuple[Notebook, ...]:
        return tuple(n for e in self.entries for n in e.notebooks)

    def notebook(self, notebook_id: str) -> Notebook | None:
        """A notebook by current ID, previous (flat, legacy or merged) ID or unique old stem."""
        for n in self.notebooks:
            if notebook_id == n.id:
                return n
        for n in self.notebooks:
            if notebook_id in n.aliases:
                return n
        stems = [n for n in self.notebooks if any(x.split("/")[-1] == notebook_id for x in n.aliases)]
        return stems[0] if len(stems) == 1 else None

    def previous_notebook(self, identity: str) -> Notebook | None:
        """The current notebook whose previous, legacy or merged IDs contain exactly `identity`."""
        return next((n for n in self.notebooks if identity in n.aliases), None)

    def config_index(self) -> dict[str, str]:
        """Registered config file stem (or data filename) -> config directory under config/."""
        return {name: e.config_dir for e in self.entries for name in e.configs}

    def config_entry(self, name: str) -> Entry | None:
        return next((e for e in self.entries if name in e.configs), None)


def _load(path: Path) -> Registry:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA:
        raise ValueError("Unknown layout registry schema")
    entries = []
    for item in raw["entries"]:
        folder, group = item["folder"], item.get("group")
        if "notebooks" in item:
            raise ValueError("One notebook per entry: use `notebook:`, not `notebooks:`")
        spec = item.get("notebook")
        notebooks = ()
        if spec:
            nb_group = group or GENERAL
            name = spec.get("name", folder)
            previous = tuple(dict.fromkeys((folder, *item.get("previous_folders", ()))))
            notebooks = (Notebook(f"{nb_group}/{name}", f"{group}/{folder}" if group else folder,
                                  name, spec["phase"], tuple(spec.get("legacy", ())),
                                  tuple(spec.get("merged", ())), nb_group, previous),)
        entries.append(Entry(
            id=item["id"], folder=folder, title=item["title"],
            phases=tuple(item.get("phases", ())),
            routes={k: tuple(v) for k, v in (item.get("routes") or {}).items()},
            nested=dict(item.get("nested") or {}),
            notebooks=notebooks, configs=tuple(item.get("configs", ())),
            previous_folders=tuple(item.get("previous_folders", ())),
            group=group, notebook_only=bool(item.get("notebook_only", False)),
        ))
    planned = tuple(Planned(p["id"], p["folder"], p["title"], p["status"])
                    for p in raw.get("planned") or ())
    groups = tuple(raw.get("notebook_groups") or (GENERAL, *GROUPS))
    registry = Registry(tuple(entries), dict(raw.get("paper") or {}), planned, groups)
    validate(registry)
    return registry


def validate(registry: Registry) -> None:
    ids = [e.id for e in registry.entries] + [p.id for p in registry.planned]
    # Bare folder names stay globally unique: previous flat paths resolve by name alone.
    folders = ([e.folder for e in registry.entries] + [p.folder for p in registry.planned]
               + [old for e in registry.entries for old in e.previous_folders])
    if len(set(ids)) != len(ids) or len(set(folders)) != len(folders):
        raise ValueError("Layout IDs and folders (current, planned and previous) must be unique")
    for folder in folders:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]*", folder):
            raise ValueError(f"Unsafe layout folder name: {folder}")
    if set(folders) & {*GROUPS, GENERAL, FIGURES, ARCHIVE_DIR_NAME, *NOTEBOOK_RECORD_KINDS}:
        raise ValueError("A layout folder cannot reuse a group, figures, archive or record-kind name")
    if any(e.group is not None and e.group not in GROUPS for e in registry.entries):
        raise ValueError(f"Layout groups must be one of {GROUPS}")
    if GENERAL not in registry.notebook_groups or set(GROUPS) - set(registry.notebook_groups):
        raise ValueError("Notebook groups must include general and every letter group")
    homes = [p for e in registry.entries for p in (*e.phases, *e.nested)]
    if sorted(homes) != sorted(PHASES):
        raise ValueError("Every internal phase needs exactly one home folder")
    for e in registry.entries:
        if e.notebook_only:
            if e.phases or e.routes or e.nested or e.configs or not e.notebooks:
                raise ValueError(f"Notebook-only entry {e.id} must own one notebook and nothing else")
        elif not e.phases and not e.routes:
            raise ValueError(f"Layout entry {e.id} owns no phase or route")
        if set(e.routes) - set(PHASES) or any(n.phase not in PHASES for n in e.notebooks):
            raise ValueError(f"Layout entry {e.id} names an unknown phase")
    configs = [c for e in registry.entries for c in e.configs]
    if len(set(configs)) != len(configs):
        raise ValueError("Registered config names must be unique")
    names = [n.id for n in registry.notebooks]
    legacy = [x for n in registry.notebooks for x in n.aliases]
    if len(set(names)) != len(names) or len(set(legacy)) != len(legacy):
        raise ValueError("Notebook IDs and previous notebook IDs must be unique")
    if set(legacy) & set(names):
        raise ValueError("A previous notebook ID cannot equal a current notebook ID")
    if any(n.group not in registry.notebook_groups for n in registry.notebooks):
        raise ValueError("Notebook group is not declared in notebook_groups")
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


def _legacy_output_prefixes() -> list[tuple[tuple[str, ...], str]]:
    """Numbered and research-question trees before 18-11-2026 -> internal phase."""
    prefixes = [((folder,), phase) for phase, folder in LEGACY_PHASE_FOLDERS.items()]
    prefixes += [(("rq2", LEGACY_PHASE_FOLDERS[p]), p) for p in LEGACY_RQ2_RESULT_PHASES]
    prefixes += [(("rq2", "regression_readout"), "regression_readout"),
                 (("regression_readout",), "regression_readout"),
                 (("rq1",), "rq1"), (("rq2",), "rq2"), (("rq3",), "rq3")]
    return sorted(prefixes, key=lambda p: -len(p[0]))


def _normalize(parts, reg: Registry) -> tuple[str, ...]:
    """A previous flat top-level folder (`exp11_data_efficiency/...`) -> its current path."""
    parts = tuple(parts)
    flat = reg.flat_paths
    if parts and parts[0] in flat:
        return (*flat[parts[0]].split("/"), *parts[1:])
    return parts


def logical_output(parts, reg: Registry | None = None) -> tuple[str, tuple[str, ...]] | None:
    """(internal phase, sub-path) of a phase tree path, in any layout; None otherwise."""
    reg = reg or registry()
    parts = tuple(parts)
    for prefix, phase in _legacy_output_prefixes():
        if parts[: len(prefix)] == prefix:
            return phase, parts[len(prefix):]
    current = _normalize(parts, reg)
    owned = reg.owner_of(current)
    if owned:
        phase, consumed = owned
        return phase, current[consumed:]
    return None


def _notebook_record(parts, reg: Registry) -> tuple[bool, tuple[str, ...] | None]:
    """(is a notebook-record path, translated parts or None when unmapped).

    `figures/<x>/...` and kind-first `{logs,manifests,results}/notebooks/<x>/...`. Current
    records are `figures/<notebook group>/<name>/...`. A previous notebook ID (`00_data`,
    `exp04_decision_features/decisions`, `rq2/01_decisions`, `general/01_data`) or a previous
    flat folder (`figures/exp11_data_efficiency/...`) moves to the current notebook's group;
    records of notebooks merged into one and of retired notebooks are archived. Kind-first
    records go to `<folder>/<kind>/notebook/...`, except `_figures.json`/`_stdout.txt`/
    `_failure.txt`, which sit beside the PDFs.
    """
    if parts[:1] == (FIGURES,):
        kind, rest = None, parts[1:]
    elif len(parts) >= 2 and parts[0] in NOTEBOOK_RECORD_KINDS and parts[1] == "notebooks":
        kind, rest = parts[0], parts[2:]
    else:
        return False, None
    if not rest:
        return True, parts
    if len(rest) == 1:
        # A previous figure folder moves as one unit when every file inside translates alike.
        entry = reg.flat_entry(rest[0]) if kind is None else None
        return True, ((FIGURES, *entry.figure_group.split("/")) if entry is not None else None)
    identity = f"{rest[0]}/{rest[1]}"
    if kind is None and any(e.figure_group == identity for e in reg.entries):
        return True, parts  # A current record.
    previous = reg.previous_notebook(identity)
    if previous is not None and identity in previous.merged:
        return True, (*MERGED_ARCHIVE.parts, *parts)
    if previous is not None:
        entry, sub = reg.locate(previous.folder)[0], rest[2:]
    elif (entry := reg.flat_entry(rest[0])) is not None:
        sub = rest[1:]
    elif rest[0] in LEGACY_GROUPS or legacy_phase(rest[0]) is not None:
        return True, (*ARCHIVE.parts, *parts)  # Records of a retired notebook.
    else:
        return True, None
    if kind is None or (sub and sub[-1] in FIGURE_RECORDS):
        return True, (FIGURES, *entry.figure_group.split("/"), *sub)
    return True, (*entry.parts, kind, NOTEBOOK_RECORDS, *sub)


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
    is_record, record = _notebook_record(parts, reg)
    if is_record:
        return PurePosixPath(*record) if record is not None else None
    if parts[0] in NOTEBOOK_RECORD_KINDS:
        return None
    for prefix, phase in _legacy_output_prefixes():
        if parts[: len(prefix)] == prefix:
            sub = parts[len(prefix):]
            return PurePosixPath(*reg.folder_for(phase, sub).split("/"), *sub)
    current = _normalize(parts, reg)
    hit = reg.entry_at(current)
    if hit is None:
        return None
    entry, nested, consumed = hit
    rest = current[consumed:]
    if entry.notebook_only:
        return PurePosixPath(*current)  # Notebook records only; nothing routes elsewhere.
    if nested is None and rest[:1] == (FIGURES,):
        # `<folder>/figures/...` (briefly used on 19-11-2026) -> `figures/<group>/...`.
        return PurePosixPath(FIGURES, *entry.figure_group.split("/"), *rest[1:])
    if rest in FOLDER_REPORTS or _own_notebook_record(rest):
        return PurePosixPath(*current)  # Each folder owns its index and notebook records.
    return PurePosixPath(*reg.folder_for(nested or entry.owner_phase, rest).split("/"), *rest)


def _own_notebook_record(sub) -> bool:
    """`<folder>/<kind>/notebook/...` (possibly in a nested phase)."""
    return len(sub) >= 2 and sub[0] in NOTEBOOK_RECORD_KINDS and sub[1] == NOTEBOOK_RECORDS


def recorded_output_path(value, output_dir_name: str = "output_JEVPFN") -> Path:
    """Locate a path recorded before a layout migration, e.g. in reviewed acceptances.

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

    Accepts current keys (`B/main`), registered stems (`main`), internal phase keys
    (`experiment_1/main`, `feature_creation/jev.local`), flat-layout keys
    (`exp04_decision_features/main`, `synthesis/analysis`) and older folders (`4. experiment_1/main`,
    `rq2/4. experiment_1/main`, `general/3. experiment_0/debug`, `rq1/behavior`,
    `exp09_adaptation/adaptation`). Unregistered bare names stay at the config root (`structure`).
    """
    reg = reg or registry()
    index = reg.config_index()
    parts = name.split("/")
    if len(parts) == 1:
        return f"{index[name]}/{name}" if name in index else name
    if parts[0] in {"general", "rq2"} and len(parts) >= 3:
        parts = parts[1:]
    prefix, rest = parts[0], "/".join(parts[1:])
    if prefix in reg.config_dirs:
        return "/".join(parts)
    entry = reg.flat_entry(prefix)
    if entry is not None:
        return f"{index[rest]}/{rest}" if rest in index else f"{entry.config_dir}/{rest}"
    phase = prefix if prefix in PHASES else legacy_phase(prefix)
    if rest in index and (phase or prefix in LEGACY_GROUPS):
        return f"{index[rest]}/{rest}"
    if phase:
        return f"{reg.home_entry(phase).config_dir}/{rest}"
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


def notebook_aliases(old_files, reg: Registry | None = None) -> dict[str, str]:
    """Map previous `notebooks/...ipynb` source paths to their registry paths."""
    reg = reg or registry()
    moved = {}
    for name in old_files:
        if not (name.startswith("notebooks/") and name.endswith(".ipynb")):
            continue
        found = reg.notebook(name.removeprefix("notebooks/").removesuffix(".ipynb"))
        if found is not None and f"notebooks/{found.id}.ipynb" != name:
            moved[name] = f"notebooks/{found.id}.ipynb"
    return moved


def legacy_literals(reg: Registry | None = None) -> dict[str, str]:
    """Physical output/config folders of previous layouts -> current, for literal source rewrites."""
    reg = reg or registry()
    pairs = {f"output_JEVPFN/rq2/{LEGACY_PHASE_FOLDERS[p]}/": f"output_JEVPFN/{reg.home(p)}/"
             for p in LEGACY_RQ2_RESULT_PHASES}
    pairs.update({f"output_JEVPFN/{folder}/": f"output_JEVPFN/{reg.home(phase)}/"
                  for phase, folder in LEGACY_PHASE_FOLDERS.items()})
    pairs.update({f"output_JEVPFN/{group}/logs/": f"output_JEVPFN/{reg.home(group)}/logs/"
                  for group in ("rq1", "rq2", "rq3")})
    pairs.update({f"output_JEVPFN/{old}/": f"output_JEVPFN/{new}/" for old, new in reg.flat_paths.items()})
    for e in reg.entries:
        for old in (e.folder, *e.previous_folders):
            if e.figure_group != old:
                pairs[f"output_JEVPFN/figures/{old}/"] = f"output_JEVPFN/figures/{e.figure_group}/"
            if e.configs and e.config_dir != old:
                pairs[f"config/{old}/"] = f"config/{e.config_dir}/"
    return pairs


def rewrite_legacy_literals(text: str, reg: Registry | None = None) -> str:
    """One pass, longest literal first, so a rewritten folder is never rewritten again."""
    pairs = legacy_literals(reg)
    pattern = re.compile("|".join(re.escape(k) for k in sorted(pairs, key=len, reverse=True)))
    return pattern.sub(lambda m: pairs[m.group(0)], text)


def layout_only_changes(old_files: dict, current_files: dict, read_old, read_current) -> set[str]:
    """Source paths whose difference is only this layout move.

    A config or notebook moved byte-for-byte to its registry path, a moved config whose only
    change is previous folders rewritten in its text, or a file whose previous content with
    previous folders rewritten equals its current content. Anything else stays changed.
    """
    explained = set()
    for old, new in {**config_aliases(old_files), **notebook_aliases(old_files)}.items():
        if old in current_files or current_files.get(new) is None:
            continue
        if old_files.get(old) == current_files[new]:
            explained |= {old, new}
        elif new.endswith(".yaml") and (before := read_old(old)) is not None:
            before = before.replace("\r\n", "\n")
            if rewrite_legacy_literals(before) == read_current(new).replace("\r\n", "\n"):
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
