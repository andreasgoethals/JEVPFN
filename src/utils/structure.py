"""The one registry of physical names: `config/structure.yaml` (schema v4, 26-11-2026).

Every experiment has exactly one name, used literally as its phase ID, config file, output
folder and notebook name (`B3`: phase `B3`, `config/B/B3.yaml`, `output_JEVPFN/B/B3/`,
`notebooks/B/B3.ipynb`). Four top-level groups, in study order:

    general/{0, 1_data_and_questions, 2_feature_creation, 3_study_audit, 4_synthesis}/
    A/{A0, A1, A2, A3, A_synthesis}/   B/{B0, ..., B6, B_synthesis}/   C/{C0, C1, C2, C_synthesis}/

    <group>/<folder>/{logs,manifests,results}/[notebook/]   per registry entry
    figures/<group>/<folder>/   one folder per notebook: PDFs and `_figures.json`
    figures/CAPTIONS.md         every caption;  All_Results.md   every notebook's report
    ../archive/                 previous-layout records, beside (not inside) the output root

Earlier names (layouts v1-v3, phase IDs such as `experiment_1` or `rq1`, config keys and
notebook IDs) live ONLY in the checksummed `config/layout_aliases.yaml`. They are used to
locate paths recorded in earlier receipts (`recorded_output_path`) and by
`layout_migration`; no other code path accepts an old name.

Import-light on purpose: `paths` imports this module, so it must not import `paths`.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path, PurePosixPath

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "config" / "structure.yaml"

#: Internal phase identities. The general preparation stages keep descriptive IDs; every
#: qualification gate and experiment uses its one name. `regression_readout` is a retired
#: phase nested in B1.
PHASES = (
    "0",
    "exploration",
    "question_preparation",
    "feature_creation",
    "synthesis",
    "A0", "A1", "A2", "A3",
    "B0", "B1", "B2", "B3", "B4", "B5", "B6",
    "C0", "C1", "C2",
    "regression_readout",
)

#: Top-level groups in study order (output `<group>/<folder>/`, notebooks `notebooks/<group>/`).
GENERAL = "general"
GROUPS = ("A", "B", "C")
ALL_GROUPS = (GENERAL, *GROUPS)

#: Previous-layout records live in `archive/`, a sibling of the output root (git-ignored at
#: the repository root; on VSC beside the migrated DATA or PROJECT output root, so moving there
#: is a same-filesystem rename). Paths below are relative to the output root.
ARCHIVE_DIR_NAME = "archive"
ARCHIVE_ROOT = PurePosixPath("..", ARCHIVE_DIR_NAME)
SCHEMA = "jevpfn.structure.v4"
ALIASES_SCHEMA = "jevpfn.layout-aliases.v1"
NOTEBOOK_RECORD_KINDS = ("logs", "manifests", "results")
#: Sub-folder of `<folder>/{logs,manifests,results}/` holding the folder notebook's records,
#: apart from the experiment's own manifests/logs/results.
NOTEBOOK_RECORDS = "notebook"
#: The one top-level figures folder: `figures/<group>/<name>/` per notebook, plus CAPTIONS.md.
FIGURES = "figures"
#: Notebook records kept beside the PDFs in `figures/<group>/<name>/`.
FIGURE_RECORDS = ("_figures.json", "_stdout.txt", "_failure.txt")
#: The two generated project documents: `All_Results.md` and `figures/CAPTIONS.md`.
ALL_RESULTS = "All_Results.md"
CAPTIONS = "CAPTIONS.md"
REPORT_NAMES = frozenset({ALL_RESULTS, CAPTIONS})
PLACEMENTS = ("main", "appendix", "undecided")
#: Generated per-folder index (run_notebooks); it stays in the experiment folder that wrote it.
FOLDER_REPORTS = {("manifests", "reports.json")}
#: Output-root entries outside the layout (left in place by migrations).
ROOT_ENTRIES = ("manifests", "transfers", "workspaces", "compute_ledger.json", ".reports.lock")


def figure_group(key: str, reg: Registry | None = None) -> str:
    """THE figure grouping: the `figures/<group>/<name>` key of a registry folder or notebook.

    `key` is an entry path (`B/B1`, optionally with a nested phase sub-folder), an entry ID or
    folder, or a notebook ID. Unknown keys are returned unchanged.
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
    """The one notebook of an entry: `notebooks/<group>/<name>.ipynb`, ID `<group>/<name>`."""

    id: str
    folder: str
    name: str
    phase: str
    group: str = GENERAL


@dataclass(frozen=True)
class Entry:
    id: str
    folder: str
    title: str
    group: str
    phases: tuple[str, ...]
    nested: dict = field(default_factory=dict)
    notebooks: tuple[Notebook, ...] = ()
    configs: tuple[str, ...] = ()
    config: bool = False
    data_files: tuple[str, ...] = ()
    notebook_only: bool = False

    @property
    def path(self) -> str:
        """Physical folder relative to an output root: `<group>/<folder>`."""
        return f"{self.group}/{self.folder}"

    @property
    def parts(self) -> tuple[str, ...]:
        return (self.group, self.folder)

    @property
    def config_dir(self) -> str:
        """`config/general/<folder>/` for a general stage, `config/<group>/` otherwise."""
        return f"{GENERAL}/{self.folder}" if self.group == GENERAL else self.group

    @property
    def config_file(self) -> str | None:
        """`<group>/<folder>` (no suffix) for an experiment's one config file."""
        return f"{self.group}/{self.folder}" if self.config else None

    @property
    def notebook_group(self) -> str:
        return self.group

    @property
    def figure_group(self) -> str:
        return self.notebooks[0].id if self.notebooks else self.path

    @property
    def owner_phase(self) -> str:
        """The internal phase owning this folder's tree (read-only phase for notebook-only)."""
        return self.phases[0] if self.phases else self.notebooks[0].phase


@dataclass(frozen=True)
class Registry:
    entries: tuple[Entry, ...]
    paper: dict
    notebook_groups: tuple[str, ...] = (*ALL_GROUPS,)

    # -- lookups -----------------------------------------------------------------------
    @property
    def folders(self) -> tuple[str, ...]:
        """Physical entry paths in registry order (`general/0`, ..., `C/C_synthesis`)."""
        return tuple(e.path for e in self.entries)

    @property
    def top_level(self) -> tuple[str, ...]:
        """Top-level output directories holding registry folders (general, A, B, C)."""
        return tuple(dict.fromkeys(e.group for e in self.entries))

    @property
    def config_dirs(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(e.config_dir for e in self.entries if e.configs or e.config))

    def entry(self, key: str) -> Entry:
        for e in self.entries:
            if key in {e.id, e.folder, e.path}:
                return e
        raise KeyError(f"Unknown layout entry: {key}")

    def locate(self, key: str) -> tuple[Entry, tuple[str, ...]] | None:
        """(entry, remaining parts) of an entry path, folder or ID, optionally with sub-folders."""
        parts = tuple(p for p in str(key).split("/") if p)
        if not parts:
            return None
        for e in self.entries:
            if parts[:2] == e.parts:
                return e, parts[2:]
        for e in self.entries:
            if parts[0] in {e.id, e.folder}:
                return e, parts[1:]
        return None

    def home_entry(self, phase: str) -> Entry:
        for e in self.entries:
            if phase in e.phases or phase in e.nested:
                return e
        raise ValueError(f"Phase has no registered folder: {phase}")

    def home(self, phase: str) -> str:
        """The folder path of an internal phase (nested for the retired regression readout)."""
        e = self.home_entry(phase)
        return e.path if phase in e.phases else f"{e.path}/{e.nested[phase]}"

    def folder_for(self, phase: str, parts=()) -> str:
        """The folder holding `<phase>/<parts>`: every phase owns exactly one folder."""
        return self.home(phase)

    def entry_at(self, folder_parts) -> tuple[Entry, str | None, int] | None:
        """(entry, nested phase or None, consumed parts) for a current physical path prefix."""
        folder_parts = tuple(folder_parts)
        for e in self.entries:
            if folder_parts[:2] != e.parts:
                continue
            for phase, sub in e.nested.items():
                if folder_parts[2:3] == (sub,):
                    return e, phase, 3
            return e, None, 2
        return None

    def owner_of(self, folder_parts) -> tuple[str, int] | None:
        """(phase, consumed parts) for a current physical folder prefix."""
        hit = self.entry_at(folder_parts)
        if hit is None:
            return None
        entry, nested, consumed = hit
        return nested or entry.owner_phase, consumed

    @property
    def notebooks(self) -> tuple[Notebook, ...]:
        return tuple(n for e in self.entries for n in e.notebooks)

    def notebook(self, notebook_id: str) -> Notebook | None:
        """A notebook by its ID (`B/B3`) or, when unique, its bare name (`B3`)."""
        for n in self.notebooks:
            if notebook_id == n.id:
                return n
        named = [n for n in self.notebooks if n.name == notebook_id]
        return named[0] if len(named) == 1 else None

    def config_index(self) -> dict[str, str]:
        """Registered general config stem or data filename -> its directory under config/."""
        index = {name: e.config_dir for e in self.entries for name in e.configs}
        index.update({name: e.config_dir for e in self.entries for name in e.data_files})
        return index

    def config_entry(self, name: str) -> Entry | None:
        return next((e for e in self.entries if name in e.configs or name == e.config_file), None)


def _load(path: Path) -> Registry:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA:
        raise ValueError("Unknown layout registry schema")
    entries = []
    for item in raw["entries"]:
        folder, group = str(item["folder"]), item["group"]
        if "notebooks" in item:
            raise ValueError("One notebook per entry: use `notebook:`, not `notebooks:`")
        spec = item.get("notebook")
        notebooks = ()
        if spec:
            notebooks = (Notebook(f"{group}/{folder}", f"{group}/{folder}", folder,
                                  str(spec["phase"]), group),)
        entries.append(Entry(
            id=str(item["id"]), folder=folder, title=item["title"], group=group,
            phases=tuple(str(p) for p in item.get("phases", ())),
            nested=dict(item.get("nested") or {}),
            notebooks=notebooks, configs=tuple(item.get("configs", ())),
            config=bool(item.get("config", False)),
            data_files=tuple(item.get("data_files", ())),
            notebook_only=bool(item.get("notebook_only", False)),
        ))
    groups = tuple(raw.get("notebook_groups") or ALL_GROUPS)
    reg = Registry(tuple(entries), {str(k): v for k, v in (raw.get("paper") or {}).items()}, groups)
    validate(reg)
    _verify_aliases(path, raw.get("aliases") or {})
    return reg


def validate(registry: Registry) -> None:
    ids = [e.id for e in registry.entries]
    folders = [e.folder for e in registry.entries]
    if len(set(ids)) != len(ids) or len(set(folders)) != len(folders):
        raise ValueError("Layout IDs and folders must be unique")
    for folder in folders:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]*", folder):
            raise ValueError(f"Unsafe layout folder name: {folder}")
    if set(folders) & {*ALL_GROUPS, FIGURES, ARCHIVE_DIR_NAME, *NOTEBOOK_RECORD_KINDS}:
        raise ValueError("A layout folder cannot reuse a group, figures, archive or record-kind name")
    if any(e.group not in ALL_GROUPS for e in registry.entries):
        raise ValueError(f"Layout groups must be one of {ALL_GROUPS}")
    if set(ALL_GROUPS) - set(registry.notebook_groups):
        raise ValueError("Notebook groups must include general and every letter group")
    homes = [p for e in registry.entries for p in (*e.phases, *e.nested)]
    if sorted(homes) != sorted(PHASES):
        raise ValueError("Every internal phase needs exactly one home folder")
    for e in registry.entries:
        if e.notebook_only:
            if e.phases or e.nested or e.configs or e.config or not e.notebooks:
                raise ValueError(f"Notebook-only entry {e.id} must own one notebook and nothing else")
        elif not e.phases:
            raise ValueError(f"Layout entry {e.id} owns no phase")
        if any(n.phase not in PHASES for n in e.notebooks):
            raise ValueError(f"Layout entry {e.id} names an unknown phase")
        if e.group != GENERAL and e.phases and e.phases[0] != e.id:
            raise ValueError(f"Experiment {e.id} must use its name as its phase ID")
        if e.configs and e.group != GENERAL:
            raise ValueError(f"Experiment {e.id} has one config file (`config: true`)")
    configs = [c for e in registry.entries for c in (*e.configs, *e.data_files)]
    if len(set(configs)) != len(configs):
        raise ValueError("Registered config names must be unique")
    names = [n.id for n in registry.notebooks]
    if len(set(names)) != len(names):
        raise ValueError("Notebook IDs must be unique")
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
    return {e.id: {"placement": reg.paper.get(e.id, {}).get("placement", "undecided")}
            for e in reg.entries}


# ---------------------------------------------------------------------------
# The one old -> new map (config/layout_aliases.yaml).
# ---------------------------------------------------------------------------


def aliases_sha256(path: Path) -> str:
    """SHA-256 of the alias map, line endings normalised (identical on Windows and VSC)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _verify_aliases(registry_path: Path, spec: dict) -> None:
    if not spec:
        return
    path = registry_path.parent / spec["file"]
    if aliases_sha256(path) != spec["sha256"]:
        raise ValueError(f"{spec['file']} differs from the checksum recorded in {registry_path.name}")


def aliases_path(path: Path | None = None) -> Path:
    path = Path(path or REGISTRY)
    spec = yaml.safe_load(path.read_text(encoding="utf-8")).get("aliases") or {}
    return path.parent / spec.get("file", "layout_aliases.yaml")


@lru_cache(maxsize=4)
def _cached_aliases(path: str, mtime: float) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema") != ALIASES_SCHEMA:
        raise ValueError("Unknown layout alias schema")
    raw["output_prefixes"] = {str(k): str(v) for k, v in raw["output_prefixes"].items()}
    raw["phases"] = {str(k): str(v) for k, v in raw["phases"].items()}
    return raw


def aliases(path: Path | None = None) -> dict:
    """The checksummed old -> new map (provenance and migration only)."""
    registry(path)  # Verifies the recorded checksum.
    file = aliases_path(path)
    return _cached_aliases(str(file), file.stat().st_mtime)


def renamed_phase(old: str, sub: str = "") -> str | None:
    """The current phase of an old phase ID (and sub-path), or None when unknown."""
    table = aliases()
    for route in table.get("phase_routes", ()):
        if route["phase"] == old and (sub == route["prefix"] or sub.startswith(route["prefix"] + "/")):
            return str(route["to"])
    return table["phases"].get(old)


# ---------------------------------------------------------------------------
# Output paths: current layout, and earlier recorded paths through the alias map.
# ---------------------------------------------------------------------------


def _alias_target(parts: tuple[str, ...]) -> tuple[str, ...] | None:
    """Longest old output prefix of `parts` -> its current parts, or None."""
    prefixes = aliases()["output_prefixes"]
    for length in range(len(parts), 0, -1):
        new = prefixes.get("/".join(parts[:length]))
        if new is not None:
            return (*PurePosixPath(new).parts, *parts[length:])
    return None


def is_current(parts, reg: Registry | None = None) -> bool:
    """Whether a relative output path already uses the current layout."""
    reg = reg or registry()
    parts = tuple(parts)
    if not parts:
        return False
    if parts[0] == FIGURES:
        return len(parts) == 2 and parts[1] == CAPTIONS or (
            len(parts) >= 3 and f"{parts[1]}/{parts[2]}" in {e.figure_group for e in reg.entries})
    return reg.entry_at(parts) is not None


def translate_output(relative, reg: Registry | None = None) -> PurePosixPath | None:
    """Current location of a path relative to an output root; None when not a layout path.

    A current path maps to itself. An earlier path (layouts v1-v3, see layout_aliases.yaml)
    maps by its longest recorded prefix; archived records translate to `../archive/...`,
    outside the output root.
    """
    reg = reg or registry()
    parts = PurePosixPath(*Path(relative).parts).parts if str(relative) not in {"", "."} else ()
    if not parts:
        return None
    if parts in {(ALL_RESULTS,), (FIGURES, CAPTIONS)} or is_current(parts, reg):
        return PurePosixPath(*parts)
    moved = _alias_target(parts)
    if moved is None:
        return None
    return PurePosixPath(os.path.normpath(PurePosixPath(*moved)).replace("\\", "/"))


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


def config_location(name: str, reg: Registry | None = None) -> tuple[str, str | None]:
    """Logical config key -> (path under config/ without `.yaml`, section or None).

    Keys: an experiment's name (`B1`) or one of its sections (`B1/models`); a general stage's
    folder, entry ID or phase plus file stem (`0/debug`, `feature_creation/extractors`,
    `synthesis/analysis`); a registered general stem (`datasets`); or a literal path under
    config/ (`general/0/debug`, `B/B1`). Unregistered bare names stay at the config root.
    """
    reg = reg or registry()
    index = reg.config_index()
    parts = name.split("/")
    experiments = {e.id: e for e in reg.entries if e.config}
    if parts[0] in experiments:
        if len(parts) > 2:
            raise ValueError(f"Unknown config section: {name}")
        return experiments[parts[0]].config_file, parts[1] if len(parts) == 2 else None
    if len(parts) == 1:
        return (f"{index[name]}/{name}" if name in index else name), None
    if parts[0] in ALL_GROUPS:
        return name, None
    general = next((e for e in reg.entries if e.group == GENERAL and e.configs
                    and parts[0] in {e.id, e.folder, *e.phases}), None)
    if general is not None and len(parts) == 2:
        return f"{general.config_dir}/{parts[1]}", None
    return name, None


def config_relative(name: str, reg: Registry | None = None) -> str:
    """Path under config/ (without `.yaml`) of the file holding a logical config key."""
    return config_location(name, reg)[0]


def legacy_literals() -> dict[str, str]:
    """Earlier physical output/config folders -> current, for literal source rewrites."""
    table = aliases()
    pairs = {f"output_JEVPFN/{old}/": f"output_JEVPFN/{new}/"
             for old, new in table["output_prefixes"].items() if not new.startswith("..")}
    for old, new in table.get("config_files", {}).items():
        if isinstance(new, str) and "#" not in new and not old.endswith((".yaml", ".csv")):
            pairs[f"{old}/"] = f"{new}/"
    return pairs


def rewrite_legacy_literals(text: str) -> str:
    """One pass, longest literal first, so a rewritten folder is never rewritten again."""
    pairs = legacy_literals()
    pattern = re.compile("|".join(re.escape(k) for k in sorted(pairs, key=len, reverse=True)))
    return pattern.sub(lambda m: pairs[m.group(0)], text)


def layout_only_changes(old_files: dict, current_files: dict, read_old, read_current) -> set[str]:
    """Source paths whose difference is only an earlier layout's folder literals.

    A file present in both snapshots whose previous content, with earlier output and config
    folders rewritten through the alias map, equals its current content. Anything else
    (including renamed files) stays changed.
    """
    explained = set()
    for name in old_files.keys() & current_files.keys():
        if old_files[name] == current_files[name]:
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
