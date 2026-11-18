"""The one registry of physical folder names: `config/structure.yaml`.

Internal phase IDs stay the keys of every hash, recipe, receipt and manifest. This module
only decides WHERE a phase's config, notebooks and outputs live, and how the previous
(numbered `general/rq1/rq2/rq3`) layout translates into the current one. `paths` builds every
path from it; `layout_migration` moves existing trees with the same translation, so code and
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
#: Notebook records of retired notebook IDs move here, preserving their relative path.
ARCHIVE = PurePosixPath("archives", "layout_20261118")
NOTEBOOK_RECORD_KINDS = ("logs", "manifests", "results")
PLACEMENTS = ("main", "appendix", "undecided")
#: Generated per-folder documents (run_notebooks); they stay in the folder that wrote them.
FOLDER_REPORTS = {("All Results.md",), ("manifests", "reports.json")}


@dataclass(frozen=True)
class Notebook:
    id: str
    folder: str
    name: str
    phase: str
    legacy: tuple[str, ...]


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

    @property
    def owner_phase(self) -> str:
        """The internal phase that owns this folder's output tree."""
        return self.phases[0] if self.phases else next(iter(self.routes))


@dataclass(frozen=True)
class Registry:
    entries: tuple[Entry, ...]
    paper: dict

    # -- lookups -----------------------------------------------------------------------
    @property
    def folders(self) -> tuple[str, ...]:
        return tuple(e.folder for e in self.entries)

    def entry(self, key: str) -> Entry:
        for e in self.entries:
            if key in {e.id, e.folder}:
                return e
        raise KeyError(f"Unknown layout entry: {key}")

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
        """A current notebook by `<folder>/<name>`, legacy ID or unique bare name."""
        for n in self.notebooks:
            if notebook_id in {n.id, *n.legacy}:
                return n
        matches = [n for n in self.notebooks if n.name == notebook_id]
        if len(matches) == 1:
            return matches[0]
        legacy_stems = [n for n in self.notebooks if any(x.split("/")[-1] == notebook_id for x in n.legacy)]
        return legacy_stems[0] if len(legacy_stems) == 1 else None

    def config_index(self) -> dict[str, str]:
        """Registered config file stem (or data filename) -> folder."""
        return {name: e.folder for e in self.entries for name in e.configs}


def _load(path: Path) -> Registry:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("schema") != "jevpfn.structure.v1":
        raise ValueError("Unknown layout registry schema")
    entries = []
    for item in raw["entries"]:
        folder = item["folder"]
        notebooks = tuple(
            Notebook(f"{folder}/{name}", folder, name, spec["phase"], tuple(spec.get("legacy", ())))
            for name, spec in (item.get("notebooks") or {}).items()
        )
        entries.append(Entry(
            id=item["id"], folder=folder, title=item["title"],
            phases=tuple(item.get("phases", ())),
            routes={k: tuple(v) for k, v in (item.get("routes") or {}).items()},
            nested=dict(item.get("nested") or {}),
            notebooks=notebooks, configs=tuple(item.get("configs", ())),
        ))
    registry = Registry(tuple(entries), dict(raw.get("paper") or {}))
    validate(registry)
    return registry


def validate(registry: Registry) -> None:
    ids = [e.id for e in registry.entries]
    folders = [e.folder for e in registry.entries]
    if len(set(ids)) != len(ids) or len(set(folders)) != len(folders):
        raise ValueError("Layout IDs and folders must be unique")
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
    legacy = [x for n in registry.notebooks for x in n.legacy]
    if len(set(names)) != len(names) or len(set(legacy)) != len(legacy):
        raise ValueError("Notebook names and legacy IDs must be unique")
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
               "rq": reg.paper.get(e.id, {}).get("rq")}
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
    prefixes = [((folder,), phase) for phase, folder in LEGACY_PHASE_FOLDERS.items()]
    prefixes += [(("rq2", LEGACY_PHASE_FOLDERS[p]), p) for p in LEGACY_RQ2_RESULT_PHASES]
    prefixes += [(("rq2", "regression_readout"), "regression_readout"),
                 (("regression_readout",), "regression_readout"),
                 (("rq1",), "rq1"), (("rq2",), "rq2"), (("rq3",), "rq3")]
    return sorted(prefixes, key=lambda p: -len(p[0]))


def logical_output(parts, reg: Registry | None = None) -> tuple[str, tuple[str, ...]] | None:
    """(internal phase, sub-path) of a phase tree path, in either layout; None otherwise."""
    reg = reg or registry()
    parts = tuple(parts)
    for prefix, phase in _legacy_output_prefixes():
        if parts[: len(prefix)] == prefix:
            return phase, parts[len(prefix):]
    owned = reg.owner_of(parts)
    if owned:
        phase, consumed = owned
        return phase, parts[consumed:]
    return None


def _notebook_record(parts, reg: Registry) -> tuple[str, ...] | None:
    """figures/<group>/<name>/... and {logs,manifests,results}/notebooks/<group>/<name>/..."""
    if parts[:1] == ("figures",):
        head, rest = parts[:1], parts[1:]
    elif len(parts) >= 2 and parts[0] in NOTEBOOK_RECORD_KINDS and parts[1] == "notebooks":
        head, rest = parts[:2], parts[2:]
    else:
        return None
    if not rest or (head == ("figures",) and rest == ("Captions.md",)):
        return parts
    group = rest[0]
    known_group = group in reg.folders or group in LEGACY_GROUPS or legacy_phase(group) is not None
    if len(rest) == 1:
        return None  # A group directory: its notebooks translate individually.
    if head == ("figures",) and rest[1:] == ("Captions.md",):
        phase = legacy_phase(group)
        # Experiment reports follow their phase's `All Results.md` route (0B -> 03_exp0b).
        return ("figures", reg.folder_for(phase, ("All Results.md",)), "Captions.md") if phase else parts
    identity = f"{group}/{rest[1]}"
    notebook = reg.notebook(identity)
    if notebook and notebook.id == identity:
        return parts
    if notebook and identity in notebook.legacy:
        return (*head, notebook.folder, notebook.name, *rest[2:])
    return (*ARCHIVE.parts, *parts) if known_group else None  # Records of a retired notebook.


def translate_output(relative, reg: Registry | None = None) -> PurePosixPath | None:
    """Current location of a path relative to an output root; None when not a layout path.

    Accepts previous-layout and current-layout paths, so a path recorded before or after the
    migration (including a relative `..` reference across a routed folder) resolves the same.
    """
    reg = reg or registry()
    parts = PurePosixPath(*Path(relative).parts).parts if str(relative) not in {"", "."} else ()
    if not parts:
        return None
    if parts in {("All Results.md",), (".reports.lock",)}:
        return PurePosixPath(*parts)
    record = _notebook_record(parts, reg)
    if record is not None or parts[0] in {"figures", *NOTEBOOK_RECORD_KINDS}:
        return PurePosixPath(*record) if record is not None else None
    logical = logical_output(parts, reg)
    if logical is None:
        return None
    phase, sub = logical
    if parts[0] in reg.folders and sub in FOLDER_REPORTS:
        return PurePosixPath(*parts)  # Each current folder owns its generated report.
    return PurePosixPath(reg.folder_for(phase, sub), *sub)


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
    candidate = Path(*parts[: index + 1], *moved.parts)
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
