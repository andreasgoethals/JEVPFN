"""Planned paper figures (`config/synthesis/paper_figures.yaml`) -> owning notebook and saved figures.

A read-only map: it never moves notebooks, figures or results, and placement stays
``undecided`` until the owner decides. ``resolve`` reports, per planned figure, which saved
figure files already exist under ``output_JEVPFN/figures/<notebook>/`` (pending otherwise).
"""

from __future__ import annotations

import re
from fnmatch import fnmatch
from pathlib import Path

import pandas as pd

from src.utils import structure

SCHEMA = "jevpfn.paper_figures.v1"
CONFIG = "paper_figures"
STATUSES = ("planned", "available_when_results_exist")
KINDS = ("figure", "table")
_STEM = re.compile(r"^[a-z0-9_]+\*?$")
_SAVED = re.compile(r"^\d+_(?P<stem>.+)\.pdf$")


def load(raw: dict | None = None, registry: structure.Registry | None = None) -> dict:
    """The validated map; notebooks must be registered or reserved (planned) folders."""
    if raw is None:
        from src.utils.config import load_yaml

        raw = load_yaml(CONFIG)
    registry = registry or structure.registry()
    if raw.get("schema") != SCHEMA:
        raise ValueError("Unknown paper-figure schema")
    current = {n.folder for n in registry.notebooks}
    reserved = {p.folder for p in registry.planned}
    seen = set()
    for item in raw.get("figures") or []:
        if set(item) - {"id", "title", "notebook", "section", "figures", "placement", "status",
                        "note", "kind"}:
            raise ValueError(f"{item.get('id')}: unknown fields")
        if not item.get("id") or item["id"] in seen:
            raise ValueError("Paper figure IDs must be present and unique")
        seen.add(item["id"])
        if item["notebook"] not in current | reserved:
            raise ValueError(f"{item['id']}: unknown notebook {item['notebook']}")
        if item["notebook"] in reserved and item.get("status") != "planned":
            raise ValueError(f"{item['id']}: a reserved experiment can only be planned")
        if item.get("placement", "undecided") not in structure.PLACEMENTS:
            raise ValueError(f"{item['id']}: placement must be one of {structure.PLACEMENTS}")
        if item.get("status") not in STATUSES:
            raise ValueError(f"{item['id']}: status must be one of {STATUSES}")
        if item.get("kind", "figure") not in KINDS:
            raise ValueError(f"{item['id']}: kind must be one of {KINDS}")
        if any(not _STEM.match(stem) for stem in item.get("figures") or []):
            raise ValueError(f"{item['id']}: figure stems are lowercase names, optionally ending in *")
    return raw


def saved_stems(folder: Path) -> list[str]:
    """Saved figure stems in one notebook's figure folder (``NN_<stem>.pdf``)."""
    if not folder.is_dir():
        return []
    return sorted(m["stem"] for p in folder.glob("*.pdf") if (m := _SAVED.match(p.name)))


def resolve(raw: dict | None = None, *, figures_root: Path | None = None) -> pd.DataFrame:
    """One row per planned figure with the matching saved files, or a pending status."""
    plan = load(raw)
    if figures_root is None:
        from src.utils import paths

        figures_root = paths.figures_dir()
    rows = []
    for item in plan["figures"]:
        saved = saved_stems(Path(figures_root) / structure.figure_group(item["notebook"]))
        found = sorted({s for s in saved for pattern in item.get("figures") or []
                        if fnmatch(s, pattern)})
        rows.append({
            "id": item["id"], "title": item["title"], "notebook": item["notebook"],
            "kind": item.get("kind", "figure"), "placement": item.get("placement", "undecided"),
            "planned_figures": ", ".join(item.get("figures") or []) or "—",
            "saved_figures": ", ".join(found) or "pending",
            "status": item["status"], "note": item.get("note") or "",
        })
    return pd.DataFrame(rows)
