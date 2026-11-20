"""A3 (exp03_geometry): one compact export of the completed geometry summaries.

The analysis itself is unchanged (`geometry_study`, `config/exp03_geometry/geometry.yaml`).
The Mindwell CPU job (`scripts/slurm/rq1_geometry_export.slurm`) packages every full
(non-pilot) `results/geometry/<dataset>.json` on PROJECT into
`results/geometry/geometry.zip`; sampled text keys stay on PROJECT, only counts travel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation import study_export
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import digest, file_sha256

SCHEMA = "jevpfn.rq1c.export.v1"
TABLES = ("sources", "comparisons", "encoder_references", "coverage")


def export_path() -> Path:
    return paths.results_dir("geometry", "geometry.zip", phase="rq1")


def _flatten(rows):
    frame = pd.DataFrame(rows)
    if "text_block_spearman_range" in frame:
        bounds = frame.pop("text_block_spearman_range")
        frame["text_block_spearman_min"] = [b[0] if b else None for b in bounds]
        frame["text_block_spearman_max"] = [b[1] if b else None for b in bounds]
    return frame


def collect(root: Path) -> dict[str, pd.DataFrame]:
    sources, comparisons, references, coverage = [], [], [], []
    for file in sorted(Path(root).glob("*.json")):
        body = json.loads(file.read_text(encoding="utf-8"))
        if body.get("pilot") is not False:
            continue
        sources.append({"dataset": body["dataset"], "file": file.name, "json_sha256": file_sha256(file),
                        "config": body["config"], "data_identity": body["data_identity"]})
        comparisons += body["comparisons"]
        references += body.get("encoder_references", [])
        coverage += [{"dataset": body["dataset"], "field": s["field"], "texts": len(s["text_keys"]),
                      "missing_providers": ", ".join(body["missing_providers"])}
                     for s in body["samples"]]
    return {"sources": pd.DataFrame(sources), "comparisons": _flatten(comparisons),
            "encoder_references": _flatten(references), "coverage": pd.DataFrame(coverage)}


def analysis_identity(cfg: dict) -> str:
    return digest({"geometry": cfg, "schema": SCHEMA})


def read_export(target=None):
    return study_export.read_export(target or export_path(), schema=SCHEMA, tables=TABLES,
                                    identity=analysis_identity(load_yaml("rq1/geometry")))


def build():
    study_export.require_mindwell_cpu("A3 geometry")
    cfg = load_yaml("rq1/geometry")
    tables = collect(paths.results_dir("geometry", phase="rq1"))
    if tables["sources"].empty:
        raise ValueError("No completed full geometry summary to export")
    if set(tables["sources"].config) != {digest(cfg)}:
        raise ValueError("Geometry summaries use a configuration other than geometry.yaml")
    target = study_export.write_export(export_path(), tables, schema=SCHEMA,
                                       identity=analysis_identity(cfg), origin=study_export.job_origin())
    print(f"{target}  SHA256 {file_sha256(target)}", flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()
