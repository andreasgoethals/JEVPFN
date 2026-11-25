"""A3 (exp03_geometry): one compact export of the completed geometry summaries.

The analysis itself is unchanged (`geometry_study`, `config/A/geometry.yaml`).
The Mindwell CPU job (`scripts/slurm/rq1_geometry_export.slurm`) packages every full
(non-pilot) dataset summary in the selected roster directory into its `geometry.zip`.
Sampled text keys stay on PROJECT; only counts travel.
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

SCHEMA = "jevpfn.rq1c.export.v2"
TABLES = ("sources", "comparisons", "encoder_references", "coverage")


def export_path(cohort=None) -> Path:
    from src.evaluation.benchmark_export import cohort_key

    return paths.results_dir("geometry", cohort_key(cohort), "geometry.zip", phase="rq1")


def _flatten(rows):
    frame = pd.DataFrame(rows)
    if "text_block_spearman_range" in frame:
        bounds = frame.pop("text_block_spearman_range")
        frame["text_block_spearman_min"] = [b[0] if b else None for b in bounds]
        frame["text_block_spearman_max"] = [b[1] if b else None for b in bounds]
    return frame


def collect(root: Path, *, members=None) -> dict[str, pd.DataFrame]:
    sources, comparisons, references, coverage = [], [], [], []
    for file in sorted(Path(root).glob("*.json")):
        body = json.loads(file.read_text(encoding="utf-8"))
        if body.get("pilot") is not False:
            continue
        if members is not None:
            from src.evaluation.geometry_study import geometry_protocol

            expected = geometry_protocol(body["dataset"], body["data_identity"], load_yaml("rq1/geometry"),
                                         load_yaml("experiment_1/main"), members)
            if (body.get("protocol") != expected or body["missing_providers"]
                    or {row["provider"] for row in body["comparisons"]} != set(members)):
                raise ValueError(f"Geometry summary has incompatible protocol or provider coverage: {file.name}")
        sources.append({"dataset": body["dataset"], "file": file.name, "json_sha256": file_sha256(file),
                        "config": body["config"], "data_identity": body["data_identity"]})
        comparisons += body["comparisons"]
        references += body.get("encoder_references", [])
        coverage += [{"dataset": body["dataset"], "field": s["field"], "texts": len(s["text_keys"]),
                      "missing_providers": ", ".join(body["missing_providers"])}
                     for s in body["samples"]]
    return {"sources": pd.DataFrame(sources), "comparisons": _flatten(comparisons),
            "encoder_references": _flatten(references), "coverage": pd.DataFrame(coverage)}


def analysis_identity(cfg: dict, cohort=None) -> str:
    from src.evaluation.benchmark_export import cohort_record
    from src.utils.provenance import code_identity

    return digest({"geometry": cfg, "schema": SCHEMA, **cohort_record(cohort),
                   "code": code_identity(["src/evaluation/geometry.py", "src/evaluation/decision_comparison.py",
                                          "src/evaluation/geometry_study.py", "src/evaluation/geometry_export.py"])})


def read_export(target=None, *, cohort=None):
    return study_export.read_export(target or export_path(cohort), schema=SCHEMA, tables=TABLES,
                                    identity=analysis_identity(load_yaml("rq1/geometry"), cohort))


def build(cohort=None):
    from src.evaluation.benchmark_export import cohort_record

    study_export.require_mindwell_cpu("A3 geometry")
    cfg = load_yaml("rq1/geometry")
    record = cohort_record(cohort)
    target_path = export_path(cohort)
    tables = collect(target_path.parent, members=record["decision_roster"])
    if tables["sources"].empty:
        raise ValueError("No completed full geometry summary to export")
    if set(tables["sources"].config) != {digest(cfg)}:
        raise ValueError("Geometry summaries use a configuration other than geometry.yaml")
    target = study_export.write_export(target_path, tables, schema=SCHEMA, extra=record,
                                       identity=analysis_identity(cfg, cohort), origin=study_export.job_origin())
    print(f"{target}  SHA256 {file_sha256(target)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", metavar="COHORT")
    build(parser.parse_args().cohort)
