"""Read compact RQ1 exports. Keep validation pilots out of scientific summaries."""

import json

import pandas as pd

from src.utils import paths


def robustness_tables():
    rows, coverage, seen = [], [], set()
    for file in sorted(paths.results_dir("robustness", phase="rq1").glob("*/*/*.json")):
        body = json.loads(file.read_text(encoding="utf-8"))
        design = body["design"]
        if design["pilot"] or body.get("completed") is not True:
            continue
        identity = design["dataset"], design["provider"]
        if identity in seen:
            raise ValueError("Select one robustness campaign for each dataset and provider")
        seen.add(identity)
        labels = {"dataset": identity[0], "provider": identity[1]}
        coverage.extend({**labels, **row} for row in body["counts"])
        rows.extend(
            {**labels, "variant": variant, **item, **body.get("replay_contrasts", {}).get(variant, {})} for variant, item in body["effects"].items()
        )
    return pd.DataFrame(rows), pd.DataFrame(coverage)


def geometry_tables():
    rows, coverage = [], []
    for file in sorted(paths.results_dir("geometry", phase="rq1").glob("*.json")):
        body = json.loads(file.read_text(encoding="utf-8"))
        rows.extend(body["comparisons"])
        coverage.append(
            dict(
                dataset=body["dataset"],
                fields=len(body["samples"]),
                missing_providers=", ".join(body["missing_providers"]),
            )
        )
    return pd.DataFrame(rows), pd.DataFrame(coverage)


def equal_dataset_replay(effects):
    """Each dataset contributes one matched perturbation-minus-replay contrast."""
    if effects.empty or "excess_tv" not in effects:
        return pd.DataFrame(columns=["provider", "variant", "datasets", "mean_excess_tv"])
    return (effects.dropna(subset=["excess_tv"]).groupby(["provider", "variant"])
            .agg(datasets=("dataset", "nunique"), mean_excess_tv=("excess_tv", "mean")).reset_index())


def encoder_references():
    rows = []
    for file in sorted(paths.results_dir("geometry", phase="rq1").glob("*.json")):
        rows.extend(json.loads(file.read_text(encoding="utf-8")).get("encoder_references", []))
    return pd.DataFrame(rows)
