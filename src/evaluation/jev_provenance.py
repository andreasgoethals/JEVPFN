"""Dated version and replay evidence for the hosted Jev model, which can change over time.

Read-only over the local Jev stores; never calls the API and never reads credentials.

* version ledger: per dataset, requested version x server-reported model, with response
  counts and the first/last response time; flags any request answered by another model.
* replay evidence: the saved robustness rounds re-send identical synthetic requests
  ("repeat") next to the original ("base"); report the largest probability change per case,
  within each round (replay noise) and between rounds of different versions (version drift).

    python -m src.evaluation.jev_provenance --output PATH.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from src.utils import paths


def _stores(root: Path):
    return sorted(p for p in root.glob("*.sqlite3") if p.is_file())


def version_ledger(root: Path | None = None) -> pd.DataFrame:
    root = Path(root or paths.data_root() / "data" / "jev_cache")
    rows = []
    for store in _stores(root):
        with closing(sqlite3.connect(store.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            for kind, requested, reported, count, first, last in db.execute(
                "SELECT kind, json_extract(settings_json,'$.version'), json_extract(response_json,'$.model'),"
                " COUNT(*), MIN(created_at), MAX(created_at) FROM responses"
                " WHERE json_extract(settings_json,'$.provider')='jev' GROUP BY 1,2,3"
            ):
                rows.append({"dataset_id": store.stem, "kind": kind, "requested_version": requested,
                             "reported_model": reported, "responses": count,
                             "first_response_utc": first, "last_response_utc": last,
                             "version_mismatch": reported != f"jev-{requested}"})
    return pd.DataFrame(rows)


def _numbers(value, prefix=""):
    """Flatten numeric leaves of an answers object into {path: value}."""
    if isinstance(value, bool):
        return {}
    if isinstance(value, (int, float)):
        return {prefix: float(value)}
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(_numbers(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, list):
        out = {}
        for i, item in enumerate(value):
            out.update(_numbers(item, f"{prefix}[{i}]"))
        return out
    return {}


def _max_change(a: dict, b: dict) -> float | None:
    left, right = _numbers(a.get("answers", {})), _numbers(b.get("answers", {}))
    if not left or left.keys() != right.keys():
        return None
    return max(abs(left[k] - right[k]) for k in left)


def replay_evidence(root: Path | None = None) -> pd.DataFrame:
    root = Path(root or paths.data_root() / "data" / "jev_cache" / "_control")
    rows = []
    for round_dir in sorted(root.glob("*/rounds/*")):
        for base in sorted(round_dir.glob("*__base.json")):
            case = base.name.removesuffix("__base.json")
            repeat = round_dir / f"{case}__repeat.json"
            if not repeat.is_file():
                continue
            a = json.loads(base.read_text(encoding="utf-8"))
            b = json.loads(repeat.read_text(encoding="utf-8"))
            rows.append({"suite": round_dir.parent.parent.name, "round": round_dir.name[:12],
                         "case": case, "model": a.get("model"), "repeat_model": b.get("model"),
                         "max_probability_change": _max_change(a, b),
                         "files_modified_utc": pd.Timestamp(base.stat().st_mtime, unit="s", tz="UTC").isoformat()})
    return pd.DataFrame(rows)


def version_drift(replays: pd.DataFrame, root: Path | None = None) -> pd.DataFrame:
    """Same synthetic base request answered by different Jev versions."""
    root = Path(root or paths.data_root() / "data" / "jev_cache" / "_control")
    if replays.empty:
        return pd.DataFrame()
    rows = []
    for (suite, case), group in replays.groupby(["suite", "case"]):
        versions = group.drop_duplicates("model")
        if len(versions) < 2:
            continue
        loaded = {}
        for _, r in versions.iterrows():
            folder = next(root.glob(f"{suite}/rounds/{r['round']}*"))
            loaded[r["model"]] = json.loads((folder / f"{case}__base.json").read_text(encoding="utf-8"))
        models = sorted(loaded)
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                rows.append({"suite": suite, "case": case, "model_a": m1, "model_b": m2,
                             "max_probability_change": _max_change(loaded[m1], loaded[m2])})
    return pd.DataFrame(rows)


def report() -> dict:
    ledger = version_ledger()
    replays = replay_evidence()
    drift = version_drift(replays)
    return {
        "schema": "jevpfn.jev-provenance.v1",
        "versions": ledger.groupby(["requested_version", "reported_model"], dropna=False)
        .agg(responses=("responses", "sum"), datasets=("dataset_id", "nunique"),
             first_response_utc=("first_response_utc", "min"),
             last_response_utc=("last_response_utc", "max"))
        .reset_index().to_dict("records") if not ledger.empty else [],
        "version_mismatches": int(ledger.version_mismatch.sum()) if not ledger.empty else 0,
        "per_dataset": ledger.to_dict("records"),
        "replays": replays.to_dict("records"),
        "version_drift": drift.to_dict("records"),
        "interpretation": "Dated responses per server-reported version; replay rows are identical "
                          "synthetic requests re-sent within a round; drift rows compare versions.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from src.utils.serialization import write_json

    body = report()
    write_json(args.output, body)
    print(json.dumps({"versions": body["versions"], "version_mismatches": body["version_mismatches"],
                      "replays": len(body["replays"]), "version_drift": len(body["version_drift"])},
                     default=str))


if __name__ == "__main__":
    main()
