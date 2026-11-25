"""Dated version and replay evidence for hosted decision models (default Jev), which can change.

Read-only over the local hosted stores; never calls the API and never reads credentials.

* version ledger: per dataset, requested version x server-reported model, with response
  counts and the first/last response time; flags any request answered by another model.
* replay evidence: the saved robustness rounds re-send identical synthetic requests
  ("repeat") next to the original ("base"); report the largest probability change per case,
  within each round (replay noise) and between rounds of different versions (version drift).

    python -m src.evaluation.jev_provenance --output PATH.json [--provider liquid_ultra]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from src.utils import paths

#: Hosted store folders under data/ and the model a pinned version must report.
#: Jev pins "1.14.0" and reports "jev-1.14.0"; Liquid pins "d1.2-ultra@<release_date>"
#: (src.jev.liquid_hosted) and its response reports only "d1.2-ultra".
HOSTED_PROVIDERS = {
    "jev": ("jev_cache", lambda version: f"jev-{version}"),
    "liquid_ultra": ("liquid_ultra_cache", lambda version: str(version).split("@", 1)[0]),
}


def _stores(root: Path):
    return sorted(p for p in root.glob("*.sqlite3") if p.is_file())


def _version_status(provider, requested_model, requested_version, reported):
    if not reported:
        return "model_unreported"
    if not requested_version:
        return "request_unpinned"
    expected = HOSTED_PROVIDERS[provider][1](requested_version)
    if provider == "jev":
        from src.jev.provider import pinned_jev_model

        try:
            pinned_jev_model({"version": requested_version, "model": expected})
        except ValueError:
            return "request_unpinned"
    elif provider == "liquid_ultra":
        from src.jev.liquid_hosted import pinned_version_ok

        if not pinned_version_ok(expected, requested_version):
            return "request_unpinned"
    if reported != expected or (requested_model is not None and requested_model != reported):
        return "mismatch"
    # The Liquid decision reply names the model; catalog observations are separate evidence.
    return "model_matches_release_unreported" if provider == "liquid_ultra" else "match"


def version_ledger(root: Path | None = None, provider: str = "jev") -> pd.DataFrame:
    folder, _ = HOSTED_PROVIDERS[provider]
    root = Path(root) if root is not None else paths.jev_cache_path(folder)
    rows = []
    for store in _stores(root):
        with closing(sqlite3.connect(store.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            job_columns = {col[1] for col in db.execute("PRAGMA table_info(extraction_jobs)")}
            release = "job.release_evidence_json" if "release_evidence_json" in job_columns else "NULL"
            join = ("LEFT JOIN extraction_jobs job ON job.request_key=r.request_key" if release != "NULL" else "")
            evidence = f"""
                SELECT r.kind, r.settings_json AS settings, r.response_json AS response, r.created_at,
                       {release} AS release_evidence FROM responses r {join} WHERE r.kind='verified_response'
            """
            if {"request_key", "request_json", "raw_response_json", "status"} <= job_columns:
                # A rejected model reply is billed and retained here, never admitted to
                # verified responses. The job schema has no receipt timestamp.
                evidence += f"""
                    UNION ALL
                    SELECT 'saved_' || job.status, json_extract(job.request_json,'$.settings'),
                           job.raw_response_json, NULL, {release}
                    FROM extraction_jobs AS job
                    WHERE job.raw_response_json IS NOT NULL AND NOT EXISTS (
                        SELECT 1 FROM responses AS r WHERE r.request_key=job.request_key
                        AND r.kind='verified_response' AND (
                            r.response_json=job.raw_response_json OR (job.status='completed' AND
                            json_extract(r.response_json,'$.model') IS json_extract(job.raw_response_json,'$.model'))
                        )
                    )
                """
            for kind, model, requested, reported, catalog, count, first, last in db.execute(f"""
                SELECT kind, json_extract(settings,'$.model'), json_extract(settings,'$.version'),
                       json_extract(response,'$.model'), release_evidence, COUNT(*), MIN(created_at), MAX(created_at)
                FROM ({evidence}) WHERE json_extract(settings,'$.provider')=? GROUP BY 1,2,3,4,5
            """, (provider,)):
                status = _version_status(provider, model, requested, reported)
                rows.append({"dataset_id": store.stem, "kind": kind, "requested_version": requested,
                             "requested_model": model, "reported_model": reported, "responses": count,
                             **({"catalog_release_evidence": json.loads(catalog) if catalog else None}
                                if provider == "liquid_ultra" else {}),
                             "first_response_utc": first, "last_response_utc": last,
                             "version_status": status, "version_mismatch": status == "mismatch"})
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
    root = Path(root) if root is not None else paths.jev_cache_path("jev_cache/_control")
    rows = []
    for round_dir in sorted(root.glob("*/rounds/*")):
        for base in sorted(round_dir.glob("*__base.json")):
            case = base.name.removesuffix("__base.json")
            repeat = round_dir / f"{case}__repeat.json"
            if not repeat.is_file():
                continue
            a = json.loads(base.read_text(encoding="utf-8"))
            b = json.loads(repeat.read_text(encoding="utf-8"))
            same_model = bool(a.get("model") and a["model"] == b.get("model"))
            rows.append({"suite": round_dir.parent.parent.name, "round": round_dir.name[:12],
                         "round_id": round_dir.name, "same_reported_model": same_model,
                         "case": case, "model": a.get("model"), "repeat_model": b.get("model"),
                         "max_probability_change": _max_change(a, b) if same_model else None,
                         "files_modified_utc": pd.Timestamp(base.stat().st_mtime, unit="s", tz="UTC").isoformat()})
    return pd.DataFrame(rows)


def version_drift(replays: pd.DataFrame, root: Path | None = None) -> pd.DataFrame:
    """Same synthetic base request answered by different Jev versions."""
    root = Path(root) if root is not None else paths.jev_cache_path("jev_cache/_control")
    if replays.empty:
        return pd.DataFrame()
    rows = []
    for (suite, case), group in replays.groupby(["suite", "case"]):
        versions = group.dropna(subset=["model"]).drop_duplicates("model")
        if len(versions) < 2:
            continue
        loaded = {}
        for _, r in versions.iterrows():
            if "round_id" in r:
                folder = root / suite / "rounds" / r["round_id"]
            else:  # Older exported reports only carried a 12-character round prefix.
                matches = list(root.glob(f"{suite}/rounds/{r['round']}*"))
                if len(matches) != 1:
                    raise ValueError("Historical replay round prefix does not identify one recording")
                folder = matches[0]
            loaded[r["model"]] = json.loads((folder / f"{case}__base.json").read_text(encoding="utf-8"))
        models = sorted(loaded)
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                rows.append({"suite": suite, "case": case, "model_a": m1, "model_b": m2,
                             "max_probability_change": _max_change(loaded[m1], loaded[m2])})
    return pd.DataFrame(rows)


def report(provider: str = "jev") -> dict:
    ledger = version_ledger(provider=provider)
    # Replay rounds exist only for Jev's saved robustness suites.
    replays = replay_evidence() if provider == "jev" else pd.DataFrame()
    drift = version_drift(replays)
    def records(frame):
        return frame.astype(object).where(frame.notna(), None).to_dict("records")

    versions = pd.DataFrame()
    if not ledger.empty:
        dates = ["first_response_utc", "last_response_utc"]
        ledger[dates] = ledger[dates].astype("string")
        versions = ledger.groupby(
            ["requested_version", "requested_model", "reported_model", "version_status"], dropna=False
        ).agg(
            responses=("responses", "sum"), datasets=("dataset_id", "nunique"),
            first_response_utc=("first_response_utc", "min"), last_response_utc=("last_response_utc", "max"),
        ).reset_index()
    body = {
        "schema": "jevpfn.jev-provenance.v1",
        "versions": records(versions),
        "version_mismatches": int(ledger.loc[ledger.version_mismatch, "responses"].sum()) if not ledger.empty else 0,
        "version_unconfirmed": int(ledger.loc[~ledger.version_status.isin(["match", "mismatch"]), "responses"].sum())
        if not ledger.empty else 0,
        "per_dataset": records(ledger),
        "replays": records(replays),
        "version_drift": records(drift),
        "interpretation": "Verified replies and retained unverified replies are grouped by requested and "
                          "reported model. Dates are recorded only where available. Replay noise compares "
                          "identical synthetic requests answered by the same reported model.",
    }
    if provider != "jev":
        body["provider"] = provider
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=tuple(HOSTED_PROVIDERS), default="jev")
    args = parser.parse_args()
    from src.utils.serialization import write_json

    body = report(args.provider)
    write_json(args.output, body)
    print(json.dumps({"versions": body["versions"], "version_mismatches": body["version_mismatches"],
                      "replays": len(body["replays"]), "version_drift": len(body["version_drift"])},
                     default=str))


if __name__ == "__main__":
    main()
