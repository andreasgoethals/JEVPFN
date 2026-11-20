"""A2 (exp02_robustness): one compact export of the completed stability campaigns.

Reads only already-collected responses: every completed, non-pilot `robustness.run` store
under `results/robustness/<dataset>/<provider>/` on PROJECT. No model is called. The Mindwell
CPU job (`scripts/slurm/rq1_robustness_export.slurm`) writes `results/robustness/robustness.zip`:

* ``effects`` / ``coverage``: each run's own summary (one factor at a time; changes and their
  excess over the unchanged replays).
* ``mode_effects``: the same per request mode (per-column or joint request), with the paired
  excess over replay noise computed on requests that have both unchanged repeats.
* ``wording_averaging``: how much averaging over wordings reduces wording sensitivity — the
  distance between the averages of two disjoint pairs {original, p1} and {p2, p3}, against the
  mean distance between single versions across those pairs ({o,p1} × {p2,p3}).
* ``averaging_quality``: stand-alone quality on the sampled JOINT requests of the original
  presentation versus the average of the four wordings, of original and reversed option order
  (multiclass, realigned by class identity) and of original and reversed field order. Target
  labels are used only for scoring, never for sampling or selection.
* ``historical_0a`` / ``historical_0a_sources``: the dated 0A sensitivity probes (three real
  inputs) and their receipts (file name, SHA-256), so the notebook reads nothing else.

Within-dataset results are combined with equal dataset weight in the notebook.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from itertools import product
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

from src.evaluation import study_export
from src.evaluation.decision_comparison import earth_movers
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import digest, file_sha256

SCHEMA = "jevpfn.rq1b.export.v2"  # v2: historical 0A sensitivity probes travel in the ZIP
COLLECTED = ("sources", "effects", "coverage", "mode_effects", "wording_averaging", "averaging_quality")
#: The dated 0A sensitivity receipts (DATA manifests), so the notebook reads only this ZIP.
HISTORICAL = ("historical_0a", "historical_0a_sources")
TABLES = COLLECTED + HISTORICAL
WORDINGS = ("original", "paraphrase_1", "paraphrase_2", "paraphrase_3")
REPEATS = ("repeat_1", "repeat_2")


def export_path() -> Path:
    return paths.results_dir("robustness", "robustness.zip", phase="rq1")


def completed_runs(root: Path):
    """One completed scientific (non-pilot) campaign per dataset and provider."""
    seen = set()
    for file in sorted(Path(root).glob("*/*/*.json")):
        body = json.loads(file.read_text(encoding="utf-8"))
        design = body["design"]
        if design["pilot"] or body.get("completed") is not True:
            continue
        identity = design["dataset"], design["provider"]
        if identity in seen:
            raise ValueError("Select one robustness campaign for each dataset and provider")
        seen.add(identity)
        yield file, body


def stored_results(store: Path) -> dict[int, dict[str, dict]]:
    with closing(sqlite3.connect(Path(store).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        rows = db.execute("SELECT input_no,variant,data FROM results").fetchall()
    found: dict[int, dict[str, dict]] = {}
    for number, variant, data in rows:
        found.setdefault(int(number), {})[variant] = json.loads(data)
    return found


def request_links(question_path: Path, input_set_id: str, numbers) -> pd.DataFrame:
    """Source rows and request modes (per_column/joint) of sampled distinct requests."""
    from src.jev.inputs import _read_set

    numbers = sorted({int(n) for n in numbers})
    with closing(sqlite3.connect(Path(question_path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        _, recipe, _, _ = _read_set(db, input_set_id)
        rows = []
        for start in range(0, len(numbers), 500):
            batch = numbers[start:start + 500]
            rows += db.execute(
                "SELECT input_no,row_position,slot FROM row_map WHERE input_set_id=? AND input_no IN ("
                + ",".join("?" for _ in batch) + ")", [input_set_id, *batch]).fetchall()
    slots = recipe["slots"]
    return pd.DataFrame([{"input_no": n, "row_position": pos, "mode": slots[slot]["representation"]}
                         for n, pos, slot in rows], columns=["input_no", "row_position", "mode"])


def _probabilities(results, variant):
    item = results.get(variant)
    return np.asarray(item["probabilities"], float) if item and item.get("status") == "measured" else None


def _distance(p, q, regression):
    return {"tv": float(np.abs(p - q).sum() / 2), "emd": float(earth_movers(p, q)) if regression else np.nan}


def wording_records(trials: dict[int, dict], regression: bool) -> pd.DataFrame:
    """Per request: averaged-pair distance versus single-version distances, plus replay."""
    rows = []
    for number, results in trials.items():
        versions = [_probabilities(results, v) for v in WORDINGS]
        if any(v is None for v in versions):
            continue
        o, p1, p2, p3 = versions
        singles = [_distance(a, b, regression) for a, b in product((o, p1), (p2, p3))]
        averaged = _distance((o + p1) / 2, (p2 + p3) / 2, regression)
        record = {"input_no": number,
                  "single_tv": np.mean([s["tv"] for s in singles]), "averaged_tv": averaged["tv"],
                  "single_emd": np.mean([s["emd"] for s in singles]), "averaged_emd": averaged["emd"]}
        repeats = [_probabilities(results, v) for v in REPEATS]
        if all(r is not None for r in repeats):
            record["replay_tv"] = np.mean([_distance(o, r, regression)["tv"] for r in repeats])
        rows.append(record)
    return pd.DataFrame(rows, columns=["input_no", "single_tv", "averaged_tv", "single_emd",
                                       "averaged_emd", "replay_tv"])


def averaged_presentations(trials: dict[int, dict]) -> dict[str, dict[int, tuple]]:
    """(original, average) probability pairs per factor and request; incomplete sets excluded."""
    factors = {"wording": WORDINGS, "option_order": ("original", "option_order"),
               "field_order": ("original", "field_order")}
    found: dict[str, dict[int, tuple]] = {factor: {} for factor in factors}
    for number, results in trials.items():
        for factor, variants in factors.items():
            if variants[-1] not in results:
                continue
            versions = [_probabilities(results, v) for v in variants]
            if all(v is not None for v in versions):
                found[factor][number] = (versions[0], np.mean(versions, axis=0))
    return found


def quality_records(found, links, targets, task, bins) -> list[dict]:
    """Stand-alone scores of original versus averaged presentations on joint source rows."""
    from src.evaluation.zero_shot import score_classification, score_regression

    joint = links[links["mode"].eq("joint")]
    rows = []
    for factor, pairs in found.items():
        rows_used = joint[joint.input_no.isin(pairs)]
        if rows_used.empty:
            continue
        y = targets[rows_used.row_position.to_numpy()]
        for position, presentation in enumerate(("original", "average")):
            p = np.stack([pairs[n][position] for n in rows_used.input_no])
            p = p / p.sum(axis=1, keepdims=True)
            scores = (score_regression(y, p) if task == "regression"
                      else score_classification(y, p, task, bins)[0])
            rows += [{"factor": factor, "presentation": presentation, "metric": metric, "value": value,
                      "requests": int(rows_used.input_no.nunique()), "rows": len(rows_used)}
                     for metric, value in scores.items()]
    return rows


def _mode_effects(trials, links, base):
    modes = links.groupby("input_no")["mode"].agg(set)
    rows = []
    for mode in ("per_column", "joint"):
        numbers = [n for n in trials if mode in modes.get(n, set())]
        variants = sorted({v for n in numbers for v, r in trials[n].items() if r.get("effect")})
        for variant in variants:
            effects = [(n, trials[n][variant]["effect"]) for n in numbers
                       if trials[n].get(variant, {}).get("effect")]
            flips = [e["flip"] for _, e in effects if e["flip"] is not None]
            excess = [e["tv"] - np.mean([trials[n][r]["effect"]["tv"] for r in REPEATS])
                      for n, e in effects
                      if variant not in REPEATS and all(trials[n].get(r, {}).get("effect") for r in REPEATS)]
            rows.append({**base, "mode": mode, "variant": variant, "measured": len(effects),
                         "mean_tv": float(np.mean([e["tv"] for _, e in effects])),
                         "flip_rate": float(np.mean(flips)) if flips else np.nan,
                         "unique_argmax_pairs": len(flips), "matched_requests": len(excess),
                         "excess_tv": float(np.mean(excess)) if excess else np.nan})
    return rows


def collect(root: Path, cfg: dict, *, load_targets=None, question_path=None) -> dict[str, pd.DataFrame]:
    """All export tables from completed campaigns (labels only for the quality scores)."""
    from src.evaluation.zero_shot import target_values

    load_targets = load_targets or _dataset_targets
    question_path = question_path or paths.dataset_question_path
    bins = cfg["averaging"]["calibration_bins"]
    tables = {name: [] for name in COLLECTED}
    targets_cache = {}
    for file, body in completed_runs(root):
        design = body["design"]
        dataset, provider = design["dataset"], design["provider"]
        base = {"dataset": dataset, "provider": provider}
        trials = stored_results(file.with_suffix(".sqlite3"))
        if set(map(int, body["sample"])) != set(trials):
            raise ValueError(f"{dataset}/{provider}: stored results differ from the frozen sample")
        if dataset not in targets_cache:
            targets_cache[dataset] = load_targets(dataset, target_values)
        task, targets = targets_cache[dataset]
        links = request_links(question_path(dataset), design["source"], trials)
        tables["sources"].append({**base, "file": file.name, "json_sha256": file_sha256(file),
                                  "design_sha256": digest(design), "requests": len(trials),
                                  "task": task})
        tables["coverage"] += [{**base, **row} for row in body["counts"]]
        tables["effects"] += [{**base, "variant": variant, **item,
                               **body.get("replay_contrasts", {}).get(variant, {})}
                              for variant, item in body["effects"].items()]
        tables["mode_effects"] += _mode_effects(trials, links, base)
        wording = wording_records(trials, task == "regression")
        modes = links.groupby("input_no")["mode"].agg(set)
        for mode in ("per_column", "joint"):
            members = {n for n, found in modes.items() if mode in found}
            subset = wording[wording.input_no.isin(members)]
            if subset.empty:
                continue
            tables["wording_averaging"].append({
                **base, "task": task, "mode": mode, "requests": len(subset),
                "single_tv": subset.single_tv.mean(), "averaged_tv": subset.averaged_tv.mean(),
                "averaged_to_single_tv": subset.averaged_tv.mean() / subset.single_tv.mean()
                if subset.single_tv.mean() > 0 else np.nan,
                "single_emd": subset.single_emd.mean(), "averaged_emd": subset.averaged_emd.mean(),
                "replay_requests": int(subset.replay_tv.notna().sum()),
                "replay_tv": subset.replay_tv.mean()})
        found = averaged_presentations(trials)
        canonicalized = sum(r.get("field_order", {}).get("status") == "canonicalized_by_runtime"
                            for r in trials.values())
        tables["averaging_quality"] += [
            {**base, "task": task, **row, "field_order_canonicalized": canonicalized}
            for row in quality_records(found, links, targets, task, bins)]
    columns = {"averaging_quality": ["dataset", "provider", "task", "factor", "presentation", "metric",
                                     "value", "requests", "rows", "field_order_canonicalized"]}
    return {name: pd.DataFrame(rows, columns=columns.get(name)) if rows or name in columns
            else pd.DataFrame(rows) for name, rows in tables.items()}


def historical_tables(evidence=None) -> dict[str, pd.DataFrame]:
    """The dated 0A sensitivity probes (three real inputs; not the 500-request study).

    Receipts are identified by file name and SHA-256; absolute local paths are not exported.
    """
    if evidence is None:
        from src.evaluation.notebook_evidence import prompt_evidence

        evidence = prompt_evidence()
    changes, _, sources = evidence
    changes = changes.drop(columns=["source"], errors="ignore").reset_index(drop=True)
    sources = pd.DataFrame([{**item, "path": PureWindowsPath(item["path"]).name,  # handles / and \ on every OS
                             "input_sets": json.dumps(item.get("input_sets", []))}
                            for item in sources],
                           columns=["path", "sha256", "created_utc", "status", "scope", "input_sets"])
    return {"historical_0a": changes, "historical_0a_sources": sources}


def _dataset_targets(dataset_id, target_values):
    from src.data.loaders import load_catalog, load_dataset
    from src.data.selection import selected_entries

    entry = next(e for e in selected_entries(load_catalog(), load_yaml("experiment_1/main"))
                 if e["id"] == dataset_id)
    dataset = load_dataset(entry, allow_download=False)
    return dataset.metadata.task_type, target_values(dataset)


def analysis_identity(cfg: dict) -> str:
    return digest({"behavior": cfg, "schema": SCHEMA})


def read_export(target=None):
    cfg = load_yaml("rq1/behavior")
    return study_export.read_export(target or export_path(), schema=SCHEMA, tables=TABLES,
                                    identity=analysis_identity(cfg))


def build():
    study_export.require_mindwell_cpu("A2 stability")
    cfg = load_yaml("rq1/behavior")
    root = paths.results_dir("robustness", phase="rq1")
    tables = collect(root, cfg)
    if tables["sources"].empty:
        raise ValueError("No completed scientific robustness campaign to export")
    tables.update(historical_tables())
    target = study_export.write_export(export_path(), tables, schema=SCHEMA,
                                       identity=analysis_identity(cfg), origin=study_export.job_origin())
    print(f"{target}  SHA256 {file_sha256(target)}", flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()
