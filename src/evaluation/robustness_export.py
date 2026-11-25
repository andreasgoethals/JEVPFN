"""A2 (exp02_robustness): one compact export of the completed stability campaigns.

Reads only already-collected responses: every completed, non-pilot `robustness.run` store
under `results/robustness/<dataset>/<provider>/` on PROJECT. No model is called. The Mindwell
CPU job (`scripts/slurm/rq1_robustness_export.slurm`) writes a ZIP keyed by the selected decision-model roster:

* ``release_evidence`` identifies the completed receipts examined for each selected
  provider and records the outcome of checking its release and settings.
* ``effects`` / ``coverage``: each run's own summary (one factor at a time; changes and their
  excess over the unchanged replays).
* ``mode_effects``: the same per request mode (per-column or joint request), with the paired
  excess over replay noise computed on requests that have both unchanged repeats.
* ``wording_averaging``: how much averaging over wordings reduces wording sensitivity — the
  distance between the averages of two disjoint pairs {original, p1} and {p2, p3}, against the
  mean distance between single versions across those pairs ({o,p1} × {p2,p3}).
* ``averaging_quality``: stand-alone quality on the sampled JOINT requests of three
  presentations per factor: the standard (``original``) presentation used in production, the
  ``average`` over the factor's presentations (four wordings; original and reversed option order,
  multiclass, realigned by class identity; original and reversed field order) and
  ``mean_single`` (each presentation scored on its own, then averaged). Target labels are used
  only for scoring, never for sampling or selection.

  Scores follow strictly proper scoring rules (Gneiting & Raftery 2007): log loss (primary) and
  Brier for classes; the ranked probability score over the nine ordered levels for regression
  (primary), with the observed level from nine equal-frequency bins of the dataset's targets.
  ROC-AUC (binary) / macro one-vs-rest AUC (multiclass) and Spearman of the expected level are
  rank measures. The primary contrast is ``average`` versus ``original``: nothing guarantees it.
  ``average`` versus ``mean_single`` is the Jensen gap; for the convex proper scores it is never
  negative and equals ensemble diversity (Wood et al., JMLR 2023), so it is reported as
  guaranteed and is not evidence that averaging helps. Rank measures are not convex, so no
  contrast on them is guaranteed (``CONTRASTS``/``METRIC_ROLES``).
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

SCHEMA = "jevpfn.rq1b.export.v4"  # v4 includes per-campaign model release evidence.
#: Ordered regression levels (lowest first) that the decision answers distribute over.
REGRESSION_LEVELS = 9
#: Scoring roles: lower_is_better, and convex in the probabilities (averaging gain guaranteed).
METRIC_ROLES = {
    "log_loss": {"role": "primary", "lower_is_better": True, "convex": True},
    "ranked_probability_score": {"role": "primary", "lower_is_better": True, "convex": True},
    "brier": {"role": "secondary_proper", "lower_is_better": True, "convex": True},
    "roc_auc": {"role": "secondary_rank", "lower_is_better": False, "convex": False},
    "roc_auc_ovr_macro": {"role": "secondary_rank", "lower_is_better": False, "convex": False},
    "spearman_expected_level": {"role": "secondary_rank", "lower_is_better": False, "convex": False},
    "ece": {"role": "descriptive", "lower_is_better": True, "convex": False},
}
#: (contrast, presentation, reference): ``average`` minus reference, per dataset.
CONTRASTS = (("average_vs_original", "average", "original"),
             ("average_vs_mean_single", "average", "mean_single"))
COLLECTED = ("sources", "release_evidence", "effects", "coverage", "mode_effects", "wording_averaging", "averaging_quality")
#: The dated 0A sensitivity receipts (DATA manifests), so the notebook reads only this ZIP.
HISTORICAL = ("historical_0a", "historical_0a_sources")
TABLES = COLLECTED + HISTORICAL
WORDINGS = ("original", "paraphrase_1", "paraphrase_2", "paraphrase_3")
REPEATS = ("repeat_1", "repeat_2")


def export_path(cohort=None) -> Path:
    from src.evaluation.benchmark_export import cohort_key

    return paths.results_dir("robustness", f"robustness.{cohort_key(cohort)}.zip", phase="rq1")


def _release_selectors(roster=None):
    from src.evaluation.benchmark_export import cohort_record
    from src.evaluation.design import decision_versions
    from src.evaluation.qualification import decision_selector

    members = cohort_record()["decision_roster"] if roster is None else roster
    main = load_yaml("experiment_1/main")
    out = {}
    for release in decision_versions(members):
        provider = release["provider"]
        selector = decision_selector(main, provider) if main["feature_selectors"].get(provider) else {}
        out[provider] = {**release, "settings_sha256": selector.get("settings_sha256")}
    return out


def completed_runs(root: Path, *, roster=None, release_evidence=None):
    for file, body, _, _ in _selected_runs(root, roster=roster, release_evidence=release_evidence):
        yield file, body


def _selected_runs(root, *, roster, release_evidence):
    """Select one campaign for each exact release and retain the basis for selection."""
    expected = _release_selectors(roster)
    seen = set()
    for file in sorted(Path(root).glob("*/*/*.json")):
        body = json.loads(file.read_text(encoding="utf-8"))
        design = body["design"]
        if design["pilot"] or body.get("completed") is not True:
            continue
        provider = design["provider"]
        if provider not in expected:
            continue
        pin = expected[provider]
        settings = body.get("decision_settings")
        fingerprint = design.get("settings")
        known_hash = pin["settings_sha256"]
        if settings is None:
            status = ("matched_legacy_settings" if known_hash and fingerprint == known_hash
                      else "unidentified_release")
        elif not isinstance(settings, dict) or digest(settings) != fingerprint or settings.get("provider") != provider:
            status = "inconsistent_settings"
        elif any(settings.get(k) != pin[k] for k in ("model", "version")):
            status = "other_release"
        elif known_hash and known_hash != fingerprint:
            status = "different_settings"
        else:
            status = "recorded_settings"
        if release_evidence is not None:
            named = settings if isinstance(settings, dict) else {}
            release_evidence.append({
                "dataset": design["dataset"], "provider": provider, "file": file.name, "status": status,
                "settings_sha256": fingerprint, "recorded_model": named.get("model"),
                "recorded_version": named.get("version"), "selected_model": pin["model"],
                "selected_version": pin["version"],
            })
        if status not in {"recorded_settings", "matched_legacy_settings"}:
            continue
        identity = design["dataset"], provider
        if identity in seen:
            raise ValueError("Select one robustness campaign for each dataset and provider")
        seen.add(identity)
        yield file, body, pin, status


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
    """(original, average, all versions) per factor and request; incomplete sets excluded."""
    factors = {"wording": WORDINGS, "option_order": ("original", "option_order"),
               "field_order": ("original", "field_order")}
    found: dict[str, dict[int, tuple]] = {factor: {} for factor in factors}
    for number, results in trials.items():
        for factor, variants in factors.items():
            if variants[-1] not in results:
                continue
            versions = [_probabilities(results, v) for v in variants]
            if all(v is not None for v in versions):
                found[factor][number] = (versions[0], np.mean(versions, axis=0), versions)
    return found


def outcome_levels(all_targets, y, levels=REGRESSION_LEVELS):
    """Observed ordered level 0..levels-1 of each target: equal-frequency bins of the dataset's
    targets (evaluation only). Tied cut-offs are kept, so some levels may stay empty."""
    reference = np.asarray(all_targets, float)
    reference = reference[np.isfinite(reference)]
    edges = np.quantile(reference, np.arange(1, levels) / levels)
    return np.searchsorted(edges, np.asarray(y, float), side="right")


def ranked_probability_score(observed_level, p):
    """Mean ranked probability score (Epstein 1969; strictly proper for ordered outcomes) in [0, 1]."""
    p = np.asarray(p, float)
    observed = np.arange(p.shape[1])[None, :] >= np.asarray(observed_level)[:, None]
    return float(np.mean(((np.cumsum(p, axis=1) - observed) ** 2).sum(axis=1) / (p.shape[1] - 1)))


def _scores(y, p, task, bins, observed_level):
    from src.evaluation.zero_shot import score_classification, score_regression

    p = p / p.sum(axis=1, keepdims=True)
    if task != "regression":
        return score_classification(y, p, task, bins)[0]
    return {**score_regression(y, p),
            "ranked_probability_score": ranked_probability_score(observed_level, p)}


def quality_records(found, links, targets, task, bins) -> list[dict]:
    """Stand-alone scores of the original, averaged and single presentations on joint rows."""
    joint = links[links["mode"].eq("joint")]
    rows = []
    for factor, sets in found.items():
        rows_used = joint[joint.input_no.isin(sets)]
        if rows_used.empty:
            continue
        y = targets[rows_used.row_position.to_numpy()]
        observed = outcome_levels(targets, y) if task == "regression" else None
        numbers = rows_used.input_no.tolist()

        stacks = {"original": [np.stack([sets[n][0] for n in numbers])],
                  "average": [np.stack([sets[n][1] for n in numbers])],
                  "mean_single": [np.stack([sets[n][2][i] for n in numbers])
                                  for i in range(len(sets[numbers[0]][2]))]}
        presentations = {name: [_scores(y, p, task, bins, observed) for p in stack]
                         for name, stack in stacks.items()}
        for presentation, scored in presentations.items():
            for metric in scored[0]:
                rows.append({"factor": factor, "presentation": presentation, "metric": metric,
                             "value": float(np.mean([s[metric] for s in scored])),
                             "requests": int(rows_used.input_no.nunique()), "rows": len(rows_used),
                             "versions": len(scored) if presentation == "mean_single" else 1})
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


def collect(root: Path, cfg: dict, *, roster=None, load_targets=None, question_path=None) -> dict[str, pd.DataFrame]:
    """All export tables from completed campaigns (labels only for the quality scores)."""
    from src.evaluation.zero_shot import target_values

    load_targets = load_targets or _dataset_targets
    question_path = question_path or paths.dataset_question_path
    bins = cfg["averaging"]["calibration_bins"]
    tables = {name: [] for name in COLLECTED}
    targets_cache = {}
    for file, body, pin, release_status in _selected_runs(root, roster=roster, release_evidence=tables["release_evidence"]):
        design = body["design"]
        dataset, provider = design["dataset"], design["provider"]
        base = {"dataset": dataset, "provider": provider}
        trials = stored_results(file.with_suffix(".sqlite3"))
        if set(map(int, body["sample"])) != set(trials):
            raise ValueError(f"{dataset}/{provider}: stored results differ from the frozen sample")
        reported = set()
        for variants in trials.values():
            for trial in variants.values():
                if trial.get("status") != "measured":
                    continue
                model, version = trial.get("reported_model"), trial.get("reported_version")
                if ((model is not None and model != pin["model"]) or (version is not None and version != pin["version"])
                        or (release_status == "recorded_settings" and
                            (not model or (pin.get("delivery") == "self_hosted" and not version)))):
                    raise ValueError(f"{dataset}/{provider}: stored reply differs from its requested decision release")
                if model:
                    reported.add(model)
        if dataset not in targets_cache:
            targets_cache[dataset] = load_targets(dataset, target_values)
        task, targets = targets_cache[dataset]
        links = request_links(question_path(dataset), design["source"], trials)
        tables["sources"].append({**base, "file": file.name, "json_sha256": file_sha256(file),
                                  "design_sha256": digest(design), "requests": len(trials),
                                  "model": pin["model"], "version": pin["version"],
                                  "settings_sha256": design.get("settings"), "release_evidence": release_status,
                                  "reported_model": next(iter(reported), None),
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
                                     "value", "requests", "rows", "versions",
                                     "field_order_canonicalized"]}
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


def analysis_identity(cfg: dict, cohort=None) -> str:
    from src.evaluation.acceptance import EXECUTION_FIELDS
    from src.evaluation.benchmark_export import cohort_record
    from src.utils.provenance import code_identity

    settings = {k: v for k, v in cfg.items() if k not in EXECUTION_FIELDS and k != "scientific_dispatch"}
    record = cohort_record(cohort)
    return digest({"behavior": settings, "schema": SCHEMA, "cohort": record,
                   "selectors": _release_selectors(record["decision_roster"]),
                   "analysis": code_identity(["src/evaluation/robustness_export.py", "src/evaluation/robustness.py",
                                              "src/evaluation/decision_comparison.py", "src/evaluation/zero_shot.py",
                                              "src/evaluation/notebook_evidence.py", "src/evaluation/study_export.py"])})


def read_export(target=None, *, cohort=None):
    cfg = load_yaml("rq1/behavior")
    return study_export.read_export(target or export_path(cohort), schema=SCHEMA, tables=TABLES,
                                    identity=analysis_identity(cfg, cohort))


def build(*, cohort=None):
    from src.evaluation.benchmark_export import cohort_record

    study_export.require_mindwell_cpu("A2 stability")
    cfg = load_yaml("rq1/behavior")
    record = cohort_record(cohort)
    root = paths.results_dir("robustness", phase="rq1")
    tables = collect(root, cfg, roster=record["decision_roster"])
    if tables["sources"].empty:
        raise ValueError("No completed scientific robustness campaign matches the selected releases")
    tables.update(historical_tables())
    target = study_export.write_export(export_path(cohort), tables, schema=SCHEMA,
                                       identity=analysis_identity(cfg, cohort), origin=study_export.job_origin(),
                                       extra=record)
    print(f"{target}  SHA256 {file_sha256(target)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", metavar="COHORT")
    build(cohort=parser.parse_args().cohort)
