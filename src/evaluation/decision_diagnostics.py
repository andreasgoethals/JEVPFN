"""Descriptive, outcome-free diversity diagnostics on verified aligned features.

Agreement is not accuracy; probability concentration is not calibration. Exact
ties are never promoted to unambiguous consensus. No model is selected here.
"""

from __future__ import annotations

import html
import json
from itertools import combinations

import numpy as np
import pandas as pd

from src.utils.serialization import canonical_json


def correlation_vectors(probabilities, task):
    """Standardized columns and tied ranks; constant option columns stay undefined."""
    if task == "regression":
        values = (probabilities @ np.arange(-4, 5))[..., None]
    elif task == "binary_classification":
        values = probabilities[..., 1:2]
    else:
        values = probabilities
    standardized, ranks, varying = [], [], []
    for value in values:
        rank = pd.DataFrame(value).rank(method="average").to_numpy()
        good = np.ptp(value, axis=0) > 0
        centered = value - value.mean(axis=0)
        ranked = rank - rank.mean(axis=0)
        std = np.sqrt((centered ** 2).sum(axis=0))
        rank_std = np.sqrt((ranked ** 2).sum(axis=0))
        standardized.append(centered / np.where(good, std, 1))
        ranks.append(ranked / np.where(good, rank_std, 1))
        varying.append(good)
    return np.array(standardized), np.array(ranks), np.array(varying)


def matching_correlations(cache, i, j):
    values, ranks, varying = cache
    good = varying[i] & varying[j]
    return {
        "representation_pearson": float(np.clip((values[i, :, good] * values[j, :, good]).sum(axis=1).mean(), -1, 1)) if good.any() else np.nan,
        "representation_spearman": float(np.clip((ranks[i, :, good] * ranks[j, :, good]).sum(axis=1).mean(), -1, 1)) if good.any() else np.nan,
        "correlation_dimensions": int(good.sum()),
        "correlation_total_dimensions": len(good),
    }


def add_diagnostics(result, base, names, reference, ps, common, hashes, choices, cfg):
    """All-row metrics, unique-input sensitivity, vote patterns and extreme examples."""
    idx = np.flatnonzero(common)
    if not len(idx):
        if base["representation"] == "joint":
            result.frames.setdefault("row_diagnostics", []).append(pd.DataFrame({
                "dataset_id": base["dataset_id"], "dataset": base["dataset"],
                "task": reference.task, "row_id": reference.row_ids, "input_hash": hashes,
                "status": "missing_text", "models": len(names),
            }))
        return
    p, decisions = ps[:, idx], choices[:, idx]
    count = len(names)
    pair_tv = np.stack([np.abs(p[i] - p[j]).sum(axis=1) / 2
                        for i, j in combinations(range(count), 2)])
    vote_counts = np.stack([(decisions == c).sum(axis=0) for c in np.unique(decisions)])
    largest = vote_counts.max(axis=0)
    patterns = np.sort(vote_counts, axis=0)[::-1].T
    patterns = np.array([":".join(str(v) for v in row if v) for row in patterns])
    ambiguous = (p == p.max(axis=2, keepdims=True)).sum(axis=2).max(axis=0) > 1
    if reference.task == "regression":
        # We compare the sign of the expected direction, not the modal rubric option.
        ambiguous = np.zeros(len(idx), dtype=bool)
    strict = (largest == count) & ~ambiguous
    scores = p @ np.arange(-4, 5) if reference.task == "regression" else None
    score_range = np.ptp(scores, axis=0) if scores is not None else np.full(len(idx), np.nan)
    confidence = p.max(axis=2).mean(axis=0)
    unique, first, inverse, repeats = np.unique(hashes[idx], return_index=True, return_inverse=True, return_counts=True)
    mean_tv = pair_tv.mean(axis=0)
    max_tv = pair_tv.max(axis=0)
    tied_patterns = np.where(ambiguous, "tied top probability", patterns)
    for weighting, select in (("rows", np.arange(len(idx))), ("unique_inputs", first)):
        summary = {
            **base, "weighting": weighting, "models": count, "n": len(select),
            "source_rows": len(idx), "unique_inputs": len(unique),
            "unanimous_count": int(strict[select].sum()),
            "unanimous": float(strict[select].mean()),
            "ambiguous_fraction": float(ambiguous[select].mean()),
            "mean_pair_tv": float(mean_tv[select].mean()),
            "mean_max_probability": float(confidence[select].mean()),
            "identical_vectors": float((max_tv[select] <= 1e-12).mean()),
            "scalar_range_median": float(np.median(score_range[select])) if scores is not None else np.nan,
            "within_tolerance": float((score_range[select] <= cfg["regression_agreement_tolerance"]).mean()) if scores is not None else np.nan,
        }
        result.add("row_summary", summary)
        patterns_unique, counts = np.unique(tied_patterns[select], return_counts=True)
        for pattern, n in zip(patterns_unique, counts):
            result.add("vote_patterns", {**base, "weighting": weighting, "models": count,
                                        "pattern": pattern, "count": int(n),
                                        "fraction": n / len(select), "n": len(select)})
        if base["representation"] == "joint":
            # Feature validation permits 1e-6 summation roundoff. Keep boundary
            # observations in the final bin without changing their exported values.
            histogram, _, _ = np.histogram2d(confidence[select], np.clip(mean_tv[select], 0, 1), bins=20, range=((0, 1), (0, 1)))
            for x, y in zip(*np.nonzero(histogram)):
                result.add("confidence_density", {**base, "weighting": weighting,
                    "confidence_bin": int(x), "tv_bin": int(y), "count": int(histogram[x, y]),
                    "fraction": histogram[x, y] / len(select), "bins": 20})
    if base["representation"] != "joint":
        return
    # Keep missing source rows in the export, but outside every comparison denominator.
    rows = pd.DataFrame({"dataset_id": base["dataset_id"], "dataset": base["dataset"],
                         "task": reference.task, "row_id": reference.row_ids,
                         "input_hash": hashes, "status": np.where(common, "compared", "missing_text"),
                         "models": count})
    for name, value in {"largest_vote": largest, "vote_pattern": tied_patterns,
                        "unanimous": strict, "ambiguous": ambiguous,
                        "mean_pair_tv": mean_tv, "maximum_pair_tv": max_tv,
                        "mean_max_probability": confidence, "score_range": score_range,
                        "input_multiplicity": repeats[inverse]}.items():
        column = pd.Series(value, index=idx)
        rows[name] = column
    for i, name in enumerate(names):
        rows[f"choice_{name}"] = pd.Series(decisions[i], index=idx)
        if scores is not None:
            rows[f"score_{name}"] = pd.Series(scores[i], index=idx)
    result.frames.setdefault("row_diagnostics", []).append(rows)
    # Three complementary case types; repeated texts cannot crowd out other examples.
    for category, eligible, objective in (
        ("disputed", first, mean_tv),
        ("consensus_close", first[strict[first]], -mean_tv),
        ("consensus_different_probabilities", first[strict[first]], mean_tv),
    ):
        if not len(eligible):
            continue
        chosen = eligible[np.argsort(-objective[eligible], kind="stable")[0]]
        result.add("cases", {**base, "category": category,
            "row_id": str(reference.row_ids[idx[chosen]]), "input_hash": str(hashes[idx[chosen]]),
            "input_multiplicity": int(repeats[inverse[chosen]]), "vote_pattern": tied_patterns[chosen],
            "mean_pair_tv": float(mean_tv[chosen]), "score_range": float(score_range[chosen]),
            "labels": canonical_json(list(reference.labels) if reference.labels else list(range(-4, 5))),
            "probabilities": canonical_json({name: p[i, chosen].tolist() for i, name in enumerate(names)})})


def peer_majority_agreement(result):
    """Per model and dataset: does its answer equal the plurality answer of the others?

    Uses each distinct joint input once. Inputs on which the other models tie are counted
    separately rather than resolved by option order.
    """
    rows = result.table("row_diagnostics")
    if rows.empty or "status" not in rows:
        return pd.DataFrame()
    rows = rows[rows.status.eq("compared")].drop_duplicates(["dataset_id", "input_hash"])
    records = []
    for (dataset_id, dataset, task), frame in rows.groupby(["dataset_id", "dataset", "task"], sort=False):
        columns = [c for c in frame.columns if c.startswith("choice_") and frame[c].notna().all()]
        if len(columns) < 3:
            continue
        choices = frame[columns].to_numpy(dtype=int)
        values = np.unique(choices)
        counts = np.stack([(choices == v).sum(axis=1) for v in values], axis=1)
        for i, column in enumerate(columns):
            others = counts.copy()
            others[np.arange(len(choices)), np.searchsorted(values, choices[:, i])] -= 1
            top = others.max(axis=1, keepdims=True)
            decided = (others == top).sum(axis=1) == 1
            agree = values[others.argmax(axis=1)] == choices[:, i]
            records.append({"dataset_id": dataset_id, "dataset": dataset, "task": task,
                            "provider": column.removeprefix("choice_"),
                            "agreement": float(agree[decided].mean()) if decided.any() else np.nan,
                            "n": int(decided.sum()), "tied_peers": int((~decided).sum())})
    return pd.DataFrame(records)


def inspect_cases(result):
    """Resolve saved questions/text only after matching each exact semantic payload hash."""
    from src.jev.inputs import prepared_row
    from src.utils import paths
    cases = result.table("cases").copy()
    if cases.empty:
        return cases
    payloads, status = [], []
    for row in cases.itertuples():
        path = paths.dataset_question_path(row.dataset_id)
        if not path.exists():
            payloads.append(None)
            status.append("Prepared inputs unavailable locally")
            continue
        matches = [r for r in prepared_row(path, int(row.row_id.removeprefix("csv:")))
                   if r["payload_sha256"] == row.input_hash and r["payload"] is not None]
        if not matches:
            raise ValueError(f"Example text does not match saved feature input: {row.dataset_id}/{row.row_id}")
        payloads.append(canonical_json(matches[0]["payload"]))
        status.append("Exact payload hash verified")
    cases["payload_json"] = payloads
    cases["payload_status"] = status
    result.frames["cases"] = [cases]
    return cases


def case_browser(cases):
    """Expandable, escaped source text; row IDs are identifiers, never leaked target values."""
    from IPython.display import HTML
    sections = []
    for row in cases.itertuples():
        title = f"{row.dataset} · {row.category.replace('_', ' ')} · {row.row_id} · votes {row.vote_pattern} · distance {row.mean_pair_tv:.3f}"
        payload = json.dumps(json.loads(row.payload_json), ensure_ascii=False, indent=2) if row.payload_json else row.payload_status
        table = pd.DataFrame(json.loads(row.probabilities), index=json.loads(row.labels)).round(4)
        sections.append(f"<details><summary>{html.escape(title)}</summary><p>{html.escape(row.payload_status)}; exact text occurs {row.input_multiplicity} times.</p><pre style='white-space:pre-wrap'>{html.escape(payload)}</pre>{table.to_html(escape=True)}<p>Input SHA256: {row.input_hash}</p></details>")
    return HTML("\n".join(sections))


def key_findings(result):
    """Measured descriptive findings, with explicit dataset and model scope."""
    rows = result.table("row_summary")
    if rows.empty:
        return ["Two complete providers are required before agreement can be measured."]
    joint = rows[rows.representation.eq("joint") & rows.weighting.eq("unique_inputs")]
    findings = []
    for task, frame in joint.groupby("task", sort=False):
        low, high = frame.loc[frame.unanimous.idxmin()], frame.loc[frame.unanimous.idxmax()]
        findings.append(f"{task.replace('_', ' ')}: across {len(frame)} datasets, {int(frame.models.iloc[0])}-model unambiguous consensus averages {frame.unanimous.mean():.1%} (equal dataset weight); {low.dataset} {low.unanimous:.1%} to {high.dataset} {high.unanimous:.1%}.")
    pairs = result.table("pairs")
    if not pairs.empty:
        for task, frame in pairs[pairs.representation.eq("joint") & pairs.weighting.eq("unique_inputs")].groupby("task", sort=False):
            means = frame.groupby(["model_a", "model_b"])[["agreement", "representation_spearman", "total_variation"]].mean()
            a, b = means.agreement.idxmax()
            best = means.loc[(a, b)]
            findings.append(f"{task.replace('_', ' ')}: {a} and {b} have the largest mean same-answer fraction ({best.agreement:.1%}), but their rank correlation is {best.representation_spearman:.2f} and probability distance is {best.total_variation:.2f}. This identifies similar decisions, not the best predictor.")
    reuse = rows[rows.representation.eq("joint")].pivot(index="dataset", columns="weighting", values="unanimous")
    difference = reuse.rows - reuse.unique_inputs
    dataset = difference.abs().idxmax()
    findings.append(f"Exact repetition changes joint consensus most in {dataset}: row weighting minus distinct-input weighting is {100 * difference[dataset]:+.1f} percentage points.")
    fields = rows[rows.representation.eq("per_column") & rows.weighting.eq("unique_inputs")]
    if not fields.empty:
        spread = fields.groupby("dataset").unanimous.agg(["min", "max"])
        dataset = (spread["max"] - spread["min"]).idxmax()
        findings.append(f"Field-level consensus varies most within {dataset}: {spread.loc[dataset, 'min']:.1%} to {spread.loc[dataset, 'max']:.1%}. Named fields and their nonmissing denominators are retained; this does not measure field accuracy.")
    findings.append("Agreement and probability concentration do not establish correctness, calibration or downstream benefit. Benchmark results are required for those claims.")
    return findings
