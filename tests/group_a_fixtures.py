"""Deterministic Group A fixture exports built with the CURRENT export writer code.

Realistic shape: the 18 catalog datasets (3 binary, 6 multiclass, 9 regression), the 17
configured decision models, per-column and joint requests. Tables come from the same functions
the Mindwell CPU jobs call (`decision_comparison.analyze_group`, `zero_shot.dataset_scores`,
`robustness_export._mode_effects` / `quality_records`, `geometry_study.relation`), so the
figures are tested against the export schema rather than against hand-written columns.
Synthetic values only; never shown in a notebook.
"""

from __future__ import annotations

import copy
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.utils.config import load_config, load_yaml

LEVELS = np.arange(-4, 5)


@lru_cache(maxsize=1)
def design():
    """(datasets, providers): catalog datasets with task, name, classes and text-field count."""
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries

    main = load_yaml("B1")
    semantics = load_yaml("dataset_semantics")["datasets"]
    rows = []
    for entry in selected_entries(load_catalog(), main):
        task = semantics[entry["id"]]["task_type"]
        k = 2 if task == "binary_classification" else 9 if task == "regression" else 3 + len(rows) % 3
        rows.append({"dataset_id": entry["id"], "dataset": entry.get("name", entry["id"]), "task": task,
                     "k": k, "text_columns": entry["paper_summary"]["text_columns"]})
    return pd.DataFrame(rows), list(main["candidate_extractors"])


def _model_traits(providers, rng):
    """Per model: signal weight, temperature and a bias, so the fixture has visible spread."""
    return {p: (rng.uniform(0.3, 1.6), rng.uniform(0.4, 2.0), rng.normal(0, 0.3)) for p in providers}


def _probabilities(latent, k, traits, rng):
    weight, temperature, bias = traits
    if k == 9:
        centre = np.clip(latent * 2.5 * weight + bias + rng.normal(0, 0.6, len(latent)), -4, 4)
        logits = -((LEVELS[None, :] - centre[:, None]) ** 2) / (2 * temperature)
    else:
        prototypes = np.linspace(-1.5, 1.5, k)
        logits = -(np.abs(prototypes[None, :] - (latent * weight + bias)[:, None]) * 2.5 / temperature)
        logits = logits + rng.normal(0, 0.4, logits.shape)
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def _options(providers):
    result = copy.deepcopy(load_config("A1")["comparison"])
    result["providers"] = list(providers)
    result["scatter_rows_per_group"] = 1
    return result


@lru_cache(maxsize=2)
def a1_tables(seed: int = 0, rows: int = 40):
    """``{output_shape, pairs, loo_consensus, quality, reliability}`` as in `comparison.zip`."""
    from src.evaluation import zero_shot
    from src.evaluation.decision_comparison import Comparison, GroupFeatures, analyze_group

    datasets, providers = design()
    rng = np.random.default_rng(seed)
    traits = _model_traits(providers, rng)
    options = _options(providers)
    result = Comparison()
    scores, curves = [], []
    for d in datasets.itertuples():
        latent = rng.normal(0, 1, rows)
        groups = {"joint": ["all_text"], "per_column": [f"text_{i:03d}" for i in range(min(2, d.text_columns))]}
        for mode, names in groups.items():
            for group in names:
                field_latent = latent + (rng.normal(0, 0.7, rows) if mode == "per_column" else 0)
                features = []
                for provider in providers:
                    p = _probabilities(field_latent, d.k, traits[provider], rng)
                    labels = () if d.task == "regression" else tuple(range(d.k))
                    features.append(GroupFeatures(provider, np.array([f"csv:{i}" for i in range(rows)]), p,
                                                  np.array([f"{d.dataset_id}:{group}:{i}" for i in range(rows)],
                                                           dtype=object), labels, d.task, "recipe"))
                analyze_group(features, {"dataset_id": d.dataset_id, "dataset": d.dataset, "representation": mode,
                                         "group": group, "text_column": group}, options, result)
                if mode != "joint":
                    continue
                if d.task == "regression":
                    target = latent * 10 + rng.normal(0, 4, rows)
                    classes = ()
                else:
                    target = np.digitize(latent + rng.normal(0, 0.8, rows), np.linspace(-1.5, 1.5, d.k + 1)[1:-1])
                    classes = tuple(range(d.k))
                blocks = {}
                for provider, feature in zip(providers, features):
                    suffixes = (["p_positive"] if d.k == 2 else [f"p_{i}" for i in range(9)] if d.task == "regression"
                                else [f"p_c{i:03d}" for i in range(d.k)])
                    values = feature.probabilities[:, 1:] if d.k == 2 else feature.probabilities
                    names_ = [f"{provider}__joint__text_only__all_text__{s}" for s in suffixes]
                    table = pd.DataFrame({"row_id": [f"csv:{i}" for i in range(rows)],
                                          **dict(zip(names_, values.T))})
                    blocks[provider] = (SimpleNamespace(values=table, manifest={
                        "feature_columns": names_, "text_columns": ["text"],
                        "identity": {"representation": "joint"}}), "fixture")
                metadata = SimpleNamespace(dataset_id=d.dataset_id, task_type=d.task, target="y",
                                           classes=list(classes), text_columns=["text"])
                dataset = SimpleNamespace(metadata=metadata, frame=pd.DataFrame({"y": target}))
                one, bins = zero_shot.dataset_scores(dataset, blocks, bins=10)
                scores.append(one)
                if not bins.empty:
                    curves.append(bins)
    return {"output_shape": result.table("output_shape"), "pairs": result.table("pairs"),
            "loo_consensus": result.table("loo_consensus"),
            "quality": pd.concat(scores, ignore_index=True),
            "reliability": pd.concat(curves, ignore_index=True)}


def _measured(p):
    return {"status": "measured", "probabilities": [float(x) for x in p]}


@lru_cache(maxsize=2)
def a2_tables(seed: int = 0, requests: int = 24):
    """``{sources, mode_effects, wording_averaging, averaging_quality}`` as in `robustness.zip`."""
    from src.evaluation import robustness_export as rx

    datasets, providers = design()
    rng = np.random.default_rng(seed)
    traits = _model_traits(providers, rng)
    sensitivity = {p: rng.uniform(0.0, 0.08) for p in providers}
    sources, mode_effects, wording, quality = [], [], [], []
    for d in datasets.itertuples():
        latent = rng.normal(0, 1, requests)
        modes = ["joint" if i % 2 == 0 else "per_column" for i in range(requests)]
        links = pd.DataFrame({"input_no": range(requests), "row_position": range(requests), "mode": modes})
        targets = (latent * 10 + rng.normal(0, 4, requests) if d.task == "regression" else
                   np.digitize(latent + rng.normal(0, 0.8, requests), np.linspace(-1.5, 1.5, d.k + 1)[1:-1]))
        for provider in providers:
            base = _probabilities(latent, d.k, traits[provider], rng)
            trials = {}
            for n in range(requests):
                original = base[n]
                item = {"original": _measured(original)}
                variants = {"repeat_1": 0.02, "repeat_2": 0.02, "paraphrase_1": 0.05 + sensitivity[provider] * 4,
                            "paraphrase_2": 0.05 + sensitivity[provider] * 3, "paraphrase_3": 0.05 + sensitivity[provider] * 5}
                if d.task == "multiclass_classification":
                    variants["option_order"] = 0.04 + sensitivity[provider] * 6
                if modes[n] == "joint" and d.text_columns >= 2:
                    variants["field_order"] = 0.03 + sensitivity[provider] * 2
                for name, scale in variants.items():
                    q = original * np.exp(rng.normal(0, scale, d.k))
                    q = q / q.sum()
                    item[name] = _measured(q) | {"effect": {"tv": float(np.abs(q - original).sum() / 2),
                                                            "flip": bool(q.argmax() != original.argmax())}}
                trials[n] = item
            context = {"dataset": d.dataset_id, "provider": provider}
            sources.append(context | {"file": "fixture.json", "requests": requests, "task": d.task})
            mode_effects += rx._mode_effects(trials, links, context)
            records = rx.wording_records(trials, d.task == "regression")
            for mode in ("per_column", "joint"):
                subset = records[records.input_no.isin(links.input_no[links["mode"].eq(mode)])]
                wording.append(context | {"task": d.task, "mode": mode, "requests": len(subset),
                                          "single_tv": subset.single_tv.mean(), "averaged_tv": subset.averaged_tv.mean(),
                                          "replay_tv": subset.replay_tv.mean()})
            found = rx.averaged_presentations(trials)
            quality += [context | {"task": d.task, **row, "field_order_canonicalized": 0}
                        for row in rx.quality_records(found, links, targets, d.task, 10)]
    return {"sources": pd.DataFrame(sources), "mode_effects": pd.DataFrame(mode_effects),
            "wording_averaging": pd.DataFrame(wording), "averaging_quality": pd.DataFrame(quality)}


@lru_cache(maxsize=2)
def a3_tables(seed: int = 0, texts: int = 30):
    """``{comparisons, encoder_references, coverage}`` as in `geometry.zip`."""
    from itertools import combinations

    from scipy.spatial.distance import pdist

    from src.evaluation import geometry_export
    from src.evaluation.geometry import decision_distances
    from src.evaluation.geometry_study import relation

    datasets, providers = design()
    encoders = load_yaml("A3")["encoders"]
    rng = np.random.default_rng(seed)
    traits = _model_traits(providers, rng)
    comparisons, references, coverage = [], [], []
    for d in datasets.itertuples():
        for f in range(min(2, d.text_columns)):
            field = f"field_{f}"
            semantic = rng.normal(0, 1, (texts, 6))
            enc = {e: pdist(semantic + rng.normal(0, 0.6 + 0.3 * i, semantic.shape), "cosine")
                   for i, e in enumerate(encoders)}
            coverage.append({"dataset": d.dataset_id, "field": field, "texts": texts, "missing_providers": ""})
            for a, b in combinations(encoders, 2):
                references.append({"dataset": d.dataset_id, "field": field, "left_encoder": a, "right_encoder": b,
                                   **relation(enc[a], enc[b], 42)})
            for provider in providers:
                p = _probabilities(semantic[:, 0], d.k, traits[provider], rng)
                decision = decision_distances(p)
                for e in encoders:
                    comparisons.append({"dataset": d.dataset_id, "field": field, "encoder": e, "provider": provider,
                                        "artifact_id": "fixture", **relation(enc[e], decision, 42)})
    flat = geometry_export._flatten
    return {"comparisons": flat(comparisons), "encoder_references": flat(references),
            "coverage": pd.DataFrame(coverage)}
