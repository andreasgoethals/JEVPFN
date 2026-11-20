"""B4: how much labelled data do decision features replace? (exp11_data_efficiency)

This study DELIBERATELY subsamples outer-training rows, as an explicit owner-approved design
(20 November 2026); AGENTS.md forbids only *silent* caps. Every subsample is recorded in its
split identity (parent split, size, draw, row positions); test rows, classes, columns and
features are unchanged, and the full-data comparison is B1/B2 itself.

* Sizes 128 ... 16,384 (no full set). Draws: 5 for sizes <= 512, 3 for 1,024-2,048, 1 above.
* Within one outer fold and draw the subsets are NESTED across sizes (a prefix of one fixed
  ordering). Classification orderings are stratified: one example of every class first, then
  every class proportionally (systematic sampling inside a random within-class order), so no
  class is dropped whenever the size is at least the number of classes. Missing classes are
  recorded (and refused by the predictors), never merged. Regression orderings are uniform.
* Two fixed dataset sets: the main curve uses all 18 datasets at 128-2,048; the extended
  curve uses the seven datasets with at least 16,384 rows in every outer-training fold at
  all sizes. Membership is computed from the training-fold sizes and must equal the config.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.utils.config import load_yaml
from src.utils.serialization import canonical_json, digest, scalar

CONFIG = "data_efficiency/data_efficiency"


def config():
    cfg = load_yaml(CONFIG)
    sizes = cfg["training_sizes"]
    if sizes != sorted(set(sizes)) or sizes[0] < 1:
        raise ValueError("Training sizes must be increasing and positive")
    return cfg


def draws_for(size, cfg=None):
    cfg = cfg or config()
    for rule in cfg["draws"]:
        if size <= rule["max_size"]:
            return rule["draws"]
    raise ValueError(f"No draw rule for training size {size}")


def min_training_rows(rows, folds=5):
    """Smallest outer-training partition of (stratified) K-fold over `rows` rows."""
    return rows - math.ceil(rows / folds)


def curve_membership(entries, cfg=None):
    """{dataset_id: sizes}; asserts the extended set equals the configured seven datasets."""
    cfg = cfg or config()
    extended = cfg["curves"]["extended"]
    observed = sorted(e["id"] for e in entries
                      if min_training_rows(e["paper_summary"]["rows"]) >= extended["min_training_rows"])
    if observed != sorted(extended["datasets"]):
        raise ValueError(f"Extended-curve membership differs from the configuration: {observed}")
    largest_main = max(cfg["curves"]["main"]["sizes"])
    sizes = {}
    for e in entries:
        if min_training_rows(e["paper_summary"]["rows"]) < largest_main:
            raise ValueError(f"{e['id']} cannot supply {largest_main} training rows in every fold")
        sizes[e["id"]] = list(extended["sizes"] if e["id"] in extended["datasets"]
                              else cfg["curves"]["main"]["sizes"])
    return sizes


def curve_of(dataset_id, size, cfg=None):
    cfg = cfg or config()
    if size in cfg["curves"]["main"]["sizes"]:
        return "main"
    return "extended"


def nested_order(train, labels, *, seed):
    """Fixed ordering of outer-training positions; every prefix is one nested subsample.

    `labels` are class codes (classification) or None (regression). Classification: every
    class appears once before any class appears twice, then rows follow systematic
    proportional allocation (key (rank + u) / n_class), so prefixes stay stratified.
    """
    train = np.asarray(train)
    rng = np.random.default_rng(seed)
    if labels is None:
        return train[rng.permutation(len(train))]
    labels = np.asarray(labels)
    keys = np.empty(len(train))
    first = np.zeros(len(train), dtype=bool)
    for value in np.unique(labels):
        members = np.flatnonzero(labels == value)
        members = members[rng.permutation(len(members))]
        offset = rng.random()
        keys[members] = (np.arange(len(members)) + offset) / len(members)
        first[members[0]] = True
    order = np.lexsort((keys, ~first))  # all first-of-class rows, then by proportional key
    return train[order]


def subsample_specification(parent, labels, size, draw, *, seed, classes=None):
    """A recorded B4 split: outer test rows unchanged, nested stratified training subsample.

    `labels`: per-row target values of the whole dataset (classification) or None.
    """
    folds, missing = [], []
    labels = None if labels is None else np.asarray(labels, dtype=object)
    for number, fold in enumerate(parent["folds"]):
        train = np.asarray(fold["train"])
        if len(train) < size:
            raise ValueError(f"Outer fold {number} has {len(train)} training rows, fewer than {size}")
        codes = None
        if labels is not None:
            codes = np.array([canonical_json(scalar(v)) for v in labels[train]])
        order = nested_order(train, codes, seed=[int(seed), int(draw), number])
        chosen = sorted(int(i) for i in order[:size])
        if classes is not None and labels is not None:
            present = {canonical_json(scalar(labels[i])) for i in chosen}
            absent = [c for c in classes if canonical_json(c) not in present]
            missing.append(absent)
        folds.append({"train": chosen, "test": list(fold["test"])})
    body = {k: v for k, v in parent.items() if k not in ("split_id", "folds")}
    body["folds"] = folds
    body["subsample"] = {
        "parent_split_id": parent["split_id"],
        "training_size": int(size),
        "draw": int(draw),
        "sampling_seed": int(seed),
        "method": "nested_stratified_one_per_class_first" if labels is not None else "nested_uniform",
        "missing_classes_by_fold": missing,
    }
    return {**body, "split_id": digest(body)}


def data_efficiency_arms(entries, models, providers, cfg=None):
    """Plan rows of the B4 grid (one row = dataset x predictor x arm x size x draw)."""
    cfg = cfg or config()
    membership = curve_membership(entries, cfg)
    arms_cfg = cfg["arms"]
    rows = []
    for entry in entries:
        for size in membership[entry["id"]]:
            for draw in range(draws_for(size, cfg)):
                extra = dict(training_size=size, draw=draw, curve=curve_of(entry["id"], size, cfg))
                for model in models:
                    base = dict(dataset_id=entry["id"], model=model, phase="data_efficiency", **extra)
                    rows.append({**base, "feature_set": "non_text", "provider": None})
                    for provider in providers:
                        rows.append({**base, "feature_set": arms_cfg["decision_feature_set"],
                                     "provider": provider})
                    for encoder in arms_cfg["encoders"]:
                        rows.append({**base, "feature_set": "non_text_plus_frozen_e5", "provider": None,
                                     "encoder": encoder, "pca_scope": arms_cfg["pca"]["scope"],
                                     "pca_components": arms_cfg["pca"]["components"]})
                    rows.append({**base, "feature_set": "non_text_plus_supervised_probe",
                                 "provider": None, "encoder": arms_cfg["probe"]["encoder"],
                                 "probe_view": arms_cfg["probe"]["view"]})
    return rows


# --------------------------------------------------------------------------- analysis


def size_gains(headline):
    """Per dataset x predictor x arm x size: raw gain and share of headroom over the
    same-size, same-draw baseline, averaged over folds then draws (draws are not datasets).

    `headline`: one row per arm x fold with columns dataset_id, task_type, model_id,
    representation, training_size, draw, outer_fold, value (headline metric).
    """
    if headline.empty:
        return pd.DataFrame()
    keys = ["dataset_id", "task_type", "model_id", "training_size", "draw", "outer_fold"]
    base = headline[headline.representation.eq("baseline")][keys + ["value"]].rename(
        columns={"value": "baseline"})
    if base.duplicated(keys).any():
        raise ValueError("Several baselines for one dataset, predictor, size, draw and fold")
    data = headline[~headline.representation.eq("baseline")].merge(base, on=keys, how="inner")
    data["raw_gain"] = data.value - data.baseline
    data["headroom_share"] = data.raw_gain / np.maximum(1 - data.baseline, 1e-3)
    per_draw = data.groupby(["dataset_id", "task_type", "model_id", "representation",
                             "training_size", "draw"], as_index=False).agg(
        raw_gain=("raw_gain", "mean"), headroom_share=("headroom_share", "mean"),
        baseline=("baseline", "mean"), folds=("outer_fold", "nunique"))
    per_draw = per_draw[per_draw.folds.eq(5)]
    out = per_draw.groupby(["dataset_id", "task_type", "model_id", "representation",
                            "training_size"], as_index=False).agg(
        raw_gain=("raw_gain", "mean"), headroom_share=("headroom_share", "mean"),
        baseline=("baseline", "mean"), draws=("draw", "nunique"),
        draw_variance=("raw_gain", "var"))
    out["log2_size"] = np.log2(out.training_size)
    return out


def pooled_curve(gains, *, outcome="raw_gain"):
    """Random-effects (DerSimonian-Laird) combination over datasets per predictor, arm, size.

    Within-dataset variance: variance over draws / draws where several draws exist; with one
    draw, the between-dataset spread alone is reported (fixed zero within variance is not
    assumed) and the status says so.
    """
    from src.evaluation.moderator_statistics import dersimonian_laird

    rows = []
    if gains.empty:
        return pd.DataFrame()
    for (model, representation, size), group in gains.groupby(
            ["model_id", "representation", "training_size"], sort=True):
        values = group[outcome].to_numpy(float)
        variances = (group.draw_variance / group.draws).to_numpy(float)
        row = dict(model_id=model, representation=representation, training_size=size,
                   log2_size=math.log2(size), datasets=len(group), mean=float(np.mean(values)))
        if np.isfinite(variances).all() and (variances > 0).all():
            result = dersimonian_laird(values, variances)
            row.update(pooled=result["pooled"], ci_low=result["ci_low"], ci_high=result["ci_high"],
                       tau2=result["tau2"], status=result["status"])
        else:
            se = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else np.nan
            row.update(pooled=row["mean"], ci_low=row["mean"] - 1.96 * se,
                       ci_high=row["mean"] + 1.96 * se, tau2=np.nan,
                       status="single_draw_equal_weight_normal_interval")
        rows.append(row)
    return pd.DataFrame(rows)


def gain_slopes(gains, *, outcome="raw_gain"):
    """Per dataset x predictor x arm: least-squares slope of gain on log2(size)."""
    rows = []
    for key, group in gains.groupby(["dataset_id", "task_type", "model_id", "representation"]):
        if group.training_size.nunique() < 2:
            continue
        slope = np.polyfit(group.log2_size, group[outcome], 1)[0]
        rows.append(dict(zip(["dataset_id", "task_type", "model_id", "representation"], key),
                         slope_per_doubling=float(slope), sizes=int(group.training_size.nunique())))
    return pd.DataFrame(rows)
