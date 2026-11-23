"""Deterministic Group B export fixtures, built through the real export code.

Synthetic fold records in the exact `benchmark_export.collect_folds` schema (RECIPE_COLUMNS,
`recipe_id`, `metric`, `value`, category labels) for the real 18 selected datasets (3 binary,
6 multiclass, 9 regression, with their real text-column and class counts), the 17 configured
decision models, the 7 predictors and 5 outer folds. B1/B2/B4/B5 tables come from
`benchmark_export.build_tables`; B3 tables from the same `group_b` functions the export calls
(its dataset properties need the raw data, so they are synthesised in the exported columns).
`exports()` round-trips every table through `study_export.write_export` and
`benchmark_export.read`, so figures see exactly the Parquet schema a notebook reads.

Values are invented; they only exercise shapes, scales and edge cases (negative gains, a
near-perfect baseline, one-text-column datasets). Never use them as results.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np
import pandas as pd

from src.evaluation import benchmark_export, complementarity, group_b
from src.evaluation.benchmark_export import RECIPE_COLUMNS
from src.evaluation.campaign import PROBE_VIEWS
from src.evaluation.endpoints import HEADLINES
from src.evaluation.models import MAIN_MODELS, TEXT_MODELS

SEED = 20261123
ENCODERS = ("e5_small_v2", "e5_large_v2", "qwen3_embedding_0_6b")
CATEGORY_COLUMNS = ("phase", "dataset_id", "task_type", "model_id", "feature_set", "provider",
                    "encoder", "pca_scope", "probe_view", "metric", "view", "checkpoint_id",
                    "split_id", "hidden_view", "hidden_position", "adaptation_cell", "recipe_id")


@lru_cache(maxsize=1)
def datasets() -> pd.DataFrame:
    """The 18 selected datasets with task, text columns, non-text columns and classes."""
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries
    from src.utils.config import load_yaml

    semantics = load_yaml("dataset_semantics")["datasets"]
    rows = []
    for entry in selected_entries(load_catalog(), load_yaml("experiment_1/main")):
        task = semantics[entry["id"]]["task_type"]
        summary = entry["paper_summary"]
        rows.append({"dataset_id": entry["id"], "task_type": task, "rows": summary["rows"],
                     "text_columns": summary["text_columns"],
                     "non_text_columns": summary["non_text_columns"],
                     "classes": len(semantics[entry["id"]].get("classes", []))})
    return pd.DataFrame(rows)


def roster() -> list[str]:
    return benchmark_export.decision_roster()


class _Generator:
    """Shared latent structure: dataset text value, model quality, predictor sensitivity."""

    def __init__(self, seed=SEED):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.data = datasets().set_index("dataset_id")
        n = len(self.data)
        cls = self.data.task_type.ne("regression").to_numpy()
        self.base = pd.Series(np.where(cls, rng.uniform(0.66, 0.93, n), rng.uniform(0.18, 0.78, n)),
                              index=self.data.index)
        self.base.iloc[1] = 0.985  # one near-perfect baseline: the headroom share explodes
        scale = np.where(cls, 0.018, 0.07)
        self.value = pd.Series(rng.normal(0.7, 0.8, n) * scale, index=self.data.index)
        self.noise = pd.Series(np.where(cls, 0.004, 0.015), index=self.data.index)
        models = roster()
        self.quality = pd.Series(np.sort(rng.normal(1.0, 0.35, len(models)))[::-1], index=models)
        self.sensitivity = pd.Series(rng.uniform(0.6, 1.3, len(MAIN_MODELS)), index=MAIN_MODELS)
        self.offset = {(d, m): rng.normal(0, 0.015) for d in self.data.index for m in MAIN_MODELS}

    def width(self, dataset, kind):
        row = self.data.loc[dataset]
        per = 1 if row.task_type == "binary_classification" else (
            9 if row.task_type == "regression" else row.classes)
        return {"per_column": per * row.text_columns, "joint": per,
                "both": per * row.text_columns + (per if row.text_columns > 1 else 0)}[kind]

    def baseline(self, dataset, model):
        return float(np.clip(self.base[dataset] + self.offset[dataset, model], 0.05, 0.995))

    def gain(self, dataset, model, factor):
        headroom = 1 - self.baseline(dataset, model)
        raw = self.value[dataset] * self.sensitivity[model] * factor
        return float(np.clip(raw, -0.5 * headroom, 0.7 * headroom))


def _record(dataset, task, phase, model, feature_set, *, fold, metric, value, **fields):
    row = dict.fromkeys(RECIPE_COLUMNS)
    row.update(phase=phase, dataset_id=dataset, task_type=task, model_id=model,
               checkpoint_id=f"{model}-checkpoint", feature_set=feature_set, view="main",
               split_id=f"split-{dataset}", outer_fold=fold, metric=metric, value=value, **fields)
    if fields.get("provider") and task == "regression":
        row["view"] = "probabilities"
    identity = [row[c] for c in RECIPE_COLUMNS if c != "outer_fold"]
    row["recipe_id"] = "r" + hashlib.sha256("|".join(map(str, identity)).encode()).hexdigest()[:16]
    return row


def _arm(rows, gen, dataset, model, phase, feature_set, gain, width, *, seconds=1.0, **fields):
    task = gen.data.loc[dataset, "task_type"]
    for fold in range(5):
        noise = gen.rng.normal(0, gen.noise[dataset])
        value = gen.baseline(dataset, model) + gain + noise
        value = float(min(value, 0.999)) if task != "regression" else float(value)
        for metric, number in ((HEADLINES[task], value), ("columns.representation", width),
                               ("seconds.fit", seconds * (1 + 0.1 * fold))):
            rows.append(_record(dataset, task, phase, model, feature_set, fold=fold,
                                metric=metric, value=number, **fields))


def _categorise(rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=[*RECIPE_COLUMNS, "recipe_id", "metric", "value"])
    for column in CATEGORY_COLUMNS:
        frame[column] = frame[column].astype("category")
    return frame


VIEW_FACTOR = {"per_column": 0.85, "joint": 0.9, "both": 1.0}
FEATURE_SET = {"per_column": "non_text_plus_per_column", "joint": "non_text_plus_joint",
               "both": "non_text_plus_both"}


def folds(phases=("experiment_1",), *, predictors=MAIN_MODELS, seed=SEED) -> pd.DataFrame:
    """Complete five-fold records of the requested phases (B4 lives in `size_folds`)."""
    gen = _Generator(seed)
    rows = []
    for dataset, info in gen.data.iterrows():
        single = info.text_columns == 1
        for model in predictors:
            if "experiment_1" in phases:
                _arm(rows, gen, dataset, model, "experiment_1", "non_text", 0.0, 0)
                for provider, quality in gen.quality.items():
                    for view in ("per_column", "joint", "both"):
                        # One text column: the three views are the same block (deduplicated).
                        factor = 0.95 if single else VIEW_FACTOR[view]
                        _arm(rows, gen, dataset, model, "experiment_1", FEATURE_SET[view],
                             gen.gain(dataset, model, quality * factor), gen.width(dataset, view),
                             provider=provider)
            if "experiment_2" in phases:
                for i, encoder in enumerate(ENCODERS):
                    for scope, components in (("per_column", 32), ("joint", 64)):
                        width = components * (info.text_columns if scope == "per_column" else 1)
                        _arm(rows, gen, dataset, model, "experiment_2", "non_text_plus_frozen_e5",
                             gen.gain(dataset, model, 0.55 + 0.08 * i - 0.1 * (scope == "joint")),
                             width, encoder=encoder, pca_scope=scope, pca_components=components)
                for provider, quality in gen.quality.items():
                    _arm(rows, gen, dataset, model, "experiment_2", "non_text_plus_both_plus_frozen_e5",
                         gen.gain(dataset, model, quality * 1.05 + 0.15),
                         gen.width(dataset, "both") + 32 * info.text_columns, provider=provider,
                         encoder="e5_small_v2", pca_scope="per_column", pca_components=32)
            if "experiment_3" in phases:
                for scope, components in (("per_column", 32), ("joint", 64)):
                    width = components * (info.text_columns if scope == "per_column" else 1)
                    _arm(rows, gen, dataset, model, "experiment_3", "non_text_plus_tar",
                         gen.gain(dataset, model, 1.2 - 0.1 * (scope == "joint")), width,
                         encoder="e5_small_v2_tar", pca_scope=scope, pca_components=components,
                         seconds=4.0)
                for i, encoder in enumerate(ENCODERS):
                    for view in PROBE_VIEWS:
                        _arm(rows, gen, dataset, model, "experiment_3",
                             "non_text_plus_supervised_probe",
                             gen.gain(dataset, model, 1.1 + 0.05 * i - 0.1 * (view == "joint")),
                             gen.width(dataset, view), encoder=encoder, probe_view=view)
            if "rq2" in phases and info.text_columns >= 2:
                weights = gen.rng.dirichlet(np.ones(info.text_columns))
                for provider, quality in gen.quality.items():
                    full = gen.gain(dataset, model, quality * VIEW_FACTOR["per_column"])
                    for column in range(info.text_columns):
                        _arm(rows, gen, dataset, model, "rq2",
                             "non_text_plus_per_column_without_one_text",
                             full * (1 - weights[column]),
                             gen.width(dataset, "per_column") - gen.width(dataset, "joint"),
                             provider=provider, dropped_text_column=column)
        if "experiment_4" in phases:
            for i, model in enumerate(TEXT_MODELS):
                _arm(rows, gen, dataset, "catboost", "experiment_4", "original_features",
                     gen.gain(dataset, "catboost", 1.4 - 0.3 * i), 0, seconds=60.0)
                rows[-15:] = [dict(r, model_id=model, checkpoint_id=f"{model}-checkpoint")
                              for r in rows[-15:]]
    frame = _categorise(rows)
    if "experiment_4" in phases:  # recipe ids of renamed native rows stay unique per system
        frame["recipe_id"] = (frame.recipe_id.astype(str) + "-" + frame.model_id.astype(str)).astype("category")
    return frame


def size_folds(*, predictors=MAIN_MODELS, seed=SEED) -> pd.DataFrame:
    """B4 records: baseline, 17 decision 'both' arms, 3 frozen encoders and the classifier per
    training size and draw (main: 18 datasets at 128-2,048; extended: 7 datasets to 16,384)."""
    from src.evaluation import data_efficiency

    cfg = data_efficiency.config()
    gen = _Generator(seed)
    extended = set(cfg["curves"]["extended"]["datasets"])
    rows = []
    for dataset in gen.data.index:
        sizes = cfg["curves"]["extended" if dataset in extended else "main"]["sizes"]
        for size in sizes:
            shrink = np.log2(size / 128) / 7  # 0 at 128, 1 at 16,384
            for draw in range(data_efficiency.draws_for(size, cfg)):
                for model in predictors:
                    common = dict(training_size=size, draw=draw)
                    lift = -0.12 * (1 - shrink) * (1 - gen.baseline(dataset, model))
                    arms = [("non_text", {}, 0.0)]
                    arms += [("non_text_plus_both", {"provider": p}, q * (1.4 - 0.6 * shrink))
                             for p, q in gen.quality.items()]
                    arms += [("non_text_plus_frozen_e5", {"encoder": e, "pca_scope": "per_column",
                                                          "pca_components": 32}, 0.3 + 0.5 * shrink)
                             for e in ENCODERS]
                    arms += [("non_text_plus_supervised_probe",
                              {"encoder": "e5_small_v2", "probe_view": "per_column"}, 0.2 + 1.0 * shrink)]
                    for feature_set, fields, factor in arms:
                        gain = lift + (gen.gain(dataset, model, factor) if factor else 0.0)
                        _arm(rows, gen, dataset, model, "data_efficiency", feature_set,
                             gain, 0, **common, **fields)
    frame = _categorise([r for r in rows if r["metric"] != "columns.representation"])
    frame["split_id"] = (frame.split_id.astype(str) + "-" + frame.training_size.astype(str) + "-"
                         + frame.draw.astype(str)).astype("category")
    return frame


# ------------------------------------------------------------------------------- B2, B3 extras


def complementarity_pairs(gains: pd.DataFrame, seed=SEED) -> pd.DataFrame:
    """Rows in `complementarity.PAIR_COLUMNS` for every decision x (frozen | TAR) pair."""
    rng = np.random.default_rng(seed)
    decision = gains[gains.family.eq("decision")]
    others = gains[gains.family.isin(["frozen", "tar"])][["dataset_id", "model_id", "representation",
                                                           "family"]]
    pairs = decision[["dataset_id", "task_type", "model_id", "representation"]].merge(
        others, on=["dataset_id", "model_id"], suffixes=("", "_other"))
    parts = pairs.representation.str.split("/")
    n = len(pairs)
    rows = 1000
    k = int(np.ceil(0.1 * rows))
    chance = k * k / rows / (2 * k - k * k / rows)
    return pd.DataFrame({
        "dataset_id": pairs.dataset_id, "task_type": pairs.task_type, "model_id": pairs.model_id,
        "checkpoint_id": pairs.model_id + "-checkpoint", "decision_representation": pairs.representation,
        "decision_mode": parts.str[1], "provider": parts.str[2],
        "other_representation": pairs.representation_other,
        "other_family": pairs.family.map({"frozen": "embedding", "tar": "tar"}), "n_rows": rows,
        "spearman_rho": np.clip(rng.normal(0.35, 0.15, n), -1, 1), "top_k": k,
        "jaccard_top": np.clip(chance + rng.normal(0.12, 0.05, n), 0, 1), "jaccard_chance": chance,
    }, columns=complementarity.PAIR_COLUMNS)


def b3_properties(seed=SEED) -> pd.DataFrame:
    """Label-free dataset properties in the exported columns (dataset_covariates + redundancy)."""
    rng = np.random.default_rng(seed + 3)
    info = datasets()
    median_chars = np.exp(rng.uniform(np.log(8), np.log(900), len(info)))
    return pd.DataFrame({
        "dataset_id": info.dataset_id, "task_type": info.task_type, "n_rows": info.rows,
        "n_text_fields": info.text_columns, "n_non_text_fields": info.non_text_columns,
        "n_classes": np.where(info.task_type.eq("regression"), np.nan, info.classes.clip(lower=2)),
        "median_text_chars": median_chars, "missing_text_share": rng.uniform(0, 0.4, len(info)),
        "repeated_text_share": rng.uniform(0, 0.9, len(info)),
        "unseen_text_share": rng.uniform(0.1, 1, len(info)),
        "free_text_share": rng.uniform(0, 1, len(info)),
        "log10_n_rows": np.log10(info.rows), "log10_median_text_chars": np.log10(median_chars),
        "decision_redundancy_r2": rng.uniform(0.05, 0.7, len(info)),
        "pilot_redundancy_r2": rng.uniform(0.0, 0.75, len(info)),
    })


def b3_model_diagnostics(seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 4)
    models = roster()
    return pd.DataFrame({
        "provider": models, "standalone_quality": rng.uniform(0.55, 0.85, len(models)),
        "calibration_ece": rng.uniform(0.02, 0.25, len(models)),
        "mean_top_probability": rng.uniform(0.5, 0.95, len(models)),
        "consensus_distance": rng.uniform(0.05, 0.4, len(models)),
        "stability_excess_tv": rng.uniform(0.0, 0.12, len(models)),
        "log10_parameters_billion": np.log10(rng.choice([2, 3, 4, 9, 27, 31], len(models)).astype(float)),
    })


def b3_row_cohorts(seed=SEED) -> pd.DataFrame:
    """The `row_benefit.summaries()` equal-dataset table (``balanced``)."""
    rng = np.random.default_rng(seed + 5)
    axes = {"availability": ["present", "absent"], "length": [f"bin_{i}" for i in range(4)] + ["missing"],
            "baseline_entropy": ["0", "1", "2", "3"],
            "decision_confidence": ["lt_0.5", "0.5_to_0.7", "0.7_to_0.9", "ge_0.9"],
            "decision_agreement": ["lt_0.5", "0.5_to_0.75", "0.75_to_1", "unanimous"]}
    rows = []
    for task in ("binary_classification", "multiclass_classification", "regression"):
        for model in MAIN_MODELS:
            for mode in ("per_column", "joint", "both"):
                for axis, cohorts in {"all": ["all"], **axes}.items():
                    for i, cohort in enumerate(cohorts):
                        rows.append({"task": task, "phase": "experiment_1", "model": model,
                                     "checkpoint": f"{model}-checkpoint", "features": FEATURE_SET[mode],
                                     "representation": f'{{"mode": "{mode}"}}', "axis": axis,
                                     "cohort": cohort, "interpretation": "pre-specified",
                                     "mean_benefit": float(rng.normal(0.01 * (i + 1) - 0.01, 0.01)),
                                     "datasets": 3 if task != "regression" else 9})
    return pd.DataFrame(rows)


def b3_tables(e1_folds: pd.DataFrame, *, n_permutations=200, n_shuffles=50) -> dict:
    """B3 tables through the export's own `group_b` functions (raw data replaced by fixtures)."""
    from src.utils.config import load_yaml

    head = group_b.headline(e1_folds)
    gains = group_b.dataset_gains(group_b.fold_gains(head))
    gains = gains[gains.family.eq("decision")]
    medians = group_b.median_over_decision_models(gains, roster=roster())
    properties = b3_properties()
    rule = load_yaml("rq2/when_text_helps")["predict_in_advance"]
    outcome = medians[medians.representation.eq(rule["outcome_representation"])]
    lodo_rows, lodo = [], []
    for model, group in outcome.groupby("model_id"):
        predictions, summary = group_b.lodo_input_sets(
            group, properties, input_sets=rule["input_sets"], alpha=rule["alpha"],
            threshold=rule["use_threshold"], n_shuffles=n_shuffles)
        lodo_rows.append(predictions.assign(model_id=model))
        lodo.append(summary.assign(model_id=model))
    diagnostics = b3_model_diagnostics()
    usefulness = group_b.per_model_usefulness(gains)
    relations = group_b.associations(usefulness.rename(columns={"provider": "unit"}),
                                     diagnostics.rename(columns={"provider": "unit"}),
                                     by=("model_id", "mode"), unit="unit",
                                     n_permutations=n_permutations, min_units=5)
    redundancy = pd.DataFrame([{"dataset_id": d, "provider": p, "status": "computed", "rows": 1000,
                                "r2": r, "top_agreement": 0.8}
                               for d, r in zip(properties.dataset_id, properties.decision_redundancy_r2)
                               for p in roster()])
    return {
        "coverage": benchmark_export.coverage(e1_folds, benchmark_export.STUDIES["B3"]["phases"]),
        "folds": e1_folds, "dataset_gains": gains, "summary": group_b.summarize(gains),
        "properties": properties, "redundancy": redundancy,
        "associations": group_b.associations(medians, properties, by=("model_id", "representation"),
                                             n_permutations=n_permutations),
        "model_diagnostics": diagnostics, "model_usefulness": usefulness, "model_relations": relations,
        "column_importance": group_b.text_column_importance(head), "row_cohorts": b3_row_cohorts(),
        "lodo_predictions": pd.concat(lodo_rows, ignore_index=True),
        "lodo": pd.concat(lodo, ignore_index=True),
    }


def tables(study: str) -> dict[str, pd.DataFrame]:
    """Every table of one study's export, in memory."""
    if study == "B1":
        return benchmark_export.build_tables("B1", folds(("experiment_1",)))
    if study == "B2":
        built = benchmark_export.build_tables("B2", folds(("experiment_1", "experiment_2", "experiment_3")))
        pairs = complementarity_pairs(built["dataset_gains"])
        built.update(complementarity_pairs=pairs,
                     complementarity_summary=complementarity.complementarity_summary(pairs, n_boot=200))
        return built
    if study == "B3":
        return b3_tables(folds(("experiment_1", "rq2")))
    if study == "B4":
        return benchmark_export.build_tables("B4", size_folds())
    if study == "B5":
        return benchmark_export.build_tables("B5", folds(("experiment_1", "experiment_4")))
    raise KeyError(study)


def export(study: str, folder) -> tuple[dict, dict]:
    """Write one study's tables as a real export ZIP in `folder` and read it back verified."""
    from pathlib import Path

    from src.evaluation import study_export

    built = {name: benchmark_export._parquet_safe(frame) for name, frame in tables(study).items()}
    target = Path(folder) / benchmark_export.STUDIES[study]["file"]
    study_export.write_export(target, {n: built[n] for n in benchmark_export.tables_for(study)},
                              schema=benchmark_export.SCHEMA,
                              identity=benchmark_export.analysis_identity(study),
                              origin="fixture:synthetic", extra={"study": study, "headlines": HEADLINES})
    return benchmark_export.read(study, target=target)
