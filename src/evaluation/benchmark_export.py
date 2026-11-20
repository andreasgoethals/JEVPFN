"""Group B: one compact, checksummed ZIP per experiment (B1-B5), built on a Mindwell CPU job.

Follows the Group A pattern (`study_export`): fold records, predictions and row-level data stay
on VSC PROJECT; `python -m src.evaluation.benchmark_export --study B1` (Slurm:
`scripts/slurm/rq2_group_b_export.slurm B1`) writes one ZIP of Parquet tables plus
`manifest.json` (member SHA-256 values and a receipt; refused above 150 MB). The owner copies
only that ZIP to the same relative path under the local `output_JEVPFN/`; each notebook reads
nothing else and shows an explicit waiting state when it is absent.

| Study | Folder (notebook) | Phases read |
|---|---|---|
| B1 decision features | exp04_decision_features | experiment_1 |
| B2 representations | exp05_frozen_embeddings (also read by exp06_tar) | experiment_1-3 |
| B3 when and for whom | exp08_when_text_helps | experiment_1, rq2 (+ Group A exports, row cohorts) |
| B4 amount of labelled data | exp11_data_efficiency | data_efficiency |
| B5 native systems | exp07_native_text | experiment_1, experiment_4 |
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import group_b, study_export
from src.evaluation.endpoints import HEADLINES
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import digest, file_sha256

SCHEMA = "jevpfn.group_b.export.v1"
STUDIES = {
    "B1": {"phase": "experiment_1", "file": "b1_decision_features.zip",
           "phases": ("experiment_1",)},
    "B2": {"phase": "experiment_2", "file": "b2_representations.zip",
           "phases": ("experiment_1", "experiment_2", "experiment_3")},
    "B3": {"phase": "rq2", "file": "b3_when_and_for_whom.zip", "phases": ("experiment_1", "rq2")},
    "B4": {"phase": "data_efficiency", "file": "b4_data_efficiency.zip",
           "phases": ("data_efficiency",)},
    "B5": {"phase": "experiment_4", "file": "b5_native_text.zip",
           "phases": ("experiment_1", "experiment_4")},
}
COMMON_TABLES = ("coverage", "folds", "dataset_gains", "summary")
EXTRA_TABLES = {
    "B1": ("median_gains", "median_summary"),
    "B2": ("median_gains", "contrasts", "complementarity_pairs", "complementarity_summary"),
    "B3": ("properties", "redundancy", "associations", "model_diagnostics", "model_usefulness",
           "model_relations", "column_importance", "row_cohorts", "lodo_predictions", "lodo"),
    "B4": ("size_gains", "pooled_raw", "pooled_share", "slopes"),
    "B5": (),
}
#: B4's grid is large: only the headline and one proper score travel; all metrics stay on PROJECT.
B4_METRICS = ("roc_auc", "roc_auc_ovr_macro", "r2", "log_loss", "rmse")
RECIPE_COLUMNS = ["phase", "dataset_id", "task_type", "model_id", "checkpoint_id", "feature_set",
                  "provider", "view", "encoder", "pca_scope", "pca_components", "probe_view",
                  "dropped_text_column", "training_size", "draw", "hidden_view", "hidden_position",
                  "hidden_layer", "adaptation_cell", "split_id", "outer_fold"]


def tables_for(study):
    return COMMON_TABLES + EXTRA_TABLES[study]


def export_path(study) -> Path:
    spec = STUDIES[study]
    return paths.results_dir("group_b", spec["file"], phase=spec["phase"])


def analysis_identity(study) -> str:
    from src.utils.provenance import code_identity

    configs = {name: load_yaml(name) for name in (
        "experiment_1/main", "experiment_1/protocol", "experiment_2/embeddings", "experiment_3/tar",
        "experiment_4/native_text", "rq2/when_text_helps", "rq2/text_column_refits",
        "data_efficiency/data_efficiency")}
    return digest({"schema": SCHEMA, "study": study, "configs": configs,
                   "analysis": code_identity(["src/evaluation/group_b.py",
                                              "src/evaluation/data_efficiency.py"])})


def recipe_fields(recipe: dict) -> dict:
    rep = recipe.get("representation_recipe") or {}
    embedding = rep.get("embedding") or {}
    probe = rep.get("probe") or {}
    subsample = rep.get("subsample") or {}
    hidden = rep.get("hidden") or {}  # Group C1
    adaptation = rep.get("adaptation") or {}  # Group C2
    return {
        "dataset_id": recipe["dataset_id"], "task_type": recipe["task_type"],
        "model_id": recipe["model_id"], "checkpoint_id": recipe["checkpoint_id"],
        "feature_set": recipe["feature_set"], "provider": rep.get("provider"),
        "view": rep.get("view", "main"), "encoder": embedding.get("encoder_id"),
        "pca_scope": embedding.get("pca_scope"),
        "pca_components": embedding.get("pca_components_per_column"),
        "probe_view": probe.get("view"), "dropped_text_column": rep.get("dropped_text_column"),
        "training_size": subsample.get("training_size"), "draw": subsample.get("draw"),
        "hidden_view": hidden.get("view"), "hidden_position": hidden.get("position"),
        "hidden_layer": hidden.get("layer"), "adaptation_cell": adaptation.get("cell"),
        "split_id": recipe["split_id"], "outer_fold": recipe["outer_fold"],
    }


def collect_folds(phases, *, metrics=None) -> pd.DataFrame:
    """Long table of complete outer-test fold metrics (scalar metrics; cohort scores excluded)."""
    from src.evaluation.records import completed_fold
    from src.evaluation.store import all_recipes

    rows = []
    for phase in phases:
        for recipe in all_recipes(phase):
            records = completed_fold(recipe, phase)
            if records is None:
                continue
            for path in records:
                record = json.loads(path.read_text(encoding="utf-8"))
                if record["metadata"]["prediction_role"] != "outer_test":
                    continue
                base = {"phase": phase, **recipe_fields(recipe)}
                values = dict(record["scores"]["metrics"])
                values["columns.representation"] = record["metadata"].get("representation_feature_count")
                values["seconds.fit"] = record["timing"]["stages"].get("fit", {}).get("wall_seconds")
                for name, value in values.items():
                    if metrics is not None and name not in metrics:
                        continue
                    if value is None or isinstance(value, (int, float)):
                        rows.append({**base, "metric": name, "value": value})
    frame = pd.DataFrame(rows, columns=[*RECIPE_COLUMNS, "metric", "value"])
    if frame.empty:
        return frame
    keys = [c for c in RECIPE_COLUMNS if c != "outer_fold"]
    complete = frame.groupby(keys, dropna=False).outer_fold.transform("nunique").eq(5)
    for column in ("phase", "dataset_id", "task_type", "model_id", "feature_set", "provider",
                   "encoder", "pca_scope", "probe_view", "metric", "view", "checkpoint_id", "split_id",
                   "hidden_view", "hidden_position", "adaptation_cell"):
        frame[column] = frame[column].astype("category")
    return frame[complete].reset_index(drop=True)


def coverage(folds, phases) -> pd.DataFrame:
    """Planned versus complete (five-fold) arms per phase and predictor."""
    from src.evaluation.campaign import benchmark_plan

    planned = pd.DataFrame([a for a in benchmark_plan() if a["phase"] in phases])
    planned = planned.groupby(["phase", "model"]).size().rename("planned_arms").reset_index()
    planned = planned.rename(columns={"model": "model_id"})
    if folds.empty:
        return planned.assign(complete_arms=0)
    keys = [c for c in RECIPE_COLUMNS if c != "outer_fold"]
    done = (folds[keys].astype(object).drop_duplicates().groupby(["phase", "model_id"]).size()
            .rename("complete_arms").reset_index())
    return planned.merge(done, on=["phase", "model_id"], how="left").fillna({"complete_arms": 0})


def _gains(head, extra_keys=()):
    paired = group_b.fold_gains(head)
    gains = group_b.dataset_gains(paired, extra_keys=extra_keys)
    return paired, gains


def contrasts(gains: pd.DataFrame) -> pd.DataFrame:
    """B2 descriptive contrasts per dataset and predictor: the median decision model's "both"
    gain minus each frozen, decision+frozen, TAR and probe arm (positive = decision ahead)."""
    medians = group_b.median_over_decision_models(gains)
    reference = medians[medians.representation.eq("decision/both/median")][
        ["dataset_id", "model_id", "raw_gain"]].rename(columns={"raw_gain": "decision_both_median"})
    others = gains[gains.family.isin(["frozen", "tar", "probe"])]
    combined = medians[medians.representation.eq("decision+frozen/both/median")]
    others = pd.concat([others, combined], ignore_index=True)
    out = others.merge(reference, on=["dataset_id", "model_id"], how="inner")
    out["difference"] = out.decision_both_median - out.raw_gain
    return out[["dataset_id", "task_type", "model_id", "representation", "family", "raw_gain",
                "decision_both_median", "difference"]]


def _group_a_frames():
    """Group A exports on PROJECT (A1 comparison, A2 stability) when they exist."""
    from src.evaluation import comparison_report, robustness_export

    frames = {}
    target = comparison_report.report_path()
    if target.exists():
        found, _ = study_export.read_export(target, schema=comparison_report.SCHEMA,
                                            tables=comparison_report.TABLES)
        frames.update(found)
    try:
        found, _ = robustness_export.read_export()
    except ValueError:  # A2 export of another configuration: model relations stay partial.
        found = {}
    frames.update({f"a2_{k}": v for k, v in found.items()})
    return frames


def dataset_properties_and_redundancy():
    """Label-free properties of every selected dataset and decision redundancy per model."""
    from src.data.loaders import load_catalog, load_dataset
    from src.data.selection import selected_entries
    from src.evaluation.moderators import dataset_covariates
    from src.evaluation.protocol import shared_splits
    from src.evaluation.row_benefit import block_probabilities, load_decision_block

    main = load_yaml("experiment_1/main")
    properties, redundancy = [], []
    for entry in selected_entries(load_catalog(), main):
        dataset = load_dataset(entry, allow_download=False)
        row = dataset_covariates(dataset)
        row["log10_n_rows"] = float(np.log10(row["n_rows"]))
        row["log10_median_text_chars"] = float(np.log10(max(row["median_text_chars"], 1.0)))
        properties.append(row)
        splits = shared_splits(dataset, seed=main["random_seed"], folds=5)["folds"]
        non_text = dataset.features.loc[:, list(dataset.metadata.non_text_columns)]
        for provider in main["candidate_extractors"]:
            block, source = load_decision_block(entry["id"], provider, "joint", main)
            if block is None:
                redundancy.append({"dataset_id": entry["id"], "provider": provider, "status": source})
                continue
            matrix = block_probabilities(block, dataset.metadata.task_type, len(dataset.frame))["all_text"]
            result = group_b.decision_redundancy(non_text.reset_index(drop=True), matrix, splits,
                                                 task_type=dataset.metadata.task_type)
            redundancy.append({"dataset_id": entry["id"], "provider": provider, "status": "computed",
                               **result})
        print(f"{entry['id']}: properties and redundancy computed", flush=True)
    properties = pd.DataFrame(properties)
    redundancy = pd.DataFrame(redundancy)
    if "r2" in redundancy:
        per_dataset = redundancy.groupby("dataset_id").r2.median().rename("decision_redundancy_r2")
        properties = properties.merge(per_dataset.reset_index(), on="dataset_id", how="left")
    return properties, redundancy


def build_tables(study, folds) -> dict[str, pd.DataFrame]:
    cfg = load_yaml("rq2/when_text_helps")
    head = group_b.headline(folds)
    tables = {"coverage": coverage(folds, STUDIES[study]["phases"]), "folds": folds}
    if study == "B4":
        paired, gains = _gains(head, extra_keys=("training_size", "draw"))
        from src.evaluation import data_efficiency

        sizes = data_efficiency.size_gains(head.dropna(subset=["training_size"]))
        tables.update(dataset_gains=gains, summary=group_b.summarize(
            sizes, keys=("model_id", "representation", "training_size")) if not sizes.empty else sizes,
            size_gains=sizes, pooled_raw=data_efficiency.pooled_curve(sizes),
            pooled_share=data_efficiency.pooled_curve(sizes, outcome="headroom_share"),
            slopes=data_efficiency.gain_slopes(sizes))
        return tables
    paired, gains = _gains(head)
    if study == "B3":
        gains = gains[gains.family.eq("decision")]
    tables.update(dataset_gains=gains, summary=group_b.summarize(gains))
    medians = group_b.median_over_decision_models(gains) if not gains.empty else pd.DataFrame()
    if study == "B1":
        tables.update(median_gains=medians, median_summary=group_b.summarize(medians))
    elif study == "B2":
        from src.evaluation import complementarity

        pairs, _ = complementarity.summaries()
        tables.update(median_gains=medians, contrasts=contrasts(gains) if not gains.empty else gains,
                      complementarity_pairs=pairs,
                      complementarity_summary=complementarity.complementarity_summary(pairs)
                      if not pairs.empty else pd.DataFrame())
    elif study == "B3":
        from src.evaluation.row_benefit import summaries

        properties, redundancy = dataset_properties_and_redundancy()
        rule = cfg["predict_in_advance"]
        outcome = medians[medians.representation.eq(rule["outcome_representation"])]
        lodo_rows, lodo = [], []
        for model, group in outcome.groupby("model_id"):
            predictions = group_b.lodo_predictions(group, properties, features=rule["properties"],
                                                   alpha=rule["alpha"])
            lodo_rows.append(predictions.assign(model_id=model))
            lodo.append({"model_id": model, **group_b.lodo_evaluation(
                predictions, threshold=rule["use_threshold"], n_shuffles=rule["n_shuffles"],
                outcome=group, properties=properties, features=rule["properties"],
                alpha=rule["alpha"])})
        a = _group_a_frames()
        diagnostics = group_b.model_diagnostics(
            a.get("quality"), a.get("output_shape"), a.get("loo_consensus"), a.get("model_profile"),
            a.get("a2_mode_effects"))
        usefulness = group_b.per_model_usefulness(gains)
        relations = (group_b.associations(usefulness.rename(columns={"provider": "unit"}),
                                          diagnostics.rename(columns={"provider": "unit"}),
                                          by=("model_id", "mode"), unit="unit",
                                          n_permutations=cfg["n_permutations"], min_units=5)
                     if not usefulness.empty and len(diagnostics.columns) > 1 else pd.DataFrame())
        _, cohorts = summaries()
        tables.update(
            properties=properties, redundancy=redundancy,
            associations=group_b.associations(medians, properties, by=("model_id", "representation"),
                                              n_permutations=cfg["n_permutations"])
            if not medians.empty else pd.DataFrame(),
            model_diagnostics=diagnostics, model_usefulness=usefulness, model_relations=relations,
            column_importance=group_b.text_column_importance(head), row_cohorts=cohorts,
            lodo_predictions=pd.concat(lodo_rows, ignore_index=True) if lodo_rows else pd.DataFrame(),
            lodo=pd.DataFrame(lodo))
    return tables


def _parquet_safe(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(
                lambda v: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v)
    return frame


def read(study, target=None):
    """Verified tables and receipt, or an explicit waiting state (absent/superseded ZIP)."""
    return study_export.read_export(target or export_path(study), schema=SCHEMA,
                                    tables=tables_for(study))


def build(study):
    study_export.require_mindwell_cpu(f"Group B {study}")
    metrics = B4_METRICS if study == "B4" else None
    folds = collect_folds(STUDIES[study]["phases"], metrics=metrics)
    if folds.empty:
        raise ValueError(f"No complete five-fold arm exists for {study}")
    tables = {name: _parquet_safe(frame) for name, frame in build_tables(study, folds).items()}
    missing = set(tables_for(study)) - set(tables)
    if missing:
        raise ValueError(f"{study} export lacks tables: {sorted(missing)}")
    target = study_export.write_export(
        export_path(study), {name: tables[name] for name in tables_for(study)}, schema=SCHEMA,
        identity=analysis_identity(study), origin=study_export.job_origin(),
        extra={"study": study, "headlines": HEADLINES})
    print(f"{target}  SHA256 {file_sha256(target)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, choices=tuple(STUDIES))
    build(parser.parse_args().study)


if __name__ == "__main__":
    main()


def median_summary(frames) -> pd.DataFrame:
    """Per-arm summary plus the median-over-decision-models rows (notebooks)."""
    medians = frames.get("median_gains")
    extra = group_b.summarize(medians) if medians is not None and not medians.empty else pd.DataFrame()
    return pd.concat([frames["summary"], extra], ignore_index=True)
