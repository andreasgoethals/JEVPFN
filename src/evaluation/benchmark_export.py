"""Group B: one compact, checksummed ZIP per experiment (B1-B6), built on a Mindwell CPU job.

Follows the Group A pattern (`study_export`): fold records, predictions and row-level data stay
on VSC PROJECT; `python -m src.evaluation.benchmark_export --study B1` (Slurm:
`scripts/slurm/B_export.slurm B1`) writes one ZIP of Parquet tables plus
`manifest.json` (member SHA-256 values and a receipt; refused above 150 MB). The owner copies
only that ZIP to the same relative path under the local `output_JEVPFN/`; each notebook reads
nothing else and shows an explicit waiting state when it is absent.

| Study | Folder and notebook | Phases read |
|---|---|---|
| B1 decision features | B/B1 | B1 |
| B2 frozen embeddings and decision + embedding | B/B2 | B1, B2 |
| B3 TAR and the label-trained classifier | B/B3 | B1, B3 |
| B4 when decision features help | B/B4 | B1, B4 (+ Group A exports, row cohorts) |
| B5 amount of labelled data | B/B5 | B5 |
| B6 native-text systems | B/B6 | B1, B6 |
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import group_b, study_export
from src.evaluation.design import cohort_name, cohort_record
from src.evaluation.endpoints import HEADLINES
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import digest, file_sha256

SCHEMA = "jevpfn.group_b.export.v1"
#: One export per experiment; each also carries the B1 arms it is compared with.
STUDIES = {
    "B1": {"phase": "B1", "file": "b1_decision_features.zip", "phases": ("B1",)},
    "B2": {"phase": "B2", "file": "b2_frozen_embeddings.zip", "phases": ("B1", "B2")},
    "B3": {"phase": "B3", "file": "b3_label_trained.zip", "phases": ("B1", "B3")},
    "B4": {"phase": "B4", "file": "b4_when_text_helps.zip", "phases": ("B1", "B4")},
    "B5": {"phase": "B5", "file": "b5_data_efficiency.zip", "phases": ("B5",)},
    "B6": {"phase": "B6", "file": "b6_native_text.zip", "phases": ("B1", "B6")},
}
#: B2 and B3 compare decision features with an alternative representation family.
REPRESENTATION_STUDIES = ("B2", "B3")
COMMON_TABLES = ("coverage", "folds", "dataset_gains", "summary")
EXTRA_TABLES = {
    "B1": ("median_gains", "median_summary", "median_roster_gaps"),
    "B2": ("median_gains", "median_roster_gaps", "contrasts", "complementarity_pairs",
           "complementarity_summary"),
    "B3": ("median_gains", "median_roster_gaps", "contrasts", "complementarity_pairs",
           "complementarity_summary"),
    "B4": ("properties", "redundancy", "associations", "model_diagnostics", "model_usefulness",
           "model_relations", "column_importance", "row_cohorts", "lodo_predictions", "lodo"),
    "B5": ("size_gains", "pooled_raw", "pooled_share", "slopes"),
    "B6": (),
}
#: B5's grid is large: only the headline and one proper score travel; all metrics stay on PROJECT.
B5_METRICS = ("roc_auc", "roc_auc_ovr_macro", "r2", "log_loss", "rmse")
RECIPE_COLUMNS = ["phase", "dataset_id", "task_type", "model_id", "checkpoint_id", "feature_set",
                  "provider", "view", "encoder", "pca_scope", "pca_components", "probe_view",
                  "dropped_text_column", "training_size", "draw", "hidden_view", "hidden_position",
                  "hidden_layer", "adaptation_cell", "split_id", "outer_fold"]


def tables_for(study):
    return COMMON_TABLES + EXTRA_TABLES[study]


def cohort_key(cohort=None, *, main=None):
    record = cohort_record(cohort, main=main)
    return f"{record['decision_cohort']}-{record['roster_sha256'][:16]}"


def export_path(study, cohort=None) -> Path:
    spec = STUDIES[study]
    file = Path(spec["file"])
    return paths.results_dir("group_b", f"{file.stem}.{cohort_key(cohort)}{file.suffix}", phase=spec["phase"])


def scientific_configs(names, *, roster=None):
    from src.evaluation.acceptance import EXECUTION_FIELDS

    configs = {}
    for name in names:
        cfg = {k: v for k, v in load_yaml(name).items() if k not in EXECUTION_FIELDS}
        if roster is not None:
            cfg.pop("initial_decision_providers", None)
            cfg.pop("decision_extensions", None)
            if "candidate_extractors" in cfg:
                cfg["candidate_extractors"] = [p for p in cfg["candidate_extractors"] if p in roster]
            if "feature_selectors" in cfg:
                cfg["feature_selectors"] = {p: v for p, v in cfg["feature_selectors"].items() if p in roster}
            if "dataset_feature_artifacts" in cfg:
                pins = {d: {p: v for p, v in providers.items() if p in roster}
                        for d, providers in cfg["dataset_feature_artifacts"].items()}
                cfg["dataset_feature_artifacts"] = {d: v for d, v in pins.items() if v}
        configs[name] = cfg
    return configs


def analysis_identity(study, cohort=None) -> str:
    from src.utils.provenance import code_identity

    configs = scientific_configs(("B1", "B1/protocol", "B2",
                                  "B3", "B6", "B4",
                                  "B4/text_column_refits", "B5"),
                                 roster=decision_roster(cohort))
    modules = ["group_b", "benchmark_export", "campaign", "design", "data_efficiency", "endpoints",
               "records", "study_export"]
    if study in REPRESENTATION_STUDIES:
        modules.extend(["attribution", "complementarity", "row_benefit", "moderators"])
    if study == "B4":
        modules.extend(["group_a", "group_b_views", "moderators", "moderator_statistics", "row_benefit", "comparison_report",
                        "robustness_export", "study_reports"])
    return digest({"schema": SCHEMA, "study": study, "configs": configs, "cohort": cohort_record(cohort),
                   "analysis": code_identity([f"src/evaluation/{name}.py" for name in modules])})


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


def arm_recipe_id(recipe: dict) -> str:
    """Identity of one arm's full recipe (feature artifacts, protocol, settings) across folds."""
    return digest({k: v for k, v in recipe.items() if k != "outer_fold"})


def cohort_recipes(recipes, scope, main):
    """Select shared controls and decisions produced by this release's registered versions."""
    from src.evaluation.features import ADAPTED, HIDDEN_FEATURE_SETS, decision_feature_set

    selectors = {}
    producers = {}
    for version in scope["decision_model_versions"]:
        provider = version["provider"]
        registered = {k: version[k] for k in ("provider", "model", "version")}
        producers[provider] = version
        explicit = main.get("feature_selectors", {}).get(provider) or {}
        if any(k in explicit and explicit[k] != v for k, v in registered.items()):
            raise ValueError(f"The {provider} feature selector differs from the registered model or version")
        selectors[provider] = {**registered, **explicit}
    pins = main.get("dataset_feature_artifacts", {})
    for recipe in recipes:
        rep = recipe.get("representation_recipe") or {}
        if recipe["feature_set"] in (*HIDDEN_FEATURE_SETS, ADAPTED):
            component = "adaptation" if recipe["feature_set"] == ADAPTED else "hidden"
            recorded = (rep.get(component) or {}).get("decision_model")
            if (not isinstance(recorded, dict) or not isinstance(recorded.get("provider"), str)
                    or producers.get(recorded["provider"]) != recorded):
                continue
        provider = rep.get("provider")
        mode = group_b.MODE_LABELS.get(decision_feature_set(recipe["feature_set"]))
        if mode is None:
            if not provider and not rep.get("providers") and not rep.get("selector"):
                yield recipe
            continue
        saved = rep.get("selector")
        expected = dict(selectors.get(provider, {}))
        overrides = expected.pop("settings_sha256_by_dataset", {})
        if recipe["dataset_id"] in overrides:
            expected["settings_sha256"] = overrides[recipe["dataset_id"]]
        if (provider not in selectors or not isinstance(saved, dict)
                or rep.get("providers", [provider]) != [provider]
                or any(k not in saved or saved[k] != v for k, v in expected.items())):
            continue
        pinned = pins.get(recipe["dataset_id"], {}).get(provider)
        if pinned is not None:
            # campaign assembles per-column then joint decision blocks before adding embeddings.
            modes = ("per_column", "joint") if mode == "both" else (mode,)
            expected = [pinned.get(m) for m in modes]
            artifacts = recipe.get("feature_artifact_ids") or []
            if not all(expected) or artifacts[:len(modes)] != expected:
                continue
        yield recipe


def collect_folds(phases, *, metrics=None, cohort=None, providers=None) -> pd.DataFrame:
    """Long table of complete outer-test fold metrics (scalar metrics; cohort scores excluded).

    An arm is complete only when ONE full recipe (``recipe_id``: every recipe field except the
    fold, including feature-artifact identities) has all five folds; folds of different recipes
    never combine, and two complete recipes of one arm are refused as ambiguous. A named
    cohort checks producer identities before reading scores. Historical unscoped callers
    can still inspect records written before producer selectors were captured.
    """
    from src.evaluation.records import completed_fold
    from src.evaluation.store import all_recipes

    scope = None
    if cohort is not None or providers is not None:
        main = load_yaml("B1")
    if providers is not None:
        from src.evaluation.design import decision_versions

        if cohort is not None:
            raise ValueError("Select a named cohort or an explicit provider list")
        scope = {"decision_model_versions": decision_versions(providers)}
    elif cohort is not None:
        scope = cohort_record(cohort, main=main)
    rows = []
    for phase in phases:
        recipes = all_recipes(phase)
        if scope is not None:
            recipes = cohort_recipes(recipes, scope, main)
        for recipe in recipes:
            records = completed_fold(recipe, phase)
            if records is None:
                continue
            for path in records:
                record = json.loads(path.read_text(encoding="utf-8"))
                if record["metadata"]["prediction_role"] != "outer_test":
                    continue
                base = {"phase": phase, **recipe_fields(recipe), "recipe_id": arm_recipe_id(recipe),
                        "data_identity": recipe.get("data_identity"),
                        "n_train": record["metadata"].get("n_train"),
                        "pairing_identity": digest({k: recipe.get(k) for k in
                            ("data_identity", "task_schema_sha256", "protocol_sha256", "estimator_settings")})}
                values = dict(record["scores"]["metrics"])
                values["columns.representation"] = record["metadata"].get("representation_feature_count")
                values["seconds.fit"] = record["timing"]["stages"].get("fit", {}).get("wall_seconds")
                for name, value in values.items():
                    if metrics is not None and name not in metrics:
                        continue
                    if value is None or isinstance(value, (int, float)):
                        rows.append({**base, "metric": name, "value": value})
    frame = pd.DataFrame(rows, columns=[*RECIPE_COLUMNS, "recipe_id", "data_identity", "pairing_identity",
                                        "n_train", "metric", "value"])
    if frame.empty:
        return frame
    by_recipe = frame.groupby(["phase", "recipe_id", "metric"], dropna=False).outer_fold
    frame = frame[by_recipe.transform("nunique").eq(5) & by_recipe.transform("size").eq(5)].copy()
    keys = [c for c in RECIPE_COLUMNS if c != "outer_fold"]
    recipes = frame.groupby([*keys, "metric"], dropna=False).recipe_id.nunique()
    if recipes.gt(1).any():
        raise ValueError("One arm has several complete recipes; select one before exporting: "
                         f"{list(recipes[recipes.gt(1)].index[:3])}")
    for column in ("phase", "dataset_id", "task_type", "model_id", "feature_set", "provider",
                   "encoder", "pca_scope", "probe_view", "metric", "view", "checkpoint_id", "split_id",
                   "hidden_view", "hidden_position", "adaptation_cell", "recipe_id", "data_identity", "pairing_identity"):
        frame[column] = frame[column].astype("category")
    return frame.reset_index(drop=True)


def comparison_plan(cohort=None):
    """Arms needed to compare the selected cohort, including already fitted shared controls."""
    from src.evaluation.campaign import GROUP_B_PHASES, benchmark_plan
    from src.evaluation.design import incremental_cohort

    selected = cohort_name(cohort)
    if not incremental_cohort(selected):
        return benchmark_plan(cohort=selected)
    providers = set(decision_roster(selected))
    return [arm for arm in benchmark_plan(cohort="all")
            if arm["phase"] in GROUP_B_PHASES and (arm["provider"] is None or arm["provider"] in providers)]


def coverage(folds, phases, *, cohort=None) -> pd.DataFrame:
    """Planned versus complete (five-fold) comparison arms per phase and predictor."""

    planned = pd.DataFrame([a for a in comparison_plan(cohort) if a["phase"] in phases])
    providers = planned.provider.dropna().unique()
    planned = planned.groupby(["phase", "model"]).size().rename("planned_arms").reset_index()
    planned = planned.rename(columns={"model": "model_id"})
    if folds.empty:
        return planned.assign(complete_arms=0)
    folds = folds[folds.provider.isna() | folds.provider.isin(providers)]
    keys = [c for c in RECIPE_COLUMNS if c != "outer_fold"]
    done = (folds[keys].astype(object).drop_duplicates().groupby(["phase", "model_id"]).size()
            .rename("complete_arms").reset_index())
    return planned.merge(done, on=["phase", "model_id"], how="left").fillna({"complete_arms": 0})


def _gains(head, extra_keys=()):
    group_b.validate_fold_identities(head, extra=extra_keys)
    paired = group_b.fold_gains(head, extra_keys=extra_keys)
    gains = group_b.dataset_gains(paired, extra_keys=extra_keys)
    return paired, gains


CONTRAST_COLUMNS = ["dataset_id", "task_type", "model_id", "representation", "family", "raw_gain",
                    "decision_both_median", "difference"]


def decision_roster(cohort=None) -> list[str]:
    """The configured decision models every median must contain (never inferred from results)."""
    return cohort_record(cohort)["decision_roster"]


def contrasts(gains: pd.DataFrame, *, roster) -> pd.DataFrame:
    """B2 descriptive contrasts per dataset and predictor: the median decision model's "both"
    gain minus each frozen, decision+frozen, TAR and probe arm (positive = decision ahead).
    Without complete decision medians the table is empty (waiting state)."""
    medians = group_b.median_over_decision_models(gains, roster=roster)
    if medians.empty:
        return pd.DataFrame(columns=CONTRAST_COLUMNS)
    reference = medians[medians.representation.eq("decision/both/median")][
        ["dataset_id", "model_id", "raw_gain"]].rename(columns={"raw_gain": "decision_both_median"})
    others = gains[gains.family.isin(["frozen", "tar", "probe"])]
    combined = medians[medians.representation.eq("decision+frozen/both/median")]
    others = pd.concat([others, combined], ignore_index=True)
    out = others.merge(reference, on=["dataset_id", "model_id"], how="inner")
    out["difference"] = out.decision_both_median - out.raw_gain
    return out[CONTRAST_COLUMNS]


def _group_a_frames(*, cohort=None):
    """Group A exports on PROJECT (A1 comparison, A2 stability) when they exist."""
    from src.evaluation import comparison_report, robustness_export

    frames, _ = comparison_report.read_tables(comparison_report.report_config(cohort), cohort=cohort)
    # An absent or superseded A2 leaves model relations partial. Verify the archive's
    # contents before checking its analysis identity so that corruption stops the export.
    found, receipt = study_export.read_export(robustness_export.export_path(cohort),
                                              schema=robustness_export.SCHEMA, tables=robustness_export.TABLES)
    if found and receipt["config_sha256"] == robustness_export.analysis_identity(load_yaml("A2"), cohort):
        frames.update({f"a2_{k}": v for k, v in found.items()})
    return frames


def dataset_properties_and_redundancy(*, roster=None):
    """Label-free properties of every selected dataset and decision redundancy per model."""
    from sklearn.model_selection import KFold

    from src.data.loaders import load_catalog, load_dataset
    from src.data.selection import selected_entries
    from src.evaluation.moderators import dataset_covariates
    from src.evaluation.row_benefit import block_probabilities, load_decision_block

    main = load_yaml("B1")
    roster = decision_roster() if roster is None else roster
    pilot = load_yaml("B4")["predict_in_advance"]["pilot"]
    properties, redundancy = [], []
    for entry in selected_entries(load_catalog(), main):
        dataset = load_dataset(entry, allow_download=False)
        row = dataset_covariates(dataset)
        row["log10_n_rows"] = float(np.log10(row["n_rows"]))
        row["log10_median_text_chars"] = float(np.log10(max(row["median_text_chars"], 1.0)))
        properties.append(row)
        non_text = dataset.features.loc[:, list(dataset.metadata.non_text_columns)]
        # Deployment has no downstream labels to stratify this covariate's folds.
        splitter = KFold(n_splits=5, shuffle=True, random_state=main["random_seed"])
        splits = [{"train": train, "test": test} for train, test in splitter.split(non_text)]
        for provider in roster:
            block, source = load_decision_block(entry["id"], provider, "joint", main)
            if block is None:
                redundancy.append({"dataset_id": entry["id"], "provider": provider, "status": source})
                continue
            matrix = block_probabilities(block, dataset.metadata.task_type, len(dataset.frame))["all_text"]
            frame = non_text.reset_index(drop=True)
            result = group_b.decision_redundancy(frame, matrix, splits,
                                                 task_type=dataset.metadata.task_type)
            result.update(group_b.pilot_redundancy(
                frame, matrix, dataset_id=entry["id"], task_type=dataset.metadata.task_type,
                size=pilot["rows"], seed=pilot["seed"], folds=pilot["folds"]))
            redundancy.append({"dataset_id": entry["id"], "provider": provider, "status": "computed",
                               **result})
        print(f"{entry['id']}: properties and redundancy computed", flush=True)
    properties = pd.DataFrame(properties)
    redundancy = pd.DataFrame(redundancy)
    for column, name in (("r2", "decision_redundancy_r2"), ("pilot_r2", "pilot_redundancy_r2")):
        if column not in redundancy:
            properties[name] = np.nan
        else:
            per_dataset = redundancy.groupby("dataset_id")[column].agg(["median", "count"])
            per_dataset = per_dataset["median"].where(per_dataset["count"].eq(len(roster))).rename(name)
            properties = properties.merge(per_dataset.reset_index(), on="dataset_id", how="left")
    return properties, redundancy


def build_tables(study, folds, *, cohort=None) -> dict[str, pd.DataFrame]:
    cfg = load_yaml("B4")
    roster = decision_roster(cohort)
    folds = folds[folds.provider.isna() | folds.provider.isin(roster)]
    head = group_b.headline(folds)
    tables = {"coverage": coverage(folds, STUDIES[study]["phases"], cohort=cohort), "folds": folds}
    if study == "B5":
        paired, gains = _gains(head, extra_keys=("training_size", "draw"))
        from src.evaluation import data_efficiency

        sizes = data_efficiency.over_draws(gains)
        # Two fixed cohorts (main: all datasets at the main sizes; extended: the configured
        # datasets at all sizes), so no curve changes its dataset membership along the x-axis.
        curves = (data_efficiency.cohort_gains(sizes, data_efficiency.config()["curves"])
                  if not sizes.empty else sizes)
        tables.update(dataset_gains=gains, summary=group_b.summarize(
            curves, keys=("curve", "model_id", "representation", "training_size"))
            if not curves.empty else curves,
            size_gains=sizes, pooled_raw=data_efficiency.pooled_curve_by_family(curves),
            pooled_share=data_efficiency.pooled_curve(curves, outcome="headroom_share"),
            slopes=data_efficiency.gain_slopes(sizes))
        return tables
    paired, gains = _gains(head)
    if study == "B4":
        gains = gains[gains.family.eq("decision")]
    tables.update(dataset_gains=gains, summary=group_b.summarize(gains))
    medians = group_b.median_over_decision_models(gains, roster=roster)
    gaps = group_b.roster_gaps(gains, roster=roster) if not gains.empty else pd.DataFrame()
    if study == "B1":
        tables.update(median_gains=medians, median_summary=group_b.summarize(medians),
                      median_roster_gaps=gaps)
    elif study in REPRESENTATION_STUDIES:
        from src.evaluation import complementarity

        pairs, _ = complementarity.summaries(cohort=cohort, headline_folds=head)
        pairs = pairs[pairs.provider.isin(roster)] if not pairs.empty else pairs
        tables.update(median_gains=medians, median_roster_gaps=gaps,
                      contrasts=contrasts(gains, roster=roster) if not gains.empty else gains,
                      complementarity_pairs=pairs,
                      complementarity_summary=complementarity.complementarity_summary(pairs, roster=roster)
                      if not pairs.empty else pd.DataFrame())
    elif study == "B4":
        from src.evaluation.row_benefit import summaries

        properties, redundancy = dataset_properties_and_redundancy(roster=roster)
        rule = cfg["predict_in_advance"]
        outcome = medians[medians.representation.eq(rule["outcome_representation"])]
        lodo_rows, lodo = [], []
        for model, group in outcome.groupby("model_id"):
            predictions, summary = group_b.lodo_input_sets(
                group, properties, input_sets=rule["input_sets"], alpha=rule["alpha"],
                threshold=rule["use_threshold"], n_shuffles=rule["n_shuffles"], value=rule["outcome"])
            lodo_rows.append(predictions.assign(model_id=model))
            lodo.append(summary.assign(model_id=model))
        a = _group_a_frames(cohort=cohort)
        diagnostics = group_b.model_diagnostics(
            a.get("quality"), a.get("output_shape"), a.get("loo_consensus"), a.get("model_profile"),
            a.get("a2_mode_effects"))
        usefulness = group_b.per_model_usefulness(gains, datasets=properties)
        relations = group_b.model_relations(usefulness, diagnostics,
                                            n_permutations=cfg["n_permutations"], min_units=5)
        from src.evaluation.group_b_views import row_cohort_cells

        measured, _ = summaries(cohort=cohort)
        cohorts = row_cohort_cells(measured, roster=roster, headline_folds=head)
        tables.update(
            properties=properties, redundancy=redundancy,
            associations=group_b.dataset_associations(medians, properties, n_permutations=cfg["n_permutations"]),
            model_diagnostics=diagnostics, model_usefulness=usefulness, model_relations=relations,
            column_importance=group_b.text_column_importance(head), row_cohorts=cohorts,
            lodo_predictions=pd.concat(lodo_rows, ignore_index=True) if lodo_rows else pd.DataFrame(
                columns=["dataset_id", "actual", "predicted", "input_set", "claim", "outcome", "model_id"]),
            lodo=pd.concat(lodo, ignore_index=True) if lodo else pd.DataFrame(
                columns=["input_set", "claim", "outcome", "properties", "excluded_datasets", "datasets", "status", "model_id"]))
    return tables


def _parquet_safe(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(
                lambda v: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v)
    return frame


def read(study, target=None, *, cohort=None):
    """Verified tables and receipt, or an explicit waiting state (absent/superseded ZIP)."""
    return study_export.read_export(target or export_path(study, cohort), schema=SCHEMA,
                                    tables=tables_for(study), identity=analysis_identity(study, cohort))


def build(study, *, cohort=None):
    study_export.require_mindwell_cpu(f"Group B {study}")
    metrics = B5_METRICS if study == "B5" else None
    folds = collect_folds(STUDIES[study]["phases"], metrics=metrics, cohort=cohort_name(cohort))
    if folds.empty:
        raise ValueError(f"No complete five-fold arm exists for {study}")
    tables = {name: _parquet_safe(frame) for name, frame in build_tables(study, folds, cohort=cohort).items()}
    missing = set(tables_for(study)) - set(tables)
    if missing:
        raise ValueError(f"{study} export lacks tables: {sorted(missing)}")
    target = study_export.write_export(
        export_path(study, cohort), {name: tables[name] for name in tables_for(study)}, schema=SCHEMA,
        identity=analysis_identity(study, cohort), origin=study_export.job_origin(),
        extra={"study": study, "headlines": HEADLINES, **cohort_record(cohort)})
    print(f"{target}  SHA256 {file_sha256(target)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, choices=tuple(STUDIES))
    parser.add_argument("--cohort", metavar="COHORT")
    args = parser.parse_args()
    build(args.study, cohort=args.cohort)


if __name__ == "__main__":
    main()


def median_summary(frames) -> pd.DataFrame:
    """Per-arm summary plus the median-over-decision-models rows (notebooks)."""
    medians = frames.get("median_gains")
    extra = group_b.summarize(medians) if medians is not None and not medians.empty else pd.DataFrame()
    return pd.concat([frames["summary"], extra], ignore_index=True)
