"""Require complete, current 0B evidence before a benchmark arm can fit."""

import importlib.metadata
import json
import math
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from src.data.loaders import load_catalog
from src.data.selection import selected_entries
from src.evaluation.acceptance import evaluation_identity
from src.utils import paths
from src.utils.acceptance import evidence_path, reviewed_acceptance
from src.utils.config import load_yaml
from src.utils.serialization import digest, write_json

PHASE_CONFIGS = {
    "experiment_1": "experiment_1/main",
    "experiment_2": "experiment_2/embeddings",
    "experiment_3": "experiment_3/tar",
    "experiment_4": "experiment_4/native_text",
    "rq2": "rq2/text_column_refits",  # B3 text-column refits
    "data_efficiency": "data_efficiency/data_efficiency",  # B4 training-size curves
    "representations": "representations/representations",  # C1 Strands hidden states
    "rq3": "rq3/adaptation",  # C2 Strands learning from labels
}
# Full-size shape, task and class extremes; other phases use targeted real-path profiles.
PROFILED_PHASES = ("experiment_1", "experiment_2", "experiment_3", "experiment_4")
GROUP_C_PHASES = ("representations", "rq3")
#: Declared 0B case (23 Nov 2026): the widest Kumo arm above the default 500-column recipe
#: (Rotten Tomatoes decision + E5, 557 columns) runs a full-fold profile on a real B200 with
#: every column per estimator, recording peak GPU memory and fold time.
KUMO_OVERRIDE_PROFILE = "kumo_max_columns_override_b200_profile"
#: Longest Mindwell bucket (benchmark_sizing.TIME_BUCKETS_HOURS[-1]); five folds must fit.
LONGEST_JOB_SECONDS = 72 * 3600


def kumo_override_arm(arms):
    """The widest planned Kumo arm whose recipe limit is widened beyond 500 columns."""
    from src.evaluation.models import KUMO_RECIPE_COLUMNS

    wide = [
        a for a in arms
        if a["model"] == "kumo_tabular" and a["phase"] in PROFILED_PHASES
        and a.get("width_status") != "unsupported_width" and a["predictor_columns"] > KUMO_RECIPE_COLUMNS
    ]
    if not wide:
        return None
    rows = {e["id"]: e["paper_summary"]["rows"] for e in load_catalog()["datasets"]}
    # First maximum in plan order, as the width extreme of profile_plan selects it: one profile.
    return max(wide, key=lambda a: (a["predictor_columns"], rows[a["dataset_id"]]))


def decision_selector(config, provider):
    from src.evaluation.design import decision_versions

    selector = config["feature_selectors"].get(provider)
    fields = ("model", "version", "question_bank_id", "question_variant", "settings_sha256")
    if (
        not selector
        or selector.get("provider") != provider
        or any(not selector.get(k) for k in fields)
    ):
        raise ValueError(f"Select the exact model, settings and question recipe for {provider}")
    registered, = decision_versions([provider])
    if any(selector[key] != registered[key] for key in ("model", "version")):
        raise ValueError(f"The {provider} selector differs from its registered model or version")
    return selector


def decision_artifact_pins(config, dataset_id, provider):
    return config.get("dataset_feature_artifacts", {}).get(dataset_id, {}).get(provider)


def execution_runtime():
    from src.cluster.containers import active_identity

    image = active_identity() or {}
    if not image and os.environ.get("JEVPFN_CONTROL_IMAGE"):
        from src.cluster.control_runtime import MARKER

        image = json.loads(
            (Path(os.environ["JEVPFN_CONTROL_IMAGE"]) / MARKER).read_text(encoding="utf-8")
        )
    packages = {
        d.metadata["Name"].lower().replace("_", "-"): d.version
        for d in importlib.metadata.distributions()
    }
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "container_id": image.get("container_id"),
        "packages_sha256": digest(packages),
    }


def profile_plan(arms=None, *, cohort="all"):
    """Core 0B covers Group B. Explicit C1/C2 arms are profiled after their assets exist."""
    from src.evaluation.campaign import benchmark_plan
    from src.evaluation.design import decision_providers

    rows = {e["id"]: e["paper_summary"]["rows"] for e in load_catalog()["datasets"]}
    paid_providers = set(decision_providers("paid"))
    semantics = load_yaml("dataset_semantics")["datasets"]
    encoders = load_yaml("feature_creation/embedding_models")["models"]
    groups, later = {}, {}
    candidates = ([a for a in benchmark_plan(cohort=cohort) if a["phase"] not in GROUP_C_PHASES]
                  if arms is None else arms)
    for arm in candidates:
        if arm["width_status"] == "unsupported_width":
            continue
        if arm["phase"] in PROFILED_PHASES:
            key = (arm["phase"], arm["model"], arm.get("pca_scope"), arm.get("pca_components"))
            groups.setdefault(key, []).append(arm)
        else:
            later.setdefault((arm["phase"], arm["model"]), []).append(arm)

    selected = {}
    for group in groups.values():

        def height(arm):
            return rows[arm["dataset_id"]]

        def width(arm):
            return arm["predictor_columns"]

        def encoder_width(arm):
            return encoders[arm["encoder"].removesuffix("_tar")]["dimensions"] if arm.get("encoder") else 0

        def tall(arm):
            return height(arm), width(arm), encoder_width(arm)

        extremes = [
            max(group, key=tall),
            max(group, key=lambda a: (width(a), height(a), encoder_width(a))),
        ]
        # Shape extremes alone can omit an entire prediction task. In particular,
        # regression can dominate both extrema without exercising real multiclass
        # probability ordering, or TAR's binary path (which has no inner OOF fits).
        tasks = {semantics[a["dataset_id"]]["task_type"] for a in group}
        for task in sorted(tasks):
            cases = [a for a in group if semantics[a["dataset_id"]]["task_type"] == task]
            covered = [a for a in extremes if semantics[a["dataset_id"]]["task_type"] == task]
            largest_vocabulary = max(len(semantics[a["dataset_id"]].get("classes", []))
                                     for a in cases)
            if covered and max(len(semantics[a["dataset_id"]].get("classes", []))
                               for a in covered) == largest_vocabulary:
                continue
            extremes.append(max(cases, key=lambda a: (
                len(semantics[a["dataset_id"]].get("classes", [])), *tall(a)
            )))
        for arm in extremes:
            selected[digest(arm)] = arm
    # Cover distinct input paths cheaply in addition to the stress profiles.
    # One linear predictor suffices for these path checks; every predictor retains its extremes.
    for group in groups.values():
        if group[0]["model"] != "linear":
            continue
        paths_to_check = {}
        for arm in group:
            if arm["phase"] == "experiment_1":
                family = (arm["feature_set"], arm.get("provider") if arm["feature_set"] == "non_text_plus_both"
                          else arm.get("provider") in paid_providers)
            elif arm["phase"] == "experiment_2":
                family = (arm["feature_set"], arm.get("encoder"))
            elif arm["phase"] == "experiment_3" and arm.get("probe_view"):
                family = (arm["encoder"], arm["probe_view"])
            else:
                continue
            key = (rows[arm["dataset_id"]], arm["dataset_id"], arm.get("provider") or "")
            if family not in paths_to_check or key < paths_to_check[family][0]:
                paths_to_check[family] = (key, arm)
        for _, arm in paths_to_check.values():
            selected[digest(arm)] = arm
    # Declared, not left to the width extremes: the real-B200 memory/time check for the
    # widened Kumo recipe.
    wide = kumo_override_arm([a for group in groups.values() for a in group])
    if wide is not None:
        selected[digest(wide)] = wide
    for (phase, model), group in later.items():
        for arm in _phase_profiles(phase, model, group, rows, semantics):
            selected[digest(arm)] = arm
    return list(selected.values())


def _phase_profiles(phase, model, arms, rows, semantics):
    """Exercise each distinct later workload, including B4's real sparse and 30-class samples."""
    from src.evaluation.features import (
        HIDDEN,
        JOINT_PLUS_HIDDEN,
        PER_COLUMN_PLUS_HIDDEN,
        SUPERVISED_PROBE,
    )

    def small(a):
        return rows[a["dataset_id"]], a["dataset_id"]

    def task_witnesses(cases):
        if model != "linear":
            return []
        # Exercise binary threshold handling and multiclass feature shapes where the
        # cheapest or widest workload alone would select regression exclusively.
        tasks = {semantics[a["dataset_id"]]["task_type"] for a in cases}
        return [min((a for a in cases if semantics[a["dataset_id"]]["task_type"] == task),
                    key=lambda a: (-len(semantics[a["dataset_id"]].get("classes", [])), *small(a)))
                for task in sorted(tasks)]

    if phase == "rq2":
        # Stress the widest retained per-column block. A second field is dropped where possible.
        return [max(arms, key=lambda a: (a["predictor_columns"], rows[a["dataset_id"]],
                                         a["dropped_text_column"] == 1)), *task_witnesses(arms)]
    if phase == "data_efficiency":
        cases = []
        for dataset, feature, size in (
            ("REG_TEXT_BOOK_READABILITY", SUPERVISED_PROBE, 128),
            ("BIN_TEXT_FAKE_JOB_POSTING", SUPERVISED_PROBE, 128),
            ("MUL_TEXT_WINE_REVIEW", "non_text_plus_frozen_e5", 128),
            ("MUL_TEXT_WINE_REVIEW", "non_text_plus_both", 16384),
        ):
            matching = [a for a in arms if a["dataset_id"] == dataset and a["feature_set"] == feature
                        and a["training_size"] == size and a["draw"] == 0]
            if matching:
                cases.append(next((a for a in matching if a.get("provider") == "strands"), matching[0]))
        return cases
    if phase == "representations":
        cases = []
        for feature, view in ((HIDDEN, "per_column"), (HIDDEN, "joint"),
                              (PER_COLUMN_PLUS_HIDDEN, "per_column"), (JOINT_PLUS_HIDDEN, "joint")):
            matching = [a for a in arms if a["feature_set"] == feature and a["hidden_view"] == view]
            if matching:
                cases.append(min(matching, key=small))
        if model == "kumo_tabular":
            cases.append(max(arms, key=lambda a: a["predictor_columns"]))
        hidden = [a for a in arms if a["feature_set"] == HIDDEN]
        if model == "linear":
            for view, pos, layer in sorted({(a["hidden_view"], a["hidden_position"], a["hidden_layer"])
                                            for a in hidden}):
                cases.append(min((a for a in hidden if (a["hidden_view"], a["hidden_position"],
                                                        a["hidden_layer"]) == (view, pos, layer)), key=small))
        return cases + task_witnesses(hidden)
    if phase == "rq3":
        cases = [min((a for a in arms if a["adaptation_cell"] == cell), key=small)
                 for cell in sorted({a["adaptation_cell"] for a in arms})]
        return cases + task_witnesses([a for a in arms if a["adaptation_cell"] == "head_only"])
    return []


def profile_fold_number(arm):
    # Small-sample sparse probes exercise fold 0, B3 and other B4 paths also cover a later fold.
    if arm["phase"] == "rq2" or (arm["phase"] == "data_efficiency" and not arm.get("probe_view")):
        return 1
    return 0


def _needs_inner(arm, task_type):
    return task_type == "binary_classification" and arm["phase"] not in {
        "experiment_3", "data_efficiency", "rq3"
    }


def source_leakage_diagnostics(dataset, splits):
    """Count exact feature-row overlap and flag copies or renamed class labels."""
    import pandas as pd

    predictors = dataset.features
    target = dataset.frame[dataset.metadata.target]
    identities = pd.util.hash_pandas_object(predictors, index=False)
    overlap = [
        int(identities.iloc[fold["test"]].isin(identities.iloc[fold["train"]]).sum())
        for fold in splits["folds"]
    ]
    target_values = target.astype(object)
    classes = target.nunique() if dataset.metadata.task_type != "regression" else 0
    copies, recodings = [], []
    for name, column in predictors.items():
        if column.astype(object).eq(target_values).all():
            copies.append(str(name))
        elif (
            classes > 1
            and not column.isna().any()
            and column.nunique() == classes
            and len(pd.DataFrame({"predictor": column, "target": target}).drop_duplicates())
            == classes
        ):
            recodings.append(str(name))
    name_mentions = {}
    if getattr(dataset.metadata, "dataset_id", None) == "MUL_TEXT_WINE_REVIEW":
        import re
        for field in dataset.metadata.text_columns:
            texts = predictors[field]
            matches = sum(
                bool(re.search(r"(?<!\w)" + re.escape(str(label)) + r"(?!\w)", str(text), re.IGNORECASE))
                for text, label in zip(texts, target, strict=True) if pd.notna(text) and pd.notna(label)
            )
            name_mentions[field] = {"whole_label_mentions": matches, "rows": len(target)}
    return {
        "literal_target_mentions": name_mentions,
        "exact_target_copies": copies,
        "exact_class_target_recodings": recodings,
        "exact_feature_row_overlap_by_fold": overlap,
        "test_rows_by_fold": [len(fold["test"]) for fold in splits["folds"]],
    }


def verify_arm_recipe(metadata, arm, artifact_ids, runtime):
    """Match the saved fit to the planned arm even when two recipes have equal width."""
    recipe = metadata.get("representation_recipe")
    if not isinstance(recipe, dict):
        raise ValueError("Published 0B predictions differ: missing fold recipe")
    embedding = recipe.get("embedding") or {}
    probe = recipe.get("probe") or {}
    hidden = recipe.get("hidden") or {}
    adapted = recipe.get("adaptation") or {}
    sample = recipe.get("subsample") or {}
    pca = hidden if hidden else embedding
    expected = (arm["model"], arm["dataset_id"], arm["feature_set"], arm.get("provider"),
                arm.get("encoder"), arm.get("pca_scope"), arm.get("pca_components"),
                arm.get("view", "main"), arm.get("probe_view"), arm.get("probe_view"),
                arm.get("hidden_view"), arm.get("hidden_position"), arm.get("hidden_layer"),
                arm.get("adaptation_cell"), arm.get("dropped_text_column"),
                arm.get("training_size"), arm.get("draw"))
    actual = (metadata["model_id"], metadata["dataset_id"], metadata["feature_set"],
              recipe.get("provider"), embedding.get("encoder_id"), pca.get("pca_scope"),
              pca.get("pca_components" if hidden else "pca_components_per_column"),
              recipe.get("view", "main"), probe.get("view"), embedding.get("probe_view"),
              hidden.get("view"), hidden.get("position"), hidden.get("layer"),
              adapted.get("cell"), recipe.get("dropped_text_column"),
              sample.get("training_size"), sample.get("draw"))
    if (expected != actual or (probe and probe.get("encoder_id") != arm.get("encoder"))
            or (arm.get("phase") == "representations" and hidden.get("provider") != "strands")
            or (arm.get("phase") == "rq3" and (adapted.get("provider"), adapted.get("question"),
                                               adapted.get("crossfit_parts")) != ("strands", "joint", 3))
            or sorted(metadata["feature_artifact_ids"]) != sorted(artifact_ids)
            or metadata["estimator_settings"].get("execution_runtime") != runtime):
        raise ValueError("Published 0B predictions differ from the planned fold recipe, runtime or input artifacts")


def save_profile(arm, dataset, features, runtime, records, *, specification=None):
    """Read back the real profile fold, including the B4 training sample and full held-out rows."""
    from src.evaluation.protocol import shared_splits
    from src.evaluation.store import read_predictions

    cfg = load_yaml(PHASE_CONFIGS[arm["phase"]])
    split = specification or shared_splits(dataset, seed=cfg["random_seed"], folds=5)
    fold_number = profile_fold_number(arm)
    fold = split["folds"][fold_number]
    roles = {"outer_test": fold["test"]}
    if _needs_inner(arm, dataset.metadata.task_type):
        roles["inner_oof_binary_threshold"] = fold["train"]
    verified = []
    fold_ids = set()
    for record in records:
        body = json.loads(record.read_text(encoding="utf-8"))
        meta = body["metadata"]
        role = meta["prediction_role"]
        if role == "outer_test":
            outer = body
        fold_ids.add(meta["fold_recipe_id"])
        positions = roles.pop(role)
        table = read_predictions(record)
        if meta.get("representation_feature_count") != arm["predictor_columns"]:
            raise ValueError("Full-fold profile predictor width differs from the planned recipe")
        verify_arm_recipe(meta, arm, features.attrs["feature_artifact_ids"], runtime)
        if (
            meta["split_id"] != split["split_id"]
            or meta["outer_fold"] != fold_number
            or meta["n_train"] != len(fold["train"])
            or table.row_id.tolist() != [f"csv:{i}" for i in positions]
            or table.y_true.tolist()
            != dataset.frame[dataset.metadata.target].iloc[positions].tolist()
        ):
            raise ValueError(
                "Full-fold profile prediction rows/targets differ from the shared split"
            )
        verified.append(
            {
                "role": role, "rows": len(table), "record_id": record.key,
                "prediction_sha256": body["predictions"]["sha256"],
            }
        )
    if roles:
        raise ValueError("Full-fold profile lacks required prediction roles")
    if len(fold_ids) != 1:
        raise ValueError("0B roles belong to different completed folds")
    measured = profile_resources(outer)
    overrides = kumo_overrides(records)
    from src.evaluation.models import KUMO_RECIPE_COLUMNS

    if arm["model"] == "kumo_tabular" and arm["predictor_columns"] > KUMO_RECIPE_COLUMNS and (
        not overrides or min(overrides) < arm["predictor_columns"]
    ):
        raise ValueError("Wide Kumo profile lacks its recorded kumo_max_columns_override")
    report = {
        "status": "full_fold_profile_passed",
        "fold_recipe_id": fold_ids.pop(),
        "evaluation_identity": evaluation_identity(),
        "arm": arm,
        "runtime_identity": runtime,
        "job_id": outer["resources"].get("fit_job_id"),
        "cluster": outer["resources"].get("fit_cluster"),
        "data_identity": dataset.source["data_identity"],
        "feature_artifact_ids": features.attrs["feature_artifact_ids"],
        "decision_artifact_checksums": features.attrs["decision_artifact_checksums"],
        "split_id": split["split_id"],
        "parent_split_id": split.get("subsample", {}).get("parent_split_id", split["split_id"]),
        "outer_fold": fold_number,
        "training_rows": len(fold["train"]),
        "predictor_columns": arm["predictor_columns"],
        "roles": verified,
        "resources": measured,
        "kumo_max_columns_override": max(overrides) if overrides else None,
        "created_at": datetime.now(UTC).isoformat(),
    }
    write_json(
        paths.phase_path("experiment_0", "manifests", "after_extraction")
        / f"profile_{arm['model']}_{uuid4().hex}.json",
        report,
    )


def kumo_overrides(records):
    """Expanded widths recorded by KumoBackend whenever it widened SelectColumns(500)."""
    widths = []
    for record in records:
        for item in json.loads(record.read_text(encoding="utf-8")).get("warnings", []):
            message = str(item.get("message", ""))
            if message.startswith("kumo_max_columns_override="):
                widths.append(int(message.split("=", 1)[1].split(":", 1)[0]))
    return widths


def profile_resources(record):
    """Peak memory and outer-fold time of the profiled fit, plus the GPU it ran on."""
    timing, resources = record.get("timing", {}), record.get("resources", {})
    device = str(timing.get("device") or "cpu")
    return {
        "device": device,
        "gpu_name": resources.get("gpu_name"),
        "gpu_total_bytes": resources.get("gpu_total_bytes"),
        "peak_gpu_allocated_bytes": resources.get("peak_gpu_allocated_bytes"),
        "peak_gpu_reserved_bytes": resources.get("peak_gpu_reserved_bytes"),
        "peak_rss_bytes": resources.get("peak_rss_bytes"),
        "device_workers": resources.get("device_workers"),
        "fold_wall_seconds": timing.get("measured_wall_seconds"),
    }


def verify_kumo_override_profile(report, planned):
    """Real B200, every column kept, exclusive device, measured memory, five folds in time."""
    measured = report.get("resources") or {}
    seconds, reserved = measured.get("fold_wall_seconds"), measured.get("peak_gpu_reserved_bytes")
    total = measured.get("gpu_total_bytes")
    if (
        not isinstance(report.get("kumo_max_columns_override"), int)
        or report["kumo_max_columns_override"] < planned["predictor_columns"]
        or "B200" not in str(measured.get("gpu_name") or "")
        or measured.get("device_workers") != 1
        or type(reserved) is not int or type(total) is not int or not 0 < reserved <= total
        or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0
        or 1.5 * 5 * seconds + 1800 > LONGEST_JOB_SECONDS
    ):
        raise ValueError(f"0B lacks a passing {KUMO_OVERRIDE_PROFILE} for {planned['dataset_id']}")


def verify_profile_resources(report, gpu):
    """A profile qualifies a real, exclusive device and one schedulable five-fold arm."""
    from src.cluster import benchmark_sizing as sizing

    measured = report.get("resources") or {}
    wall = measured.get("fold_wall_seconds")
    rss = measured.get("peak_rss_bytes")
    if (type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0
            or wall * 5 * 1.5 + 1800 > LONGEST_JOB_SECONDS
            or type(rss) is not int or rss < 1 or measured.get("device_workers") != 1
            or sizing._memory_mib({"peak_rss_bytes": rss}, 1) > sizing._memory_ceiling(gpu)):
        raise ValueError("0B profile lacks feasible measured wall time, host memory or exclusive workers")
    device = measured.get("device", "")
    if gpu:
        reserved, total = measured.get("peak_gpu_reserved_bytes"), measured.get("gpu_total_bytes")
        if (not device.startswith("cuda") or "B200" not in str(measured.get("gpu_name"))
                or type(reserved) is not int or type(total) is not int or not 0 < reserved <= total):
            raise ValueError("0B GPU profile lacks measured B200 memory or device evidence")
    elif device != "cpu":
        raise ValueError("0B CPU profile has the wrong device")


def _complete(rows, keys, expected, label):
    seen = {tuple(row[k] for k in keys): row for row in rows}
    if (
        len(seen) != len(rows)
        or set(seen) != expected
        or any(row.get("status") != "passed" for row in rows)
    ):
        raise ValueError(f"0B requires complete passing {label} evidence")
    return seen


def _frozen_equivalent(alias, providers, scopes):
    """A growing convenience alias can still describe its original registered edition."""
    from src.evaluation.design import incremental_cohort

    members = set(providers)
    if alias not in {"all", "paid"} or not members < set(scopes[alias]):
        return None
    return next((name for name, registered in scopes.items()
                 if name not in {"all", "paid"} and incremental_cohort(name) == incremental_cohort(alias)
                 and len(registered) == len(providers) and set(registered) == members), None)


def provider_scope(providers, main, *, declared=None):
    """An owner may release a registered cohort and must name its exact ordered roster."""
    from src.evaluation.design import decision_scopes

    if (not isinstance(providers, list) or not providers
            or any(not isinstance(p, str) or not p for p in providers) or len(set(providers)) != len(providers)):
        raise ValueError("0B requires at least one named decision provider")
    scopes = decision_scopes(main)
    if declared is None:
        for cohort in ("all", *scopes):
            if list(scopes[cohort]) == providers:
                return cohort
    elif isinstance(declared, str) and declared in scopes:
        if list(scopes[declared]) == providers:
            return declared
        cohort = _frozen_equivalent(declared, providers, scopes)
        if cohort:
            return cohort
    raise ValueError("0B acceptance must name the complete ordered decision providers in its cohort")


def acceptance_for(setting, provider):
    """Route the arm to its configured receipt; benchmark_evidence verifies what it signed."""
    if not isinstance(setting, dict):
        return setting
    from src.evaluation.design import decision_scopes, incremental_cohort

    dedicated = setting.get("providers", {})
    if not isinstance(dedicated, dict):
        raise ValueError("0B provider receipt paths must be a mapping")
    if provider is not None and provider in dedicated:
        return dedicated[provider]
    scopes = decision_scopes()
    additions = [name for name in scopes if name.endswith("_added")]
    editions = [name for name in scopes if name not in {"open_weight", "paid", "all", *additions}]
    for name in ("open_weight", *additions, "paid", *editions, "all"):
        eligible = provider in scopes[name] if provider is not None else not incremental_cohort(name)
        if eligible and setting.get(name):
            return setting[name]
    return None


def feature_evidence(report):
    """Check coverage against the current selected grid, including all three encoders."""
    main = load_yaml("experiment_1/main")
    providers = report.get("qualified_decision_providers", main["candidate_extractors"])
    provider_scope(providers, main, declared=report.get("decision_cohort"))
    versions = {}
    artifacts = report.get("decision_artifacts", [])
    if ("decision_cohort" in report or "decision_model_versions" in report
            or any("model" in row or "version" in row for row in artifacts)):
        from src.evaluation.design import decision_versions
        if not report.get("decision_cohort") or "qualified_decision_providers" not in report:
            raise ValueError("0B must name the cohort and decision providers of its versioned artifacts")
        current = decision_versions(providers)
        if report.get("decision_model_versions") != current:
            raise ValueError("0B must record the exact registered model versions for this decision cohort")
        versions = {entry["provider"]: entry for entry in current}
    entries = selected_entries(load_catalog(), main)
    datasets = {entry["id"] for entry in entries}
    if not datasets:
        raise ValueError("0B requires selected datasets")
    data = _complete(
        report.get("datasets", []), ("dataset_id",), {(d,) for d in datasets}, "dataset"
    )
    tasks = {"binary_classification", "multiclass_classification", "regression"}
    for row in data.values():
        # Absent identities would otherwise compare equal (None == None) to absent profile fields.
        if (
            any(not isinstance(row.get(k), str) or not row[k]
                for k in ("data_identity", "split_id"))
            or row.get("task_type") not in tasks
            or row.get("outer_folds") != 5
            or type(row.get("rows")) is not int
            or row["rows"] < 5
        ):
            raise ValueError("0B requires the verified identity, task, rows and five-fold split of each dataset")
    decisions = _complete(
        artifacts,
        ("dataset_id", "provider"),
        {(d, p) for d in datasets for p in providers},
        "decision feature",
    )
    embeddings = _complete(
        report.get("embedding_artifacts", []),
        ("dataset_id", "encoder_id"),
        {(d, e) for d in datasets for e in load_yaml("experiment_2/embeddings")["encoders"]},
        "encoder feature",
    )
    def valid_ids(values, count):
        return (isinstance(values, list) and len(values) == count
                and all(isinstance(v, str) and v for v in values)
                and len(set(values)) == count)

    def outer_pca_keys(dataset_id):
        row = data[dataset_id,]
        columns = row.get("text_columns")
        if type(columns) is not int or columns < 1:
            raise ValueError("0B lacks the verified text-column count")
        from src.evaluation.design import pca_grid

        return 5 * sum(columns if scope == "per_column" else 1 for scope, _ in pca_grid())

    for item in decisions.values():
        if versions and any(item.get(key) != versions[item["provider"]][key] for key in ("model", "version")):
            raise ValueError("Qualified decision artifacts differ from the registered model or version")
        runtime = item.get("runtime_revision")
        if versions and runtime is not None and versions[item["provider"]].get("runtime_revision") not in (None, runtime):
            raise ValueError("Qualified decision artifacts differ from the registered runtime revision")
        ids = item.get("artifact_ids", {})
        sums = item.get("artifact_checksums", {})
        if (set(ids) != {"joint", "per_column"} or not valid_ids(list(ids.values()), 2)
                or set(sums) != set(ids.values()) or not all(sums.values())):
            raise ValueError("0B requires exact artifact IDs and checksums for both decision modes")
    for (dataset_id, _), item in embeddings.items():
        inner = 4 if data[dataset_id,]["task_type"] == "binary_classification" else 1
        if not item.get("artifact_id") or item.get("prepared_pca_keys") != inner * outer_pca_keys(dataset_id):
            raise ValueError("0B requires exact artifact IDs and complete frozen PCA partition counts")
        if not valid_ids(item.get("prepared_probe_keys"), 10):
            raise ValueError("0B requires both supervised probe views for all five outer folds")
    from src.evaluation.tar import ENCODER
    tar = _complete(report.get("tar_artifacts", []), ("dataset_id", "encoder_id"),
                    {(d, ENCODER) for d in datasets}, "TAR fold feature")
    for key, item in tar.items():
        if (not valid_ids(item.get("artifact_ids"), 5)
                or not valid_ids(item.get("prepared_pca_keys"), outer_pca_keys(key[0]))):
            raise ValueError("0B requires all five TAR folds and their distinct PCA parameters")
        embeddings[key] = item
    return data, decisions, embeddings


def verify_published_profile(report, *, dataset=None, specification=None):
    """Reopen the durable records, including their Parquet payloads, at 0B review."""
    from src.evaluation.store import RecordRef, database, read_predictions, readonly

    if paths.on_vsc() and paths.staging_override() != paths.durable_project_override():
        raise ValueError("Verify published 0B profiles against durable PROJECT, outside scratch")
    db = database(report["arm"]["dataset_id"], "experiment_0")
    with readonly(db) as connection:
        marker = connection.execute("SELECT recipe, records FROM folds WHERE id=?", (report.get("fold_recipe_id"),)).fetchone()
    if marker is None or digest(json.loads(marker[0])) != report["fold_recipe_id"]:
        raise ValueError("Published 0B predictions differ: missing completed fold recipe")
    registered = {r["id"]: r["sha256"] for r in json.loads(marker[1])}
    for item in report["roles"]:
        record = RecordRef(db, item["record_id"])
        saved = json.loads(record.read_text())
        if registered.get(item["record_id"]) != digest(saved) or saved["metadata"]["fold_recipe_id"] != report["fold_recipe_id"]:
            raise ValueError("Published 0B predictions differ from the completed fold")
        verify_arm_recipe(saved["metadata"], report["arm"], report["feature_artifact_ids"], report["runtime_identity"])
        table = read_predictions(record)
        if specification is not None:
            fold = specification["folds"][report["outer_fold"]]
            positions = fold["test" if item["role"] == "outer_test" else "train"]
            if "subsample" in specification:
                sample = saved["metadata"]["representation_recipe"].get("subsample", {})
                if any(sample.get(k) != specification["subsample"][k] for k in
                       ("parent_split_id", "training_size", "draw", "method")):
                    raise ValueError("Published 0B sampling recipe differs from the reconstructed split")
            if (report["split_id"] != specification["split_id"]
                    or table.row_id.tolist() != [f"csv:{i}" for i in positions]
                    or table.y_true.tolist() != dataset.frame[dataset.metadata.target].iloc[positions].tolist()):
                raise ValueError("Published 0B predictions differ from the independently reconstructed split")
        if item["role"] == "outer_test":
            measured = report.get("resources", {})
            resources = saved.get("resources", {})
            if (measured != profile_resources(saved)
                    or report.get("job_id") != resources.get("fit_job_id")
                    or report.get("cluster") != resources.get("fit_cluster")):
                raise ValueError("Published 0B resource use differs from the measured fold")
        if (
            saved["metadata"]["prediction_role"] != item["role"]
            or saved["metadata"]["outer_fold"] != profile_fold_number(report["arm"])
            or saved["metadata"]["data_identity"] != report["data_identity"]
            or saved["metadata"]["model_id"] != report["arm"]["model"]
            or saved["metadata"]["feature_set"] != report["arm"]["feature_set"]
            or saved["metadata"]["split_id"] != report["split_id"]
            or saved["metadata"]["n_train"] != report["training_rows"]
            or saved["metadata"].get("representation_feature_count") != report["predictor_columns"]
            or saved["predictions"]["sha256"] != item["prediction_sha256"]
            or len(table) != item["rows"]
        ):
            raise ValueError("Published 0B predictions differ from the profile receipt")
    if sorted(registered) != sorted(item["record_id"] for item in report["roles"]):
        raise ValueError("0B profile omits or duplicates a completed fold record")


def benchmark_evidence(filename, arm, *, verify_publication=False, required_cohort=None):
    """Checksums alone cannot make a partial or synthetic-only report into acceptance."""
    from src.cluster.predictor_contracts import CONTRACT_VERSION, required_fixture_cases
    from src.evaluation.design import decision_scopes, incremental_cohort
    from src.evaluation.models import MAIN_MODELS, TEXT_MODELS

    current = evaluation_identity()
    approval = reviewed_acceptance(
        filename, stage="after_extraction", expected={"evaluation_identity": current}
    )
    reports = [
        json.loads(evidence_path(filename, item).read_text(encoding="utf-8"))
        for item in approval["evidence"]
    ]
    live = [r for r in reports if r.get("evaluation_identity") == current and r.get("job_id")]
    preflight = [
        r
        for r in live
        if (
            r.get("phase") == "experiment_0_preflight"
            and r.get("stage") == "after_extraction"
            and r.get("execution_site") == "vsc_compute_node"
            and r.get("status") == "preflight_passed"
            and r.get("contract_tests", {}).get("status") == "passed"
            and r.get("runtime_parity", {}).get("matches_control") is True
        )
    ]
    if len(preflight) != 1:
        raise ValueError("0B requires one current VSC preflight with all offline contracts passing")
    main = load_yaml("experiment_1/main")
    # Historical full-roster files predate scoped releases. New preflights must be
    # counter-signed for exactly the named providers, including an all-provider release.
    providers = approval.get("qualified_decision_providers", main["candidate_extractors"])
    if ("qualified_decision_providers" in preflight[0] and "qualified_decision_providers" not in approval
            or providers != preflight[0].get("qualified_decision_providers", main["candidate_extractors"])):
        raise ValueError("The signed 0B decision providers differ from the validated preflight")
    if "qualified_decision_providers" in preflight[0]:
        from src.evaluation.acceptance import decision_input_identity
        if preflight[0].get("decision_input_identity") != decision_input_identity(providers):
            raise ValueError("The qualified decision selectors and artifact pins have changed")
    original = preflight[0].get("decision_cohort")
    cohort = provider_scope(providers, main, declared=original)
    scopes = decision_scopes(main)
    members = set(providers)
    for name, signed in ((approval.get("decision_cohort"), True), (required_cohort, False)):
        if name is None:
            continue
        if (not isinstance(name, str) or name not in scopes or incremental_cohort(name) != incremental_cohort(cohort)
                or set(scopes[name]) != members and not (signed and _frozen_equivalent(name, providers, scopes))):
            raise ValueError("The signed 0B decision providers differ from the requested cohort")
    if ("decision_model_versions" in approval
            and approval["decision_model_versions"] != preflight[0].get("decision_model_versions")):
        raise ValueError("The signed model versions differ from the validated preflight")
    if (arm.get("provider") and arm["provider"] not in providers
            or not arm.get("provider") and incremental_cohort(cohort)):
        raise ValueError("This arm is outside the signed 0B decision-provider cohort")
    features = feature_evidence(preflight[0])
    predictors = {
        r["model"]: r
        for r in live
        if (
            r.get("status") == "predictor_contracts_passed_requires_full_fold_profile"
            and r.get("contract_version") == CONTRACT_VERSION
            and r.get("fixtures")
            and all(f.get("resume_without_refitting") for f in r["fixtures"])
        )
    }
    approved_contracts = [r for r in live if r.get("status") == "predictor_contracts_passed_requires_full_fold_profile"]
    if len(approved_contracts) != len(predictors) or set(predictors) != set(MAIN_MODELS + TEXT_MODELS):
        raise ValueError(f"0B requires passing real predictor contracts for all {len(MAIN_MODELS + TEXT_MODELS)} models")
    for model, report in predictors.items():
        cases = {(f["task_type"], f["classes"], f["feature_case"]) for f in report["fixtures"]}
        # Five outer test records, plus five inner-OOF records for the binary threshold.
        if any(
            not isinstance(f.get("records"), list)
            or len(set(f["records"])) != len(f["records"])
            or len(f["records"]) != (10 if f["task_type"] == "binary_classification" else 5)
            for f in report["fixtures"]
        ):
            raise ValueError(f"0B predictor fixtures for {model} lack their saved fold records")
        if (
            len(cases) != len(report["fixtures"])
            or cases != required_fixture_cases(model)
            or not report.get("runtime_identity", {}).get("container_id")
        ):
            raise ValueError(f"0B predictor fixtures or sealed runtime are incomplete for {model}")
        if not any(f.get("partial_fold_resume_verified") and f.get("interrupted_record_id")
                   and f["task_type"] == "binary_classification"
                   and f["feature_case"] in {"original_text", "non_text"} for f in report["fixtures"]):
            raise ValueError(f"0B lacks an interrupted fixture resume for {model}")
        if verify_publication:
            from src.cluster.predictor_contracts import verify_published_fixtures
            verify_published_fixtures(report)
    profiles = {digest(r["arm"]): r for r in live if r.get("status") == "full_fold_profile_passed"}
    if len(profiles) != sum(r.get("status") == "full_fold_profile_passed" for r in live):
        raise ValueError("Select exactly one full-fold profile per planned arm")
    if original and original != cohort:
        from src.evaluation.campaign import benchmark_plan

        # Stress-profile ties use plan order. Restore the tested provider order before
        # selecting any witnesses from an older full or paid release.
        historical_main = {**main, "candidate_extractors": [*providers, *(p for p in main["candidate_extractors"]
                                                                         if p not in members)]}
        historical_arms = [a for a in benchmark_plan(main=historical_main, cohort=original)
                           if a["phase"] not in GROUP_C_PHASES and (not a.get("provider") or a["provider"] in members)]
        required = profile_plan(arms=historical_arms)
    else:
        required = profile_plan(cohort=cohort)
    if arm.get("phase") in GROUP_C_PHASES:
        # C1/C2 have their own enabling configuration and acceptance file. The scientific
        # assets for these profiles need only exist when that phase is actually released.
        from src.evaluation.campaign import benchmark_plan
        required += profile_plan([a for a in benchmark_plan(cohort=cohort) if a["phase"] == arm["phase"]])
    wide = kumo_override_arm(required)
    group_c_inputs = {}
    previous_dataset, raw, parent, samples = None, None, None, {}
    for planned in sorted(required, key=lambda a: a["dataset_id"]):
        report = profiles.get(digest(planned))
        dataset = features[0][planned["dataset_id"],]
        if verify_publication and previous_dataset != planned["dataset_id"]:
            from src.data.loaders import load_dataset
            from src.evaluation.protocol import shared_splits
            previous_dataset = planned["dataset_id"]
            entry = next(e for e in load_catalog()["datasets"] if e["id"] == previous_dataset)
            raw = load_dataset(entry, allow_download=False)
            parent = shared_splits(raw, seed=main["random_seed"])
            if parent["split_id"] != dataset["split_id"]:
                raise ValueError("Published 0B source or shared split differs from preflight")
            samples = {}
        expected_ids, expected_hashes = expected_artifacts(features, planned)
        roles = {"outer_test"}
        if _needs_inner(planned, dataset["task_type"]):
            roles.add("inner_oof_binary_threshold")
        if report and planned["phase"] in GROUP_C_PHASES:
            # Base receipts have no Group C artifacts. Bind each later profile to its sealed
            # hidden store or five trained bundles, checked again on durable publication.
            supplied = report.get("feature_artifact_ids") or []
            additional = [item for item in supplied if item not in expected_ids]
            n_assets = 5 if planned["phase"] == "rq3" else 1
            if len(set(additional)) != n_assets or not all(isinstance(i, str) and i for i in additional):
                raise ValueError("0B Group C profile lacks its producer artifacts")
            if verify_publication:
                key = (planned["dataset_id"], planned.get("adaptation_cell"))
                if key not in group_c_inputs:
                    if planned["phase"] == "representations":
                        from src.jev.strands_hidden import read_manifest
                        group_c_inputs[key] = [read_manifest(key[0])["artifact_id"]]
                    else:
                        from src.evaluation import group_c
                        group_c_inputs[key] = group_c.expected_artifacts(planned, raw, parent)
                if sorted(additional) != sorted(group_c_inputs[key]):
                    raise ValueError("0B Group C producer identity differs from the published store")
            expected_ids = [*expected_ids, *additional]
        sampled = planned["phase"] == "data_efficiency"
        if (
            not report or not report.get("cluster")
            or report.get("runtime_identity")
            != predictors[planned["model"]].get("runtime_identity")
            or report.get("data_identity") != dataset["data_identity"]
            or report.get("outer_fold") != profile_fold_number(planned)
            or report.get("parent_split_id") != dataset["split_id"]
            or not report.get("split_id")
            or (sampled and report["split_id"] == dataset["split_id"])
            or (not sampled and report["split_id"] != dataset["split_id"])
            or report.get("predictor_columns") != planned["predictor_columns"]
            or sorted(report.get("feature_artifact_ids", [])) != sorted(expected_ids)
            or report.get("decision_artifact_checksums") != expected_hashes
            or len(report.get("roles", [])) != len(roles)
            or {item["role"] for item in report["roles"]} != roles
            or any(
                not item.get("record_id")
                or not item.get("prediction_sha256") or item.get("rows", 0) < 1
                for item in report["roles"]
            )
        ):
            raise ValueError(
                f"0B lacks a current full-fold profile for {planned['model']}/{planned['phase']}"
            )
        # A full-fold profile scores the whole first outer fold: its test rows are one fifth of
        # the dataset (balanced folds differ by at most one row) and its inner OOF rows are the
        # complete outer training partition. A subsample is a silent row cap, not a profile.
        sizes = {item["role"]: item["rows"] for item in report["roles"]}
        total = dataset["rows"]
        training = planned["training_size"] if sampled else total - sizes["outer_test"]
        if (
            report.get("training_rows") != training or training > total - sizes["outer_test"]
            or not total // 5 <= sizes["outer_test"] <= -(-total // 5)
            or sizes.get("inner_oof_binary_threshold", total - sizes["outer_test"])
            != total - sizes["outer_test"]
        ):
            raise ValueError(
                f"0B full-fold profile rows differ from the complete first fold for {planned['model']}"
            )
        if planned == wide or planned.get("kumo_max_columns_override"):
            verify_kumo_override_profile(report, planned)
        from src.cluster.benchmark_jobs import CPU_ONLY_MODELS
        verify_profile_resources(report, planned["model"] not in CPU_ONLY_MODELS)
        if verify_publication:
            specification = parent
            if sampled:
                from src.evaluation.data_efficiency import subsample_specification
                key = (planned["training_size"], planned["draw"])
                if key not in samples:
                    classification = raw.metadata.task_type != "regression"
                    samples[key] = subsample_specification(
                        parent, raw.frame[raw.metadata.target].to_numpy() if classification else None,
                        *key, seed=load_yaml(PHASE_CONFIGS[planned["phase"]])["sampling_seed"],
                        classes=list(raw.metadata.classes) if classification else None)
                specification = samples[key]
            verify_published_profile(report, dataset=raw, specification=specification)
    return features, predictors[arm["model"]]["runtime_identity"]


def verify_inputs(evidence, dataset, features, arm, runtime, *, specification):
    """Reject a changed dataset, partition, artifact or sealed predictor before fitting."""
    (data, decisions, embeddings), expected_runtime = evidence
    dataset_id = arm["dataset_id"]
    if runtime != expected_runtime or not runtime.get("container_id"):
        raise ValueError("Predictor runtime differs from the qualified 0B image")
    qualified = data[dataset_id,]
    if dataset.source["data_identity"] != qualified["data_identity"]:
        raise ValueError("Dataset differs from the qualified 0B input")
    if specification["split_id"] != qualified["split_id"]:
        raise ValueError("Split differs from the qualified 0B partition")
    ids, checksums = expected_artifacts((data, decisions, embeddings), arm)
    from src.evaluation.features import GROUP_C_FEATURE_SETS

    if arm["feature_set"] in GROUP_C_FEATURE_SETS:  # verified C1 store / C2 fold bundles
        from src.evaluation.group_c import expected_artifacts as group_c_artifacts

        ids = [*ids, *group_c_artifacts(arm, dataset, specification)]
    if (
        sorted(features.attrs["feature_artifact_ids"]) != sorted(ids)
        or features.attrs["decision_artifact_checksums"] != checksums
    ):
        raise ValueError("Feature artifacts differ from the qualified 0B inputs")


def expected_artifacts(evidence, arm):
    from src.evaluation.features import decision_feature_set

    _, decisions, embeddings = evidence
    dataset_id = arm["dataset_id"]
    ids, checksums = [], {}
    if arm.get("provider"):
        # Group B: decision + frozen uses "both"; a text-column refit uses "per_column".
        view = decision_feature_set(arm["feature_set"])
        item = decisions[dataset_id, arm["provider"]]
        ids = [v for k, v in item["artifact_ids"].items() if view.endswith(("both", k))]
        checksums = {i: item["artifact_checksums"][i] for i in ids}
    if arm["feature_set"] == "non_text_plus_tar":
        return embeddings[dataset_id, arm["encoder"]]["artifact_ids"], {}
    if arm.get("encoder"):
        ids = [*ids, embeddings[dataset_id, arm["encoder"]]["artifact_id"]]
    return ids, checksums


def main():
    import argparse

    from src.evaluation.design import decision_scopes, decision_versions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path)
    scopes = decision_scopes()
    parser.add_argument("--cohort", choices=scopes, default="open_weight")
    parser.add_argument("--phase", choices=("core", *GROUP_C_PHASES), default="core",
                        help="Verify core Group B or a later C1/C2 release with its producer assets")
    args = parser.parse_args()
    arms = profile_plan(cohort=args.cohort)
    release_arms = arms
    if args.phase in GROUP_C_PHASES:
        from src.evaluation.campaign import benchmark_plan
        release_arms = profile_plan([a for a in benchmark_plan(cohort=args.cohort) if a["phase"] == args.phase])
        if not release_arms:
            parser.error("This cohort contains no predictor arms for the requested Group C phase")
        arms += release_arms
    if args.acceptance:
        if not release_arms:
            parser.error("This cohort contains no predictor arms to qualify")
        benchmark_evidence(args.acceptance, release_arms[0], verify_publication=True, required_cohort=args.cohort)
        print(json.dumps({"status": "reviewed_0b_evidence_valid", "required_profiles": len(arms)}))
    else:
        providers = list(scopes[args.cohort])
        print(json.dumps({"required_profiles": len(arms), "decision_cohort": args.cohort,
                          "qualified_decision_providers": providers,
                          "decision_model_versions": decision_versions(providers), "arms": arms}, indent=2))


if __name__ == "__main__":
    main()
