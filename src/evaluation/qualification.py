"""Require complete, current 0B evidence before a benchmark arm can fit."""

import importlib.metadata
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from src.data.loaders import load_catalog
from src.data.selection import selected_entries
from src.evaluation.acceptance import evaluation_identity
from src.utils import paths
from src.utils.acceptance import reviewed_acceptance
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
#: 0B full-fold profiles cover the full-size pipelines. B3 refits are narrower copies of the
#: per-column decision arms and B4 arms train on at most 16,384 sampled rows, so both are
#: covered by the Experiment 1-3 profiles of the same predictor and pipeline. Group C arms are
#: copies of profiled pipelines too: C1 is per-group PCA of precomputed vectors (as frozen
#: embeddings) and C2 adds a B1-width joint probability block.
PROFILED_PHASES = ("experiment_1", "experiment_2", "experiment_3", "experiment_4")


def decision_selector(config, provider):
    selector = config["feature_selectors"].get(provider)
    fields = ("model", "version", "question_bank_id", "question_variant", "settings_sha256")
    if (
        not selector
        or selector.get("provider") != provider
        or any(not selector.get(k) for k in fields)
    ):
        raise ValueError(f"Select the exact model, settings and question recipe for {provider}")
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


def profile_plan(arms=None):
    """Profile shape extremes and the real inner-OOF workload for each applicable recipe."""
    from src.evaluation.campaign import benchmark_plan

    rows = {e["id"]: e["paper_summary"]["rows"] for e in load_catalog()["datasets"]}
    semantics = load_yaml("dataset_semantics")["datasets"]
    encoders = load_yaml("feature_creation/embedding_models")["models"]
    groups = {}
    for arm in benchmark_plan() if arms is None else arms:
        if arm["width_status"] != "unsupported_width" and arm["phase"] in PROFILED_PHASES:
            key = (arm["phase"], arm["model"], arm.get("pca_scope"), arm.get("pca_components"))
            groups.setdefault(key, []).append(arm)

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
                family = (arm["feature_set"], arm.get("provider") == "jev")
            elif arm["phase"] == "experiment_2":
                family = (arm.get("encoder"),)
            else:
                continue
            key = (rows[arm["dataset_id"]], arm["dataset_id"], arm.get("provider") or "")
            if family not in paths_to_check or key < paths_to_check[family][0]:
                paths_to_check[family] = (key, arm)
        for _, arm in paths_to_check.values():
            selected[digest(arm)] = arm
    return list(selected.values())


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
    expected = (arm["model"], arm["dataset_id"], arm["feature_set"], arm.get("provider"),
                arm.get("encoder"), arm.get("pca_scope"), arm.get("pca_components"),
                arm.get("view", "main"))
    actual = (metadata["model_id"], metadata["dataset_id"], metadata["feature_set"],
              recipe.get("provider"), embedding.get("encoder_id"), embedding.get("pca_scope"),
              embedding.get("pca_components_per_column"), recipe.get("view", "main"))
    if (expected != actual or sorted(metadata["feature_artifact_ids"]) != sorted(artifact_ids)
            or metadata["estimator_settings"].get("execution_runtime") != runtime):
        raise ValueError("Published 0B predictions differ from the planned fold recipe, runtime or input artifacts")


def save_profile(arm, dataset, features, runtime, records):
    """Read back the real first-fold predictions and check their exact row identities."""
    from src.evaluation.protocol import shared_splits
    from src.evaluation.store import read_predictions

    cfg = load_yaml(PHASE_CONFIGS[arm["phase"]])
    split = shared_splits(dataset, seed=cfg["random_seed"], folds=5)
    fold = split["folds"][0]
    roles = {"outer_test": fold["test"]}
    if dataset.metadata.task_type == "binary_classification" and arm["phase"] != "experiment_3":
        roles["inner_oof_binary_threshold"] = fold["train"]
    verified = []
    fold_ids = set()
    for record in records:
        body = json.loads(record.read_text(encoding="utf-8"))
        meta = body["metadata"]
        role = meta["prediction_role"]
        fold_ids.add(meta["fold_recipe_id"])
        positions = roles.pop(role)
        table = read_predictions(record)
        if meta.get("representation_feature_count") != arm["predictor_columns"]:
            raise ValueError("Full-fold profile predictor width differs from the planned recipe")
        verify_arm_recipe(meta, arm, features.attrs["feature_artifact_ids"], runtime)
        if (
            meta["split_id"] != split["split_id"]
            or meta["outer_fold"] != 0
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
    report = {
        "status": "full_fold_profile_passed",
        "fold_recipe_id": fold_ids.pop(),
        "evaluation_identity": evaluation_identity(),
        "arm": arm,
        "runtime_identity": runtime,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
        "data_identity": dataset.source["data_identity"],
        "feature_artifact_ids": features.attrs["feature_artifact_ids"],
        "decision_artifact_checksums": features.attrs["decision_artifact_checksums"],
        "split_id": split["split_id"],
        "training_rows": len(fold["train"]),
        "predictor_columns": arm["predictor_columns"],
        "roles": verified,
        "created_at": datetime.now(UTC).isoformat(),
    }
    write_json(
        paths.phase_path("experiment_0", "manifests", "after_extraction")
        / f"profile_{arm['model']}_{uuid4().hex}.json",
        report,
    )


def _complete(rows, keys, expected, label):
    seen = {tuple(row[k] for k in keys): row for row in rows}
    if (
        len(seen) != len(rows)
        or set(seen) != expected
        or any(row.get("status") != "passed" for row in rows)
    ):
        raise ValueError(f"0B requires complete passing {label} evidence")
    return seen


def feature_evidence(report):
    """Check coverage against the current selected grid, including all three encoders."""
    main = load_yaml("experiment_1/main")
    entries = selected_entries(load_catalog(), main)
    datasets = {entry["id"] for entry in entries}
    if not datasets:
        raise ValueError("0B requires selected datasets")
    data = _complete(
        report.get("datasets", []), ("dataset_id",), {(d,) for d in datasets}, "dataset"
    )
    decisions = _complete(
        report.get("decision_artifacts", []),
        ("dataset_id", "provider"),
        {(d, p) for d in datasets for p in main["candidate_extractors"]},
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
        ids = item.get("artifact_ids", {})
        sums = item.get("artifact_checksums", {})
        if (set(ids) != {"joint", "per_column"} or not valid_ids(list(ids.values()), 2)
                or set(sums) != set(ids.values()) or not all(sums.values())):
            raise ValueError("0B requires exact artifact IDs and checksums for both decision modes")
    for (dataset_id, _), item in embeddings.items():
        inner = 4 if data[dataset_id,]["task_type"] == "binary_classification" else 1
        if not item.get("artifact_id") or item.get("prepared_pca_keys") != inner * outer_pca_keys(dataset_id):
            raise ValueError("0B requires exact artifact IDs and complete frozen PCA partition counts")
    from src.evaluation.tar import ENCODER
    tar = _complete(report.get("tar_artifacts", []), ("dataset_id", "encoder_id"),
                    {(d, ENCODER) for d in datasets}, "TAR fold feature")
    for key, item in tar.items():
        if (not valid_ids(item.get("artifact_ids"), 5)
                or not valid_ids(item.get("prepared_pca_keys"), outer_pca_keys(key[0]))):
            raise ValueError("0B requires all five TAR folds and their distinct PCA parameters")
        embeddings[key] = item
    return data, decisions, embeddings


def verify_published_profile(report):
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
        if (
            saved["metadata"]["prediction_role"] != item["role"]
            or saved["metadata"]["outer_fold"] != 0
            or saved["metadata"]["data_identity"] != report["data_identity"]
            or saved["metadata"]["model_id"] != report["arm"]["model"]
            or saved["metadata"]["feature_set"] != report["arm"]["feature_set"]
            or saved["metadata"]["split_id"] != report["split_id"]
            or saved["metadata"].get("representation_feature_count") != report["predictor_columns"]
            or saved["predictions"]["sha256"] != item["prediction_sha256"]
            or len(read_predictions(record)) != item["rows"]
        ):
            raise ValueError("Published 0B predictions differ from the profile receipt")
    if sorted(registered) != sorted(item["record_id"] for item in report["roles"]):
        raise ValueError("0B profile omits or duplicates a completed fold record")


def benchmark_evidence(filename, arm, *, verify_publication=False):
    """Checksums alone cannot make a partial or synthetic-only report into acceptance."""
    from src.cluster.predictor_contracts import CONTRACT_VERSION, required_fixture_cases
    from src.evaluation.models import MAIN_MODELS, TEXT_MODELS

    current = evaluation_identity()
    approval = reviewed_acceptance(
        filename, stage="after_extraction", expected={"evaluation_identity": current}
    )
    root = Path(filename).parent
    reports = [
        json.loads((root / item["path"]).read_text(encoding="utf-8"))
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
        raise ValueError("0B requires passing real predictor contracts for all nine models")
    for model, report in predictors.items():
        cases = {(f["task_type"], f["classes"], f["feature_case"]) for f in report["fixtures"]}
        if (
            len(cases) != len(report["fixtures"])
            or cases != required_fixture_cases(model)
            or not report.get("runtime_identity", {}).get("container_id")
        ):
            raise ValueError(f"0B predictor fixtures or sealed runtime are incomplete for {model}")
    profiles = {digest(r["arm"]): r for r in live if r.get("status") == "full_fold_profile_passed"}
    if len(profiles) != sum(r.get("status") == "full_fold_profile_passed" for r in live):
        raise ValueError("Select exactly one full-fold profile per planned arm")
    for planned in profile_plan():
        report = profiles.get(digest(planned))
        dataset = features[0][planned["dataset_id"],]
        expected_ids, expected_hashes = expected_artifacts(features, planned)
        roles = {"outer_test"}
        if dataset["task_type"] == "binary_classification" and planned["phase"] != "experiment_3":
            roles.add("inner_oof_binary_threshold")
        if (
            not report
            or report.get("runtime_identity")
            != predictors[planned["model"]].get("runtime_identity")
            or report.get("data_identity") != dataset["data_identity"]
            or report.get("split_id") != dataset["split_id"]
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
        if verify_publication:
            verify_published_profile(report)
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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path)
    args = parser.parse_args()
    arms = profile_plan()
    if args.acceptance:
        benchmark_evidence(args.acceptance, arms[0], verify_publication=True)
        print(json.dumps({"status": "reviewed_0b_evidence_valid", "required_profiles": len(arms)}))
    else:
        print(json.dumps({"required_profiles": len(arms), "arms": arms}, indent=2))


if __name__ == "__main__":
    main()
