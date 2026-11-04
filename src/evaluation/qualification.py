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
    "experiment_3": "experiment_3/native_text",
}


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
    """Profile the tall and wide extremes for each predictor/representation phase."""
    from src.evaluation.campaign import benchmark_plan

    rows = {e["id"]: e["paper_summary"]["rows"] for e in load_catalog()["datasets"]}
    groups = {}
    for arm in benchmark_plan() if arms is None else arms:
        if arm["width_status"] != "unsupported_width":
            groups.setdefault((arm["phase"], arm["model"]), []).append(arm)

    selected = {}
    for group in groups.values():

        def height(arm):
            return rows[arm["dataset_id"]]

        def width(arm):
            return arm["predictor_columns"]

        for arm in (
            max(group, key=lambda a: (height(a), width(a))),
            max(group, key=lambda a: (width(a), height(a))),
        ):
            selected[digest(arm)] = arm
    return list(selected.values())


def source_leakage_diagnostics(dataset, splits):
    """Count exact feature-row overlap and flag exact raw target copies."""
    import pandas as pd

    predictors = dataset.features
    target = dataset.frame[dataset.metadata.target]
    identities = pd.util.hash_pandas_object(predictors, index=False)
    overlap = [
        int(identities.iloc[fold["test"]].isin(identities.iloc[fold["train"]]).sum())
        for fold in splits["folds"]
    ]
    return {
        "exact_target_copies": [
            str(name) for name in predictors if predictors[name].eq(target).fillna(False).all()
        ],
        "exact_feature_row_overlap_by_fold": overlap,
        "test_rows_by_fold": [len(fold["test"]) for fold in splits["folds"]],
    }


def save_profile(arm, dataset, features, runtime, records):
    """Read back the real first-fold predictions and check their exact row identities."""
    from src.evaluation.protocol import shared_splits
    from src.evaluation.store import read_predictions

    cfg = load_yaml(PHASE_CONFIGS[arm["phase"]])
    split = shared_splits(dataset, seed=cfg["random_seed"], folds=5)
    fold = split["folds"][0]
    roles = {"outer_test": fold["test"]}
    if dataset.metadata.task_type == "binary_classification":
        roles["inner_oof_binary_threshold"] = fold["train"]
    verified = []
    for record in records:
        body = json.loads(record.read_text(encoding="utf-8"))
        meta = body["metadata"]
        role = meta["prediction_role"]
        positions = roles.pop(role)
        table = read_predictions(record)
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
    report = {
        "status": "full_fold_profile_passed",
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
        "roles": verified,
        "created_at": datetime.now(UTC).isoformat(),
    }
    write_json(
        paths.manifests_dir("experiment_0")
        / "after_extraction"
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
    if any(
        set(item.get("artifact_ids", {})) != {"joint", "per_column"}
        or len(set(item["artifact_ids"].values())) != 2
        or not all(item["artifact_ids"].values())
        or set(item.get("artifact_checksums", {})) != set(item["artifact_ids"].values())
        for item in decisions.values()
    ) or any(
        not item.get("artifact_id") or not item.get("prepared_pca_keys")
        for item in embeddings.values()
    ):
        raise ValueError("0B requires exact artifact IDs for both decision modes and every encoder")
    return data, decisions, embeddings


def verify_published_profile(report):
    """Reopen the durable records, including their Parquet payloads, at 0B review."""
    from src.evaluation.store import RecordRef, database, read_predictions

    if paths.on_vsc() and paths.staging_override() != paths.durable_project_override():
        raise ValueError("Verify published 0B profiles against durable PROJECT, outside scratch")
    db = database(report["arm"]["dataset_id"], "experiment_0")
    for item in report["roles"]:
        record = RecordRef(db, item["record_id"])
        saved = json.loads(record.read_text())
        if (
            saved["metadata"]["prediction_role"] != item["role"]
            or saved["metadata"]["split_id"] != report["split_id"]
            or saved["predictions"]["sha256"] != item["prediction_sha256"]
            or len(read_predictions(record)) != item["rows"]
        ):
            raise ValueError("Published 0B predictions differ from the profile receipt")


def benchmark_evidence(filename, arm, *, verify_publication=False):
    """Checksums alone cannot make a partial or synthetic-only report into acceptance."""
    from src.cluster.predictor_contracts import required_fixture_cases
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
            and r.get("contract_version") == 6
            and r.get("fixtures")
            and all(f.get("resume_without_refitting") for f in r["fixtures"])
        )
    }
    if set(predictors) != set(MAIN_MODELS + TEXT_MODELS):
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
    for planned in profile_plan():
        report = profiles.get(digest(planned))
        dataset = features[0][planned["dataset_id"],]
        expected_ids, expected_hashes = expected_artifacts(features, planned)
        roles = {"outer_test"}
        if dataset["task_type"] == "binary_classification":
            roles.add("inner_oof_binary_threshold")
        if (
            not report
            or report.get("runtime_identity")
            != predictors[planned["model"]].get("runtime_identity")
            or report.get("data_identity") != dataset["data_identity"]
            or report.get("split_id") != dataset["split_id"]
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


def verify_inputs(evidence, dataset, features, arm, runtime):
    """Reject a changed dataset, artifact or sealed predictor before fitting."""
    (data, decisions, embeddings), expected_runtime = evidence
    dataset_id = arm["dataset_id"]
    if runtime != expected_runtime or not runtime.get("container_id"):
        raise ValueError("Predictor runtime differs from the qualified 0B image")
    if dataset.source["data_identity"] != data[dataset_id,]["data_identity"]:
        raise ValueError("Dataset differs from the qualified 0B input")
    ids, checksums = expected_artifacts((data, decisions, embeddings), arm)
    if (
        sorted(features.attrs["feature_artifact_ids"]) != sorted(ids)
        or features.attrs["decision_artifact_checksums"] != checksums
    ):
        raise ValueError("Feature artifacts differ from the qualified 0B inputs")


def expected_artifacts(evidence, arm):
    _, decisions, embeddings = evidence
    dataset_id = arm["dataset_id"]
    if arm.get("provider"):
        item = decisions[dataset_id, arm["provider"]]
        ids = [
            v for k, v in item["artifact_ids"].items() if arm["feature_set"].endswith(("both", k))
        ]
        return ids, {i: item["artifact_checksums"][i] for i in ids}
    if arm.get("encoder"):
        return [embeddings[dataset_id, arm["encoder"]]["artifact_id"]], {}
    return [], {}


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
