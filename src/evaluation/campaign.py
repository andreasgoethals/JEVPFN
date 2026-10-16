"""Plan the paired representation benchmark; run one selected arm only after explicit approval.

No-argument CLI is a read-only plan. Execution also requires enabled config, recorded
experiment-0 acceptance, staged runtime pins and exact feature artifact selection.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import UTC, datetime

from src.data.loaders import load_catalog, load_dataset
from src.data.selection import selected_entries
from src.evaluation.features import (
    MAIN_FEATURE_SETS,
    PROVIDER_INDEPENDENT_SETS,
    add_embeddings,
    assemble_feature_set,
    decision_feature_set,
    load_feature_selection,
    original_features,
    uses_embeddings,
)
from src.evaluation.models import MAIN_MODELS, TEXT_MODELS, estimator_factory
from src.evaluation.protocol import evaluate_cv, shared_splits
from src.utils import paths
from src.utils.acceptance import reviewed_acceptance
from src.utils.config import load_yaml
from src.utils.events import append_event
from src.utils.provenance import code_identity
from src.utils.serialization import digest, write_json


def benchmark_plan(main=None, text=None, *, include_controls=False):
    main = main or load_yaml("experiment_1/main")
    text = text or load_yaml("experiment_2/native_text")
    if (
        tuple(main["candidate_models"]) != MAIN_MODELS
        or tuple(text["candidate_models"]) != TEXT_MODELS
    ):
        raise ValueError("Model roster differs from the reviewed six predictors/two text baselines")
    if tuple(main["candidate_combinations"]) != MAIN_FEATURE_SETS:
        raise ValueError("The main experiment has exactly eight feature configurations")
    rows = []
    for entry in selected_entries(load_catalog(), main):
        for model in MAIN_MODELS:
            # Both baselines are independent of the decision extractor and fitted once.
            rows.extend(
                {
                    "dataset_id": entry["id"],
                    "model": model,
                    "feature_set": arm,
                    "provider": None,
                    "phase": "experiment_1",
                }
                for arm in PROVIDER_INDEPENDENT_SETS
            )
            for provider in main["candidate_extractors"]:
                for arm in MAIN_FEATURE_SETS:
                    if arm in PROVIDER_INDEPENDENT_SETS:
                        continue
                    rows.append(
                        {
                            "dataset_id": entry["id"],
                            "model": model,
                            "feature_set": arm,
                            "provider": provider,
                            "phase": "experiment_1",
                        }
                    )
        for model in TEXT_MODELS:
            rows.append(
                {
                    "dataset_id": entry["id"],
                    "model": model,
                    "feature_set": "original_features",
                    "provider": None,
                    "phase": "experiment_2",
                }
            )
    if include_controls:
        cfg = load_yaml("experiment_3/output_ablation")
        semantics = load_yaml("dataset_semantics")["datasets"]
        for entry in selected_entries(load_catalog(), main):
            for model in cfg["candidate_models"]:
                rows.append(
                    dict(
                        dataset_id=entry["id"],
                        model=model,
                        feature_set="non_text_plus_text_statistics",
                        provider=None,
                        phase="experiment_3",
                        control="text_statistics",
                        view="main",
                    )
                )
                for provider in main["candidate_extractors"]:
                    task = semantics[entry["id"]]["task_type"]
                    controls = [("permuted", "main")]
                    if task == "regression":
                        controls.append(("retained_view", "probabilities"))
                    if task != "binary_classification":
                        controls.append(("retained_view", "probabilities_plus_confidence"))
                    for control, view in controls:
                        rows.append(
                            dict(
                                dataset_id=entry["id"],
                                model=model,
                                feature_set="non_text_plus_both",
                                provider=provider,
                                phase="experiment_3",
                                control=control,
                                view=view,
                            )
                        )
    return rows


def run_arm(arm, *, allow_model_runs=False, profile=False):
    cfg = (
        load_yaml("experiment_3/output_ablation")
        if arm["phase"] == "experiment_3"
        else load_yaml(
            "experiment_2/native_text" if arm["model"] in TEXT_MODELS else "experiment_1/main"
        )
    )
    enabled = (
        load_yaml("experiment_0/debug")["full_fold_profile_enabled"] if profile else cfg["enabled"]
    )
    if allow_model_runs is not True or enabled is not True:
        raise RuntimeError("Benchmark execution is disabled; plan only")
    from src.cluster.freeze import check_if_present

    check_if_present()
    if not profile:
        reviewed_acceptance(cfg.get("experiment_0_acceptance"), stage="after_extraction")
    if arm not in benchmark_plan(include_controls=True):
        raise ValueError("Arm is outside the selected datasets/models/feature combinations")
    catalog = load_yaml("experiment_1/models")
    settings = copy.deepcopy(catalog["models"][arm["model"]]["runtime"])
    settings["seed"] = cfg["random_seed"]
    settings["code_sha256"] = code_identity(
        [
            "src/evaluation/models.py",
            "src/evaluation/protocol.py",
            "src/evaluation/features.py",
            "src/evaluation/metrics.py",
            "src/evaluation/records.py",
            "src/evaluation/text_overlap.py",
            "src/evaluation/campaign.py",
            "src/evaluation/embedding_control.py",
        ]
    )
    entry = next(e for e in load_catalog()["datasets"] if e["id"] == arm["dataset_id"])
    dataset = load_dataset(entry, allow_download=False)
    blocks = []
    selector = None
    decision_set = decision_feature_set(arm["feature_set"])
    if arm["provider"] is not None:
        selector = cfg["feature_selectors"].get(arm["provider"])
        if not selector or any(
            selector.get(k) is None
            for k in ("provider", "model", "version", "question_bank_id", "question_variant")
        ):
            raise ValueError(
                "Select an exact completed provider/checkpoint/question feature recipe"
            )
        if selector["provider"] != arm["provider"]:
            raise ValueError("Feature selector provider differs from the requested benchmark arm")
        modes = (
            ("per_column", "joint")
            if decision_set == "non_text_plus_both"
            else ("joint",)
            if decision_set == "non_text_plus_joint"
            else ("per_column",)
        )
        blocks = load_feature_selection(paths.dataset_cache_path(entry["id"]), selector, modes)
    embedding_cfg = None
    features = (
        original_features(dataset)
        if arm["model"] in TEXT_MODELS
        else assemble_feature_set(
            dataset, blocks, decision_set, selector=selector, view=arm.get("view", "main")
        )
    )
    if uses_embeddings(arm["feature_set"]):
        from src.data.text_embeddings import load

        embedding_cfg = load_yaml("feature_creation/text_embeddings")
        embeddings = load(dataset, embedding_cfg)
        if arm["model"] not in embedding_cfg["models"]:
            raise ValueError("Predictor is not enabled for the embedding feature configurations")
        features = add_embeddings(features, embeddings)
    factory = estimator_factory(arm["model"], dataset.metadata, settings, allow_model_runs=True)
    representation = {
        "provider": arm["provider"],
        "selector": selector,
        "control": arm.get("control"),
        "view": arm.get("view", "main"),
        "embedding": embedding_cfg,
    }
    if embedding_cfg:
        from src.evaluation.embedding_control import EmbeddingControl

        base_factory = factory

        def factory():
            return EmbeddingControl(
                base_factory(),
                features.attrs["embedding_groups"],
                components=embedding_cfg["pca_components_per_column"],
                seed=cfg["random_seed"],
            )

    synchronize = None
    if settings["device"].startswith("cuda"):
        import torch

        synchronize = torch.cuda.synchronize
    phase = "experiment_0" if profile else arm["phase"]
    log = paths.logs_dir(phase) / "benchmark.jsonl"
    append_event(log, "arm_started", **arm)
    try:
        records = evaluate_cv(
            dataset,
            features,
            factory,
            specification=shared_splits(dataset, seed=cfg["random_seed"], folds=5),
            model_id=arm["model"],
            checkpoint_id=digest(settings),
            feature_set=arm["feature_set"]
            + ("__" + arm["control"] + "__" + arm["view"] if "control" in arm else ""),
            allow_model_runs=True,
            inner_folds=3,
            phase=phase,
            synchronize=synchronize,
            device=settings["device"],
            estimator_settings=settings,
            representation_recipe=representation,
            permutation_columns=[c for c in features if c not in dataset.metadata.non_text_columns]
            if arm.get("control") == "permuted"
            else None,
            fold_numbers=[0] if profile else None,
        )
    except Exception as exc:
        # Do not record raw text/row values from third-party exception messages.
        failure = {
            "arm": arm,
            "status": "failed",
            "error_type": type(exc).__name__,
            "created_utc": datetime.now(UTC).isoformat(),
            "settings": settings,
        }
        write_json(paths.manifests_dir(phase) / "failures" / f"{digest(failure)}.json", failure)
        append_event(log, "arm_failed", error_type=type(exc).__name__, **arm)
        raise
    append_event(log, "arm_completed", records=len(records), **arm)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--model", choices=MAIN_MODELS + TEXT_MODELS)
    parser.add_argument(
        "--feature-set",
        choices=MAIN_FEATURE_SETS + ("original_features", "non_text_plus_text_statistics"),
    )
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--control", choices=("text_statistics", "permuted", "retained_view"))
    parser.add_argument(
        "--view", choices=("main", "probabilities", "probabilities_plus_confidence")
    )
    parser.add_argument("--provider", choices=tuple(load_yaml("extractors")["candidates"]))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--profile-fold",
        action="store_true",
        help="Experiment 0B: full first outer fold, no row cap",
    )
    args = parser.parse_args()
    plan = benchmark_plan(include_controls=args.include_controls or args.control is not None)
    selected = [
        a
        for a in plan
        if all(
            v is None or a.get(k) == v
            for k, v in {
                "dataset_id": args.dataset,
                "model": args.model,
                "feature_set": args.feature_set,
                "provider": args.provider,
                "control": args.control,
                "view": args.view,
            }.items()
        )
    ]
    if args.execute:
        if len(selected) != 1:
            parser.error("Execution requires exactly one dataset/model/feature-set/provider arm")
        run_arm(selected[0], allow_model_runs=True, profile=args.profile_fold)
    else:
        print(
            json.dumps(
                {
                    "execution_enabled": False,
                    "datasets": 18,
                    "main_models": MAIN_MODELS,
                    "main_feature_sets": MAIN_FEATURE_SETS,
                    "original_text_models": TEXT_MODELS,
                    "outer_folds": 5,
                    "inner_binary_threshold_folds": 3,
                    "selected_arm_count": len(selected),
                    "outer_fits": 5 * len(selected),
                    "non_text_includes_categorical": True,
                    "training_row_cap": None,
                    "selected_arms": selected
                    if len(selected) <= 30
                    else "Filter with --dataset/--model/--feature-set/--provider to inspect",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
