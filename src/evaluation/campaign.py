"""Plan the paired representation benchmark; run one selected arm only after explicit approval.

No-argument CLI is a read-only plan. Execution also requires enabled config, recorded
experiment-0 acceptance, staged runtime pins and exact feature artifact selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import UTC, datetime

from src.data.loaders import load_catalog, load_dataset
from src.data.selection import selected_entries
from src.evaluation.features import (
    DECISION_FEATURE_SETS,
    EMBEDDING_FEATURE_SETS,
    MAIN_FEATURE_SETS,
    NATIVE_FEATURE_SETS,
    add_embeddings,
    assemble_feature_set,
    combine_representations,
    decision_feature_set,
    load_feature_selection,
    original_features,
    uses_embeddings,
)
from src.evaluation.models import (
    MAIN_MODELS,
    TEXT_MODELS,
    estimator_factory,
    predictor_settings_identity,
)
from src.evaluation.protocol import evaluate_cv, shared_splits
from src.utils import paths
from src.utils.acceptance import reviewed_acceptance
from src.utils.config import load_yaml
from src.utils.events import append_event
from src.utils.provenance import code_identity
from src.utils.serialization import digest, write_json


def benchmark_plan(main=None, text=None):
    main = main or load_yaml("experiment_1/main")
    text = text or load_yaml("experiment_3/native_text")
    embedding = load_yaml("experiment_2/embeddings")
    if (
        tuple(main["candidate_models"]) != MAIN_MODELS
        or tuple(text["candidate_models"]) != TEXT_MODELS
    ):
        raise ValueError(
            "Model roster differs from the reviewed six predictors/three text baselines"
        )
    if tuple(main["candidate_combinations"]) != DECISION_FEATURE_SETS:
        raise ValueError("Experiment 1 has exactly four decision feature configurations")
    semantics = load_yaml("dataset_semantics")["datasets"]
    rows = []
    from src.evaluation.design import pca_grid

    def add(entry, model, feature_set, phase, **extra):
        rows.append(
            dict(
                dataset_id=entry["id"],
                model=model,
                feature_set=feature_set,
                provider=None,
                phase=phase,
                **extra,
            )
        )

    def arms(entry, models, combinations, phase, **extra):
        for model in models:
            add(entry, model, combinations[0], phase, **extra)
            for provider in main["candidate_extractors"]:
                for feature_set in combinations[1:]:
                    add(entry, model, feature_set, phase, **extra)
                    rows[-1]["provider"] = provider

    for entry in selected_entries(load_catalog(), main):
        arms(entry, MAIN_MODELS, DECISION_FEATURE_SETS, "experiment_1")
        for encoder in embedding["encoders"]:
            for scope, components in pca_grid():
                for model in MAIN_MODELS:
                    extra = dict(encoder=encoder, pca_scope=scope, pca_components=components)
                    add(entry, model, EMBEDDING_FEATURE_SETS[0], "experiment_2", **extra)
        for model in TEXT_MODELS:
            add(entry, model, NATIVE_FEATURE_SETS[0], "experiment_3")
    from types import SimpleNamespace

    from src.evaluation.widths import width_status

    entries = {e["id"]: e for e in selected_entries(load_catalog(), main)}
    for row in rows:
        entry = entries[row["dataset_id"]]
        stats, task = entry["paper_summary"], semantics[row["dataset_id"]]
        metadata = SimpleNamespace(
            text_columns=range(stats["text_columns"]),
            non_text_columns=range(stats["non_text_columns"]),
            classes=task.get("classes", []),
            task_type=task["task_type"],
        )
        row.update(width_status(metadata, row))
    return rows


def run_arm(arm, *, allow_model_runs=False, profile=False):
    config_names = {
        "experiment_1": "main",
        "experiment_2": "embeddings",
        "experiment_3": "native_text",
    }
    cfg = load_yaml(arm["phase"] + "/" + config_names[arm["phase"]])
    enabled = (
        load_yaml("experiment_0/debug")["full_fold_profile_enabled"] if profile else cfg["enabled"]
    )
    if allow_model_runs is not True or enabled is not True:
        raise RuntimeError("Benchmark execution is disabled; plan only")
    from src.cluster.freeze import check_if_present

    check_if_present()
    if not profile:
        from src.evaluation.acceptance import evaluation_identity

        reviewed_acceptance(
            cfg.get("experiment_0_acceptance"), stage="after_extraction",
            expected={"evaluation_identity": evaluation_identity()},
        )
    if arm not in benchmark_plan():
        raise ValueError("Arm is outside the selected datasets/models/feature combinations")
    catalog = load_yaml("experiment_1/models")
    settings = copy.deepcopy(catalog["models"][arm["model"]]["runtime"])
    settings["threads"] = min(
        settings["threads"], int(os.environ.get("SLURM_CPUS_PER_TASK", settings["threads"]))
    )
    settings["seed"] = cfg["random_seed"]
    if arm["model"] == "tabstar":
        settings["output_phase"] = "experiment_0" if profile else arm["phase"]
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
            "src/evaluation/ag_mm.py",
            "src/evaluation/widths.py",
            "src/evaluation/design.py",
            "src/evaluation/pca_cache.py",
            "src/evaluation/prepare_pca.py",
        ]
    )
    entry = next(e for e in load_catalog()["datasets"] if e["id"] == arm["dataset_id"])
    dataset = load_dataset(entry, allow_download=False)
    from src.evaluation.widths import width_status

    status = width_status(dataset.metadata, arm)
    if status["width_status"] == "unsupported_width":
        raise ValueError(f"Unsupported width: {status}; no columns are silently discarded")
    decision_set = decision_feature_set(arm["feature_set"])
    providers = [arm["provider"]] if arm["provider"] else []
    selectors, additions = {}, []
    for provider in providers:
        selector = cfg["feature_selectors"].get(provider)
        if not selector or any(
            selector.get(k) is None
            for k in ("provider", "model", "version", "question_bank_id", "question_variant")
        ):
            raise ValueError(
                "Select an exact completed provider/checkpoint/question feature recipe"
            )
        if selector["provider"] != provider:
            raise ValueError("Feature selector provider differs from requested arm")
        modes = (
            ("per_column", "joint")
            if decision_set == "non_text_plus_both"
            else ("joint",)
            if decision_set == "non_text_plus_joint"
            else ("per_column",)
        )
        blocks = load_feature_selection(paths.dataset_cache_path(entry["id"]), selector, modes)
        additions.append(
            (
                provider,
                assemble_feature_set(
                    dataset, blocks, decision_set, selector=selector, view=arm.get("view", "main")
                ),
            )
        )
        selectors[provider] = selector
    base = (
        original_features(dataset)
        if arm["model"] in TEXT_MODELS
        else assemble_feature_set(dataset, [], "non_text")
    )
    features = combine_representations(base, additions) if additions else base
    selector = selectors.get(arm["provider"]) if arm["provider"] else selectors or None
    embedding_cfg = None
    if uses_embeddings(arm["feature_set"]):
        from src.data.text_embeddings import encoder_config, load

        embedding_cfg = encoder_config(arm["encoder"])
        embedding_cfg["pca_components_per_column"] = arm["pca_components"]
        embedding_cfg["pca_scope"] = arm["pca_scope"]
        embeddings = load(dataset, embedding_cfg)
        embedding_artifacts = embeddings.attrs["feature_artifact_ids"]
        if arm["model"] not in embedding_cfg["models"]:
            raise ValueError("Predictor is not enabled for the embedding feature configurations")
        features = add_embeddings(features, embeddings)
    factory = estimator_factory(arm["model"], dataset.metadata, settings, allow_model_runs=True)
    representation = {
        "missingness_policy": features.attrs.get("missingness_policy"),
        "uncertainty_summary": None,
        "provider": arm["provider"],
        "providers": providers,
        "selector": selector,
        "control": None,
        "view": "main",
        "embedding": embedding_cfg,
    }
    if embedding_cfg:
        from src.evaluation.embedding_control import EmbeddingControl

        base_factory = factory

        def factory():
            from src.evaluation.pca_cache import cache_path

            return EmbeddingControl(
                base_factory(),
                features.attrs["embedding_groups"],
                components=embedding_cfg["pca_components_per_column"],
                seed=cfg["random_seed"],
                scope=embedding_cfg["pca_scope"],
                prepared_cache=cache_path(entry["id"], arm["encoder"]),
                embedding_artifacts=embedding_artifacts,
            )

    synchronize = None
    if settings["device"].startswith("cuda"):
        import torch

        synchronize = torch.cuda.synchronize
    phase = "experiment_0" if profile else arm["phase"]
    log = paths.logs_dir(phase) / "benchmark.log"
    append_event(log, "arm_started", **arm)
    try:
        records = evaluate_cv(
            dataset,
            features,
            factory,
            specification=shared_splits(dataset, seed=cfg["random_seed"], folds=5),
            model_id=arm["model"],
            checkpoint_id=predictor_settings_identity(settings),
            feature_set=arm["feature_set"],
            allow_model_runs=True,
            inner_folds=3,
            phase=phase,
            synchronize=synchronize,
            device=settings["device"],
            estimator_settings=settings,
            representation_recipe=representation,
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
    parser.add_argument(
        "--phase", choices=("experiment_1", "experiment_2", "experiment_3")
    )
    parser.add_argument("--encoder")
    parser.add_argument("--pca-components", type=int)
    parser.add_argument("--pca-scope", choices=("per_column", "joint"))
    parser.add_argument("--model", choices=MAIN_MODELS + TEXT_MODELS)
    parser.add_argument(
        "--feature-set",
        choices=MAIN_FEATURE_SETS + NATIVE_FEATURE_SETS,
    )
    parser.add_argument("--provider", choices=tuple(load_yaml("extractors")["candidates"]))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--profile-fold",
        action="store_true",
        help="Experiment 0B: full first outer fold, no row cap",
    )
    args = parser.parse_args()
    plan = benchmark_plan()
    selected = [
        a
        for a in plan
        if all(
            v is None or a.get(k) == v
            for k, v in {
                "phase": args.phase,
                "encoder": args.encoder,
                "pca_components": args.pca_components,
                "pca_scope": args.pca_scope,
                "dataset_id": args.dataset,
                "model": args.model,
                "feature_set": args.feature_set,
                "provider": args.provider,
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
                    "feature_sets_by_experiment": {
                        "experiment_1": DECISION_FEATURE_SETS,
                        "experiment_2": EMBEDDING_FEATURE_SETS,
                        "experiment_3": NATIVE_FEATURE_SETS,
                    },
                    "configurations_by_experiment": {
                        phase: sum(a["phase"] == phase for a in selected)
                        for phase in sorted({a["phase"] for a in selected})
                    },
                    "original_text_models": TEXT_MODELS,
                    "outer_folds": 5,
                    "inner_binary_threshold_folds": 3,
                    "selected_arm_count": len(selected),
                    "unsupported_width_arms": sum(
                        a["width_status"] == "unsupported_width" for a in selected
                    ),
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
