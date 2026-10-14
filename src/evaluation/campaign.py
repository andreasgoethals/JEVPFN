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
    assemble_feature_set,
    load_feature_selection,
    original_features,
)
from src.evaluation.models import MAIN_MODELS, TEXT_MODELS, estimator_factory
from src.evaluation.protocol import evaluate_cv, shared_splits
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.events import append_event
from src.utils.provenance import code_identity
from src.utils.serialization import digest, write_json


def benchmark_plan(main=None, text=None):
    main = main or load_yaml("experiment_1/main")
    text = text or load_yaml("experiment_2/native_text")
    if (
        tuple(main["candidate_models"]) != MAIN_MODELS
        or tuple(text["candidate_models"]) != TEXT_MODELS
    ):
        raise ValueError("Model roster differs from the reviewed six predictors/two text baselines")
    if tuple(main["candidate_combinations"]) != MAIN_FEATURE_SETS:
        raise ValueError("The main experiment has exactly four feature configurations")
    rows = []
    for entry in selected_entries(load_catalog(), main):
        for model in MAIN_MODELS:
            # Reuse the no-extractor baseline across all selected decision-model comparisons.
            rows.append(
                {
                    "dataset_id": entry["id"],
                    "model": model,
                    "feature_set": "non_text",
                    "provider": None,
                    "phase": "experiment_1",
                }
            )
            for provider in main["candidate_extractors"]:
                for arm in MAIN_FEATURE_SETS[1:]:
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
    return rows


def run_arm(arm, *, allow_model_runs=False):
    cfg = load_yaml(
        "experiment_2/native_text" if arm["model"] in TEXT_MODELS else "experiment_1/main"
    )
    if allow_model_runs is not True or cfg["enabled"] is not True:
        raise RuntimeError("Benchmark execution is disabled; plan only")
    if not cfg.get("experiment_0_acceptance"):
        raise ValueError("Record accepted experiment-0 evidence before running benchmark models")
    if arm not in benchmark_plan():
        raise ValueError("Arm is outside the selected datasets/models/feature combinations")
    catalog = load_yaml("experiment_1/models")
    settings = copy.deepcopy(catalog["models"][arm["model"]]["runtime"])
    settings["seed"] = cfg["random_seed"]
    settings["code_sha256"] = code_identity()
    entry = next(e for e in load_catalog()["datasets"] if e["id"] == arm["dataset_id"])
    dataset = load_dataset(entry, allow_download=False)
    blocks = []
    selector = None
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
            if arm["feature_set"] == "non_text_plus_both"
            else ("joint",)
            if arm["feature_set"] == "non_text_plus_joint"
            else ("per_column",)
        )
        blocks = load_feature_selection(paths.dataset_cache_path(entry["id"]), selector, modes)
    features = (
        original_features(dataset)
        if arm["model"] in TEXT_MODELS
        else assemble_feature_set(dataset, blocks, arm["feature_set"], selector=selector)
    )
    factory = estimator_factory(arm["model"], dataset.metadata, settings, allow_model_runs=True)
    synchronize = None
    if settings["device"].startswith("cuda"):
        import torch

        synchronize = torch.cuda.synchronize
    log = paths.logs_dir(arm["phase"]) / "benchmark.jsonl"
    append_event(log, "arm_started", **arm)
    try:
        records = evaluate_cv(
            dataset,
            features,
            factory,
            specification=shared_splits(dataset, seed=cfg["random_seed"], folds=5),
            model_id=arm["model"],
            checkpoint_id=digest(settings),
            feature_set=arm["feature_set"],
            allow_model_runs=True,
            inner_folds=3,
            phase=arm["phase"],
            synchronize=synchronize,
            device=settings["device"],
            estimator_settings=settings,
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
        write_json(
            paths.manifests_dir(arm["phase"]) / "failures" / f"{digest(failure)}.json", failure
        )
        append_event(log, "arm_failed", error_type=type(exc).__name__, **arm)
        raise
    append_event(log, "arm_completed", records=len(records), **arm)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--model", choices=MAIN_MODELS + TEXT_MODELS)
    parser.add_argument("--feature-set", choices=MAIN_FEATURE_SETS + ("original_features",))
    parser.add_argument("--provider", choices=tuple(load_yaml("extractors")["candidates"]))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = benchmark_plan()
    selected = [
        a
        for a in plan
        if all(
            v is None or a[k] == v
            for k, v in {
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
        run_arm(selected[0], allow_model_runs=True)
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
