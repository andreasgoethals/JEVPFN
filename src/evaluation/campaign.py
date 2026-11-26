"""Plan the paired representation benchmark; run one selected arm only after explicit approval.

No-argument CLI is a read-only plan. Execution also requires enabled config, recorded
experiment-0 acceptance, staged runtime pins and exact feature artifact selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime

from src.data.loaders import load_catalog, load_dataset
from src.data.selection import selected_entries
from src.evaluation.features import (
    ADAPTED,
    DECISION_FEATURE_SETS,
    DECISION_PLUS_EMBEDDING,
    EMBEDDING_FEATURE_SETS,
    GROUP_B_FEATURE_SETS,
    GROUP_C_FEATURE_SETS,
    HIDDEN_FEATURE_SETS,
    MAIN_FEATURE_SETS,
    NATIVE_FEATURE_SETS,
    PER_COLUMN_WITHOUT_ONE,
    SUPERVISED_PROBE,
    add_embeddings,
    assemble_feature_set,
    combine_representations,
    decision_feature_set,
    load_feature_selection,
    original_features,
    uses_embeddings,
    without_text_column_block,
)
from src.evaluation.models import (
    MAIN_MODELS,
    TEXT_MODELS,
    estimator_factory,
    predictor_settings_identity,
)
from src.evaluation.protocol import evaluate_cv, shared_splits
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.events import append_event
from src.utils.provenance import code_identity
from src.utils.serialization import digest, write_json

#: Group B (approved 20 November 2026; one name per experiment since 26 November 2026):
#: B1 decision features, B2 frozen embeddings (and decision + frozen), B3 TAR and the
#: label-trained classifier, B4 text-column refits, B5 training-size curves, B6 native text.
GROUP_B_PHASES = {"B1": "B1", "B2": "B2", "B3": "B3", "B4": "B4", "B5": "B5", "B6": "B6"}
#: Group C (approved 20 November 2026, Strands only): C1 hidden states, C2 learning from labels.
GROUP_C_PHASES = {"C1": "C1", "C2": "C2"}
#: The config of each phase's arms (B4's arms are its text-column refits).
ARM_CONFIGS = {"B4": "B4/text_column_refits"}
PROBE_VIEWS = ("per_column", "joint")
B2_DECISION_EMBEDDING = {"encoder": "e5_small_v2", "pca_scope": "per_column", "pca_components": 32}


def benchmark_plan(main=None, text=None, *, cohort="all"):
    from src.evaluation.design import decision_providers, incremental_cohort, pca_grid

    main = main or load_yaml("B1")
    providers = decision_providers(cohort, main)
    text = text or load_yaml("B6")
    embedding = load_yaml("B2")
    tar = load_yaml("B3")
    if (
        tuple(main["candidate_models"]) != MAIN_MODELS
        or tuple(text["candidate_models"]) != TEXT_MODELS
    ):
        raise ValueError(
            "Model roster differs from the reviewed structured predictors or native-text systems"
        )
    if tuple(main["candidate_combinations"]) != DECISION_FEATURE_SETS:
        raise ValueError("Experiment 1 has exactly four decision feature configurations")
    selected = selected_entries(load_catalog(), main)
    for config in (embedding, tar, text):
        if (
            config["random_seed"] != main["random_seed"]
            or selected_entries(load_catalog(), config) != selected
        ):
            raise ValueError("Benchmark phases must share the same datasets and split seed")
    if (tar["encoders"] != ["e5_small_v2_tar"] or tar["pca_grid"] != embedding["pca_grid"]
            or tar["binary_threshold_policy"] != "fixed_0_5" or tar["inner_encoder_refits"] != 0):
        raise ValueError("TAR must use the reviewed encoder, matched PCA grid and fixed threshold")
    semantics = load_yaml("dataset_semantics")["datasets"]
    rows = []
    grid = pca_grid()  # read once: the reviewed per-column PCA32 / joint PCA64 recipes

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
            for provider in providers:
                for feature_set in combinations[1:]:
                    add(entry, model, feature_set, phase, **extra)
                    rows[-1]["provider"] = provider

    for entry in selected:
        arms(entry, MAIN_MODELS, DECISION_FEATURE_SETS, "B1")
        for encoder in embedding["encoders"]:
            for scope, components in grid:
                for model in MAIN_MODELS:
                    extra = dict(encoder=encoder, pca_scope=scope, pca_components=components)
                    add(entry, model, EMBEDDING_FEATURE_SETS[0], "B2", **extra)
        for scope, components in grid:
            for model in MAIN_MODELS:
                add(entry, model, "non_text_plus_tar", "B3",
                    encoder="e5_small_v2_tar", pca_scope=scope, pca_components=components)
        for model in TEXT_MODELS:
            add(entry, model, NATIVE_FEATURE_SETS[0], "B6")
        # B2: each decision model's "both" block with frozen E5-small per-column PCA32.
        for model in MAIN_MODELS:
            for provider in providers:
                add(entry, model, DECISION_PLUS_EMBEDDING, "B2", **B2_DECISION_EMBEDDING)
                rows[-1]["provider"] = provider
        # B3: label-trained classifier on each frozen encoder, per text column and joint.
        for encoder in embedding["encoders"]:
            for view in PROBE_VIEWS:
                for model in MAIN_MODELS:
                    add(entry, model, SUPERVISED_PROBE, "B3", encoder=encoder,
                        probe_view=view)
        # B4: per-column decision arm refitted without one text column's block.
        if entry["paper_summary"]["text_columns"] >= 2:
            for column in range(entry["paper_summary"]["text_columns"]):
                for model in MAIN_MODELS:
                    for provider in providers:
                        add(entry, model, PER_COLUMN_WITHOUT_ONE, "B4",
                            dropped_text_column=column)
                        rows[-1]["provider"] = provider
    # B5: training-size curves (explicit, recorded subsamples; see data_efficiency).
    from src.evaluation.data_efficiency import data_efficiency_arms

    rows.extend(data_efficiency_arms(selected, MAIN_MODELS, providers))
    # Group C: C1a hidden states (12), C1b hidden + probabilities (2), C2 adapted cells (3).
    from src.evaluation.group_c import arms as group_c_arms

    rows.extend(group_c_arms(selected, MAIN_MODELS))
    if incremental_cohort(cohort):
        rows = [row for row in rows if row["phase"] in GROUP_B_PHASES and row.get("provider") in providers]
    for arm in rows:
        # Explicit view distinguishes the nine-probability regression protocol from historical
        # records whose implicit "main" view selected the mean.
        if arm.get("provider") and semantics[arm["dataset_id"]]["task_type"] == "regression":
            arm["view"] = "probabilities"
    from types import SimpleNamespace

    from src.evaluation.widths import width_status

    entries = {e["id"]: e for e in selected}
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


# Bundled tasks run many arms of one dataset in one process. Pure, deterministic
# inputs are loaded once and every arm receives its own deep copy, so no arm can
# observe another arm's mutations. Estimators are still constructed fresh per fit.
# The cache exists only inside task_cache(); a lone run_arm call loads everything fresh.
_TASK_CACHE = None


def _cached(key, load, *, copy_result=True):
    if _TASK_CACHE is None:
        return load()
    if key not in _TASK_CACHE:
        _TASK_CACHE[key] = load()
    result = _TASK_CACHE[key]
    return copy.deepcopy(result) if copy_result else result


def _retain_representation(provider, encoder, hidden_key=None):
    if _TASK_CACHE is None:
        return
    for key in list(_TASK_CACHE):
        if (key[0] == "decision" and key[1] != provider) or (
            key[0] == "embeddings" and key[1] != encoder
        ) or (key[0] == "hidden" and key != hidden_key):
            del _TASK_CACHE[key]


def _representation_storage():
    from src.cluster.scratch_stage import staging_environment

    root = paths.durable_project_override()
    if os.environ.get("SLURM_CLUSTER_NAME") == "mindwell" and paths.native_gpfs_project(root):
        return staging_environment(root / paths.PROJECT_NAME)
    return nullcontext()


def _retain_for_arm(arm):
    hidden_key = None
    if arm["feature_set"] in HIDDEN_FEATURE_SETS:
        from src.jev.strands_hidden import active_artifact_id

        with _representation_storage():
            artifact = active_artifact_id(arm["dataset_id"])
        hidden_key = ("hidden", arm["dataset_id"], arm["hidden_view"],
                      arm["hidden_position"], arm["hidden_layer"], artifact)
    _retain_representation(arm.get("provider"), arm.get("encoder"), hidden_key)
    return hidden_key


@contextmanager
def task_cache():
    """Share pure inputs between the arms of one bundled task, then discard them."""
    global _TASK_CACHE
    if _TASK_CACHE is not None:
        raise RuntimeError("Bundled task caches do not nest")
    _TASK_CACHE = {}
    try:
        yield
    finally:
        _TASK_CACHE = None


def run_arm(arm, *, allow_model_runs=False, profile=False):
    cfg = load_yaml(ARM_CONFIGS.get(arm["phase"], arm["phase"]))
    enabled = (
        load_yaml("0/debug")["full_fold_profile_enabled"] if profile else cfg["enabled"]
    )
    if allow_model_runs is not True or enabled is not True:
        raise RuntimeError("Benchmark execution is disabled; plan only")
    from src.cluster.freeze import check_if_present

    frozen = check_if_present()
    if not profile:
        if not frozen["frozen"]:
            raise ValueError("Freeze the reviewed source before executing a production benchmark arm")
        from src.evaluation.qualification import acceptance_for, benchmark_evidence

        acceptance = acceptance_for(cfg.get("gate_acceptance"), arm.get("provider"))
        qualified = _cached(
            ("evidence", acceptance, arm["phase"], arm.get("model"), arm.get("provider")),
            lambda: benchmark_evidence(acceptance, arm),
            copy_result=False,
        )
    planned = _cached(("plan_keys",), lambda: {digest(a) for a in benchmark_plan()},
                      copy_result=False)
    if digest(arm) not in planned:
        raise ValueError("Arm is outside the selected datasets/models/feature combinations")
    hidden_key = _retain_for_arm(arm)
    catalog = load_yaml("B1/models")
    settings = copy.deepcopy(catalog["models"][arm["model"]]["runtime"])
    settings["threads"] = min(
        settings["threads"], int(os.environ.get("SLURM_CPUS_PER_TASK", settings["threads"]))
    )
    settings["seed"] = cfg["random_seed"]
    if arm["model"] == "tabstar":
        settings["output_phase"] = "0" if profile else arm["phase"]
    settings["code_sha256"] = code_identity(
        [
            "src/evaluation/models.py",
            "src/evaluation/limix_preprocessing.py",
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
            "src/evaluation/tar.py",
            "src/evaluation/label_probe.py",
            "src/evaluation/data_efficiency.py",
            "src/evaluation/group_c.py",
            "src/evaluation/strands_adaptation.py",
            "src/jev/strands_hidden.py",
        ]
    )
    entry = next(e for e in load_catalog()["datasets"] if e["id"] == arm["dataset_id"])
    dataset = _cached(("dataset", entry["id"]), lambda: load_dataset(entry, allow_download=False))
    from src.evaluation.widths import width_status

    status = width_status(dataset.metadata, arm)
    if status["width_status"] == "unsupported_width":
        raise ValueError(f"Unsupported width: {status}; no columns are silently discarded")
    decision_set = decision_feature_set(arm["feature_set"])
    providers = [arm["provider"]] if arm["provider"] else []
    from src.evaluation.qualification import decision_artifact_pins, decision_selector

    selectors, additions, checksums = {}, [], {}
    for provider in providers:
        selector = decision_selector(cfg, provider, entry["id"])
        pins = decision_artifact_pins(cfg, entry["id"], provider)
        modes = (
            ("per_column", "joint")
            if decision_set == "non_text_plus_both"
            else ("joint",)
            if decision_set == "non_text_plus_joint"
            else ("per_column",)
        )
        with _representation_storage():
            blocks = [
                block
                for mode in modes
                for block in _cached(
                    ("decision", provider, entry["id"], digest(selector), digest(pins), mode),
                    lambda mode=mode, selector=selector, pins=pins: load_feature_selection(
                        paths.dataset_cache_path(entry["id"]), selector, (mode,), artifact_pins=pins,
                    ),
                )
            ]
        checksums.update({b.manifest["artifact_id"]: digest(b.manifest["sha256"]) for b in blocks})
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
    if arm["feature_set"] == PER_COLUMN_WITHOUT_ONE:
        features = without_text_column_block(features, arm["dropped_text_column"])
    hidden_arm = arm["feature_set"] in HIDDEN_FEATURE_SETS
    adapted_arm = arm["feature_set"] == ADAPTED
    if hidden_arm:
        from src.evaluation import group_c
        from src.jev.strands_hidden import vector_frame

        with _representation_storage():
            vectors = _cached(
                hidden_key,
                lambda: vector_frame(dataset, view=arm["hidden_view"], position=arm["hidden_position"],
                                     layer=arm["hidden_layer"]),
            )
            if vectors.attrs["feature_artifact_ids"] != [hidden_key[-1]]:
                raise ValueError("Hidden-state artifact changed while its view was being read")
        features = group_c.add_hidden(features, vectors)
    selector = selectors.get(arm["provider"]) if arm["provider"] else selectors or None
    embedding_cfg = None
    tar_arm = arm["feature_set"] == "non_text_plus_tar"
    probe_arm = arm["feature_set"] == SUPERVISED_PROBE
    subsampled = arm["phase"] == "B5"
    if uses_embeddings(arm["feature_set"]) and not tar_arm:
        from src.data.text_embeddings import encoder_config, load

        embedding_cfg = encoder_config(arm["encoder"])
        with _representation_storage():
            embeddings = _cached(
                ("embeddings", arm["encoder"], entry["id"], digest(embedding_cfg)),
                lambda: load(dataset, embedding_cfg),
            )
        embedding_cfg["pca_components_per_column"] = arm.get("pca_components")
        embedding_cfg["pca_scope"] = arm.get("pca_scope")
        embedding_artifacts = embeddings.attrs["feature_artifact_ids"]
        if arm["model"] not in embedding_cfg["models"]:
            raise ValueError("Predictor is not enabled for the embedding feature configurations")
        features = add_embeddings(features, embeddings)
    from src.evaluation.qualification import execution_runtime, profile_fold_number, verify_inputs

    features.attrs["decision_artifact_checksums"] = checksums
    settings["execution_runtime"] = _cached(("runtime",), execution_runtime)
    specification = parent_specification = shared_splits(dataset, seed=cfg["random_seed"], folds=5)
    if subsampled:
        from src.evaluation.data_efficiency import subsample_specification

        classification = dataset.metadata.task_type != "regression"
        specification = subsample_specification(
            parent_specification,
            dataset.frame[dataset.metadata.target].to_numpy() if classification else None,
            arm["training_size"], arm["draw"], seed=cfg["sampling_seed"],
            classes=list(dataset.metadata.classes) if classification else None,
        )
        if any(specification["subsample"]["missing_classes_by_fold"]):
            raise ValueError("A training subsample lacks a class; classes are never dropped")
    if tar_arm:
        from src.evaluation.tar import read_bundle
        bundles = [_cached(("tar", entry["id"], digest(specification["split_id"]), i),
                           lambda i=i: read_bundle(dataset, specification, i)) for i in range(5)]
        features.attrs["feature_artifact_ids"] = [b["artifact_id"] for b in bundles]
    if adapted_arm:
        from src.evaluation.strands_adaptation import read_bundle as read_adapted

        with _representation_storage():
            adapted = [_cached(("adapted", entry["id"], arm["adaptation_cell"], i),
                               lambda i=i: read_adapted(dataset, specification, i, arm["adaptation_cell"]))
                       for i in range(5)]
        features.attrs["feature_artifact_ids"] = [
            *features.attrs.get("feature_artifact_ids", []), *(b["artifact_id"] for b in adapted)]
    if not profile:
        verify_inputs(
            qualified, dataset, features, arm, settings["execution_runtime"],
            specification=parent_specification,
        )
    factory = estimator_factory(arm["model"], dataset.metadata, settings, allow_model_runs=True)
    representation = {
        "missingness_policy": features.attrs.get("missingness_policy"),
        "uncertainty_summary": None,
        "provider": arm["provider"],
        "providers": providers,
        "selector": selector,
        "control": None,
        "view": arm.get("view", "main"),
        "embedding": embedding_cfg,
    }
    if arm["feature_set"] == PER_COLUMN_WITHOUT_ONE:
        representation["dropped_text_column"] = arm["dropped_text_column"]
    if subsampled:
        representation["subsample"] = {
            k: specification["subsample"][k]
            for k in ("parent_split_id", "training_size", "draw", "method")
        }
    if probe_arm:
        from src.evaluation import label_probe

        base_factory = factory
        with _representation_storage():
            probe_cache = None if subsampled else label_probe.cache_path(entry["id"], arm["encoder"])
        representation["probe"] = {
            "version": label_probe.VERSION, "view": arm["probe_view"], "encoder_id": arm["encoder"],
            "c_grid": list(label_probe.C_GRID), "inner_folds": label_probe.INNER_FOLDS,
            "crossfit_folds": label_probe.CROSSFIT_FOLDS,
            "regression_bins": label_probe.REGRESSION_BINS,
            "fitted_on": "sampled_training_rows_inline" if subsampled else "cpu_prepared_outer_training",
        }
        representation["embedding"] = {"encoder_id": arm["encoder"], "probe_view": arm["probe_view"]}

        def factory():
            return label_probe.ProbeControl(
                base_factory(), features.attrs["embedding_groups"], view=arm["probe_view"],
                task_type=dataset.metadata.task_type, classes=list(dataset.metadata.classes),
                seed=cfg["random_seed"], prepared_cache=probe_cache,
                embedding_artifacts=embedding_artifacts,
            )
    elif embedding_cfg:
        from src.evaluation.embedding_control import EmbeddingControl
        from src.evaluation.pca_cache import cache_path

        base_factory = factory
        with _representation_storage():
            # B4 samples are new training partitions: PCA is fitted inline on the sampled rows.
            prepared_cache = None if subsampled else cache_path(entry["id"], arm["encoder"])

        def factory():
            return EmbeddingControl(
                base_factory(),
                features.attrs["embedding_groups"],
                components=embedding_cfg["pca_components_per_column"],
                seed=cfg["random_seed"],
                scope=embedding_cfg["pca_scope"],
                prepared_cache=prepared_cache,
                embedding_artifacts=embedding_artifacts,
            )

    if hidden_arm:
        base_factory = factory

        def factory():
            return group_c.hidden_control(base_factory(), features.attrs["embedding_groups"], arm,
                                          cfg["random_seed"])

        representation["hidden"] = {
            "provider": "strands", "view": arm["hidden_view"], "position": arm["hidden_position"],
            "decision_model": group_c.decision_model_for_fit([vectors.attrs.get("decision_model")]),
            "layer": arm["hidden_layer"], "pca_scope": arm["pca_scope"],
            "pca_components": arm["pca_components"], "fitted_on": "outer_training_rows_inline",
            "artifact_ids": vectors.attrs["feature_artifact_ids"],
        }
    if adapted_arm:
        from src.evaluation import group_c
        from src.evaluation.strands_adaptation import AdaptedControl

        base_factory = factory

        def factory():
            return AdaptedControl(base_factory(), dataset, specification, adapted, arm["adaptation_cell"])

        representation["adaptation"] = {
            "provider": "strands", "question": "joint", "cell": arm["adaptation_cell"],
            "decision_model": group_c.decision_model_for_fit(
                [bundle["recipe"]["sources"].get("decision_model") for bundle in adapted]),
            "crossfit_parts": 3, "artifact_ids": [b["artifact_id"] for b in adapted],
        }
    if tar_arm:
        from src.evaluation.tar import TarControl
        base_factory = factory
        def factory():
            return TarControl(base_factory(), dataset, specification, bundles, arm)
        representation["embedding"] = {
            "encoder_id": arm["encoder"], "model": "intfloat/e5-small-v2",
            "pca_scope": arm["pca_scope"], "pca_components_per_column": arm["pca_components"],
            "training": "multabench_lora_outer_partition_only",
            "input_policy": cfg["text_input"], "length_policy": cfg["text_length_policy"],
        }
    synchronize = None
    if settings["device"].startswith("cuda"):
        import torch

        synchronize = torch.cuda.synchronize
    phase = "0" if profile else arm["phase"]
    # Group B (20 November 2026) replaced permutation attribution by B4 text-column refits:
    # no shuffled re-predictions are made (protocol field_attribution.enabled is false).
    fixed_threshold = tar_arm or probe_arm or subsampled or adapted_arm
    log = paths.logs_dir(phase) / "benchmark.log"
    append_event(log, "arm_started", **arm)
    try:
        records = evaluate_cv(
            dataset,
            features,
            factory,
            specification=specification,
            model_id=arm["model"],
            checkpoint_id=predictor_settings_identity(settings),
            feature_set=arm["feature_set"],
            allow_model_runs=True,
            inner_folds=3,
            binary_threshold_policy="fixed_0_5" if fixed_threshold else "inner_oof_f1",
            phase=phase,
            synchronize=synchronize,
            device=settings["device"],
            estimator_settings=settings,
            representation_recipe=representation,
            fold_numbers=[profile_fold_number(arm)] if profile else None,
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
    if profile:
        from src.evaluation.qualification import save_profile

        save_profile(arm, dataset, features, settings["execution_runtime"], records,
                     specification=specification)
    append_event(log, "arm_completed", records=len(records), **arm)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--cohort", metavar="COHORT", default="open_weight",
                      help="Plan a registered full release, or use its <name>_added scope for new fits only.")
    parser.add_argument("--phase", choices=(*GROUP_B_PHASES, *GROUP_C_PHASES))
    parser.add_argument("--encoder")
    parser.add_argument("--pca-components", type=int)
    parser.add_argument("--pca-scope", choices=("per_column", "joint"))
    parser.add_argument("--model", choices=MAIN_MODELS + TEXT_MODELS)
    parser.add_argument(
        "--feature-set",
        choices=MAIN_FEATURE_SETS + NATIVE_FEATURE_SETS + ("non_text_plus_tar",) + GROUP_B_FEATURE_SETS
        + GROUP_C_FEATURE_SETS,
    )
    parser.add_argument("--hidden-view", choices=("per_column", "joint"))
    parser.add_argument("--hidden-position", choices=("state_end", "question_end", "answer"))
    parser.add_argument("--hidden-layer", type=int, choices=(12, 24))
    parser.add_argument("--adaptation-cell", choices=("head_only", "lora_only", "lora_head"))
    parser.add_argument("--probe-view", choices=PROBE_VIEWS)
    parser.add_argument("--dropped-text-column", type=int)
    parser.add_argument("--training-size", type=int)
    parser.add_argument("--draw", type=int)
    parser.add_argument("--provider", choices=tuple(load_yaml("extractors")["candidates"]))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--profile-fold",
        action="store_true",
        help="B0: full first outer fold, no row cap",
    )
    args = parser.parse_args()
    plan = benchmark_plan(cohort=args.cohort)
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
                "probe_view": args.probe_view,
                "dropped_text_column": args.dropped_text_column,
                "training_size": args.training_size,
                "draw": args.draw,
                "hidden_view": args.hidden_view,
                "hidden_position": args.hidden_position,
                "hidden_layer": args.hidden_layer,
                "adaptation_cell": args.adaptation_cell,
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
                    "group_b_study_by_phase": GROUP_B_PHASES,
                    "group_c_study_by_phase": GROUP_C_PHASES,
                    "feature_sets_by_experiment": {
                        "B1": DECISION_FEATURE_SETS,
                        "B2": EMBEDDING_FEATURE_SETS + (DECISION_PLUS_EMBEDDING,),
                        "B3": ("non_text_plus_tar", SUPERVISED_PROBE),
                        "B6": NATIVE_FEATURE_SETS,
                        "B4": (PER_COLUMN_WITHOUT_ONE,),
                        "B5": ("non_text", "non_text_plus_both",
                                            "non_text_plus_frozen_e5", SUPERVISED_PROBE),
                        "C1": HIDDEN_FEATURE_SETS,
                        "C2": (ADAPTED,),
                    },
                    "configurations_by_experiment": {
                        phase: sum(a["phase"] == phase for a in selected)
                        for phase in sorted({a["phase"] for a in selected})
                    },
                    "original_text_models": TEXT_MODELS,
                    "outer_folds": 5,
                    "inner_binary_threshold_folds_by_phase": {
                        "B1": 3, "B2": 3, "B3": 0,
                        "B6": 3, "B4": 3, "B5": 0,
                        "C1": 3, "C2": 0},
                    "selected_arm_count": len(selected),
                    "unsupported_width_arms": sum(
                        a["width_status"] == "unsupported_width" for a in selected
                    ),
                    "outer_fits": 5 * len(selected),
                    "non_text_includes_categorical": True,
                    "training_row_cap": None,
                    "explicit_training_subsamples": "B5 only (training-size design)",
                    "selected_arms": selected
                    if len(selected) <= 30
                    else "Filter with --dataset/--model/--feature-set/--provider to inspect",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
