"""Identity of the tested evaluation pipeline, independent of execution switches.

B0 reports this identity. Its reviewed acceptance must carry the same
value when benchmarking starts. Printing it never grants approval or runs models.
"""

import json

from src.cluster.freeze import source_hash
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import digest

GROUP_B_C_CONFIGS = ("B4/text_column_refits", "B5",
                     "C1", "C2")
PROVIDER_FIELDS = frozenset({"candidate_extractors", "initial_decision_providers", "decision_extensions",
                             "feature_selectors", "dataset_feature_artifacts"})
EXECUTION_FIELDS = frozenset({"enabled", "prepare_enabled", "training_enabled", "pilot_enabled",
                              "gate_acceptance", "pilot_acceptance", "status"})

# Code that constructs, fits, scores, stores, or verifies accepted benchmark folds.
# Decision extraction backends are identified by the producer artifacts and selectors.
# Report notebooks, exports, and analysis-only modules do not participate in a fit.
PIPELINE_MODULES = {
    "cluster": ("benchmark_jobs", "benchmark_sizing", "containers", "control_runtime", "predictor_contracts",
                "preflight", "runtime_profiles", "scheduling", "scratch_stage"),
    "data": ("loaders", "metadata", "official_types", "selection", "text_embeddings"),
    "evaluation": ("acceptance", "ag_mm", "attribution", "campaign", "checkpoint_paths", "data_efficiency",
                   "design", "embedding_control", "features", "group_c", "label_probe", "limix_preprocessing",
                   "metrics", "models", "pca_cache", "prepare_pca", "protocol", "qualification", "records",
                   "store", "strands_adaptation", "tabstar_assets", "tar", "text_overlap", "widths"),
    "jev": ("cache", "checkpoint_hashes", "checkpoints", "features", "feature_retirement", "inputs",
            "provider", "question_bank", "requests", "strands_hidden", "templates"),
    "utils": ("acceptance", "config", "files", "locking", "paths", "provenance", "serialization", "structure"),
}


def pipeline_sources():
    root = paths.REPO_ROOT
    files = [root / "src" / package / f"{name}.py"
             for package, modules in PIPELINE_MODULES.items() for name in modules]
    vendor = root / "src/evaluation/vendor"
    files.extend(vendor.rglob("*.py"))
    files.append(vendor / "SOURCE.json")
    files.extend((root / "src/data/upstream").rglob("*.py"))
    return {p.relative_to(root).as_posix(): source_hash(p) for p in files if p.is_file()}


def evaluation_identity():
    main = load_yaml("B1")
    catalog = load_yaml("B1/models")["models"]
    phases = {"B1": main}
    for phase in ("B2", "B3", "B6"):
        phases[phase] = load_yaml(phase)
    feature_fields = (
        "random_seed",
        "candidate_models",
        "candidate_combinations",
        "modes",
        "views",
        "dataset_selection",
    )
    body = {
        "schema": "jevpfn.evaluation_contract.v3",
        "dataset_catalog": load_yaml("datasets")["datasets"],
        "dataset_semantics": load_yaml("dataset_semantics")["datasets"],
        "environment": source_hash(paths.REPO_ROOT / "pyproject.toml"),
        "source": pipeline_sources(),
        "models": {
            name: {key: cfg.get(key) for key in ("runtime", "interface", "model_generation")}
            for name, cfg in catalog.items()
        },
        "evaluation": load_yaml("B1/protocol")["evaluation"],
        "features": {
            phase: {key: cfg.get(key) for key in feature_fields} for phase, cfg in phases.items()
        },
        "embeddings": {
            key: load_yaml("B2")[key] for key in ("encoders", "pca_grid")
        },

        "a2_questions": source_hash(paths.config_file("A2_question_versions.csv")),

        "downstream_contracts": {
            name: {k:v for k,v in load_yaml(name).items() if k not in EXECUTION_FIELDS | PROVIDER_FIELDS}
            for name in ("A0", "A2", "A3", "B0", "C0")
        },
        "tar": {k: v for k, v in load_yaml("B3").items()
                if k not in EXECUTION_FIELDS | PROVIDER_FIELDS},
        "native_models": load_yaml("B6")["candidate_models"],
        # B4/B5 and Group C scientific settings (e.g. C2 regression bins); execution
        # switches and status labels stay outside, as for the other phases.
        "group_b_c": {
            name: {k: v for k, v in load_yaml(name).items() if k not in EXECUTION_FIELDS | PROVIDER_FIELDS}
            for name in GROUP_B_C_CONFIGS
        },
        "encoder_definitions": {
            name: {key: value for key, value in cfg.items() if key != "enabled"}
            for name, cfg in load_yaml("feature_creation/embedding_models")["models"].items()
        },
    }
    # C1 can start after the core B0 review. Keep its extraction recipe bound
    # when the owner turns on the separate producer.
    c1 = body["group_b_c"]["C1"]
    c1["extraction"] = {key: value for key, value in c1["extraction"].items() if key != "enabled"}
    return digest(body)


def decision_input_identity(providers):
    """Bind the scoped selectors and artifact pins, including inherited per-phase overrides."""
    from src.evaluation.design import decision_versions

    scoped = {}
    for phase in ("B1", "B2", "B3",
                  "B6", *GROUP_B_C_CONFIGS):
        config = load_yaml(phase)
        selectors = config.get("feature_selectors", {})
        pins = config.get("dataset_feature_artifacts", {})
        scoped[phase] = {
            "selectors": {p: selectors.get(p) for p in providers},
            "pins": {dataset: {p: items[p] for p in providers if p in items}
                     for dataset, items in pins.items() if any(p in items for p in providers)},
        }
    return digest({"decision_model_versions": decision_versions(providers), "phase_inputs": scoped})


if __name__ == "__main__":
    print(json.dumps({"evaluation_identity": evaluation_identity()}))
