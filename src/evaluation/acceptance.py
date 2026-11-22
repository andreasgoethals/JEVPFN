"""Identity of the tested evaluation pipeline, independent of execution switches.

Experiment 0B reports this identity. Its reviewed acceptance must carry the same
value when benchmarking starts. Printing it never grants approval or runs models.
"""

import json

from src.cluster.freeze import source_hash
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import digest

GROUP_B_C_CONFIGS = ("rq2/text_column_refits", "data_efficiency/data_efficiency",
                     "representations/representations", "rq3/adaptation")
EXECUTION_FIELDS = frozenset({"enabled", "prepare_enabled", "training_enabled",
                              "experiment_0_acceptance", "status"})


def evaluation_identity():
    main = load_yaml("experiment_1/main")
    catalog = load_yaml("experiment_1/models")["models"]
    phases = {"experiment_1": main}
    for phase, name in (
        ("experiment_2", "embeddings"),
        ("experiment_3", "tar"),
        ("experiment_4", "native_text"),
    ):
        phases[phase] = load_yaml(phase + "/" + name)
    feature_fields = (
        "random_seed",
        "candidate_models",
        "candidate_extractors",
        "candidate_combinations",
        "modes",
        "views",
        "feature_selectors",
        "dataset_feature_artifacts",
        "dataset_selection",
    )
    body = {
        "schema": "jevpfn.evaluation_contract.v2",
        "dataset_catalog": load_yaml("datasets")["datasets"],
        "dataset_semantics": load_yaml("dataset_semantics")["datasets"],
        "environment": source_hash(paths.REPO_ROOT / "pyproject.toml"),
        "source": {
            p.relative_to(paths.REPO_ROOT).as_posix(): source_hash(p)
            for p in sorted((paths.REPO_ROOT / "src").rglob("*.py"))
        },
        "models": {
            name: {key: cfg.get(key) for key in ("runtime", "interface", "model_generation")}
            for name, cfg in catalog.items()
        },
        "evaluation": load_yaml("experiment_1/protocol")["evaluation"],
        "features": {
            phase: {key: cfg.get(key) for key in feature_fields} for phase, cfg in phases.items()
        },
        "embeddings": {
            key: load_yaml("experiment_2/embeddings")[key] for key in ("encoders", "pca_grid")
        },

        "rq1_questions": source_hash(paths.config_file("question_versions.csv")),

        "downstream_contracts": {
            name: {k:v for k,v in load_yaml(name).items() if k not in {"enabled","experiment_0_acceptance"}}
            for name in ("rq1/behavior","rq1/geometry","experiment_0/research_questions")
        },
        "tar": {k: v for k, v in load_yaml("experiment_3/tar").items()
                if k not in {"enabled", "prepare_enabled", "experiment_0_acceptance"}},
        "native_models": load_yaml("experiment_4/native_text")["candidate_models"],
        # Group B3/B4 and Group C scientific settings (e.g. C2 regression bins); execution
        # switches and status labels stay outside, as for the other phases.
        "group_b_c": {
            name: {k: v for k, v in load_yaml(name).items() if k not in EXECUTION_FIELDS}
            for name in GROUP_B_C_CONFIGS
        },
        "encoder_definitions": {
            name: {key: value for key, value in cfg.items() if key != "enabled"}
            for name, cfg in load_yaml("feature_creation/embedding_models")["models"].items()
        },
    }
    return digest(body)


if __name__ == "__main__":
    print(json.dumps({"evaluation_identity": evaluation_identity()}))
