"""Identity of the tested evaluation pipeline, independent of execution switches.

Experiment 0B reports this identity. Its reviewed acceptance must carry the same
value when benchmarking starts. Printing it never grants approval or runs models.
"""

import json

from src.cluster.freeze import source_hash
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import digest


def evaluation_identity():
    main = load_yaml("experiment_1/main")
    catalog = load_yaml("experiment_1/models")["models"]
    phases = {"experiment_1": main}
    for phase, name in (
        ("experiment_2", "embeddings"),
        ("experiment_3", "native_text"),
        ("experiment_5", "output_ablation"),
    ):
        phases[phase] = load_yaml(phase + "/" + name)
    feature_fields = (
        "random_seed",
        "candidate_models",
        "candidate_extractors",
        "candidate_combinations",
        "feature_selectors",
        "dataset_selection",
        "combined_extractor_groups",
    )
    body = {
        "schema": "jevpfn.evaluation_contract.v1",
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
        "native_models": load_yaml("experiment_3/native_text")["candidate_models"],
        "encoder_definitions": {
            name: {key: value for key, value in cfg.items() if key != "enabled"}
            for name, cfg in load_yaml("feature_creation/embedding_models")["models"].items()
        },
    }
    return digest(body)


if __name__ == "__main__":
    print(json.dumps({"evaluation_identity": evaluation_identity()}))
