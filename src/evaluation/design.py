"""Reviewed PCA grid and explicit, descriptive post-selection follow-ups."""

from pathlib import Path

from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import digest, file_sha256


def pca_grid():
    grid = load_yaml("experiment_2/embeddings")["pca_grid"]
    expected = {"per_column": [32], "joint": [256, 128, 64, 32, 16, 8]}
    if grid != expected:
        raise ValueError("PCA grid differs from the reviewed per-column32/joint256-to8 design")
    return [(scope, k) for scope, dimensions in grid.items() for k in dimensions]


def followup_selection(*, require_complete=False):
    cfg = load_yaml("experiment_1/followup_selection")
    fields = ("provider", "mode", "evidence", "evidence_sha256", "rationale", "embedding")
    if not any(cfg.get(k) for k in fields):
        if require_complete:
            raise ValueError("Review and record the Experiment 1 winner before exploratory fits")
        return None
    if not all(cfg.get(k) for k in fields):
        raise ValueError("Incomplete follow-up selection; record provider, mode and evidence")
    if cfg["provider"] not in load_yaml("experiment_1/main")["candidate_extractors"]:
        raise ValueError("Unselected decision provider")
    if cfg["mode"] not in {"joint", "per_column", "both"}:
        raise ValueError("Unknown decision mode")
    if cfg["inference_status"] != "exploratory_post_selection":
        raise ValueError("Test-selected follow-ups must remain exploratory")
    embedding = cfg["embedding"]
    if (
        not isinstance(embedding, dict)
        or embedding.get("encoder") not in load_yaml("experiment_2/embeddings")["encoders"]
        or (embedding.get("scope"), embedding.get("components")) not in pca_grid()
    ):
        raise ValueError("Select one reviewed encoder/PCA recipe for exploratory combinations")
    evidence = Path(cfg["evidence"])
    if not evidence.is_absolute():
        evidence = paths.REPO_ROOT / evidence
    if not evidence.is_file() or file_sha256(evidence) != cfg["evidence_sha256"]:
        raise ValueError("Reviewed selection evidence is missing or changed")
    return {**cfg, "selection_id": digest(cfg)}
