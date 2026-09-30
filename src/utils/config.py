"""Read and validate the initial research configuration using the template's YAML convention.

Resolved copies go to output_JEVPFN/<phase>/manifests/. Live Jev and modelling switches remain disabled.
Phase configs may extend one other YAML file; package and credential loading stay separate.
"""

from __future__ import annotations

import copy
import math
import os

import yaml

from src.utils import paths
from src.utils.serialization import digest, write_json


def load_yaml(name: str, *, _seen: tuple[str, ...] = ()) -> dict:
    path = paths.config_path(name)
    key = str(path)
    if key in _seen:
        raise ValueError("Cyclic configuration inheritance.")
    with path.open(encoding="utf-8") as handle:
        child = yaml.safe_load(handle)
    if not isinstance(child, dict):
        raise ValueError("Configuration must be a mapping.")
    parent = child.pop("extends", None)
    if parent is None:
        return child
    return _merge(load_yaml(parent, _seen=(*_seen, key)), child)


def _merge(parent: dict, child: dict) -> dict:
    result = copy.deepcopy(parent)
    for key, value in child.items():
        result[key] = (
            _merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def load_config(name: str = "exploration/audit") -> dict:
    """Load an exploration/feature config; never read credentials or load .env."""
    cfg = load_yaml(name)
    price = os.environ.get("JEV_INPUT_PRICE_PER_MILLION")
    if price is not None:
        cfg["cost"]["input_price_per_million"] = float(price)
        cfg["cost"]["price_source"] = "environment"
    else:
        cfg["cost"]["price_source"] = "config"
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    if cfg["dataset_collection"] != "multabench_core_text":
        raise ValueError("Only the 20 core text datasets are enabled in this phase.")
    if cfg["jev"]["enabled"] is not False or cfg["downstream"]["enabled"] is not False:
        raise ValueError("Jev API calls and modelling are disabled in the initial phase.")
    if cfg.get("feature_creation", {}).get("enabled", False) is not False:
        raise ValueError("Live feature creation is disabled in the initial phase.")
    if "feature_creation" in cfg:
        feature = cfg["feature_creation"]
        modes = feature["modes"]
        if not modes or len(modes) != len(set(modes)) or not set(modes) <= {"per_column", "joint"}:
            raise ValueError("Feature modes must be distinct per_column/joint blocks.")
        if feature["missing_text_policy"] != "skip_empty_store_nan":
            raise ValueError("Empty inputs must be skipped and stored as missing features.")
        paths.jev_cache_path(feature["feature_directory"])
    if cfg["jev"]["representation"] not in {"per_column", "joint", "per_column_and_joint"}:
        raise ValueError("Unknown representation mode.")
    if cfg["jev"]["include_non_text_features"] is not False:
        raise ValueError("Non-text features are excluded from Jev requests.")
    if cfg["jev"]["missing_text_policy"] != "skip_empty_store_nan":
        raise ValueError("Empty inputs must be skipped and stored as missing features.")
    if cfg["jev"]["regression_scores"] != list(range(-4, 5)):
        raise ValueError("The agreed regression scale is exactly -4 through +4.")
    if cfg["jev"]["binary_positive_class"] != "canonical_last":
        raise ValueError("Only the documented mechanical binary orientation is implemented.")
    if cfg["token_estimation"]["method"] != "characters_divided_by_four":
        raise ValueError("Only the documented character-count approximation is implemented.")
    ratio = cfg["token_estimation"]["characters_per_token"]
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("characters_per_token must be finite and positive.")
    price = cfg["cost"]["input_price_per_million"]
    if price is not None and (not math.isfinite(price) or price < 0):
        raise ValueError("Input price must be null or a finite nonnegative number.")
    overhead = cfg["cost"]["provider_overhead_tokens_per_request"]
    if not math.isfinite(overhead) or overhead < 0:
        raise ValueError("Hypothetical provider overhead must be finite and nonnegative.")
    paths.jev_cache_path(cfg["jev"]["cache_location"])


def save_resolved_config(cfg: dict, notebook: str) -> str:
    resolved = copy.deepcopy(cfg)
    resolved["resolved_cache_path"] = str(paths.jev_cache_path(cfg["jev"]["cache_location"]))
    resolved["config_sha256"] = digest(cfg)
    write_json(
        paths.manifest_path(f"{notebook}_config", phase=cfg.get("phase", "exploration")), resolved
    )
    return resolved["config_sha256"]
