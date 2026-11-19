"""Sourced decision-model properties (`config/02_feature_creation/model_properties.yaml`).

One row per provider in `extractors.yaml`. Every field is ``{value, verified, source}``;
a value that the local sources (extractors config, dm-library summaries, pinned tokenizer
configs) do not state is ``null`` with ``verified: false`` and is never guessed.

The table is an interpretation aid: it explains *who* the decision models are when their
outputs or downstream gains differ (Exp 1, Exp 8). It never selects providers, features or
predictors, and it never enters a decision request.
"""

from __future__ import annotations

import math

import pandas as pd

from src.utils.config import load_yaml

SCHEMA = "jevpfn.model_properties.v1"
CONFIG = "feature_creation/model_properties"
FIELDS = (
    "display_name", "publisher", "release", "revision", "parameters_billion", "parameters_note",
    "base_model", "architecture_family", "readout_type", "hidden_size", "num_layers",
    "context_tokens", "license", "training", "source_url",
)
#: Fields whose value must equal the provider's entry in extractors.yaml.
EXTRACTOR_KEYS = {"context_tokens": "context_tokens", "source_url": "reference"}
NUMERIC = {"parameters_billion": (int, float), "hidden_size": (int,), "num_layers": (int,),
           "context_tokens": (int,)}
#: Model-level covariates offered to Exp 8 moderators.
COVARIATES = ("parameters_billion", "log10_parameters", "architecture_family", "readout_type")


def load(raw: dict | None = None, extractors: dict | None = None) -> dict:
    """The validated properties document."""
    raw = load_yaml(CONFIG) if raw is None else raw
    extractors = load_yaml("feature_creation/extractors") if extractors is None else extractors
    validate(raw, extractors)
    return raw


def validate(raw: dict, extractors: dict) -> None:
    if raw.get("schema") != SCHEMA:
        raise ValueError("Unknown model-properties schema")
    candidates = extractors["candidates"]
    providers = raw.get("providers") or {}
    if list(providers) != list(candidates):
        missing = sorted(set(candidates) - set(providers))
        extra = sorted(set(providers) - set(candidates))
        raise ValueError(f"Providers must match extractors.yaml in order; missing {missing}, extra {extra}")
    families, readouts = set(raw["architecture_families"]), set(raw["readout_types"])
    for provider, row in providers.items():
        if tuple(row) != FIELDS:
            raise ValueError(f"{provider}: fields must be exactly {FIELDS}")
        for name, cell in row.items():
            if set(cell) != {"value", "verified", "source"} or not isinstance(cell["verified"], bool):
                raise ValueError(f"{provider}.{name}: needs value, boolean verified and source")
            value = cell["value"]
            if value is None:
                if cell["verified"] or cell["source"] is not None:
                    raise ValueError(f"{provider}.{name}: a missing value is unverified and unsourced")
                continue
            if not cell["source"]:
                raise ValueError(f"{provider}.{name}: every stated value needs a local source")
            if name in NUMERIC and (isinstance(value, bool) or not isinstance(value, NUMERIC[name])
                                    or value <= 0):
                raise ValueError(f"{provider}.{name}: must be a positive number")
        if (family := row["architecture_family"]["value"]) is not None and family not in families:
            raise ValueError(f"{provider}: unknown architecture family {family}")
        if (readout := row["readout_type"]["value"]) is not None and readout not in readouts:
            raise ValueError(f"{provider}: unknown readout type {readout}")
        candidate = candidates[provider]
        expected_revision = candidate.get("revision") or candidate.get("model")
        if row["revision"]["value"] not in {expected_revision, candidate.get("version")}:
            raise ValueError(f"{provider}: revision differs from extractors.yaml")
        for field, key in EXTRACTOR_KEYS.items():
            if row[field]["value"] is not None and row[field]["value"] != candidate.get(key):
                raise ValueError(f"{provider}: {field} differs from extractors.yaml")


def table(raw: dict | None = None, *, verified_only: bool = False) -> pd.DataFrame:
    """One row per provider with plain values; unverified values become missing on request."""
    raw = load() if raw is None else raw
    rows = []
    for provider, row in raw["providers"].items():
        values = {"provider": provider}
        for name, cell in row.items():
            values[name] = cell["value"] if cell["verified"] or not verified_only else None
        values["verified_fields"] = sum(cell["verified"] for cell in row.values())
        rows.append(values)
    return pd.DataFrame(rows, columns=["provider", *FIELDS, "verified_fields"])


def covariates(raw: dict | None = None) -> pd.DataFrame:
    """Verified model-level covariates per provider (parameters, architecture, readout)."""
    frame = table(raw, verified_only=True)
    params = pd.to_numeric(frame.parameters_billion, errors="coerce")
    frame["parameters_billion"] = params
    frame["log10_parameters"] = [math.log10(v * 1e9) if pd.notna(v) else float("nan") for v in params]
    return frame[["provider", *COVARIATES]]


def display_table(raw: dict | None = None) -> pd.DataFrame:
    """Compact, readable view for notebooks: unverified values are marked, never filled."""
    raw = load() if raw is None else raw

    def shown(cell, digits=None):
        value = cell["value"]
        if value is None:
            return "not verified"
        if digits is not None and isinstance(value, float):
            value = f"{value:.{digits}f}"
        return f"{value}" if cell["verified"] else f"{value} (unverified)"

    rows = []
    for provider, row in raw["providers"].items():
        rows.append({
            "provider": provider,
            "model": shown(row["display_name"]),
            "publisher": shown(row["publisher"]),
            "parameters (B)": shown(row["parameters_billion"], 2),
            "architecture": shown(row["architecture_family"]),
            "readout": shown(row["readout_type"]),
            "context tokens": shown(row["context_tokens"]),
            "license": shown(row["license"]),
            "verified fields": f"{sum(c['verified'] for c in row.values())}/{len(FIELDS)}",
        })
    return pd.DataFrame(rows)
