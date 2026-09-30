"""Label-free states and immutable serialisable intended requests."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from src.data.metadata import TaskMetadata
from src.jev.templates import TEMPLATE_VERSION, make_question
from src.utils.serialization import canonical_json, digest, text_value


def text_groups(metadata: TaskMetadata, mode: str) -> tuple[tuple[str, ...], ...]:
    per_column = tuple((c,) for c in metadata.text_columns)
    joint = (metadata.text_columns,)
    if mode == "per_column":
        return per_column
    if mode == "joint":
        return joint
    if mode == "per_column_and_joint":
        # Intentional duplicate for a single-text dataset: planned slots differ, but cache reuses it.
        return per_column + joint
    raise ValueError(f"Unknown representation: {mode}")


def make_payload(
    metadata: TaskMetadata, text: dict, structured: dict | None, *, scores: list[int]
) -> dict:
    state = {"task": metadata.task_context(), "text": text}
    if structured is not None:
        raise ValueError("Non-text features are excluded from Jev requests.")
    return {"state": state, "question": make_question(metadata, scores=scores)}


@dataclass(frozen=True)
class IntendedRequest:
    payload_json: str
    settings_json: str
    provenance_json: str

    @property
    def cache_key(self) -> str:
        # Row IDs, timestamps and code revisions are trace data, not semantic API inputs.
        # Identical payload + pinned provider settings can reuse a response across rows/folds.
        return digest(
            {"payload": json.loads(self.payload_json), "settings": json.loads(self.settings_json)}
        )

    def to_dict(self) -> dict:
        return {
            "schema": "jevpfn.intended_request.v1",
            "api_mapping_verified": False,
            "payload": json.loads(self.payload_json),
            "settings": json.loads(self.settings_json),
            "provenance": json.loads(self.provenance_json),
            "cache_key": self.cache_key,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> IntendedRequest:
        value = json.loads(text)
        if (
            value["schema"] != "jevpfn.intended_request.v1"
            or value["api_mapping_verified"] is not False
        ):
            raise ValueError("Unsupported request schema.")
        result = cls(
            canonical_json(value["payload"]),
            canonical_json(value["settings"]),
            canonical_json(value["provenance"]),
        )
        if value["cache_key"] != result.cache_key:
            raise ValueError("Request checksum mismatch.")
        return result


def build_requests(
    features: pd.DataFrame,
    metadata: TaskMetadata,
    row_position: int,
    cfg: dict,
    *,
    source: dict | None = None,
    code_version: str = "unrecorded",
) -> list[IntendedRequest]:
    """The builder accepts FEATURES ONLY and refuses target/extra columns.

    CSV positional row IDs remain stable within the pinned, hashed data artifact. A caller
    must pass the full feature table, not a reindexed fold. Neither row ID nor file hash is
    sent as semantic context. No text is truncated. Empty text groups are skipped; original slot IDs stay stable.
    """
    expected = set(metadata.text_columns + metadata.non_text_columns)
    if not features.columns.is_unique or set(features.columns) != expected:
        raise ValueError(
            "Pass exactly the feature columns, never the target or derived statistics."
        )
    if not 0 <= row_position < len(features):
        raise IndexError(row_position)
    jev = cfg["jev"]
    if jev["enabled"] is not False:
        raise ValueError("Only dry-run construction is enabled.")
    if jev["template_version"] != TEMPLATE_VERSION:
        raise ValueError("Unknown template version.")
    row = features.iloc[row_position]
    if jev["include_non_text_features"] is not False:
        raise ValueError("Non-text features are excluded from Jev requests.")
    settings = {
        "provider": "jev",
        "model": jev["model"],
        "version": jev["version"],
        "request_configuration": jev["request_configuration"],
        "template_version": TEMPLATE_VERSION,
        "api_mapping_version": "typesafe-systemone-v1",
    }
    requests = []
    groups = text_groups(metadata, jev["representation"])
    for slot, group in enumerate(groups):
        text = {c: value for c in group if (value := text_value(row[c])) is not None}
        if not text:
            continue
        payload = make_payload(
            metadata,
            text,
            None,
            scores=jev["regression_scores"],
        )
        provenance = {
            "dataset_id": metadata.dataset_id,
            "row_id": f"csv:{row_position}",
            "text_columns": list(text),
            "source_text_columns": list(group),
            "missing_text_policy": "skip_empty_store_nan",
            "representation": jev["representation"],
            "representation_slot": slot,
            "include_non_text_features": jev["include_non_text_features"],
            "dataset_source": source or {},
            "task_schema_sha256": metadata.fingerprint,
            "code_version": code_version,
            "config_sha256": digest(cfg),
        }
        requests.append(
            IntendedRequest(
                canonical_json(payload), canonical_json(settings), canonical_json(provenance)
            )
        )
    return requests
