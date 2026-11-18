"""Validate an owner's explicit acceptance against preserved test evidence."""

import json
from pathlib import Path

from src.utils.serialization import file_sha256
from src.utils.structure import recorded_output_path


def reviewed_acceptance(filename, *, stage, expected=None):
    if not filename:
        raise ValueError(f"A reviewed {stage} acceptance file is required")
    # Paths recorded before the 18-11-2026 layout migration resolve to the moved file;
    # the checksum below still binds the exact reviewed evidence.
    path = recorded_output_path(filename)
    body = json.loads(path.read_text(encoding="utf-8"))
    if (
        body.get("accepted") is not True
        or body.get("stage") != stage
        or not body.get("reviewed_by")
    ):
        raise ValueError(f"Acceptance must explicitly approve {stage} and identify its reviewer")
    if not body.get("evidence"):
        raise ValueError("Acceptance requires checksummed test reports")
    for item in body["evidence"]:
        evidence = Path(item["path"])
        if not evidence.is_absolute():
            evidence = path.parent / evidence
        evidence = recorded_output_path(evidence)
        if not evidence.is_file() or file_sha256(evidence) != item["sha256"]:
            raise ValueError("Accepted evidence is missing or changed")
    if any(body.get(k) != v for k, v in (expected or {}).items()):
        raise ValueError("Acceptance does not match the current runtime/inputs")
    return body
