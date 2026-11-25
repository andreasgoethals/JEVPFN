"""Validate an owner's explicit acceptance against preserved test evidence."""

import json
import re
from pathlib import Path, PurePosixPath

from src.utils import paths
from src.utils.serialization import file_sha256
from src.utils.structure import recorded_output_path


def evidence_path(acceptance, item):
    """Resolve relative and historical paths exactly as they were checked at review."""
    path = recorded_output_path(acceptance)
    found = recorded_output_path(path.parent / item["path"])
    if not found.is_file() and path != Path(acceptance):
        found = recorded_output_path(Path(acceptance).parent / item["path"])
    # A copied native approval can contain absolute VSC paths. Resolve its signed
    # bytes on the desktop, retaining both versions when a local audit has the same name.
    remote = PurePosixPath(item["path"])
    if found.is_file() or paths.on_vsc() or not remote.is_absolute() or ".." in remote.parts:
        return found
    output = next((p for p in path.resolve().parents if p.name == paths.OUTPUT_DIR_NAME), None)
    if output is None or paths.OUTPUT_DIR_NAME not in remote.parts:
        return found
    index = remote.parts.index(paths.OUTPUT_DIR_NAME)
    if index < 2 or remote.parts[index - 1] != "JEVPFN" or not re.fullmatch("[a-f0-9]{64}", item["sha256"]):
        return found
    local = recorded_output_path(output.joinpath(*remote.parts[index + 1:]))
    if local.is_file() and file_sha256(local) == item["sha256"]:
        return local
    manifests = next((p for p in path.resolve().parents if p.name == "manifests" and output in p.parents), None)
    if manifests is not None:
        imported = manifests / "imported_revisions" / "evidence" / (item["sha256"] + remote.suffix)
        if imported.is_file():
            return imported
    return found


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
        evidence = evidence_path(filename, item)
        if not evidence.is_file() or file_sha256(evidence) != item["sha256"]:
            raise ValueError("Accepted evidence is missing or changed")
    if any(body.get(k) != v for k, v in (expected or {}).items()):
        raise ValueError("Acceptance does not match the current runtime/inputs")
    return body
