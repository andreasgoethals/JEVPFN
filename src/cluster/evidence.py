"""Bind live checks to normalized code/configuration and pinned runtime artifacts."""

from src.cluster.freeze import runtime_pins, source_snapshot
from src.utils import paths
from src.utils.serialization import digest, file_sha256


def current_identity():
    return {
        "source_sha256": source_snapshot()["source_sha256"],
        "runtime_pins_sha256": digest(runtime_pins()),
        "prepared_inputs_sha256": digest(
            {p.name: file_sha256(p) for p in sorted(paths.decision_inputs_dir().glob("*.sqlite3"))}
        ),
    }
