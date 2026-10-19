"""Explicit, bounded offline E5 test; never creates features for a benchmark dataset."""

import argparse
import json
from uuid import uuid4

import numpy as np
import pandas as pd

from src.data import text_embeddings as embeddings
from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.events import append_event
from src.utils.serialization import digest, write_json


def exercise(cfg, encoder, destination):
    """Exercise the real persistence path with repeated, missing and over-512-token text."""
    long_text = "A detailed description of a research observation. " * 160
    short = "The observation describes a useful product."
    frame = pd.DataFrame(
        {
            "a": [short, short, None, long_text],
            "b": ["", short, "A different product.", None],
            "target": [0.0, 1.0, 2.0, 3.0],
        }
    )
    official = {"target": "target", "task_type": "reg"}
    metadata = extract_metadata(
        frame, official, dataset_id="EXP0_E5", dataset_name="E5 fixture", text_columns=["a", "b"]
    )
    dataset = LoadedDataset(
        frame, metadata, official, {"data_identity": digest(frame.to_dict())}, []
    )
    _, chunks = embeddings.token_chunks(
        encoder.tokenizer, long_text, prefix=cfg["prefix"], max_tokens=cfg["max_tokens"]
    )
    if len(chunks) <= 1:
        raise AssertionError("Long fixture did not exercise multiple encoder windows")

    class Counted:
        identity = encoder.identity
        texts = 0

        def __call__(self, texts):
            self.texts += len(texts)
            return encoder(texts)

    counted = Counted()
    artifact = embeddings.build(dataset, cfg, counted, destination=destination)
    table = embeddings.load(dataset, cfg, source=destination)
    first = table[table.attrs["embedding_groups"][0]].to_numpy()
    if counted.texts != 3 or not np.isnan(first[2]).all():
        raise AssertionError("Exact reuse or empty-input skipping failed")
    np.testing.assert_array_equal(first[0], first[1])
    np.testing.assert_allclose(np.linalg.norm(first[[0, 1, 3]], axis=1), 1.0, atol=1e-5)
    if (
        embeddings.build(dataset, cfg, counted, destination=destination) != artifact
        or counted.texts != 3
    ):
        raise AssertionError("Resume called the encoder or changed its artifact")
    return {
        "artifact_id": artifact,
        "unique_texts_encoded": counted.texts,
        "long_text_chunks": len(chunks),
        "missing_inputs_skipped": True,
        "resume_without_inference": True,
        "rows": len(table),
        "dimensions_per_text": cfg["dimensions"],
        "runtime": encoder.identity,
    }


def run(*, execute=False, encoder_factory=embeddings.E5Encoder):
    if not execute:
        raise RuntimeError("Pass --execute for the bounded E5 contract test")
    cfg = load_yaml("feature_creation/text_embeddings")
    run_id = uuid4().hex
    manifest = (
        paths.manifests_dir("experiment_0") / "before_extraction" / "embeddings" / f"{run_id}.json"
    )
    destination = (
        paths.results_dir("embedding_contracts", phase="experiment_0") / run_id / "EXP0_E5.sqlite3"
    )
    report = {
        "status": "running",
        "config": cfg,
        "fixture_cache": str(destination),
        "full_feature_creation_enabled": False,
    }
    log = paths.logs_dir("experiment_0") / "embeddings.log"
    write_json(manifest, report)
    append_event(log, "embedding_contract_started", run_id=run_id)
    try:
        report.update(exercise(cfg, encoder_factory(cfg), destination))
        report["status"] = "embedding_contracts_passed_requires_full_dataset_review"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        write_json(manifest, report)
        append_event(log, "embedding_contract_finished", run_id=run_id, status=report["status"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(execute=True)
            if args.execute
            else {
                "execution_enabled": False,
                "scope": "Three unique synthetic texts; long text, missing inputs, reuse and resume",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
