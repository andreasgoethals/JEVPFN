"""Prepare a real-data compatibility manifest, WITHOUT network calls or model loads.

Run ``python -m src.jev.pilot``. Source rows can be reconstructed from their dataset
fingerprints and CSV positions. Operational files contain no raw text or targets.
Published context windows are screens, not deployed-account/tokenizer guarantees.
"""

import copy
import json
import math
from collections import Counter

import numpy as np

from src.data.audit import request_lengths
from src.data.loaders import load_catalog, load_dataset
from src.data.selection import selected_entries
from src.jev.alternatives import preview
from src.jev.identity import bind_provider
from src.jev.provider import api_preview
from src.jev.question_bank import build_question_bank, compile_frozen_question
from src.jev.requests import build_requests
from src.utils import paths
from src.utils.config import load_config, load_yaml
from src.utils.events import append_event
from src.utils.provenance import code_identity
from src.utils.serialization import canonical_json, digest, file_sha256, write_json


def dataset_pilot_requests(dataset, cfg):
    """Select by CSV position and text length, never target values or performance."""
    lengths = request_lengths(dataset)
    count = len(dataset.features)
    positions = {0, count // 2, count - 1}
    for i in range(len(dataset.metadata.text_columns)):
        if lengths[f"present_{i}"].any():
            positions.add(int(np.argmax(lengths[f"text_{i}"])))
    variant = copy.deepcopy(cfg)
    variant["jev"]["representation"] = "per_column_and_joint"
    requests = []
    code_version = code_identity()
    for position in sorted(positions):
        requests.extend(
            build_requests(
                dataset.features,
                dataset.metadata,
                position,
                variant,
                source=dataset.source,
                code_version=code_version,
            )
        )
    return positions, requests


def prepare_pilot(cfg, pilot_cfg, catalog, *, entries=None, question_bank=None):
    """Generate structural previews and distinct provider identities; no live readiness claim."""
    if pilot_cfg["enabled"] is not False or catalog["enabled"] is not False:
        raise ValueError("This preparer is offline only; live candidates must remain disabled")
    if set(pilot_cfg["question_variants"]) - {"accepted", "proposed_direct", "documented"}:
        raise ValueError("Unknown question variant")
    if set(pilot_cfg["providers"]) - set(catalog["candidates"]):
        raise ValueError("Unknown extractor")
    feature_cfg = cfg
    entries = selected_entries(load_catalog(cfg["catalog"]), cfg) if entries is None else entries
    design = load_yaml(pilot_cfg["question_design"])
    if question_bank is None:
        # Metadata-only first pass avoids retaining all dataset frames in RAM.
        question_bank = build_question_bank(
            [load_dataset(entry, allow_download=False).metadata for entry in entries], design
        )
    records, tasks, counts = [], set(), Counter()
    dataset_ids = []
    for entry in entries:
        dataset = load_dataset(entry, allow_download=False)
        dataset_ids.append(dataset.metadata.dataset_id)
        tasks.add(dataset.metadata.task_type)
        positions, intended = dataset_pilot_requests(dataset, feature_cfg)
        counts["sampled_rows"] += len(positions)
        transforms = {
            wording: compile_frozen_question(dataset.metadata, question_bank, wording)
            for wording in pilot_cfg["question_variants"]
        }
        for accepted in intended:
            for wording in pilot_cfg["question_variants"]:
                request = transforms[wording](accepted)
                trace = json.loads(request.provenance_json)
                wire_tokens_proxy = math.ceil(len(canonical_json(api_preview(request))) / 3)
                for provider in pilot_cfg["providers"]:
                    candidate = catalog["candidates"][provider]
                    if candidate["enabled"] is not False:
                        raise ValueError("Every candidate must remain disabled")
                    mapping = preview(request, provider)
                    identity = {
                        "provider": provider,
                        "model": candidate["model"],
                        "revision": candidate.get("revision"),
                        "version": candidate.get("version"),
                        "semantic_payload_sha256": digest(json.loads(request.payload_json)),
                        "mapping_sha256": digest(mapping),
                    }
                    limit = candidate["context_tokens"]
                    records.append(
                        {
                            "dataset_id": dataset.metadata.dataset_id,
                            "task": dataset.metadata.task_type,
                            "row_id": trace["row_id"],
                            "dataset_source": dataset.source,
                            "provider": provider,
                            "question_variant": wording,
                            "question_bank_id": question_bank["bank_id"],
                            "request_key": bind_provider(request, provider, catalog).cache_key,
                            "representation": trace["representation"],
                            "text_columns": trace["text_columns"],
                            "main_features_per_input": len(dataset.metadata.classes)
                            if dataset.metadata.task_type == "multiclass_classification"
                            else 1,
                            "provider_request_identity": digest(identity),
                            "identity": identity,
                            "input_tokens_proxy": wire_tokens_proxy,
                            "published_context_reference": limit,
                            "over_context_proxy": None
                            if limit is None
                            else wire_tokens_proxy > limit,
                            "live_verified": False,
                        }
                    )
        print(f"Prepared {dataset.metadata.dataset_name}: {len(positions)} rows", flush=True)
    unique = {r["provider_request_identity"]: r for r in records}
    provider_summary = {}
    for provider in pilot_cfg["providers"]:
        part = [r for r in unique.values() if r["provider"] == provider]
        provider_summary[provider] = {
            "unique_planned_requests": len(part),
            "input_tokens_proxy": sum(r["input_tokens_proxy"] for r in part),
            "over_context_proxy": sum(r["over_context_proxy"] is True for r in part),
            "live_verified": False,
        }
    return {
        "schema": "jevpfn.offline_pilot_manifest.v1",
        "network_calls": 0,
        "models_loaded": 0,
        "dataset_ids": dataset_ids,
        "tasks": sorted(tasks),
        "sampled_rows": counts["sampled_rows"],
        "question_variants": pilot_cfg["question_variants"],
        "question_bank_id": question_bank["bank_id"],
        "providers": provider_summary,
        "context_estimator": "complete Jev JSON characters / 3; not a provider tokenizer",
        "live_checks_required": pilot_cfg["live_contract_checks"],
        "records": records,
        "spending_limit_usd": pilot_cfg["spending_limit_usd"],
    }


def save_pilot_manifest(manifest):
    """Large trace manifest on PROJECT; compact discovery summary/log on DATA.

    Locally both resolve into output_JEVPFN, with distinct full/summary filenames.
    """
    name = digest(manifest)
    destination = (
        paths.large_outputs_dir()
        / paths.phase_folder("feature_creation")
        / "manifests"
        / "pilots"
        / f"{name}.json"
    )
    write_json(destination, manifest)
    summary = {k: v for k, v in manifest.items() if k != "records"}
    summary.update(
        full_manifest_path=str(destination),
        full_manifest_sha256=file_sha256(destination),
        source_trace_records=len(manifest["records"]),
    )
    write_json(paths.manifests_dir("feature_creation") / "pilots" / f"{name}.summary.json", summary)
    append_event(
        paths.logs_dir("feature_creation") / "pilot_preparation.jsonl",
        "offline_pilot_prepared",
        manifest_sha256=name,
        dataset_count=len(manifest["dataset_ids"]),
        network_calls=0,
    )
    return destination


def main():
    from src.jev.preparation import prepare_questions

    pilot_cfg = load_yaml("decision_pilot")
    cfg = load_config(pilot_cfg["feature_config"])
    catalog = load_yaml(pilot_cfg["extractor_catalog"])
    bank, _, _ = prepare_questions(cfg)
    manifest = prepare_pilot(cfg, pilot_cfg, catalog, question_bank=bank)
    destination = save_pilot_manifest(manifest)
    print(json.dumps(manifest["providers"], indent=2))
    print(f"Offline manifest: {destination}")
    print(
        "No API calls, model loads or Slurm submissions. Live pilot budget/access remain unapproved."
    )


if __name__ == "__main__":
    main()
