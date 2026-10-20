"""Real, bounded GPU acceptance tests; ordinary pytest/notebooks never execute them.

python -m src.cluster.gpu_contracts --provider cloudflare --execute
Requires an allocated VSC GPU, staged pinned weights and a complete exact token audit.
Use --check-inputs on the login host to validate existing token evidence without inference.
Synthetic outputs stay under Experiment 0; real pilot responses are permanently reusable.
"""

from __future__ import annotations

import argparse
import copy
import os
import sqlite3
import time
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd

from src.cluster.preflight import storage_policy_failures
from src.cluster.semantic_contracts import exercise_robustness, exercise_semantics, probes
from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.jev.cache import ResponseCache
from src.jev.decision_tokenization import SELF_HOSTED, ExactCounter
from src.jev.features import load_block, materialize_cached_block, save_block
from src.jev.inputs import iter_prepared_inputs, prepared_request
from src.jev.provider import response_features
from src.jev.requests import IntendedRequest, build_requests
from src.jev.self_hosted import (
    SelfHostedClient,
    cached_evaluate,
    inference_environment,
    token_evidence,
)
from src.utils import paths
from src.utils.config import load_config, load_yaml
from src.utils.events import append_event
from src.utils.locking import file_lock
from src.utils.serialization import canonical_json, digest, write_json


class MeasuredClient:
    """A hard ceiling counts attempted forwards, including attempts that fail."""

    def __init__(self, client, maximum):
        if type(maximum) is not int or maximum < 1:
            raise ValueError("GPU test request budget must be a positive integer")
        self.client, self.maximum = client, maximum
        self.settings = client.settings
        self.calls, self.seconds = 0, []

    def evaluate(self, request):
        if self.calls >= self.maximum:
            raise RuntimeError("GPU contract-test request ceiling reached")
        self.calls += 1
        started = time.perf_counter()
        try:
            # SelfHostedClient synchronizes CUDA before returning.
            return self.client.evaluate(request)
        finally:
            self.seconds.append(time.perf_counter() - started)

    def evaluate_many(self, requests):
        if self.calls + len(requests) > self.maximum:
            raise RuntimeError("GPU contract-test request ceiling reached")
        self.calls += len(requests)
        started = time.perf_counter()
        try:
            return self.client.evaluate_many(requests)
        finally:
            self.seconds.append(time.perf_counter() - started)


def exercise_batch_parity(client, *, tolerance):
    size = client.settings["request_configuration"].get("batch_size", 1)
    if size == 1:
        return {"passed": True, "mode": "serial_only", "batch_size": 1}
    # Different tasks/lengths catch padding and per-row option mapping errors.
    requests = [
        request
        for name, request, _ in probes(client.settings)
        if name in {"binary_True", "score_4"}
    ]
    serial = [response_features(r, client.evaluate(r)) for r in requests]
    batch = [
        response_features(r, raw)
        for r, raw in zip(requests, client.evaluate_many(requests), strict=True)
    ]
    maximum = max(
        abs(a[key] - b[key])
        for a, b in zip(serial, batch, strict=True)
        for key in a
        if key.startswith("p_")
    )
    if maximum > tolerance:
        raise ValueError(
            f"Serial/batch probability difference {maximum} exceeds {tolerance}; review batching"
        )
    return {
        "passed": True,
        "mode": "native_batch",
        "batch_size": size,
        "maximum_probability_difference": maximum,
        "tolerance": tolerance,
    }


class CacheOnlyClient:
    def evaluate(self, request):
        raise AssertionError("A repeated request attempted GPU inference instead of cache reuse")


def synthetic_datasets():
    """Known duplicate/empty/Unicode inputs; arbitrary fixture targets never enter requests."""
    for kind, target in (
        ("binary", [0, 1, 0, 1, 0, 1]),
        ("multiclass", ["arts", "science", "sports"] * 2),
        ("regression", list(range(6))),
    ):
        frame = pd.DataFrame(
            {
                "text_a": [
                    "A positive review.",
                    "A positive review.",
                    None,
                    None,
                    "Résumé 世界",
                    "Unusual punctuation:\n{ } [ ]",
                ],
                "text_b": [
                    "Clear evidence.",
                    "Clear evidence.",
                    "One available text.",
                    " ",
                    "Second field",
                    "An independent description.",
                ],
                "non_text": list(range(6)),
                "target": target,
            }
        )
        meta = extract_metadata(
            frame,
            {"task_type": "reg" if kind == "regression" else "cls", "target": "target"},
            dataset_id=f"EXP0_SYNTHETIC_{kind.upper()}",
            dataset_name=f"GPU fixture: {kind}",
            text_columns=["text_a", "text_b"],
        )
        yield LoadedDataset(frame, meta, {}, {"data_identity": digest(frame.to_dict("list"))}, [])


def exercise_feature_pipeline(client, dataset, directory):
    """Actual GPU responses -> restart reuse -> both feature blocks -> checksum reload."""
    cfg = load_config("feature_creation")
    cfg["jev"]["representation"] = "per_column_and_joint"
    requests = [
        replace(r, settings_json=canonical_json(client.settings))
        for row in range(len(dataset.frame))
        for r in build_requests(dataset.features, dataset.metadata, row, cfg, source=dataset.source)
    ]
    cache_path = directory / f"{dataset.metadata.dataset_id}.sqlite3"
    cache = ResponseCache(cache_path, dataset.metadata.dataset_id)
    # Commit the first result and reopen the database, as after an interrupted process.
    cached_evaluate(cache, client, requests[0])
    cache = ResponseCache(cache_path, dataset.metadata.dataset_id)
    for request in requests:
        cached_evaluate(cache, client, request)
    for request in requests:
        if cached_evaluate(cache, CacheOnlyClient(), request) != "cache_hit":
            raise AssertionError("Restart did not reuse the saved response")
    artifacts = []
    for mode in ("per_column", "joint"):
        cfg["jev"]["representation"] = mode
        block = materialize_cached_block(
            dataset, cfg, cache, provider_settings=client.settings, phase="experiment_0"
        )
        path = save_block(block, cache_path, phase="experiment_0")
        loaded = load_block(path, block.manifest["artifact_id"])
        values = loaded.values.drop(columns="row_id")
        if loaded.values.row_id.tolist() != [f"csv:{i}" for i in range(len(dataset.frame))]:
            raise AssertionError("Feature row identity changed")
        if not values.iloc[0].equals(values.iloc[1]) or not values.iloc[3].isna().all():
            raise AssertionError("Repeated or empty text features differ from their contract")
        if not loaded.values.equals(block.values) or not loaded.requests.equals(block.requests):
            raise AssertionError("Saved/reloaded feature tables differ")
        artifacts.append(
            {"mode": mode, "artifact_id": block.manifest["artifact_id"], "path": str(path)}
        )
    return {
        "task_type": dataset.metadata.task_type,
        "unique_requests": len({r.cache_key for r in requests}),
        "original_rows": len(dataset.frame),
        "cache_reopen_and_reuse": True,
        "duplicate_rows_equal": True,
        "empty_features_missing": True,
        "artifacts": artifacts,
    }


def exercise_boundaries(client, directory):
    """A real 30-option forward; actual overlong/empty payloads must fail before forward."""
    raw = client.client
    payload = {
        "state": {"text": {"body": "The record describes category 12."}},
        "question": {
            "kind": "choice",
            "text": "Which category does the text describe?",
            "allowed_answers": [f"category {i}" for i in range(30)],
        },
    }
    request = IntendedRequest(
        canonical_json(payload),
        canonical_json(client.settings),
        canonical_json({"dataset_id": "EXP0_SYNTHETIC_WIDTH"}),
    )
    cache = ResponseCache(directory / "EXP0_SYNTHETIC_WIDTH.sqlite3", "EXP0_SYNTHETIC_WIDTH")
    cached_evaluate(cache, client, request)
    features = response_features(request, cache.get(request, kind="verified_response"))
    if len([k for k in features if k.startswith("p_")]) != 30:
        raise AssertionError("The 30-class test lost output probabilities")
    checks = {}
    for name, text in (
        ("empty", {}),
        ("overlong", {"body": "unrelatedword " * (raw.counter.limit + 1)}),
    ):
        changed = copy.deepcopy(payload)
        changed["state"]["text"] = text
        if name == "overlong" and raw.counter(changed) <= raw.counter.limit:
            raise AssertionError("Overflow fixture is not actually over the model's window")
        try:
            raw.evaluate(replace(request, payload_json=canonical_json(changed)))
        except ValueError as exc:
            expected = "Empty text" if name == "empty" else "Context overflow"
            if expected not in str(exc):
                raise
            checks[f"{name}_rejected"] = True
        else:
            raise AssertionError(f"{name} input reached the model instead of being rejected")
    return {"complete_30_class_vector": True, **checks}


def real_sample(evidence, counts):
    """One first and one longest input per dataset, selected without target information."""
    with closing(sqlite3.connect(counts.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        return [
            (
                item,
                {
                    0,
                    db.execute(
                        "SELECT input_no FROM counts WHERE dataset_id=? ORDER BY input_tokens DESC,input_no LIMIT 1",
                        (item["dataset_id"],),
                    ).fetchone()[0],
                },
            )
            for item in evidence["datasets"]
            if item["unique_inputs"]
        ]


def exercise_real_inputs(client, sample, log):
    records = []
    for item, numbers in sample:
        dataset_id = item["dataset_id"]
        cache = ResponseCache.for_dataset(dataset_id)
        added = reused = 0
        with file_lock(cache.path.with_suffix(".extraction.lock"), timeout=1):
            for record in iter_prepared_inputs(
                paths.dataset_question_path(dataset_id),
                input_set_id=item["input_set_id"],
                input_numbers=numbers,
            ):
                request = prepared_request(
                    record, settings=client.settings, provenance={"dataset_id": dataset_id}
                )
                status = cached_evaluate(cache, client, request)
                added += status == "completed"
                reused += status == "cache_hit"
                reopened = ResponseCache.for_dataset(dataset_id)
                cached_evaluate(reopened, CacheOnlyClient(), request)
        if added + reused != len(numbers):
            raise AssertionError("Real GPU sample coverage is incomplete")
        result = {
            "dataset_id": dataset_id,
            "new": added,
            "reused": reused,
            "first_and_longest_verified": True,
            "reopened_cache_hit": True,
        }
        records.append(result)
        append_event(log, "dataset_checked", **result)
        print(
            f"{dataset_id}: {added} GPU responses, {reused} cached; restart check passed",
            flush=True,
        )
    return records


def checked_inputs(provider):
    """Read-only prerequisite check, shared by the login CLI and GPU execution."""
    evidence, counts = token_evidence(provider, ExactCounter(provider))
    from src.cluster.token_audit import selected_inputs

    if {i["dataset_id"] for i in selected_inputs()} != {
        i["dataset_id"] for i in evidence["datasets"]
    }:
        raise ValueError("Token evidence does not cover every selected dataset")
    return evidence, counts


def run(provider, *, execute=False):
    if not execute or not paths.on_vsc() or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU contracts require --execute inside a VSC GPU allocation")
    cfg = load_yaml("experiment_0/gpu_contracts")
    if provider not in cfg["providers"] or cfg["full_extraction_authorized"] is not False:
        raise ValueError("Unsupported GPU test configuration")
    if failures := storage_policy_failures(os.environ.get("SLURM_CLUSTER_NAME")):
        raise ValueError("; ".join(failures))
    inference_environment()
    run_id = f"{provider}_{uuid4().hex}"
    manifest = (
        paths.manifests_dir("experiment_0")
        / "before_extraction"
        / "gpu_contracts"
        / f"{run_id}.json"
    )
    log = paths.logs_dir("experiment_0") / "before_extraction" / "gpu_contracts" / f"{run_id}.log"
    directory = paths.require_project_output(
        paths.results_dir("gpu_contracts", run_id, phase="experiment_0")
    )
    report = {
        "run_id": run_id,
        "provider": provider,
        "status": "running",
        "created_utc": datetime.now(UTC).isoformat(),
        "job_id": os.environ["SLURM_JOB_ID"],
        "node": os.environ.get("SLURMD_NODENAME"),
        "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
        "configuration": cfg,
        "configuration_sha256": digest(cfg),
        "full_experiment_0_passed": False,
        "full_extraction_authorized": False,
        "results_directory": str(directory),
        "synthetic": [],
        "real_datasets": [],
    }
    client = metered = None
    from src.cluster.evidence import current_identity

    report["validation_identity"] = current_identity()
    started = time.perf_counter()
    write_json(manifest, report)
    append_event(log, "started", run_id=run_id, provider=provider)
    try:
        evidence, counts = checked_inputs(provider)
        sample = real_sample(evidence, counts)
        # Existing contracts plus 13 label-blind real-text robustness forwards.
        if 70 + sum(len(numbers) for _, numbers in sample) > cfg["max_live_requests"]:
            raise ValueError("GPU contract plan exceeds the configured request budget")
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Loading pinned {provider} on the allocated GPU (offline)", flush=True)
        client = SelfHostedClient(provider, allow_model_runs=True)
        metered = MeasuredClient(client, cfg["max_live_requests"])
        report.update(
            settings=client.settings,
            settings_sha256=digest(client.settings),
            token_audit_id=evidence["audit_id"],
            load_seconds=client.load_seconds,
            gpu={
                "name": client.torch.cuda.get_device_name(),
                "torch_cuda": client.torch.version.cuda,
                "memory_bytes": client.torch.cuda.get_device_properties(0).total_memory,
            },
        )
        write_json(manifest, report)
        for dataset in synthetic_datasets():
            report["synthetic"].append(exercise_feature_pipeline(metered, dataset, directory))
            print(
                f"{provider}: {dataset.metadata.task_type} GPU, cache and feature checks passed",
                flush=True,
            )
            write_json(manifest, report)
        report["boundaries"] = exercise_boundaries(metered, directory)
        semantic_cache = ResponseCache(directory / "EXP0_SEMANTICS.sqlite3", "EXP0_SEMANTICS")
        report["semantic_diagnostics"] = exercise_semantics(metered, semantic_cache)
        write_json(manifest, report)
        if not report["semantic_diagnostics"]["passed"]:
            raise ValueError(
                "Known-answer diagnostics need review; inspect saved raw answers before blaming the decoder"
            )
        report["prompt_robustness"] = exercise_robustness(metered)
        from src.cluster.semantic_contracts import exercise_real_robustness

        report["real_prompt_robustness"] = exercise_real_robustness(
            metered, directory / "real_robustness"
        )
        report["batch_parity"] = exercise_batch_parity(
            metered, tolerance=cfg["batch_probability_tolerance"]
        )
        write_json(manifest, report)
        report["real_datasets"] = exercise_real_inputs(metered, sample, log)
        report["status"] = "gpu_contracts_passed_requires_review"
        report["runtime_token_count_parity"] = True  # SelfHostedClient rejects every mismatch.
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        append_event(log, "failed", error_type=type(exc).__name__)
        raise
    finally:
        report["finished_utc"] = datetime.now(UTC).isoformat()
        report["seconds"] = time.perf_counter() - started
        if metered:
            report.update(live_requests=metered.calls, synchronized_request_seconds=metered.seconds)
        if client:
            report.update(
                peak_gpu_allocated_bytes=client.torch.cuda.max_memory_allocated(),
                peak_gpu_reserved_bytes=client.torch.cuda.max_memory_reserved(),
            )
        write_json(manifest, report)
        print(f"GPU contract report: {manifest}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=SELF_HOSTED)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument(
        "--check-inputs",
        action="store_true",
        help="Read-only validation of saved counts and input hashes; no GPU or inference",
    )
    args = parser.parse_args()
    if args.check_inputs:
        evidence, counts = checked_inputs(args.provider)
        print(
            canonical_json(
                {
                    "status": "token_input_evidence_passed",
                    "provider": args.provider,
                    "token_audit_id": evidence["audit_id"],
                    "datasets": len(evidence["datasets"]),
                    "counts_database": str(counts),
                    "inference_executed": False,
                    "gpu_execution_verified": False,
                }
            )
        )
    elif args.execute:
        run(args.provider, execute=True)
    else:
        print(
            "Prepared real GPU tests. --execute requires an allocated GPU, staged weights and complete token evidence. No inference performed."
        )


if __name__ == "__main__":
    main()
