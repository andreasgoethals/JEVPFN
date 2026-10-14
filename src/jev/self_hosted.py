"""Guarded, serial GPU pilot/extraction for five pinned open-weight candidates.

Default CLI only explains readiness. Actual inference requires --execute, enabled pilot
config, a VSC GPU allocation, staged/hash-verified weights and matching complete exact
token evidence. Full extraction has a separate switch and reviewed acceptance manifest.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import inspect
import json
import os
import sqlite3
import sys
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from src.jev.cache import ResponseCache
from src.jev.checkpoints import verify_checkpoint
from src.jev.decision_tokenization import PINS, SELF_HOSTED, ExactCounter
from src.jev.inputs import _read_set, iter_prepared_inputs, prepared_request
from src.jev.provider import QUESTION_ID, response_features, semantic_question
from src.jev.templates import TEMPLATE_VERSION
from src.utils import paths
from src.utils.config import load_config, load_yaml
from src.utils.events import append_event
from src.utils.locking import file_lock
from src.utils.serialization import canonical_json, digest, file_sha256, write_json


def inference_environment():
    """Redirect all implicit library caches before importing a model runtime."""
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    os.environ["HF_HOME"] = str(paths.checkpoints_dir("huggingface"))
    os.environ["HF_HUB_CACHE"] = str(paths.checkpoints_dir("huggingface", "hub"))
    os.environ["HF_XET_CACHE"] = str(paths.checkpoints_dir("huggingface", "xet"))
    os.environ["TORCH_HOME"] = str(paths.checkpoints_dir("torch"))


class SelfHostedClient:
    def __init__(self, provider, *, allow_model_runs=False):
        if not allow_model_runs or not paths.on_vsc() or not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError(
                "Self-hosted inference requires explicit execution and a VSC compute allocation"
            )
        from src.cluster.preflight import storage_policy_failures

        failures = storage_policy_failures(os.environ.get("SLURM_CLUSTER_NAME"))
        if failures:
            raise ValueError("; ".join(failures))
        if provider not in SELF_HOSTED:
            raise ValueError("Unsupported self-hosted provider")
        inference_environment()
        self.counter = ExactCounter(provider)
        folder, checkpoint_id = verify_checkpoint(provider)
        self.provider = provider
        self.candidate = load_yaml("extractors")["candidates"][provider]
        self.settings = {
            "provider": provider,
            "model": self.candidate["model"],
            "version": PINS[provider],
            "template_version": TEMPLATE_VERSION,
            "api_mapping_version": f"{provider}-self-hosted-v1",
            "request_configuration": {
                "checkpoint_sha256": checkpoint_id,
                "counter_id": self.counter.identity["counter_id"],
                "runtime_revision": self.candidate.get("runtime_revision"),
                "packages": {
                    d.metadata["Name"]: d.version
                    for d in importlib.metadata.distributions()
                    if d.metadata["Name"]
                },
            },
        }
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("The allocated environment has no usable CUDA GPU")
        self.torch = torch
        self.settings["request_configuration"]["torch_cuda"] = torch.version.cuda
        self.settings["request_configuration"]["inference_adapter_sha256"] = file_sha256(
            paths.REPO_ROOT / "src/jev/self_hosted.py"
        )
        started = time.perf_counter()
        if provider == "liquid":
            from transformers import AutoModel

            self.model = (
                AutoModel.from_pretrained(
                    str(folder), local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16
                )
                .to("cuda")
                .eval()
            )
        elif provider == "perplexity":
            source = folder / "source" / "src"
            sys.path.insert(0, str(source))
            from autojev.model import DecisionModel, answer

            if (
                Path(inspect.getfile(DecisionModel)).resolve()
                != (source / "autojev" / "model.py").resolve()
            ):
                raise ValueError("A different Perplexity runtime is already imported")
            if file_sha256(source / "autojev" / "model.py") != file_sha256(
                paths.tokenizer_dir(provider) / "publisher_model.py"
            ):
                raise ValueError("Perplexity runtime differs from the audited renderer")
            self.model, self.decode = DecisionModel(folder, device="cuda"), answer
        elif provider in {"cloudflare", "intern"}:
            from src.jev.open_runtimes import load_runtime

            self.model = load_runtime(provider, folder, self.counter.limit)
            self.settings["request_configuration"]["additional_runtime_sha256"] = file_sha256(
                paths.REPO_ROOT / "src/jev/open_runtimes.py"
            )
        else:
            from strands_decider import infer, prompting
            from strands_decider.modeling import StrandsDeciderModel
            from strands_decider.schema import SystemOneRequest

            for module, name in ((infer, "infer.py"), (prompting, "prompting.py")):
                if file_sha256(Path(inspect.getfile(module))) != file_sha256(
                    paths.tokenizer_dir(provider) / name
                ):
                    raise ValueError(
                        "Installed Strands runtime differs from the pinned audited source"
                    )
            self.model = infer.SystemOneEngine(
                StrandsDeciderModel.load(str(folder)),
                infer.EngineConfig(
                    device="cuda",
                    strict_window=True,
                    max_batch=1,
                    model_name=self.candidate["model"],
                ),
            )
            self.request_class = SystemOneRequest
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started

    def evaluate(self, request):
        if json.loads(request.settings_json) != self.settings:
            raise ValueError("Request differs from the loaded checkpoint/runtime")
        payload = json.loads(request.payload_json)
        count = self.counter(payload)
        if count > self.counter.limit:
            raise ValueError("Context overflow; no truncation or row removal")
        question = semantic_question(payload["question"])
        with self.torch.inference_mode():
            if self.provider == "liquid":
                raw = self.model.system_one(
                    state=canonical_json(payload["state"]), questions={QUESTION_ID: question}
                )
            elif self.provider == "strands":
                raw = self.model.evaluate(
                    self.request_class(state=payload["state"], questions={QUESTION_ID: question})
                ).model_dump(mode="json")
            elif self.provider in {"cloudflare", "intern"}:
                raw = self.model.predict(
                    {"state": payload["state"], "questions": {QUESTION_ID: question}}
                )
            else:
                row = {"state": canonical_json(payload["state"]), "question": question}
                runtime_count = self.model.prepare([row]).input_tokens
                probabilities = self.model.predict([row], batch_size=1)[0]
                raw = {
                    "answers": {QUESTION_ID: self.decode(question, probabilities)},
                    "usage": {"input_tokens": runtime_count, "output_tokens": 0},
                    "raw_probabilities": probabilities,
                }
        self.torch.cuda.synchronize()
        if raw["usage"]["input_tokens"] != count:
            raise ValueError("Actual inference token count differs from the offline audit; stop")
        response = {
            "provider": self.provider,
            "model": self.settings["model"],
            "version": self.settings["version"],
            "settings_sha256": digest(self.settings),
            "answers": raw["answers"],
            "usage": raw["usage"],
            "publisher_response": raw,
        }
        response_features(request, response)
        return response


def cached_evaluate(cache, client, request):
    """Persist a validated response before advancing. A restart reuses its exact identity."""
    cached = cache.get(request, kind="verified_response")
    if cached is not None:
        response_features(request, cached)
        return "cache_hit"
    response = client.evaluate(request)
    response_features(request, response)
    cache.put(request, response, kind="verified_response")
    return "completed"


def token_evidence(provider, counter):
    path = paths.manifests_dir("experiment_0") / "before_extraction" / "token_audit.json"
    combined = json.loads(path.read_text())
    matches = [r for r in combined["self_hosted"] if r["provider"] == provider]
    if len(matches) != 1:
        raise ValueError("Run the exact tokenizer audit for this provider first")
    report = matches[0]
    if (
        report["counter_id"] != counter.identity["counter_id"]
        or report["status"] != "counted_all_inputs"
        or report["overflow_unique_inputs"]
    ):
        raise ValueError(
            "Exact token evidence is stale, incomplete or contains unresolved overflows"
        )
    for item in report["datasets"]:
        if (
            file_sha256(paths.dataset_question_path(item["dataset_id"]))
            != item["input_file_sha256"]
        ):
            raise ValueError("Prepared input changed after token audit")
    database = paths.token_audit_dir() / f"{provider}_{report['audit_id']}.sqlite3"
    if not database.is_file():
        raise ValueError("Detailed exact counts are missing from PROJECT")
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        saved = db.execute("SELECT recipe_json FROM identity WHERE id=1").fetchone()
        if not saved or digest(json.loads(saved[0])) != report["audit_id"]:
            raise ValueError("Detailed counts do not match the token-audit identity")
        coverage = dict(db.execute("SELECT dataset_id,COUNT(*) FROM counts GROUP BY dataset_id"))
        expected = {d["dataset_id"]: d["unique_inputs"] for d in report["datasets"]}
        if coverage != {k: v for k, v in expected.items() if v}:
            raise ValueError("Detailed token counts are incomplete")
    return report, database


def pilot_numbers(source, counts, dataset_id):
    """First/middle/last source rows plus ten longest complete requests; no label access."""
    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        identity, recipe, _, _ = _read_set(db, None)
        positions = {0, recipe["row_count"] // 2, recipe["row_count"] - 1}
        numbers = {
            n
            for pos in positions
            for (n,) in db.execute(
                "SELECT input_no FROM row_map WHERE input_set_id=? AND row_position=? AND input_no IS NOT NULL",
                (identity, pos),
            )
        }
    with closing(sqlite3.connect(counts.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        numbers.update(
            r[0]
            for r in db.execute(
                "SELECT input_no FROM counts WHERE dataset_id=? ORDER BY input_tokens DESC,input_no LIMIT 10",
                (dataset_id,),
            )
        )
    return numbers


def run(provider, *, execute=False, full=False):
    cfg = load_yaml("experiment_0/self_hosted")
    if not execute or cfg["enabled"] is not True:
        raise RuntimeError("GPU pilot is disabled; explicitly enable its config and pass --execute")
    if provider not in cfg["providers"]:
        raise ValueError("Provider is not selected for this pilot")
    if cfg["batch_size"] != 1:
        raise ValueError("The initial GPU pilot currently supports serialized batch size one")
    if not paths.on_vsc() or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU extraction must run in a VSC compute allocation")
    if full and (
        cfg["full_feature_creation_enabled"] is not True
        or not cfg["accepted_before_extraction_manifest"]
    ):
        raise RuntimeError(
            "Full extraction requires reviewed Experiment 0A acceptance and its own enable switch"
        )
    inference_environment()  # Must precede the first Transformers/HF import in this process.
    counter = ExactCounter(provider)
    evidence, counts_path = token_evidence(provider, counter)
    from src.cluster.token_audit import selected_inputs

    inputs = selected_inputs()
    if {i["dataset_id"] for i in inputs} != {i["dataset_id"] for i in evidence["datasets"]}:
        raise ValueError("Token evidence differs from the selected 18 datasets")
    selection = {
        i["dataset_id"]: None
        if full
        else pilot_numbers(
            paths.dataset_question_path(i["dataset_id"]), counts_path, i["dataset_id"]
        )
        for i in inputs
    }
    if not full and sum(map(len, selection.values())) > cfg["max_pilot_requests"]:
        raise ValueError("Pilot exceeds its explicit request ceiling")
    # All cheap checks precede weight loading. GPU runtime is created once per job.
    client = SelfHostedClient(provider, allow_model_runs=True)
    if full:
        acceptance = json.loads(Path(cfg["accepted_before_extraction_manifest"]).read_text())
        if (
            acceptance.get("accepted") is not True
            or acceptance.get("settings_sha256") != digest(client.settings)
            or acceptance.get("token_audit_id") != evidence["audit_id"]
        ):
            raise ValueError("Reviewed acceptance does not match this exact runtime/input audit")
    phase = "feature_creation" if full else "experiment_0"
    run_id = uuid4().hex
    log = (
        paths.logs_dir(phase)
        / ("full_extraction" if full else "before_extraction")
        / f"{provider}_{run_id}.log"
    )
    summary = {
        "run_id": run_id,
        "provider": provider,
        "settings": client.settings,
        "token_audit_id": evidence["audit_id"],
        "mode": "full" if full else "pilot",
        "created_utc": datetime.now(UTC).isoformat(),
        "load_seconds": client.load_seconds,
        "datasets": [],
        "status": "running",
        "full_experiment_0_passed": False,
    }
    manifest = (
        paths.manifests_dir(phase)
        / ("full_extraction" if full else "before_extraction")
        / f"{provider}_{run_id}.json"
    )
    write_json(manifest, summary)
    append_event(log, "started", run_id=run_id, provider=provider)
    try:
        for item in inputs:
            dataset_id = item["dataset_id"]
            source = paths.dataset_question_path(dataset_id)
            cache = ResponseCache.for_dataset(dataset_id)
            started, completed, reused = time.perf_counter(), 0, 0
            # One writer per dataset across providers. OS releases the lock after a crash.
            with file_lock(cache.path.with_suffix(".extraction.lock"), timeout=1):
                for record in iter_prepared_inputs(
                    source, input_set_id=item["input_set_id"], input_numbers=selection[dataset_id]
                ):
                    request = prepared_request(
                        record, settings=client.settings, provenance={"dataset_id": dataset_id}
                    )
                    status = cached_evaluate(cache, client, request)
                    completed += status == "completed"
                    reused += status == "cache_hit"
                    if (completed + reused) % 100 == 0:
                        append_event(
                            log,
                            "progress",
                            dataset_id=dataset_id,
                            completed=completed,
                            reused=reused,
                        )
                        print(
                            f"{provider} / {dataset_id}: {completed} new, {reused} reused",
                            flush=True,
                        )
                if full:
                    _materialize(dataset_id, client.settings, cache, item["input_set_id"])
            result = {
                "dataset_id": dataset_id,
                "completed": completed,
                "reused": reused,
                "seconds": time.perf_counter() - started,
            }
            summary["datasets"].append(result)
            write_json(manifest, summary)
            append_event(log, "dataset_complete", **result)
            print(result, flush=True)
        summary["status"] = "completed_requires_review"
    except BaseException as exc:
        summary.update(status="failed", error_type=type(exc).__name__)
        append_event(log, "failed", error_type=type(exc).__name__)
        raise
    finally:
        summary["peak_gpu_allocated_bytes"] = client.torch.cuda.max_memory_allocated()
        write_json(manifest, summary)
    return summary


def _materialize(dataset_id, settings, cache, input_set_id):
    from src.data.loaders import load_catalog, load_dataset
    from src.jev.features import materialize_cached_block, save_block
    from src.jev.question_bank import load_question_bank

    entry = next(e for e in load_catalog()["datasets"] if e["id"] == dataset_id)
    dataset = load_dataset(entry, allow_download=False)
    with closing(
        sqlite3.connect(
            paths.dataset_question_path(dataset_id).resolve().as_uri() + "?mode=ro", uri=True
        )
    ) as db:
        _, recipe, _, _ = _read_set(db, input_set_id)
    if (
        dataset.source["data_identity"] != recipe["data_identity"]
        or dataset.metadata.fingerprint != recipe["task_schema_sha256"]
    ):
        raise ValueError("Raw data differs from the prepared inputs used for extraction")
    bank = load_question_bank(paths.decision_inputs_dir(), bank_id=recipe["bank_id"])
    cfg = load_config("feature_creation")
    for mode in ("per_column", "joint"):
        mode_cfg = copy.deepcopy(cfg)
        mode_cfg["jev"]["representation"] = mode
        save_block(
            materialize_cached_block(
                dataset,
                mode_cfg,
                cache,
                question_bank=bank,
                question_variant=recipe["question_variant"],
                provider_settings=settings,
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=SELF_HOSTED, default="liquid")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            "Prepared GPU pilot. Stage weights, run exact tokens, enable experiment_0/self_hosted, then submit the pilot Slurm job. No inference performed."
        )
        return 0
    run(args.provider, execute=True, full=args.full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
