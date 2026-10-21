"""Explicit local Jev pilot/full driver over frozen, deduplicated inputs.

Preview: python -m src.jev.extraction
Execution additionally requires reviewed account/context/budget configuration and --execute.
"""

import argparse
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from uuid import uuid4

from src.cluster.token_audit import selected_inputs
from src.jev.cache import ResponseCache
from src.jev.extraction import DatasetExtractor, RateGate, TypeSafeTransport, campaign_limits
from src.jev.inputs import _read_set, iter_prepared_inputs, prepared_request
from src.jev.provider import MAPPING_VERSION, api_preview
from src.jev.templates import TEMPLATE_VERSION
from src.utils import paths
from src.utils.acceptance import reviewed_acceptance
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import canonical_json, digest, write_json


class ReviewedContextEnvelope:
    """An explicitly verified request-size envelope, never a pretend exact tokenizer.

    Legacy reviewed_byte_envelope needs evidence for a byte/token upper envelope.
    provider_enforced uses the documented HTTP body ceiling and server context rejection.
    It makes no claim to count tokens locally; HTTP 400 stops without truncation or retry.
    """

    def __init__(self, cfg):
        self.contract = cfg["live_extraction"].get("context_envelope", {})
        c = self.contract
        if (
            c.get("verified") is not True
            or c.get("model") != cfg["jev"]["model"]
            or not c.get("evidence")
        ):
            raise ValueError("A reviewed model-specific context envelope and evidence are required")
        if type(c.get("max_request_utf8_bytes")) is not int or c["max_request_utf8_bytes"] < 1:
            raise ValueError("Set a verified positive complete-request byte ceiling")
        if (
            c.get("max_billed_input_tokens")
            != cfg["live_extraction"]["max_billed_input_tokens_per_attempt"]
        ):
            raise ValueError("Context and spending bounds disagree")
        mode = c.get("mode", "reviewed_byte_envelope")
        if mode not in {"reviewed_byte_envelope", "provider_enforced"}:
            raise ValueError("Unknown context policy")
        if mode == "provider_enforced" and (
            c["model"] != "jev-1.13.0"
            or c["max_request_utf8_bytes"] != 32 * 1024 * 1024
            or c["max_billed_input_tokens"] != 64000
            or c.get("per_question_context_tokens") != 32000
        ):
            raise ValueError("Provider-enforced policy differs from the pinned documented limits")

    def __call__(self, request):
        wire = api_preview(request)
        if wire["model"] != self.contract["model"]:
            raise ValueError("Context envelope belongs to a different model")
        if len(canonical_json(wire).encode("utf-8")) > self.contract["max_request_utf8_bytes"]:
            raise ValueError("Request exceeds the reviewed body-size ceiling; no truncation")


def pilot_input_numbers(path, input_set_id):
    """First/middle/last rows and the longest stored text group; no label access."""
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        _, recipe, _, _ = _read_set(db, input_set_id)
        positions = (0, recipe["row_count"] // 2, recipe["row_count"] - 1)
        numbers = {
            r[0]
            for r in db.execute(
                "SELECT input_no FROM row_map WHERE input_set_id=? AND row_position IN (?,?,?) AND input_no IS NOT NULL",
                (input_set_id, *positions),
            )
        }
        longest = db.execute(
            "SELECT input_no FROM unique_inputs WHERE input_set_id=? ORDER BY length(text_json) DESC,input_no LIMIT 1",
            (input_set_id,),
        ).fetchone()
        if longest:
            numbers.add(longest[0])
    return numbers


def run(cfg, *, execute=False, full=False, dataset_ids=None, transport_factory=TypeSafeTransport):
    if not execute or not cfg["live_extraction"].get("enabled"):
        raise RuntimeError("Jev execution requires explicit opt-in and enabled live configuration")
    if paths.on_vsc():
        raise RuntimeError("The selected Jev workflow runs locally")
    from src.cluster.freeze import check_if_present

    check_if_present()
    review = cfg["live_extraction"]
    if full and (
        review.get("full_feature_creation_enabled") is not True
        or review.get("pilot_accepted") is not True
    ):
        raise ValueError("Full Jev extraction requires accepted pilot evidence and its own switch")
    validator = ReviewedContextEnvelope(cfg)
    limits = campaign_limits(cfg)
    items = selected_inputs()
    if dataset_ids:
        if set(dataset_ids) - {i["dataset_id"] for i in items}:
            raise ValueError("Unselected dataset")
        items = [i for i in items if i["dataset_id"] in dataset_ids]
    if set(i["dataset_id"] for i in items) - limits.keys():
        raise ValueError("Every selected dataset needs an approved spending allocation")
    selection = {
        i["dataset_id"]: None
        if full
        else pilot_input_numbers(paths.dataset_question_path(i["dataset_id"]), i["input_set_id"])
        for i in items
    }
    if not full and sum(map(len, selection.values())) > review["max_pilot_requests"]:
        raise ValueError("Pilot exceeds its approved request ceiling")
    settings = {
        "provider": "jev",
        "model": cfg["jev"]["model"],
        "version": cfg["jev"]["version"],
        "template_version": TEMPLATE_VERSION,
        "api_mapping_version": MAPPING_VERSION,
        "request_configuration": cfg["jev"]["request_configuration"],
    }
    if full:
        reviewed_acceptance(
            review.get("accepted_pilot_manifest"),
            stage="jev_pilot",
            expected={
                "settings_sha256": digest(settings),
                "input_sets": {i["dataset_id"]: i["input_set_id"] for i in items},
                "context_envelope_sha256": digest(review["context_envelope"]),
            },
        )
    phase, run_id = ("feature_creation" if full else "experiment_0"), uuid4().hex
    report = {
        "run_id": run_id,
        "provider": "jev",
        "status": "running",
        "mode": "full" if full else "pilot",
        "settings": settings,
        "configuration_sha256": digest(cfg),
        "datasets": [],
        "created_utc": datetime.now(UTC).isoformat(),
    }
    report.update(
        settings_sha256=digest(settings),
        input_sets={i["dataset_id"]: i["input_set_id"] for i in items},
        context_envelope_sha256=digest(review["context_envelope"]),
    )
    manifest = paths.manifests_dir(phase) / "hosted" / f"{run_id}.json"
    write_json(manifest, report)
    gate = RateGate(next(iter(limits.values())))
    client = None
    # One local campaign owns the account rate gate; merges wait for dataset writers.
    try:
        with file_lock(paths.jev_cache_path() / ".jev_campaign.lock", timeout=1):
            client = transport_factory(cfg, allow_network=True)
            for item in items:
                dataset_id = item["dataset_id"]
                with file_lock(
                    paths.dataset_cache_path(dataset_id).with_suffix(".extraction.lock"), timeout=1
                ):
                    cache = ResponseCache.for_dataset(dataset_id)
                    extractor = DatasetExtractor(
                        cache,
                        client,
                        limits[dataset_id],
                        gate=gate,
                        validate_context=validator,
                        phase=phase,
                        policy_revision_manifest=review.get("spending_policy_revision_manifest"),
                    )
                    requests = (
                        prepared_request(
                            record, settings=settings, provenance={"dataset_id": dataset_id}
                        )
                        for record in iter_prepared_inputs(
                            paths.dataset_question_path(dataset_id),
                            input_set_id=item["input_set_id"],
                            input_numbers=selection[dataset_id],
                        )
                    )
                    counts = extractor.run(requests)
                    report["datasets"].append(
                        {
                            "dataset_id": dataset_id,
                            "input_set_id": item["input_set_id"],
                            "spending_policy_sha256": extractor.policy_sha256,
                            "counts": counts,
                        }
                    )
                    write_json(manifest, report)
                    print(f"{dataset_id}: {counts}", flush=True)
                    if set(counts) - {"completed", "cache_hit", "recovered_response"}:
                        report["status"] = "stopped_requires_review"
                        break
                    if full:
                        from src.jev.self_hosted import _materialize

                        _materialize(dataset_id, settings, cache, item["input_set_id"])
            else:
                report["status"] = "completed_requires_review"
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        write_json(manifest, report)
        if client is not None and callable(getattr(client, "close", None)):
            client.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--config", default="feature_creation")
    args = parser.parse_args(argv)
    if not args.execute:
        print(
            "Jev is disabled. Configure live_extraction account rates, budgets and verified context envelope; then --execute runs a bounded pilot. --full additionally requires pilot acceptance."
        )
        return 0
    result = run(load_yaml(args.config), execute=True, full=args.full, dataset_ids=args.dataset)
    return int(result["status"] != "completed_requires_review")


if __name__ == "__main__":
    raise SystemExit(main())
