"""Owner-run A2 Jev transport with separate, bounded ledgers for fresh repeats."""

import argparse
import hashlib
import json
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path

from src.jev.cache import ResponseCache
from src.jev.extraction import DatasetExtractor, TypeSafeTransport, campaign_limits
from src.jev.hosted import ReviewedContextEnvelope, rate_gate
from src.jev.prefix_pilot import ledger_total
from src.jev.provider import MAPPING_VERSION, api_preview, pinned_jev_model
from src.jev.requests import IntendedRequest
from src.jev.spending import JEV_CEILING_USD, campaign_committed
from src.jev.templates import TEMPLATE_VERSION
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import canonical_json, digest, write_json

VARIANTS = (
    "original",
    "repeat_1",
    "repeat_2",
    "paraphrase_1",
    "paraphrase_2",
    "paraphrase_3",
    "field_order",
    "option_order",
)


class HostedReplay:
    def __init__(self, cfg, dataset, *, pilot, transport_factory=TypeSafeTransport):
        review = cfg["live_extraction"]
        if (
            paths.on_vsc()
            or review.get("purpose") != "rq1b_controlled_replays"
            or review.get("workers") != 1
            or not review.get("enabled")
        ):
            raise ValueError("Use a separately funded local A2 configuration with one worker")
        if not review.get("accepted_pilot_manifest"):
            raise ValueError("A reviewed local Jev interface pilot is required")
        limits = campaign_limits(cfg)
        self.limit = replace(
            limits[dataset],
            dataset_budget_usd=limits[dataset].dataset_budget_usd / (2 * len(VARIANTS)),
        )
        self.gate = rate_gate(review, self.limit)
        self.validator = ReviewedContextEnvelope(cfg)
        self.cfg, self.dataset, self.pilot = cfg, dataset, pilot
        self.settings = dict(
            provider="jev",
            model=cfg["jev"]["model"],
            version=cfg["jev"]["version"],
            template_version=TEMPLATE_VERSION,
            api_mapping_version=MAPPING_VERSION,
            request_configuration=cfg["jev"]["request_configuration"],
        )
        pinned_jev_model(self.settings)
        self.factory, self.transport, self.workers = transport_factory, None, {}
        self.design_verified = False
        self.directory = paths.results_dir(
            "hosted_ledgers", dataset, "pilot" if pilot else "scientific", phase="A2"
        )

    def input_signature(self, payload):
        request = IntendedRequest(
            json.dumps(payload, ensure_ascii=False), canonical_json(self.settings), "{}"
        )
        wire = json.dumps(api_preview(request), ensure_ascii=False, separators=(",", ":")).encode()
        return hashlib.sha256(wire).hexdigest()

    def start(self, design):
        from src.utils.acceptance import reviewed_acceptance

        self.design_verified = False
        pilot_review = reviewed_acceptance(
            self.cfg["live_extraction"]["accepted_pilot_manifest"],
            stage="jev_pilot",
            expected={
                "settings_sha256": digest(self.settings),
                "context_envelope_sha256": digest(self.cfg["live_extraction"]["context_envelope"]),
            },
        )
        if pilot_review.get("input_sets", {}).get(self.dataset) != design["source"]:
            raise ValueError("Frozen input set differs from the signed Jev pilot")
        if design.get("settings") != digest(self.settings):
            raise ValueError("Hosted replay design differs from the pinned Jev settings")
        budget = self.cfg["live_extraction"]
        contract = {
            "limit": budget["campaign_spending_limit_usd"],
            "allocations": budget["dataset_budget_allocations_usd"],
        }
        budget_path = self.directory.parent.parent / "budget.json"
        if budget_path.exists() and json.loads(budget_path.read_text(encoding="utf-8")) != contract:
            raise ValueError(
                "A2 spending allocations changed; review the cumulative ledger before continuing"
            )
        write_json(budget_path, contract)
        # Retain all versions in the existing per-variant spending ledger. Its request
        # keys already include the exact settings, and its allocation is cumulative.
        pinned = self.directory / f"design.{self.settings['model']}.json"
        legacy = self.directory / "design.json"
        if not pinned.exists() and legacy.is_file():
            old = json.loads(legacy.read_text(encoding="utf-8"))
            if old != design and not self._different_legacy_release(old):
                raise ValueError("Hosted replay design changed or the legacy release is unidentified")
        if pinned.exists() and json.loads(pinned.read_text(encoding="utf-8")) != design:
            raise ValueError(
                "Hosted replay design changed; preserve existing spending and review before resuming"
            )
        write_json(pinned, design)
        self.design_verified = True

    def _different_legacy_release(self, design):
        """Identify the legacy marker through the request settings its digest binds."""
        prior_hash = design.get("settings")
        if not prior_hash or prior_hash == digest(self.settings):
            return False
        for store in sorted(self.directory.glob("*.sqlite3")):
            with closing(sqlite3.connect(store.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                if not db.execute("SELECT 1 FROM sqlite_master WHERE name='extraction_jobs'").fetchone():
                    continue
                for (raw,) in db.execute("SELECT DISTINCT json_extract(request_json,'$.settings') FROM extraction_jobs"):
                    settings = json.loads(raw)
                    if settings.get("provider") == "jev" and digest(settings) == prior_hash:
                        return pinned_jev_model(settings) != self.settings["model"]
        return False

    def evaluate_trial(self, request, variant):
        if not self.design_verified or json.loads(request.settings_json) != self.settings:
            raise ValueError("Verify the approved study design and Jev settings before dispatch")
        if variant not in VARIANTS:
            raise ValueError("Unknown hosted replay")
        cache_folder = paths.jev_cache_path()
        with file_lock(cache_folder / ".jev_campaign.lock", timeout=1):
            committed = campaign_committed(cache_folder, replay_folder=self.directory.parent.parent)
            return self._evaluate_locked(request, variant, committed)

    def _evaluate_locked(self, request, variant, committed):
        if self.transport is None:
            self.transport = self.factory(self.cfg, allow_network=True)
        if variant not in self.workers:
            cache = ResponseCache(self.directory / (variant + ".sqlite3"), dataset_id=self.dataset)
            self.workers[variant] = DatasetExtractor(
                cache,
                self.transport,
                self.limit,
                gate=self.gate,
                validate_context=self.validator,
                phase="A2",
            )
        worker = self.workers[variant]
        # The atomic reservation includes this ledger's previous usage itself.
        other_usage = committed - ledger_total(worker.cache.path)
        worker.available_usd = min(self.limit.dataset_budget_usd, JEV_CEILING_USD - other_usage)
        tagged = IntendedRequest(
            request.payload_json,
            request.settings_json,
            canonical_json({"dataset_id": self.dataset, "study": "rq1b", "variant": variant}),
        )
        status = worker.evaluate(tagged)
        response = worker.cache.get(tagged, kind="verified_response")
        if response is None:
            raise RuntimeError(f"Hosted replay stopped with ledger status {status}")
        return response


@contextmanager
def owner_client(cfg, dataset, pilot):
    # Serialize access to this independently budgeted study across restarts/datasets.
    root = paths.results_dir("hosted_ledgers", phase="A2")
    with file_lock(root / ".dispatcher.lock", timeout=1):
        client = HostedReplay(cfg, dataset, pilot=pilot)
        try:
            yield client
        finally:
            if client.transport is not None:
                client.transport.close()


def main():
    from src.evaluation.robustness import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.exit(
            message="Preview only. A dedicated owner-approved A2 budget and --execute are required.\n"
        )
    with owner_client(load_yaml(args.config), args.dataset, args.pilot) as client:
        print(run(args.dataset, "jev", pilot=args.pilot, acceptance=args.acceptance, client=client))


if __name__ == "__main__":
    main()
