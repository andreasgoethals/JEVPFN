"""Read-only, complete prepared-input context screening; never permission to truncate.

Character ratios are planning proxies, not token counts or upper bounds. Exact counters
must include a provider's full rendered prompt (question, options, markers and state).
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from contextlib import closing
from datetime import UTC, datetime

import pandas as pd

from src.jev.inputs import _read_set, input_origins, iter_prepared_inputs
from src.jev.provider import QUESTION_ID, semantic_question
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import canonical_json, write_json

CONTEXT_LIMITS = {
    "jev": 32000,
    "liquid": 32768,
    "perplexity": 8192,
    "cloudflare": 24576,
    "strands": 4096,
    "microsoft": 32768,
}


def shared_wire(payload):
    """Common semantic body; provider deployment names/transport headers are excluded."""
    return {
        "state": payload["state"],
        "questions": {QUESTION_ID: semantic_question(payload["question"])},
    }


def require_context(payload, *, counters, limits):
    """Fail before dispatch if any selected provider lacks an exact counter or cannot fit.

    The caller supplies reviewed runtime counters; char/3 is deliberately not a counter.
    No row, text, question or answer option is dropped or shortened by this function.
    """
    if not limits or set(counters) != set(limits):
        raise ValueError("Exact rendered-input counters are required for every selected provider")
    counts = {}
    for provider, limit in limits.items():
        n = counters[provider](payload)
        if isinstance(n, bool) or not isinstance(n, int) or n < 1 or limit < 1:
            raise ValueError("Invalid context token count or limit")
        if n > limit:
            raise ValueError(f"Context overflow for {provider}: {n} > {limit}; no truncation")
        counts[provider] = n
    return counts


def screen_prepared(folder=None, *, progress=print):
    """Stream all exact unique inputs, including documentary metadata and complete rubrics."""
    folder = folder or paths.decision_inputs_dir()
    # Use selected IDs only; removed datasets/legacy folders can never enter this audit.
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries

    entries = selected_entries(load_catalog(), load_yaml("feature_creation"))
    datasets = []
    for entry in entries:
        path = folder / f"{entry['id']}.sqlite3"
        counts = {str(r): {p: 0 for p in CONTEXT_LIMITS} for r in (3, 4, 5)}
        longest, total_chars, n = 0, 0, 0
        examples = []
        input_set_id = None
        for record in iter_prepared_inputs(path):
            input_set_id = record["input_set_id"]
            length = len(canonical_json(shared_wire(record["payload"])))
            n += 1
            total_chars += length
            longest = max(longest, length)
            for ratio in (3, 4, 5):
                proxy = math.ceil(length / ratio)
                for provider, limit in CONTEXT_LIMITS.items():
                    counts[str(ratio)][provider] += int(proxy > limit)
            if math.ceil(length / 3) > min(CONTEXT_LIMITS.values()):
                examples.append(
                    {
                        "input_no": record["input_no"],
                        "payload_sha256": record["payload_sha256"],
                        "characters": length,
                    }
                )
            if progress and n % 100000 == 0:
                progress(f"  {entry['id']}: {n:,} unique inputs screened", flush=True)
        datasets.append(
            {
                "dataset_id": entry["id"],
                "input_set_id": input_set_id,
                "unique_inputs": n,
                "total_semantic_wire_characters": total_chars,
                "max_semantic_wire_characters": longest,
                "overflow_proxy_counts": counts,
                "longest_screened_inputs": sorted(examples, key=lambda x: -x["characters"])[:10],
            }
        )
        if progress:
            progress(f"{entry['id']}: {n:,} unique inputs; char/3 flags {counts['3']}", flush=True)
    return {
        "schema": "jevpfn.context_screen.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "method": "canonical shared state/questions JSON characters divided by 3, 4 or 5",
        "exact_token_counts": False,
        "not_a_safe_upper_bound": True,
        "provider_wrapper_overhead_included": False,
        "microsoft_limit_status": "OpenRouter reference; Foundry deployment unverified",
        "context_limits_reference": CONTEXT_LIMITS,
        "unique_inputs": sum(d["unique_inputs"] for d in datasets),
        "overflow_proxy_counts": {
            str(r): {
                p: sum(d["overflow_proxy_counts"][str(r)][p] for d in datasets)
                for p in CONTEXT_LIMITS
            }
            for r in (3, 4, 5)
        },
        "datasets": datasets,
    }


def saved_screen_tables():
    """Trace the saved full scan to rows; reject stale selection/input identities.

    This reads the scan, not every text again. Absence is explicit; neither characters
    nor their ratios are relabelled as exact tokens or confirmed overflows.
    """
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries

    path = paths.manifests_dir("feature_creation") / "context_screen.json"
    if not path.is_file():
        return (
            {"status": "not_screened", "command": "python -m src.jev.context"},
            pd.DataFrame(),
            pd.DataFrame(),
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    selected = {e["id"] for e in selected_entries(load_catalog(), load_yaml("feature_creation"))}
    if {d["dataset_id"] for d in report["datasets"]} != selected:
        raise ValueError("Context screen dataset selection is stale; rerun src.jev.context")
    examples, source_rows = [], 0
    for dataset in report["datasets"]:
        source = paths.dataset_question_path(dataset["dataset_id"])
        with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            identity, recipe, summary, _ = _read_set(db, None)
        if (
            identity != dataset["input_set_id"]
            or summary["unique_inputs"] != dataset["unique_inputs"]
        ):
            raise ValueError("Context screen input version is stale; rerun src.jev.context")
        source_rows += recipe["row_count"]
        for warning in dataset["longest_screened_inputs"]:
            origins = list(input_origins(source, warning["input_no"], input_set_id=identity))
            examples.append(
                {
                    "dataset_id": dataset["dataset_id"],
                    **warning,
                    "source_row_slots": len(origins),
                    "source_rows": len({o["row_id"] for o in origins}),
                    "first_origins": canonical_json(origins[:5]),
                    "confirmed_token_overflow": False,
                }
            )
    scenarios = pd.DataFrame(
        [
            {
                "characters_per_token": int(ratio),
                "provider": provider,
                "reference_window": report["context_limits_reference"][provider],
                "warning_unique_inputs": count,
                "warning_percent": count / report["unique_inputs"] * 100,
            }
            for ratio, counts in report["overflow_proxy_counts"].items()
            for provider, count in counts.items()
        ]
    )
    return (
        {
            "status": "approximate_screen_only",
            "datasets": len(selected),
            "source_rows": source_rows,
            "unique_inputs": report["unique_inputs"],
            "created_utc": report["created_utc"],
            "exact_token_counts": False,
            "provider_wrapper_overhead_included": False,
            "not_a_safe_upper_bound": True,
            "scope": "All frozen nonempty per-column and joint inputs; Zomato and Spotify excluded",
        },
        scenarios,
        pd.DataFrame(examples),
    )


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    report = screen_prepared()
    destination = paths.manifests_dir("feature_creation") / "context_screen.json"
    write_json(destination, report)
    print(json.dumps({k: v for k, v in report.items() if k != "datasets"}, indent=2))
    print(destination)


if __name__ == "__main__":
    main()
