"""Read-only analysis of explicit, bounded real-text diagnostic reports (Experiment 4).

Pass exact report files; never silently mix historical versions or dispatch new calls.
The inference entry points remain the bounded local Jev/GPU contracts in Experiment 0A.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.cluster.semantic_contracts import robustness_summary
from src.utils import paths
from src.utils.serialization import file_sha256, write_json


def analyze(reports):
    rows, identities = [], set()
    for filename in reports:
        path = Path(filename)
        sha = file_sha256(path)
        if sha in identities:
            raise ValueError("The same diagnostic report was supplied twice")
        identities.add(sha)
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") not in {
            "gpu_contracts_passed_requires_review",
            "jev_dataset_robustness_requires_review",
        }:
            raise ValueError("Cannot analyze a failed diagnostic as completed evidence")
        detail = report.get("real_prompt_robustness", report.get("prompt_robustness", {}))
        records = detail.get("probes", report.get("attempts", []))
        if not records or not all(r["probe"].startswith("real_") for r in records):
            raise ValueError(
                "Use completed real-text diagnostics, not synthetic known-answer tests"
            )
        comparison = robustness_summary(records)
        provider = report.get("provider") or report.get("settings", {}).get("provider", "jev")
        for item in comparison["comparisons"]:
            rows.append({"provider": provider, "source_sha256": sha, "source": str(path), **item})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    table = analyze(args.report)
    print(table.to_string(index=False))
    if args.save:
        destination = paths.require_project_output(paths.results_dir(phase="experiment_4"))
        destination.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination / "prompt_sensitivity.csv", index=False)
        write_json(
            paths.manifests_dir("experiment_4") / "prompt_sensitivity.json",
            {
                "inputs": args.report,
                "rows": len(table),
                "inference_executed": False,
                "scope": "small fixed label-blind sample; not a population robustness guarantee",
            },
        )


if __name__ == "__main__":
    main()
