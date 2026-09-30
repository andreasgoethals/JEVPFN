"""Prepare public data only: python -m src.data.prepare [--offline].

--create-catalog is an explicit initial bootstrap, never a silent upstream update.
It extracts the official core list, slugs and summary using AST literals (no remote code).
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yaml

from src.data.loaders import load_catalog, load_dataset
from src.utils import paths
from src.utils.serialization import file_sha256, write_json

COMMIT = "3bb95079a6146ed35d45187e98bcad5b063273a1"
FILES = {
    "core.py": "multabench/datasets/all_multabench_datasets.py",
    "ids.py": "multabench/datasets/all_datasets.py",
    "summary.csv": "multabench/leaderboard/results/datasets_summary.csv",
    "summary_script.py": "multabench/scripts/do_dataset_summary.py",
    "feat_types.py": "multabench/preprocessing/feat_types.py",
    "nulls.py": "multabench/utils/nulls.py",
    "load.py": "multabench/benchmark/load.py",
    "curation.py": "multabench/benchmark/utils/curation.py",
}


def _assignment(tree, name):
    return next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
    )


def create_catalog() -> None:
    target = paths.config_path("exploration/datasets")
    if target.exists():
        raise FileExistsError("Catalog already exists; review updates explicitly.")
    texts, sources = {}, {}
    for local, remote in FILES.items():
        url = f"https://raw.githubusercontent.com/alanarazi7/MulTaBench/{COMMIT}/{remote}"
        path = paths.source_snapshot_path(COMMIT, local)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        texts[local] = path.read_text(encoding="utf-8")
        sources[local] = {"url": url, "sha256": file_sha256(path)}
    core = [e.attr for e in _assignment(ast.parse(texts["core.py"]), "MULTABENCH_CORE_TEXT").elts]
    cls = next(
        n
        for n in ast.parse(texts["ids.py"]).body
        if isinstance(n, ast.ClassDef) and n.name == "MulTaBenchDatasetID"
    )
    ids = {
        n.targets[0].id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.Assign)
    }
    names = ast.literal_eval(_assignment(ast.parse(texts["summary_script.py"]), "_FRIENDLY_NAMES"))
    summaries = pd.read_csv(io.StringIO(texts["summary.csv"])).set_index("Dataset")
    entries = []
    for number, dataset_id in enumerate(core, 1):
        slug = ids[dataset_id]
        with urllib.request.urlopen(
            f"https://www.kaggle.com/api/v1/datasets/view/chico89/{slug}", timeout=60
        ) as response:
            info = json.load(response)
        summary = summaries.loc[names[dataset_id]]
        entries.append(
            {
                "id": dataset_id,
                "name": names[dataset_id],
                "slug": slug,
                "directory": f"{number:02d}_" + slug.removeprefix("multabench-").replace("-", "_"),
                "version": int(info["currentVersionNumber"]),
                "paper_summary": {
                    "rows": int(summary["N"]),
                    "text_columns": int(summary["Text cols"]),
                    "non_text_columns": int(summary["Struct."]),
                },
            }
        )
        print(f"Pinned {dataset_id} v{entries[-1]['version']}", flush=True)
    catalog = {
        "collection": "multabench_core_text",
        "upstream_commit": COMMIT,
        "upstream_tag": "paper_version",
        "tabstar_numeric_rule_version": "1.1.15",
        "sources": sources,
        "datasets": entries,
    }
    target.write_text(
        yaml.safe_dump(catalog, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n"
    )


def prepare(entry: dict, *, allow_download: bool = True) -> dict:
    try:
        dataset = load_dataset(entry, allow_download=allow_download)
        result = {
            "dataset": entry["id"],
            "status": "ok",
            "rows": len(dataset.frame),
            "text_columns": list(dataset.metadata.text_columns),
            "issues": dataset.issues,
            "source": dataset.source,
        }
    except Exception as exc:
        result = {
            "dataset": entry["id"],
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(f"{result['status']}: {entry['id']}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create-catalog", action="store_true")
    parser.add_argument(
        "--offline", action="store_true", help="Verify cached data without downloads"
    )
    args = parser.parse_args()
    if args.create_catalog:
        if args.offline:
            parser.error("--create-catalog requires network access")
        create_catalog()
    catalog = load_catalog()
    with ThreadPoolExecutor(max_workers=3) as pool:
        status = list(
            pool.map(
                lambda entry: prepare(entry, allow_download=not args.offline), catalog["datasets"]
            )
        )
    if args.create_catalog and all(s["status"] == "ok" for s in status):
        for entry, result in zip(catalog["datasets"], status):
            entry["text_columns"] = result["text_columns"]
            entry["sha256"] = result["source"]["sha256"]
        paths.config_path("exploration/datasets").write_text(
            yaml.safe_dump(catalog, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
            newline="\n",
        )
    write_json(paths.manifest_path("dataset_preparation"), status)
    print(f"{sum(s['status'] == 'ok' for s in status)}/20 datasets loaded")
    return int(any(s["status"] != "ok" for s in status))


if __name__ == "__main__":
    raise SystemExit(main())
