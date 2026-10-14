"""Match the optimized offline budget to the actual serialized request builder."""

import copy
import math
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.jev.provider import api_preview
from src.jev.requests import build_requests
from src.jev.review import review_dataset
from src.utils.config import load_config
from src.utils.serialization import canonical_json


@pytest.mark.parametrize("texts", [1, 2])
@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize(
    "task,labels",
    [
        ("cls", [0, 1, 0, 1]),
        ("cls", ["a", "b", "c", "a"]),
        ("reg", [0.3, -1.5, 9.0, 2.1]),
    ],
)
def test_budget_matches_real_wire_and_cache_identity(texts, task, labels, frozen):
    frame = pd.DataFrame(
        {
            "text": ['a"b', 'a"b', "  ", None],
            "context": [1.0, 1.0, float("inf"), float("nan")],
            "target": labels,
        }
    )
    text_columns = ["text"]
    if texts == 2:
        frame["other_text"] = ["longer", "", "longer", ""]
        text_columns.append("other_text")
    official = {"target": "target", "task_type": task}
    metadata = extract_metadata(
        frame, official, dataset_id="fixture", dataset_name="Fixture", text_columns=text_columns
    )
    dataset = LoadedDataset(frame, metadata, official, {}, [])
    cfg = load_config("feature_creation")
    entry, apply = None, lambda request: request
    if frozen:
        from src.jev.question_bank import build_question_bank, compile_frozen_question
        from src.utils.config import load_yaml

        design = load_yaml("question_design")
        design.pop("semantics_catalog")
        design["documentary_definitions"] = {
            "fixture": {
                "source_reference": "https://example.test/synthetic-data-dictionary",
                "target_definition": "A documented target, without label statistics.",
                "text_column_definitions": {
                    c: f'Meaning of {c}: Résumé, 世界 and a "quote".' for c in text_columns
                },
            }
        }
        bank = build_question_bank([metadata], design)
        entry = bank["datasets"]["fixture"]
        apply = compile_frozen_question(metadata, bank, "proposed_direct")
    records, _ = review_dataset(
        dataset,
        cfg,
        question_entry=entry,
        question_variant="proposed_direct" if frozen else "accepted",
    )
    all_requests = []
    for context in (False,):
        variant = copy.deepcopy(cfg)
        variant["jev"].update(
            representation="per_column_and_joint", include_non_text_features=context
        )
        for position in range(len(frame)):
            all_requests.extend(
                apply(r) for r in build_requests(dataset.features, metadata, position, variant)
            )
    unique = {request.cache_key: request for request in all_requests}
    for ratio in (3, 4, 5):
        subset = [record for record in records if record["characters_per_token"] == ratio]
        assert sum(r["requests_for_combined_build"] for r in subset) == len(unique)
        assert sum(r["logical_requests"] for r in subset) == len(all_requests)
        assert sum(r["wire_tokens_logical_approx"] for r in subset) == sum(
            math.ceil(len(canonical_json(api_preview(request))) / ratio) for request in all_requests
        )
        assert sum(r["wire_tokens_combined_approx"] for r in subset) == sum(
            math.ceil(len(canonical_json(api_preview(request))) / ratio)
            for request in unique.values()
        )


def test_removed_cleanup_cannot_touch_files(tmp_path):
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip("PowerShell unavailable on this platform")
    dataset = tmp_path / "data" / "raw" / "dataset"
    dataset.mkdir(parents=True)
    source = dataset / "data.csv"
    source.write_bytes(b"text,target\nexample,1\n")
    script = Path(__file__).resolve().parents[1] / "scripts" / "cleanup_empty_folders.ps1"
    before = source.read_bytes()
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-File", str(script)],
        cwd=tmp_path,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert not script.exists()
    assert source.read_bytes() == before
