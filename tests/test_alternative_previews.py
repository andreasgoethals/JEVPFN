"""Published offline inference signatures, without importing a provider runtime."""

import json

import pandas as pd
import pytest

from src.data.metadata import extract_metadata
from src.jev.alternatives import PERPLEXITY_SOURCE, preview
from src.jev.provider import api_preview
from src.jev.requests import build_requests
from src.utils.config import load_config


@pytest.fixture(params=[("cls", [0, 1, 0]), ("cls", [0, 1, 2]), ("reg", [0.1, 2.0, 4.3])])
def request_by_task(request):
    task, targets = request.param
    frame = pd.DataFrame({"text": ["Example evidence", "More text", "Third text"], "y": targets})
    metadata = extract_metadata(
        frame,
        {"target": "y", "task_type": task},
        dataset_id="fixture",
        dataset_name="Fixture",
        text_columns=["text"],
    )
    return build_requests(frame.drop(columns="y"), metadata, 0, load_config())[0]


def test_perplexity_uses_singular_question_and_saved_calibration(request_by_task):
    mapped = preview(request_by_task, "perplexity")
    arguments = mapped["arguments"]
    row = arguments["rows"][0]
    expected = api_preview(request_by_task)
    assert set(row) == {"state", "question"}
    assert json.loads(row["state"]) == expected["state"]
    assert row["question"] == expected["questions"]["semantic"]
    assert arguments["batch_size"] == 1
    assert "temperature" not in arguments
    assert mapped["reference"] == PERPLEXITY_SOURCE
    assert "model.predict(**arguments)" in mapped["call"]
    assert mapped["live_verified"] is False


def test_direct_strands_preview_requires_strict_window_and_serialization(request_by_task):
    mapped = preview(request_by_task, "strands")
    assert "strict_window=True" in mapped["initialization"]
    assert "SystemOneRequest(**arguments)" in mapped["call"]
    assert any("Serialize" in requirement for requirement in mapped["runtime_requirements"])
    assert mapped["arguments"]["questions"] == api_preview(request_by_task)["questions"]


def test_all_previews_are_inspectable_json_and_explicitly_unverified(request_by_task):
    before = request_by_task.to_json()
    for provider in ("jev", "liquid", "perplexity", "cloudflare", "strands", "intern"):
        mapped = preview(request_by_task, provider)
        assert json.loads(json.dumps(mapped)) == mapped
        assert mapped["live_verified"] is False
        assert mapped["reference"].startswith("https://")
    assert request_by_task.to_json() == before
