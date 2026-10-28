import copy

import numpy as np
import pandas as pd
import pytest

from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.jev.alternatives import preview
from src.jev.cache import ResponseCache
from src.jev.context import require_context
from src.jev.extraction import DatasetExtractor, ExtractionLimits
from src.jev.features import materialize_cached_block
from src.jev.requests import IntendedRequest, build_requests
from src.utils.config import load_config
from src.utils.serialization import canonical_json


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("mode", ["per_column", "joint"])
def test_all_empty_inputs_no_calls_all_derived_values_missing(
    tmp_path, isolated_output, task, mode
):
    y = (
        [0, 1, 0, 1]
        if task == "binary"
        else [0, 1, 2, 1]
        if task == "multiclass"
        else [0.1, 0.2, 0.3, 0.4]
    )
    frame = pd.DataFrame({"text": [None, "", " \t\n", np.nan], "number": range(4), "y": y})
    official = {"target": "y", "task_type": "reg" if task == "regression" else "cls"}
    meta = extract_metadata(
        frame, official, dataset_id="fixture", dataset_name="Fixture", text_columns=["text"]
    )
    ds = LoadedDataset(frame, meta, official, {"data_identity": "fixture"}, [])
    cfg = load_config("feature_creation")
    cfg["jev"]["representation"] = mode
    requests = [r for i in range(len(frame)) for r in build_requests(ds.features, meta, i, cfg)]

    class NeverCalled:
        def evaluate(self, request):
            pytest.fail("Empty inputs must never call a provider")

    cache = ResponseCache(tmp_path / "fixture.sqlite3", "fixture")
    extractor = DatasetExtractor(
        cache,
        NeverCalled(),
        ExtractionLimits(1, 0.042, 100, 100, 10000),
        validate_context=lambda r: None,
    )
    assert extractor.run(requests) == {}
    block = materialize_cached_block(ds, cfg, cache)
    assert block.values.drop(columns="row_id").isna().all().all()
    assert block.requests.request_key.isna().all()
    assert set(block.requests.status) == {"missing_text"}


@pytest.mark.parametrize(
    "provider",
    [
        "jev",
        "liquid",
        "liquid_lfm3",
        "perplexity",
        "strands",
        "cloudflare",
        "intern",
        "h2o",
        "sii",
        "kev",
        "decision3",
        "nimble",
    ],
)
def test_even_hand_built_empty_request_is_rejected_for_every_provider(provider):
    request = IntendedRequest(
        canonical_json({"state": {"text": {"description": " "}, "task": {}}, "question": {}}),
        "{}",
        "{}",
    )
    with pytest.raises(ValueError, match="Empty text"):
        preview(request, provider)


def test_exact_context_guard_fails_before_truncation_or_dispatch():
    payload = {"state": {"text": {"a": "unchanged"}}, "question": {"text": "unchanged"}}
    before = copy.deepcopy(payload)
    with pytest.raises(ValueError, match="overflow"):
        require_context(payload, counters={"strands": lambda _: 4097}, limits={"strands": 4096})
    assert payload == before
    with pytest.raises(ValueError, match="every selected"):
        require_context(payload, counters={}, limits={"strands": 4096})
    assert require_context(
        payload, counters={"strands": lambda _: 4096}, limits={"strands": 4096}
    ) == {"strands": 4096}
