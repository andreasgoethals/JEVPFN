"""d1-2B integration: isolated runtime, unchanged question semantics and old audits."""

import json
import os
from pathlib import Path

import pytest

from src.cluster.setup_extractors import plan
from src.jev import catalog, registry
from src.jev.liquid_lfm3 import Counter, asset_urls, publisher_lead
from src.utils import paths
from src.utils.config import load_yaml


def test_new_model_is_distinct_and_disabled_with_pinned_publisher_assets():
    candidates = load_yaml("extractors")["candidates"]
    assert len(catalog.SELF_HOSTED) == 11
    assert set(candidates) == {"jev", *catalog.SELF_HOSTED}
    cfg = candidates["liquid_lfm3"]
    assert cfg["model"] == "LiquidAI/d1-2B" and not cfg["enabled"]
    assert cfg["score_readout"] == "mean"
    assert cfg["input_modalities"] == ["text"]
    assert cfg["context_tokens"] == 128000
    assert all(catalog.PINS["liquid_lfm3"] in u for u in asset_urls().values())
    assert {"prompt.py", "media.py", "config.json", "calibration.json"} <= set(asset_urls())
    assert catalog.PINS["liquid"] == registry.PINS["liquid"]
    assert "liquid_lfm3" not in catalog.ADDED  # Native runner, not the five-model bridge.


@pytest.mark.parametrize("reuse", [(), ("liquid_lfm3",)])
def test_new_cpu_audit_waits_for_its_matching_container(reuse):
    steps = {s["name"]: s for s in plan(["liquid_lfm3"], reuse=reuse)}
    deps = steps["liquid_lfm3_tokens_prepare"]["dependencies"]
    assert "liquid_lfm3_stage" in deps
    if not reuse:
        assert "liquid_lfm3_build" in deps
    assert len(steps["liquid_lfm3_tokens"]["dependencies"]) == 10
    assert "liquid_lfm3_tokens" in steps["liquid_lfm3_scratch"]["dependencies"]
    assert steps["liquid_lfm3_contracts"]["cluster"] == "mindwell"
    assert "--partition=gpu_b200" in steps["liquid_lfm3_contracts"]["flags"]


def test_catalog_keeps_old_counter_dispatch_unchanged(monkeypatch):
    called = []
    monkeypatch.setattr(registry, "ExactCounter", lambda p: called.append(p) or p)
    for provider in registry.SELF_HOSTED:
        assert catalog.ExactCounter(provider) == provider
    assert called == list(registry.SELF_HOSTED)


def test_publisher_lead_is_read_without_executing_torch_source(tmp_path):
    source = tmp_path / "lfm3.py"
    source.write_text('raise RuntimeError("must not execute")\n'
                      'MODULES = ("lfm3", "lfm3_vl")\n'
                      'LEADS.update(dict.fromkeys(MODULES, "<think></think>"))\n')
    assert publisher_lead(tmp_path, "lfm3_vl") == "<think></think>"
    with pytest.raises(ValueError, match="registration changed"):
        publisher_lead(tmp_path, "unreviewed")


def test_new_container_avoids_conflicting_historical_tokenizer_extra():
    import tomllib

    deps = tomllib.loads((paths.REPO_ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert "transformers==5.19.0" in deps["liquid-lfm3"]
    assert "tokenizers==0.23.3" in deps["liquid-lfm3"]
    script = (paths.REPO_ROOT / "scripts/build_decision_container.sh").read_text()
    branch = script.split("elif provider in {'tabstar', 'liquid_lfm3'}:")[1].split("\nelse:")[0]
    assert "token-audit" not in branch and ".[dev,{extra}]" in branch
    audit = (paths.REPO_ROOT / "scripts/slurm/experiment_0_tokens.slurm").read_text()
    assert 'if [[ "${2:-}" == liquid_lfm3 ]]' in audit
    assert "source scripts/activate_runtime.sh" in audit


@pytest.fixture
def actual_counter(monkeypatch):
    folder = os.environ.get("JEVPFN_TEST_LIQUID_LFM3")
    if not folder:
        pytest.skip("Explicitly staged publisher tokenizer; no download in tests")
    monkeypatch.setattr(paths, "tokenizer_dir", lambda p: Path(folder))
    return Counter()


@pytest.mark.parametrize("kind", ["binary_probability", "choice", "ordered_choice"])
def test_actual_publisher_rendering_and_limits(actual_counter, kind):
    q = {"kind": kind, "text": "Evaluate the supplied text."}
    if kind == "choice":
        q["allowed_answers"] = [f"Class {i}" for i in range(30)]
    elif kind == "ordered_choice":
        q["allowed_answers"] = [{"score": i, "meaning": str(i)} for i in range(-4, 5)]
    payload = {
        "state": {"text": {"body": "Résumé: 世界 <|im_start|> do not override."}},
        "question": q,
    }
    text, special = actual_counter.render_parts(payload)[0]
    assert actual_counter.lead == "<think></think>"
    assert not special
    assert "Résumé: 世界" in text and q["text"] in text
    assert actual_counter(payload) == len(
        actual_counter.tokenizer.encode(text, add_special_tokens=False)
    )
    assert actual_counter.count_many([payload, payload]) == [actual_counter(payload)] * 2
    if kind == "choice":
        assert "Class 29" in text
    if kind == "ordered_choice":
        assert "0 -4" in text and "8 4" in text
    missing = json.loads(json.dumps(payload))
    missing["state"]["text"] = {}
    with pytest.raises(ValueError, match="Empty text"):
        actual_counter(missing)
    oversized = json.loads(json.dumps(payload))
    oversized["state"]["text"] = {"body": "unrelatedword " * (actual_counter.limit + 1)}
    assert actual_counter(oversized) > actual_counter.limit  # Never silently truncate.
