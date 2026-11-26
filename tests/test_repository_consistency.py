"""Cross-file research contracts, independent of API/model execution."""

import re
import subprocess
import tomllib
from urllib.parse import unquote

import yaml

from src.cluster.token_audit import HOSTED_UNAVAILABLE
from src.evaluation.metrics import METRICS_VERSION
from src.evaluation.models import MAIN_MODELS, TEXT_MODELS
from src.jev.catalog import PINS, SELF_HOSTED
from src.jev.context import CONTEXT_LIMITS
from src.utils import paths
from src.utils.config import load_yaml


def test_active_rosters_and_provider_pins_agree():
    catalog = load_yaml("extractors")["candidates"]
    roster = set(catalog)
    assert set(SELF_HOSTED) == {p for p, cfg in catalog.items() if cfg["delivery"] == "self_hosted"}
    assert {"jev"} <= set(HOSTED_UNAVAILABLE) <= roster - set(SELF_HOSTED)
    for providers in (
        load_yaml("decision_pilot")["providers"],
        load_yaml("feature_comparison")["comparison"]["providers"],
        load_yaml("B1")["candidate_extractors"],
        load_yaml("B1")["feature_selectors"],
    ):
        assert set(providers) == roster
    assert {k: v["context_tokens"] for k, v in catalog.items()} == CONTEXT_LIMITS
    assert all(catalog[p]["revision"] == PINS[p] for p in SELF_HOSTED)
    cfg = load_yaml("feature_creation")
    assert catalog["jev"]["model"] == cfg["jev"]["model"]
    assert catalog["jev"]["input_price_per_million_usd"] == cfg["cost"]["input_price_per_million"]
    assert (
        catalog["jev"]["context_tokens"]
        == cfg["provider_reference"]["state_plus_longest_question_tokens"]
    )


def test_execution_catalog_and_metric_version_match_implementation():
    from src.evaluation.features import (
        DECISION_FEATURE_SETS,
        EMBEDDING_FEATURE_SETS,
        NATIVE_FEATURE_SETS,
    )

    for config, combinations in (
        ("B1", DECISION_FEATURE_SETS),
        ("B2", EMBEDDING_FEATURE_SETS[:1]),
        ("B6", NATIVE_FEATURE_SETS[:1]),
    ):
        assert tuple(load_yaml(config)["candidate_combinations"]) == combinations
    assert tuple(load_yaml("feature_creation/text_embeddings")["models"]) == MAIN_MODELS
    assert set(load_yaml("B1/models")["models"]) == set(MAIN_MODELS + TEXT_MODELS)
    assert (
        load_yaml("B1/protocol")["evaluation"]["metric_conventions"]["version"]
        == METRICS_VERSION
    )
    for folder in (paths.REPO_ROOT / "config").iterdir():
        assert (folder.is_dir() and folder.name in paths.layout().top_level) or (
            folder.name in {"structure.yaml", "layout_aliases.yaml"})


def test_citation_matches_project_metadata():
    project = tomllib.loads((paths.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    citation = yaml.safe_load((paths.REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["cff-version"] == "1.2.0" and citation["type"] == "software"
    assert citation["version"] == project["version"]
    assert citation["repository-code"] == project["urls"]["Repository"]
    author = citation["authors"][0]
    assert author["given-names"] + " " + author["family-names"] == project["authors"][0]["name"]
    assert citation["license"] == (paths.REPO_ROOT / "LICENSE").read_text().split()[0]


def test_public_readme_links_resolve_without_private_manuals():
    root = paths.REPO_ROOT
    readme = (root / "README.md").read_text(encoding="utf-8")
    for match in re.finditer(r"\[[^\]]*\]\((<[^>]+>|[^)]+)\)", readme):
        target = match[1].strip("<>").split("#", 1)[0]
        if target and not re.match(r"[a-z]+:", target):
            path = (root / unquote(target)).resolve()
            assert not path.is_relative_to((root / "docs").resolve()), target
            assert path.exists(), target


def test_sensitive_and_large_artifacts_are_ignored():
    examples = [
        "docs/TEMPLATE.md",
        "docs/nested/new-manual.md",
        "data/raw/dataset/data.csv",
        "data/decision_inputs/dataset.sqlite3",
        "data/jev_cache/dataset.sqlite3",
        "checkpoints/model/model.safetensors",
        "output_JEVPFN/All_Results.md",
        "output_JEVPFN/figures/CAPTIONS.md",
        "output_JEVPFN/figures/exp04_decision_features/01_a.pdf",
        "archive/layout_20261118/figures/general/a.pdf",
        "output_JEVPFN/4. experiment_1/results/predictions.parquet",
        "archived_output_JEVPFN/old/3. experiment_0/logs/test.log",
        ".maintenance.lock",
        "config/private/.env",
        "credentials/private.pem",
        "environments/gpu/bin/python",
    ]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin", "-z"],
        input=("\0".join(examples) + "\0").encode("utf-8"),
        cwd=paths.REPO_ROOT,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0 and set(
        result.stdout.decode("utf-8").strip("\0").split("\0")
    ) == set(examples)
