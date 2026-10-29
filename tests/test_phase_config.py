import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from src.utils import paths
from src.utils.config import load_config, load_yaml, validate_config


def test_optional_environment_requirements_are_valid_pep508():
    project = tomllib.loads((paths.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    for dependencies in [project["dependencies"], *project["optional-dependencies"].values()]:
        for requirement in dependencies:
            Requirement(requirement)


def test_phase_inheritance_preserves_defaults():
    cfg = load_config("feature_creation")
    assert cfg["phase"] == "feature_creation"
    assert cfg["allow_download"] is False
    assert cfg["jev"]["model"] == "jev-1.13.0"
    assert cfg["jev"]["regression_scores"] == list(range(-4, 5))
    assert cfg["provider_reference"]["state_plus_longest_question_tokens"] == 32000
    assert load_yaml("experiment_0/debug")["cluster"]["account"] == "lp_verbekelab"
    for number, name in [
        (0, "debug"),
        (1, "main"),
        (2, "embeddings"),
        (3, "native_text"),
    ]:
        assert load_yaml(f"experiment_{number}/{name}")["enabled"] is False


def test_inheritance_cycle_and_escape_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    folder = tmp_path / "config"
    folder.mkdir()
    (folder / "a.yaml").write_text("extends: b\n", encoding="utf-8")
    (folder / "b.yaml").write_text("extends: a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Cyclic"):
        load_yaml("a")
    with pytest.raises(ValueError):
        paths.config_path("../outside")
    with pytest.raises(ValueError):
        paths.config_path(str(Path(tmp_path.anchor) / "outside"))


def test_cannot_enable_live_features():
    cfg = load_config("feature_creation")
    cfg["feature_creation"]["enabled"] = True
    with pytest.raises(ValueError, match="disabled"):
        validate_config(cfg)


def test_shared_experiment_proposals_preserve_validation_boundary_and_catalog():
    models = load_yaml("experiment_1/models")["models"]
    for name in ("experiment_1/main", "experiment_3/native_text"):
        cfg = load_yaml(name)
        assert cfg["enabled"] is False
        assert set(cfg["candidate_models"]) <= set(models)
        evaluation = cfg["evaluation"]
        assert evaluation["outer_folds"] == 5
        assert evaluation["inner_validation"]["source"] == "outer_training_only"
        assert evaluation["inner_validation"]["strategy"] == "stratified_out_of_fold"
        assert evaluation["inner_validation"]["folds"] == 3
        assert evaluation["binary"]["refit_after_threshold_selection"] is True
        assert evaluation["classification_split"] == "stratified_kfold"
        assert evaluation["regression_split"] == "kfold" and evaluation["shuffle"] is True
        assert evaluation["group_or_time_policy"] == "not_used_in_this_benchmark"
        assert evaluation["training_row_cap"] is None
        assert evaluation["model_tuning_budget"] == 0
        assert evaluation["binary"]["test_threshold_tuning"] == "forbidden"
        assert evaluation["binary"]["threshold_objective"] == "f1"
        assert evaluation["regression"]["classification_threshold"] == "not_applicable"
        assert evaluation["preprocessing_fit_scope"] == "training_partition_only"
    assert set(load_yaml("experiment_1/main")["candidate_models"]) == {
        "tabpfn_3_5",
        "catboost",
        "linear",
        "tabicl_v2",
        "limix_2",
        "kumo_tabular",
    }
    assert "tabm" not in models and "tabfm" not in models


def test_experiment_selection_excludes_zomato_but_audit_keeps_it():
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries

    catalog = load_catalog()
    assert len(catalog["datasets"]) == 20
    for name in (
        "feature_creation",
        "experiment_0/debug",
        "experiment_1/main",
        "experiment_3/native_text",
    ):
        entries = selected_entries(catalog, load_yaml(name))
        assert len(entries) == 18
        assert not {"REG_TEXT_ZOMATO_RESTAURANTS", "MUL_TEXT_SPOTIFY_GENRES"} & {
            entry["id"] for entry in entries
        }
    cfg = load_config("feature_creation")
    cfg["dataset_selection"]["exclude"] = ["unknown"]
    with pytest.raises(ValueError, match="Unknown excluded"):
        selected_entries(catalog, cfg)
