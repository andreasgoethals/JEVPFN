from pathlib import Path

import pytest

from src.utils import paths
from src.utils.config import load_config, load_yaml, validate_config


def test_phase_inheritance_preserves_defaults():
    cfg = load_config("feature_creation/default")
    assert cfg["phase"] == "feature_creation"
    assert cfg["allow_download"] is False
    assert cfg["jev"]["model"] == "jev-1.13.0"
    assert cfg["jev"]["regression_scores"] == list(range(-4, 5))
    assert cfg["provider_reference"]["state_plus_longest_question_tokens"] == 32000
    assert load_yaml("experiment_0/debug")["cluster"]["account"] == "lp_verbekelab"
    for number, name in [
        (0, "debug"),
        (1, "main"),
        (2, "native_text"),
        (3, "output_ablation"),
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
    cfg = load_config("feature_creation/default")
    cfg["feature_creation"]["enabled"] = True
    with pytest.raises(ValueError, match="disabled"):
        validate_config(cfg)


def test_shared_experiment_proposals_preserve_validation_boundary_and_catalog():
    models = load_yaml("experiment_1/models")["models"]
    for name in ("experiment_1/main", "experiment_2/native_text", "experiment_3/output_ablation"):
        cfg = load_yaml(name)
        assert cfg["enabled"] is False
        assert set(cfg["candidate_models"]) <= set(models)
        evaluation = cfg["evaluation"]
        assert evaluation["outer_folds"] == 5
        assert evaluation["inner_validation"]["source"] == "outer_training_only"
        assert evaluation["binary"]["test_threshold_tuning"] == "forbidden"
        assert evaluation["binary"]["threshold_objective"] == "f1"
        assert evaluation["regression"]["classification_threshold"] == "not_applicable"
        assert evaluation["preprocessing_fit_scope"] == "training_partition_only"
