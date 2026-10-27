"""0B approval stays valid across activation, not across untested pipeline changes."""

import copy

import pytest

from src.evaluation import acceptance
from src.utils.acceptance import reviewed_acceptance
from src.utils.serialization import file_sha256, write_json


@pytest.fixture
def configuration(monkeypatch):
    original = acceptance.load_yaml
    names = [
        "experiment_1/main",
        "experiment_1/models",
        "experiment_1/protocol",
        "experiment_2/embeddings",
        "experiment_3/native_text",
        "experiment_5/output_ablation",
        "feature_creation/embedding_models",
    ]
    configs = {name: copy.deepcopy(original(name)) for name in names}
    monkeypatch.setattr(acceptance, "load_yaml", lambda name: configs[name])
    return configs


def test_execution_switch_and_acceptance_path_do_not_change_0b_identity(configuration):
    before = acceptance.evaluation_identity()
    for cfg in configuration.values():
        cfg["enabled"] = True
        cfg["experiment_0_acceptance"] = "/new/reviewed.json"
    for cfg in configuration["feature_creation/embedding_models"]["models"].values():
        cfg["enabled"] = True
    assert acceptance.evaluation_identity() == before


@pytest.mark.parametrize("change", ["checkpoint", "seed", "pca", "classes", "selector", "encoder"])
def test_untested_recipes_invalidate_0b_identity(configuration, change):
    before = acceptance.evaluation_identity()
    if change == "checkpoint":
        configuration["experiment_1/models"]["models"]["tabpfn_3_5"]["runtime"]["checkpoints"][
            "binary_classification"
        ]["sha256"] = "different-checkpoint"
    elif change == "seed":
        configuration["experiment_2/embeddings"]["random_seed"] = 4
    elif change == "pca":
        configuration["experiment_2/embeddings"]["pca_grid"]["joint"] = [7]
    elif change == "classes":
        configuration["experiment_1/protocol"]["evaluation"]["binary"]["positive_class"] = "first"
    elif change == "selector":
        configuration["experiment_1/main"]["feature_selectors"]["jev"] = {"version": "other"}
    else:
        configuration["feature_creation/embedding_models"]["models"]["e5_small_v2"]["revision"] = (
            "other"
        )
    assert acceptance.evaluation_identity() != before


def test_changed_code_invalidates_0b_identity(configuration, monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    source = root / "src" / "pipeline.py"
    source.write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(acceptance.paths, "REPO_ROOT", root)
    before = acceptance.evaluation_identity()
    source.write_bytes(b"pass\r\n")
    assert acceptance.evaluation_identity() == before
    source.write_text("pass\n# different implementation\n", encoding="utf-8")
    assert acceptance.evaluation_identity() != before


def test_checksums_alone_do_not_authorize_a_different_pipeline(tmp_path):
    report = tmp_path / "0b.json"
    write_json(report, {"status": "passed"})
    signed = tmp_path / "accepted.json"
    body = {
        "accepted": True,
        "stage": "after_extraction",
        "reviewed_by": "reviewer",
        "evaluation_identity": "tested",
        "evidence": [{"path": str(report), "sha256": file_sha256(report)}],
    }
    write_json(signed, body)
    assert reviewed_acceptance(
        signed, stage="after_extraction", expected={"evaluation_identity": "tested"}
    )["accepted"]
    with pytest.raises(ValueError, match="current runtime/inputs"):
        reviewed_acceptance(
            signed, stage="after_extraction", expected={"evaluation_identity": "changed"}
        )
    del body["evaluation_identity"]
    write_json(signed, body)
    with pytest.raises(ValueError, match="current runtime/inputs"):
        reviewed_acceptance(
            signed, stage="after_extraction", expected={"evaluation_identity": "tested"}
        )


def test_campaign_refuses_stale_0b_before_loading_models(tmp_path, monkeypatch):
    from src.cluster import freeze
    from src.evaluation import campaign

    report = tmp_path / "report.json"
    write_json(report, {"status": "passed"})
    reviewed = tmp_path / "accepted.json"
    write_json(
        reviewed,
        {
            "accepted": True,
            "stage": "after_extraction",
            "reviewed_by": "reviewer",
            "evaluation_identity": "old",
            "evidence": [{"path": str(report), "sha256": file_sha256(report)}],
        },
    )
    monkeypatch.setattr(freeze, "check_if_present", lambda: {"frozen": False})
    monkeypatch.setattr(acceptance, "evaluation_identity", lambda: "current")
    monkeypatch.setattr(
        campaign,
        "load_yaml",
        lambda *a: {"enabled": True, "experiment_0_acceptance": str(reviewed)},
    )
    with pytest.raises(ValueError, match="current runtime/inputs"):
        campaign.run_arm({"phase": "experiment_1"}, allow_model_runs=True)
