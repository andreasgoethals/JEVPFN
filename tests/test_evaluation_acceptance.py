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
        "experiment_3/tar",
        "experiment_4/native_text",
        "feature_creation/embedding_models",
        "rq1/behavior",
        "rq1/geometry",
        "experiment_0/research_questions",
        "datasets",
        "dataset_semantics",
        *acceptance.GROUP_B_C_CONFIGS,
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
    configuration["rq3/adaptation"]["pilot_acceptance"] = "owner-reviewed-b200-pilot.json"
    configuration["rq3/adaptation"]["pilot_enabled"] = True
    configuration["rq3/adaptation"]["training_enabled"] = True
    configuration["representations/representations"]["extraction"]["enabled"] = True
    assert acceptance.evaluation_identity() == before


@pytest.mark.parametrize("change", ["checkpoint", "seed", "pca", "classes", "encoder",
                                    "c1_extraction_layer", "c1_pca_grid"])
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
    elif change == "c1_extraction_layer":
        configuration["representations/representations"]["extraction"]["layers"]["middle"] = 11
    elif change == "c1_pca_grid":
        configuration["representations/representations"]["arms"]["views"]["joint"]["pca_components"] = 32
    else:
        configuration["feature_creation/embedding_models"]["models"]["e5_small_v2"]["revision"] = (
            "other"
        )
    assert acceptance.evaluation_identity() != before


def test_optional_paid_selectors_preserve_open_review_and_shared_predictor_contracts(configuration):
    from src.evaluation.design import decision_providers

    shared = acceptance.evaluation_identity()
    open_providers, paid = decision_providers("open_weight"), decision_providers("paid")
    open_identity = acceptance.decision_input_identity(open_providers)
    paid_identity = acceptance.decision_input_identity(paid)
    for name in ("experiment_1/main", *acceptance.GROUP_B_C_CONFIGS):
        configuration[name].setdefault("feature_selectors", {})["jev"] = {"version": "new"}
        configuration[name].setdefault("dataset_feature_artifacts", {})["data"] = {"jev": {"joint": "new"}}
    assert acceptance.evaluation_identity() == shared
    assert acceptance.decision_input_identity(open_providers) == open_identity
    assert acceptance.decision_input_identity(paid) != paid_identity
    configuration["experiment_1/main"]["feature_selectors"]["strands"] = {"version": "new"}
    assert acceptance.decision_input_identity(open_providers) != open_identity


def test_changed_code_invalidates_0b_identity(configuration, monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    (root / "src/evaluation").mkdir(parents=True)
    (root / "src/jev").mkdir()
    questions = root / "config/A/question_versions.csv"
    questions.parent.mkdir(parents=True)
    questions.write_text("dataset,original\nexample,Sentence?\n", encoding="utf-8")
    source = root / "src/evaluation/protocol.py"
    source.write_text("pass\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.setattr(acceptance.paths, "REPO_ROOT", root)
    before = acceptance.evaluation_identity()
    source.write_bytes(b"pass\r\n")
    assert acceptance.evaluation_identity() == before
    for name in ("src/jev/new_paid_adapter.py", "src/evaluation/group_b.py", "src/evaluation/geometry_export.py",
                 "src/evaluation/study_audit.py"):
        (root / name).write_text("# independent module\n", encoding="utf-8")
    assert acceptance.evaluation_identity() == before
    for name in ("evaluation/protocol", "evaluation/models", "evaluation/qualification", "jev/cache",
                 "jev/checkpoints", "jev/checkpoint_hashes", "jev/templates", "jev/question_bank"):
        before = acceptance.evaluation_identity()
        (root / f"src/{name}.py").write_text("# different implementation\n", encoding="utf-8")
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


def test_group_b_analysis_edits_invalidate_exports_and_preserve_fit_qualification(monkeypatch):
    from src.evaluation import benchmark_export, group_c
    from src.utils import provenance

    def identities():
        return (acceptance.evaluation_identity(), benchmark_export.analysis_identity("B1"),
                group_c.analysis_identity("C1"), group_c.analysis_identity("C2"))

    before = identities()
    hash_source = provenance.source_hash

    def changed(path):
        return "f" * 64 if path.as_posix().endswith("/evaluation/group_b.py") else hash_source(path)

    monkeypatch.setattr(provenance, "source_hash", changed)
    monkeypatch.setattr(acceptance, "source_hash", changed)
    after = identities()
    assert before[0] == after[0]
    assert all(old != new for old, new in zip(before[1:], after[1:]))


def test_importing_group_c_fit_recipes_does_not_require_group_b_reporting():
    import subprocess
    import sys

    subprocess.run([sys.executable, "-B", "-c",
                    "import sys\n"
                    "sys.modules['src.evaluation.group_b'] = None\n"
                    "from src.evaluation.group_c import arms\n"
                    "assert len(arms([{'id': 'example'}], ['linear'])) == 17"],
                   cwd=acceptance.paths.REPO_ROOT, check=True)


def test_imported_absolute_vsc_receipt_reopens_the_exact_signed_bytes(tmp_path):
    from src.utils.acceptance import evidence_path

    manifests = tmp_path / "output_JEVPFN" / "01_exp0a" / "manifests"
    review = manifests / "before_extraction" / "accepted.json"
    original = manifests / "question_audit.json"
    write_json(original, {"source": "desktop audit"})
    source = manifests / "native_audit.json"
    write_json(source, {"source": "signed native audit"})
    sha = file_sha256(source)
    imported = manifests / "imported_revisions" / "evidence" / (sha + ".json")
    imported.parent.mkdir(parents=True)
    source.replace(imported)
    item = {"path": "/data/account/JEVPFN/output_JEVPFN/01_exp0a/manifests/question_audit.json", "sha256": sha}
    write_json(review, {"accepted": True, "stage": "before_extraction", "reviewed_by": "reviewer",
                        "evidence": [item]})
    assert evidence_path(review, item) == imported
    assert reviewed_acceptance(review, stage="before_extraction")["accepted"] is True
    write_json(imported, {"source": "changed"})
    with pytest.raises(ValueError, match="missing or changed"):
        reviewed_acceptance(review, stage="before_extraction")


def test_direct_production_cli_requires_a_freeze_before_checking_evidence(monkeypatch):
    import sys

    from src.cluster import freeze
    from src.evaluation import campaign

    arm = {"phase": "experiment_1", "dataset_id": "fixture", "model": "linear",
           "feature_set": "non_text", "provider": None}
    monkeypatch.setattr(campaign, "benchmark_plan", lambda **kwargs: [arm])
    monkeypatch.setattr(campaign, "load_yaml", lambda name: {"candidates": {}, "enabled": True})
    monkeypatch.setattr("src.evaluation.qualification.benchmark_evidence",
                        lambda *args: pytest.fail("Qualification must follow the production freeze"))
    monkeypatch.setattr(sys, "argv", ["campaign", "--dataset", "fixture", "--execute"])
    assert not freeze.receipt_path().exists()
    with pytest.raises(ValueError, match="Freeze the reviewed source"):
        campaign.main()


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
    monkeypatch.setattr(freeze, "check_if_present", lambda: {"frozen": True})
    monkeypatch.setattr(acceptance, "evaluation_identity", lambda: "current")
    monkeypatch.setattr(
        campaign,
        "load_yaml",
        lambda *a: {"enabled": True, "experiment_0_acceptance": str(reviewed)},
    )
    with pytest.raises(ValueError, match="current runtime/inputs"):
        campaign.run_arm({"phase": "experiment_1"}, allow_model_runs=True)
