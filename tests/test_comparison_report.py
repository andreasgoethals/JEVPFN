from zipfile import ZipFile

import pandas as pd
import pytest

from src.evaluation import comparison_report as reports
from src.evaluation.decision_comparison import Comparison


def test_local_report_is_snapshot_only_and_detects_tampering(tmp_path, monkeypatch):
    cfg = reports.report_config()
    target = tmp_path / "comparison.zip"
    result, receipt = reports.read_report(cfg, target)
    assert receipt["status"] == "awaiting_cpu_report" and result.table("inventory").empty
    monkeypatch.setattr(
        reports, "compare_saved_features", lambda *a, **k: pytest.fail("Rescanned inputs")
    )
    rows = Comparison()
    rows.add("inventory", {"provider": "nimble", "status": "available", "artifact_id": "verified"})
    reports.write_report(rows, cfg, target, origin="fixture", joint_rows={"rows": 1})
    saved, receipt = reports.read_report(cfg, target)
    pd.testing.assert_frame_equal(saved.table("inventory"), rows.table("inventory"))
    assert receipt["joint_rows"] == {"rows": 1}
    assert reports.read_report({**cfg, "cost": {"price": 100}}, target)[1] == receipt
    with pytest.raises(ValueError, match="configuration"):
        reports.read_report({"comparison": {}}, target)
    monkeypatch.setattr(reports, "report_path", lambda *args: target)
    changed = {**cfg, "comparison": {**cfg["comparison"], "random_seed": -1}}
    pending, info = reports.read_report(changed, allow_previous_roster=True)
    assert pending.table("inventory").empty and info["status"] == "superseded_configuration"
    assert "Waiting for a matching A1 export" in reports.describe_receipt(info)
    with ZipFile(target) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["pairs.parquet"] += b"wrong"
    with ZipFile(target, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="checksum"):
        reports.read_report(cfg, target)
    with pytest.raises(ValueError, match="checksum"):
        reports.read_report(changed)


def test_rq1a_shares_production_recovery_selection_and_refuses_login_execution(monkeypatch):
    from src.utils.config import load_yaml

    assert (
        reports.report_config()["comparison"]["artifacts"]
        == load_yaml("B1")["dataset_feature_artifacts"]
    )
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Mindwell CPU allocation"):
        reports.build()


def test_cpu_publication_binds_large_rows_to_report(tmp_path, monkeypatch):
    from src.utils.serialization import file_sha256

    cfg = reports.report_config()
    result = Comparison()
    result.add("pairs", {"agreement": 1.0})
    result.add("row_diagnostics", {"status": "missing_text"})
    root = tmp_path / "gpfs"
    target = root / "comparison.zip"
    monkeypatch.setattr(reports.paths, "on_vsc", lambda: True)
    monkeypatch.setattr(reports.paths, "native_gpfs_project", lambda p: p == root)
    monkeypatch.setattr(reports.paths, "staging_root", lambda: root)
    monkeypatch.setattr(reports.paths, "durable_project_override", lambda: root)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "mindwell")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(reports, "report_path", lambda: target)
    monkeypatch.setattr(reports, "compare_saved_features", lambda cfg, workers: result)
    monkeypatch.setattr(reports, "peer_majority_agreement", lambda _: pd.DataFrame())
    monkeypatch.setattr(reports, "direct_scores", lambda cfg: (pd.DataFrame({"metric": ["roc_auc"]}), pd.DataFrame()))
    monkeypatch.setattr(reports, "model_profile", lambda cfg: pd.DataFrame({"provider": ["nimble"]}))
    reports.build()
    data, receipt = reports.read_report(cfg)
    assert data.table("pairs").agreement.tolist() == [1.0]
    rows = receipt["joint_rows"]
    assert rows["rows"] == 1 and file_sha256(root / rows["name"]) == rows["sha256"]
    assert receipt["origin"] == "mindwell:123"
    assert data.table("quality").metric.tolist() == ["roc_auc"]
    assert data.table("model_profile").provider.tolist() == ["nimble"]

def test_historical_roster_requires_exact_original_settings(tmp_path):
    from copy import deepcopy

    cfg = reports.report_config()
    cfg["comparison"]["providers"] = ["nimble"]
    result = Comparison()
    result.add("inventory", {"provider": "nimble", "status": "available"})
    target = tmp_path / "report.zip"
    reports.write_report(result, cfg, target, origin="fixture", joint_rows={"rows": 0})
    expanded = deepcopy(cfg)
    expanded["comparison"]["providers"].append("sieve")
    with pytest.raises(ValueError, match="configuration"):
        reports.read_report(expanded, target)
    _, info = reports.read_report(expanded, target, allow_previous_roster=True)
    assert info["unassessed_additions"] == ["sieve"]
    assert info["status"] == "verified_previous_roster"
    assert [p["provider"] for p in info["decision_model_versions"]] == ["nimble"]
    repinned = deepcopy(expanded)
    for row in repinned["decision_model_versions"]:
        if row["provider"] == "nimble":
            row["version"] = "1" * 40
    with pytest.raises(ValueError, match="configuration"):
        reports.read_report(repinned, target, allow_previous_roster=True)
    expanded["comparison"]["random_seed"] += 1
    with pytest.raises(ValueError, match="configuration"):
        reports.read_report(expanded, target, allow_previous_roster=True)


def test_a1_identity_uses_only_the_selected_model_versions(tmp_path, monkeypatch):
    from copy import deepcopy

    from src.evaluation import design
    from src.utils.serialization import digest

    configuration = design.load_yaml
    candidates = deepcopy(configuration("extractors"))
    main = deepcopy(configuration("B1"))
    def modified(name):
        if name == "extractors":
            return candidates
        return main if name == "B1" else configuration(name)

    monkeypatch.setattr(design, "load_yaml", modified)
    monkeypatch.setattr(reports, "load_yaml", modified)
    first = reports.report_config("open_weight")
    expanded = reports.report_config("with_jev")
    first_id = reports.analysis_identity(first)
    first_path = reports.report_path("open_weight")
    expanded_path = reports.report_path("with_jev")
    candidates["candidates"]["jev"]["version"] = "99.0"
    candidates["candidates"]["future"] = {**candidates["candidates"]["jev"], "model": "future-1"}
    main["candidate_extractors"].append("future")
    main["decision_extensions"]["later"] = {"parent": "with_jev", "add": ["future"]}
    main["dataset_feature_artifacts"]["new-dataset"] = {"future": {"joint": "future-artifact"}}
    assert reports.analysis_identity(reports.report_config("open_weight")) == first_id
    assert reports.report_path("open_weight") == first_path
    assert reports.analysis_identity(reports.report_config("with_jev")) != reports.analysis_identity(expanded)
    assert reports.report_path("with_jev") != expanded_path

    target = tmp_path / first_path.name
    reports.write_report(Comparison(), first, target, origin="fixture", joint_rows={})
    _, meta = reports.read_report(first, target)
    assert meta["decision_model_versions"] == first["decision_model_versions"]
    assert meta["roster_sha256"] == digest(first["decision_model_versions"])
