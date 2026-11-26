"""The layout registry v4 (26-11-2026: one name per experiment), its checksummed alias map
and the verified tree migration from earlier layouts."""

import dataclasses
import json
import sqlite3
from contextlib import closing
from pathlib import Path, PurePosixPath

import pytest
import yaml

from src.utils import layout_migration, paths, structure
from src.utils.config import load_yaml
from src.utils.serialization import digest, file_sha256

FOLDERS = ("general/0", "general/1_data_and_questions", "general/2_feature_creation",
           "general/3_study_audit", "general/4_synthesis",
           "A/A0", "A/A1", "A/A2", "A/A3", "A/A_synthesis",
           "B/B0", "B/B1", "B/B2", "B/B3", "B/B4", "B/B5", "B/B6", "B/B_synthesis",
           "C/C0", "C/C1", "C/C2", "C/C_synthesis")
EXPERIMENTS = ("A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3", "B4", "B5", "B6", "C0", "C1", "C2")


def test_registry_order_folders_notebooks_and_paper_mapping():
    reg = structure.registry()
    assert reg.folders == FOLDERS
    assert reg.top_level == ("general", "A", "B", "C")
    assert tuple(n.id for n in reg.notebooks) == FOLDERS
    assert reg.notebook_groups == ("general", "A", "B", "C", "paper")
    assert {v["placement"] for v in structure.paper_mapping().values()} == {"undecided"}
    assert set(structure.paper_mapping()) == set(reg.paper) == {e.id for e in reg.entries}
    for entry in reg.entries:  # Folder names never encode paper placement.
        assert not any(word in entry.folder for word in ("main", "appendix", "paper"))


def test_one_name_per_experiment_everywhere():
    """Phase ID = entry ID = folder = notebook name = config file (owner decision, 26-11-2026)."""
    for name in EXPERIMENTS:
        entry = structure.registry().entry(name)
        assert entry.phases == (name,) and entry.folder == name
        assert paths.phase_folder(name) == f"{name[0]}/{name}"
        assert entry.notebooks[0].id == f"{name[0]}/{name}"
        assert (paths.REPO_ROOT / "notebooks" / name[0] / f"{name}.ipynb").is_file()
        if entry.config:
            assert paths.config_path(name) == paths.REPO_ROOT / "config" / name[0] / f"{name}.yaml"
            assert load_yaml(name).get("phase") in {name, None}
    assert paths.phase_folder("0") == "general/0"
    assert (paths.REPO_ROOT / "notebooks/general/0.ipynb").is_file()
    assert paths.phase_folder("regression_readout") == "B/B1/regression_readout"
    assert {paths.phase_folder(p) for p in ("exploration", "question_preparation")} == {
        "general/1_data_and_questions"}
    assert paths.phase_folder("synthesis") == "general/4_synthesis"


def test_old_identifiers_are_rejected_outside_the_alias_map():
    for old in ("experiment_0", "experiment_1", "rq1", "rq2", "rq3", "data_efficiency",
                "representations", "01_exp0a", "exp04_decision_features", "3. experiment_0"):
        with pytest.raises(ValueError):
            paths.phase_name(old)
    with pytest.raises((ValueError, FileNotFoundError)):
        load_yaml("experiment_1/main")
    table = structure.aliases()
    assert table["phases"]["experiment_0"] == "0" and table["phases"]["rq2"] == "B4"
    assert table["phases"]["experiment_4"] == "B6" and table["phases"]["data_efficiency"] == "B5"
    assert structure.renamed_phase("experiment_0", "manifests/after_extraction/x.json") == "B0"
    assert structure.renamed_phase("rq1", "results/geometry_pilots") == "A0"
    assert structure.renamed_phase("rq1", "results/robustness/x") == "A2"
    assert structure.renamed_phase("rq1", "results/outputs") == "A1"


def test_alias_map_is_checksummed_in_the_registry(tmp_path):
    for name in ("structure.yaml", "layout_aliases.yaml"):
        (tmp_path / name).write_bytes((paths.REPO_ROOT / "config" / name).read_bytes())
    structure.registry(tmp_path / "structure.yaml")
    (tmp_path / "layout_aliases.yaml").write_text(
        (tmp_path / "layout_aliases.yaml").read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        structure._load(tmp_path / "structure.yaml")


@pytest.mark.parametrize("key, expected, section", [
    ("B1", "B/B1.yaml", None),
    ("B1/models", "B/B1.yaml", "models"),
    ("B1/protocol", "B/B1.yaml", "protocol"),
    ("B4/text_column_refits", "B/B4.yaml", "text_column_refits"),
    ("A2", "A/A2.yaml", None),
    ("C2", "C/C2.yaml", None),
    ("0/debug", "general/0/debug.yaml", None),
    ("feature_creation/extractors", "general/2_feature_creation/extractors.yaml", None),
    ("synthesis/analysis", "general/4_synthesis/analysis.yaml", None),
    ("datasets", "general/1_data_and_questions/datasets.yaml", None),
    ("general/0/scheduling", "general/0/scheduling.yaml", None),
])
def test_config_keys_resolve_to_one_file(key, expected, section):
    assert paths.config_path(key) == paths.REPO_ROOT / "config" / expected
    assert paths.config_section(key) == section
    assert isinstance(load_yaml(key), dict)


def test_sections_and_inheritance_load_the_previous_dictionaries():
    models = load_yaml("B1/models")
    assert "models" in models and "sections" not in load_yaml("B1")
    assert load_yaml("B1")["model_catalog"] == "B1/models"
    assert load_yaml("B2")["random_seed"] == load_yaml("B1/protocol")["random_seed"]
    assert load_yaml("B3")["encoders"] == ["e5_small_v2_tar"]
    assert load_yaml("B4/text_column_refits")["phase"] == "B4"
    with pytest.raises(ValueError, match="section"):
        load_yaml("B1/unknown")
    assert paths.config_file("A2_question_versions.csv").is_file()
    with pytest.raises(ValueError):
        paths.config_path("../outside")
    with pytest.raises(ValueError):
        paths.config_file("../A2_question_versions.csv")


def test_config_tree_and_notebooks_follow_registry():
    reg = structure.registry()
    expected = {f"general/{e.folder}/{c}.yaml" for e in reg.entries for c in e.configs}
    expected |= {f"{e.group}/{e.folder}.yaml" for e in reg.entries if e.config}
    expected |= {f"{e.group}/{f}" for e in reg.entries for f in e.data_files}
    present = {p.relative_to(paths.REPO_ROOT / "config").as_posix()
               for p in (paths.REPO_ROOT / "config").rglob("*") if p.is_file()}
    assert expected <= present
    assert present - expected == {"structure.yaml", "layout_aliases.yaml"}
    notebooks = {p.relative_to(paths.REPO_ROOT / "notebooks").with_suffix("").as_posix()
                 for p in (paths.REPO_ROOT / "notebooks").rglob("*.ipynb")
                 if ".ipynb_checkpoints" not in p.parts}
    assert notebooks == {n.id for n in reg.notebooks}


def test_notebook_only_entries_own_records_but_no_phase():
    reg = structure.registry()
    for name in ("A_synthesis", "B_synthesis", "C_synthesis", "study_audit"):
        entry = reg.entry(name)
        assert entry.notebook_only and not entry.phases and not entry.config
    assert paths.notebook_folder("general/3_study_audit") == "general/3_study_audit"
    assert paths.notebook_phase("general/3_study_audit") == "synthesis"


def test_figure_groups_follow_notebooks():
    assert structure.figure_group("B/B3") == "B/B3"
    assert structure.figure_group("B3") == "B/B3"
    assert structure.figure_group("general/0") == "general/0"
    assert structure.figure_folder("C/C2") == "C/C2"
    assert paths.figures_dir("B/B4").parts[-3:] == ("figures", "B", "B4")


@pytest.mark.parametrize("old, new", [
    # v1: numbered phase folders and research-question trees.
    ("3. experiment_0/manifests/before_extraction/a.json", "general/0/manifests/before_extraction/a.json"),
    ("3. experiment_0/manifests/after_extraction/p.json", "B/B0/manifests/p.json"),
    ("2. feature_creation/results/timings/t.sqlite3", "general/2_feature_creation/results/timings/t.sqlite3"),
    ("rq2/4. experiment_1/results/cv_folds/a.json", "B/B1/results/cv_folds/a.json"),
    ("rq2/5. experiment_2/results/pca/e5/D.sqlite3", "B/B2/results/pca/e5/D.sqlite3"),
    ("6. experiment_3/results/tar/D/fold_0/manifest.json", "B/B3/results/tar/D/fold_0/manifest.json"),
    ("7. experiment_4/logs/x.log", "B/B6/logs/x.log"),
    ("rq1/results/outputs/comparison.zip", "A/A1/results/outputs/comparison.zip"),
    ("rq1/results/geometry_pilots/p/D.json", "A/A0/results/geometry_pilots/p/D.json"),
    ("rq1/results/geometry/D.json", "A/A3/results/geometry/D.json"),
    ("rq1/results/hosted_ledgers/D/pilot/x", "A/A2/results/hosted_ledgers/D/pilot/x"),
    ("rq2/regression_readout/x", "B/B1/regression_readout/x"),
    # v2: flat folders.
    ("01_exp0a/manifests/before_extraction/accepted_x.json", "general/0/manifests/before_extraction/accepted_x.json"),
    ("03_exp0b/manifests/after_extraction/p.json", "B/B0/manifests/p.json"),
    ("03_exp0b/manifests/reports.json", "../archive/layout_20261126/03_exp0b/manifests/reports.json"),
    ("00_data/manifests/x.json", "general/1_data_and_questions/manifests/x.json"),
    ("04_study_audit/results/notebook/_report.txt", "general/3_study_audit/results/notebook/_report.txt"),
    ("05_synthesis/results/x", "general/4_synthesis/results/x"),
    ("synthesis/results/x", "general/4_synthesis/results/x"),
    ("exp11_data_efficiency/results/group_b/b4.zip", "B/B5/results/group_b/b4.zip"),
    ("exp09_representations/logs/c1.log", "C/C1/logs/c1.log"),
    ("exp10_adaptation/results/adaptation_pilots/D/c", "C/C0/results/adaptation_pilots/D/c"),
    # v3: letter groups.
    ("A/exp03_geometry/results/geometry_pilots/a", "A/A0/results/geometry_pilots/a"),
    ("A/exp02_robustness/results/robustness/x", "A/A2/results/robustness/x"),
    ("B/exp04_decision_features/manifests/reports.json", "B/B1/manifests/reports.json"),
    ("B/exp05_frozen_embeddings/results/pca/e5/D.sqlite3", "B/B2/results/pca/e5/D.sqlite3"),
    ("B/exp06_tar/results/supervised_probe/e5/D.sqlite3", "B/B3/results/supervised_probe/e5/D.sqlite3"),
    ("B/exp07_native_text/results/x", "B/B6/results/x"),
    ("B/exp08_when_text_helps/results/x", "B/B4/results/x"),
    ("B/exp09_data_efficiency/results/x", "B/B5/results/x"),
    ("C/exp10_representations/results/x", "C/C1/results/x"),
    ("C/exp11_adaptation/results/x", "C/C2/results/x"),
    ("figures/general/5_synthesis/a.pdf", "figures/general/4_synthesis/a.pdf"),
    ("figures/general/3_model_and_feature_qualification/a.pdf",
     "../archive/layout_20261126/figures/general/3_model_and_feature_qualification/a.pdf"),
    ("figures/B/exp05_frozen_embeddings/a.pdf", "figures/B/B2/a.pdf"),
    # Current paths are unchanged.
    ("B/B3/results/tar/D/fold_0/manifest.json", "B/B3/results/tar/D/fold_0/manifest.json"),
    ("general/0/manifests/x.json", "general/0/manifests/x.json"),
    ("figures/general/1_data_and_questions/a.pdf", "figures/general/1_data_and_questions/a.pdf"),
    ("figures/CAPTIONS.md", "figures/CAPTIONS.md"),
    ("All_Results.md", "All_Results.md"),
])
def test_output_translation(old, new):
    assert structure.translate_output(PurePosixPath(old)) == PurePosixPath(new)


@pytest.mark.parametrize("unmapped", ["transfers/a.zip", "workspaces/w/x", "manifests/x.json",
                                      "compute_ledger.json", "unknown/x"])
def test_paths_outside_the_layout_are_left_alone(unmapped):
    assert structure.translate_output(PurePosixPath(unmapped)) is None


def test_registry_validation_refuses_ambiguous_layouts():
    reg = structure.registry()
    b1 = reg.entry("B1")

    def with_entries(*entries):
        return dataclasses.replace(reg, entries=tuple(entries))

    clash = dataclasses.replace(reg.entry("A1"), id="clash", group="C", folder="B1")
    with pytest.raises(ValueError, match="unique"):
        structure.validate(with_entries(*reg.entries, clash))
    with pytest.raises(ValueError, match="groups"):
        structure.validate(with_entries(*(dataclasses.replace(e, group="D") if e is b1 else e
                                          for e in reg.entries)))
    with pytest.raises(ValueError, match="its name as its phase"):
        structure.validate(with_entries(*(dataclasses.replace(e, id="B9") if e is b1 else e
                                          for e in reg.entries)))
    with pytest.raises(ValueError, match="Notebook-only"):
        structure.validate(with_entries(*(dataclasses.replace(e, config=True)
                                          if e.id == "A_synthesis" else e for e in reg.entries)))
    with pytest.raises(ValueError, match="Notebook groups"):
        structure.validate(dataclasses.replace(reg, notebook_groups=("general", "A", "B")))


def _write(root: Path, relative: str, content: bytes = b"x") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _v3_tree(tmp_path):
    root = tmp_path / "output_JEVPFN"
    files = {
        "01_exp0a/manifests/before_extraction/accepted.json": b"accept",
        "01_exp0a/logs/run.log": b"log",
        "03_exp0b/manifests/after_extraction/predictor.json": b"0b",
        "03_exp0b/manifests/reports.json": b"{}",
        "03_exp0b/results/notebook/_report.txt": b"qualification",
        "00_data/results/notebook/_report.txt": b"data",
        "02_feature_creation/results/timings/t.sqlite3": b"",
        "04_study_audit/results/notebook/_report.txt": b"audit",
        "05_synthesis/manifests/notebook/_run.json": b"{}",
        "A/exp01_outputs/results/outputs/comparison.zip": b"zip",
        "A/exp03_geometry/results/geometry/D.json": b"geo",
        "A/A_synthesis/manifests/notebook/_run.json": b"{a}",
        "B/exp04_decision_features/manifests/reports.json": b"{b1}",
        "B/exp05_frozen_embeddings/results/pca/e5/D.txt": b"pca",
        "B/exp06_tar/results/tar/D/fold_0/manifest.json": b"tar",
        "B/exp09_data_efficiency/results/group_b/b4.zip": b"b4",
        "C/exp11_adaptation/logs/c2.log": b"c2",
        "figures/general/3_model_and_feature_qualification/_figures.json": b"[0b]",
        "figures/general/5_synthesis/a.pdf": b"%PDF-s",
        "figures/B/exp05_frozen_embeddings/a.pdf": b"%PDF-b2",
        "figures/general/1_data_and_questions/01_lengths.pdf": b"%PDF-data",
        "figures/CAPTIONS.md": b"# captions",
        "All_Results.md": b"# all",
        "transfers/bundle.zip": b"bundle",
        "manifests/layout_migration_20261123T162435Z.json": b"{}",
    }
    for relative, content in files.items():
        _write(root, relative, content)
    with closing(sqlite3.connect(root / "B/exp05_frozen_embeddings/results/pca/e5/D.sqlite3")) as db:
        db.execute("create table t(x)")
        db.commit()
    return root


def test_migration_previews_by_default_then_moves_with_verified_receipt(tmp_path):
    root = _v3_tree(tmp_path)
    before = {p.relative_to(root).as_posix(): file_sha256(p) for p in root.rglob("*") if p.is_file()}
    preview = layout_migration.plan(root)
    assert not preview["conflicts"]
    assert sorted(before) == sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    with pytest.raises(ValueError, match="writer"):
        layout_migration.migrate(root)

    receipt = layout_migration.migrate(root, writers_stopped=True)
    assert receipt["verified"] and receipt["deleted_files"] == 0
    assert receipt["archive"] == str(tmp_path / "archive")
    after = {p.relative_to(tmp_path).as_posix(): file_sha256(p) for p in tmp_path.rglob("*")
             if p.is_file() and not p.name.startswith("layout_migration_2026112")}
    expected = {}
    for old, value in before.items():
        if old.startswith("manifests/layout_migration_"):
            continue
        new = structure.translate_output(PurePosixPath(old)) or PurePosixPath(old)
        expected[layout_migration.destination_path(root, new).relative_to(tmp_path).as_posix()] = value
    assert after == expected
    assert (root / "general/0/manifests/before_extraction/accepted.json").read_bytes() == b"accept"
    assert (root / "B/B0/manifests/predictor.json").read_bytes() == b"0b"
    assert (root / "A/A1/results/outputs/comparison.zip").is_file()
    assert (root / "A/A3/results/geometry/D.json").is_file()
    assert (root / "B/B2/results/pca/e5/D.sqlite3").is_file()
    assert (root / "B/B3/results/tar/D/fold_0/manifest.json").read_bytes() == b"tar"
    assert (root / "B/B5/results/group_b/b4.zip").read_bytes() == b"b4"
    assert (root / "C/C2/logs/c2.log").read_bytes() == b"c2"
    assert (root / "general/3_study_audit/results/notebook/_report.txt").read_bytes() == b"audit"
    assert (root / "figures/general/4_synthesis/a.pdf").read_bytes() == b"%PDF-s"
    assert (root / "figures/B/B2/a.pdf").read_bytes() == b"%PDF-b2"
    archive = tmp_path / "archive/layout_20261126"
    assert (archive / "03_exp0b/results/notebook/_report.txt").read_bytes() == b"qualification"
    assert (archive / "figures/general/3_model_and_feature_qualification/_figures.json").is_file()
    assert (root / "transfers/bundle.zip").is_file() and (root / "All_Results.md").is_file()
    for earlier in ("01_exp0a", "03_exp0b", "00_data", "02_feature_creation", "04_study_audit",
                    "05_synthesis", "A/exp01_outputs", "B/exp05_frozen_embeddings",
                    "figures/general/5_synthesis", "figures/general/3_model_and_feature_qualification"):
        assert not (root / earlier).exists(), earlier
    assert layout_migration.plan(root)["moves"] == []  # Idempotent.


def test_migration_refuses_existing_destinations_and_sidecars(tmp_path):
    root = _v3_tree(tmp_path)
    _write(root, "A/A1/results/outputs/comparison.zip", b"other")
    assert any("comparison.zip" in c for c in layout_migration.plan(root)["conflicts"])
    with pytest.raises(ValueError, match="refused"):
        layout_migration.migrate(root, writers_stopped=True)
    assert (root / "A/exp01_outputs/results/outputs/comparison.zip").read_bytes() == b"zip"

    root2 = _v3_tree(tmp_path / "second")
    _write(root2, "B/exp05_frozen_embeddings/results/pca/e5/D.sqlite3-wal", b"")
    with pytest.raises(ValueError, match="sidecar"):
        layout_migration.migrate(root2, writers_stopped=True)
    with pytest.raises(ValueError, match="output_JEVPFN"):
        layout_migration.plan(tmp_path)


def test_recorded_paths_of_earlier_layouts_resolve_after_migration(tmp_path):
    from src.utils.acceptance import reviewed_acceptance

    root = _v3_tree(tmp_path)
    evidence = root / "01_exp0a/manifests/preflight_x.json"
    _write(root, "01_exp0a/manifests/preflight_x.json", b"{}")
    accepted = root / "01_exp0a/manifests/before_extraction/accepted_x.json"
    accepted.write_text(json.dumps({"accepted": True, "stage": "before_extraction", "reviewed_by": "owner",
                                    "evidence": [{"path": str(evidence), "sha256": file_sha256(evidence)}]}))
    layout_migration.migrate(root, writers_stopped=True)
    # A signed review recorded with v3 paths keeps validating without a rerun.
    assert reviewed_acceptance(str(accepted), stage="before_extraction")["accepted"] is True
    # The numbered v1 spelling of the same file resolves too.
    numbered = root / "3. experiment_0/manifests/before_extraction/accepted_x.json"
    assert structure.recorded_output_path(numbered) == root / "general/0/manifests/before_extraction/accepted_x.json"
    assert structure.recorded_output_path(root / "03_exp0b/manifests/after_extraction/predictor.json").read_bytes() == b"0b"
    assert structure.recorded_output_path(tmp_path / "elsewhere.json") == tmp_path / "elsewhere.json"


def test_rekey_rewrites_only_the_tar_source_hash(tmp_path):
    from src.evaluation import tar

    folder = tmp_path / "fold_0"
    folder.mkdir()
    body = {"status": "complete", "recipe": {"adapter_source": "old", "fold": 0}, "files": {"text_0.npy": "h"}}
    body["artifact_id"] = digest(body)
    (folder / "manifest.json").write_text(json.dumps(body))
    preview = tar.rekey_manifest(folder, "old")
    assert not preview["applied"] and json.loads((folder / "manifest.json").read_text()) == body
    result = tar.rekey_manifest(folder, "old", apply=True)
    saved = json.loads((folder / "manifest.json").read_text())
    assert result["applied"] and saved["recipe"]["adapter_source"] == file_sha256(Path(tar.__file__))
    assert saved["files"] == body["files"] and saved["artifact_id"] == result["new_artifact_id"]
    assert saved["artifact_id"] == digest({k: v for k, v in saved.items() if k != "artifact_id"})
    with pytest.raises(ValueError, match="unknown adapter source"):
        body["recipe"]["adapter_source"] = "foreign"
        body["artifact_id"] = digest({k: v for k, v in body.items() if k != "artifact_id"})
        (folder / "manifest.json").write_text(json.dumps(body))
        tar.rekey_manifest(folder, "old")


def test_rekey_values_are_recorded_in_the_alias_map():
    spec = structure.aliases()["rekey"]
    assert len(spec["pca_implementation"]["old_code_identity"]) == 64
    assert len(spec["tar_adapter_source"]["old_sha256_lf"]) == 64
    raw = yaml.safe_load((paths.REPO_ROOT / "config/structure.yaml").read_text(encoding="utf-8"))
    assert raw["aliases"]["sha256"] == structure.aliases_sha256(paths.REPO_ROOT / "config/layout_aliases.yaml")


def test_literal_rewrite_is_single_pass_and_never_chains():
    text = ("output_JEVPFN/01_exp0a/ output_JEVPFN/B/exp06_tar/ output_JEVPFN/exp11_data_efficiency/ "
            "output_JEVPFN/synthesis/ config/05_synthesis/ output_JEVPFN/general/0/")
    assert structure.rewrite_legacy_literals(text) == (
        "output_JEVPFN/general/0/ output_JEVPFN/B/B3/ output_JEVPFN/B/B5/ "
        "output_JEVPFN/general/4_synthesis/ config/general/4_synthesis/ output_JEVPFN/general/0/")
    current = structure.rewrite_legacy_literals(text)
    assert structure.rewrite_legacy_literals(current) == current


def test_slurm_scripts_use_registry_folders_only():
    import re

    for script in (paths.REPO_ROOT / "scripts").rglob("*.slurm"):
        text = script.read_text(encoding="utf-8")
        for folder in re.findall(r"output_JEVPFN/([^\"\s]+?)/logs/", text):
            assert folder in FOLDERS, (script.name, folder)
