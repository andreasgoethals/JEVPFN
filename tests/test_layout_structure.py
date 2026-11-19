"""The layout registry (18-11-2026, revised 19-11-2026), its aliases and the verified tree migration."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path, PurePosixPath

import pytest

from src.utils import layout_migration, paths, structure
from src.utils.config import load_yaml
from src.utils.serialization import file_sha256

FOLDERS = ("00_data", "01_exp0a", "02_feature_creation", "03_exp0b", "exp01_outputs",
           "exp02_robustness", "exp03_geometry", "exp04_decision_features",
           "exp05_frozen_embeddings", "exp06_tar", "exp07_native_text",
           "exp08_when_text_helps", "exp10_adaptation", "synthesis")


def test_registry_order_folders_and_paper_mapping_default_undecided():
    reg = structure.registry()
    assert reg.folders == FOLDERS
    assert sorted(FOLDERS) == list(FOLDERS)  # Digits sort before "exp"; exp10 after exp08.
    assert {v["placement"] for v in structure.paper_mapping().values()} == {"undecided"}
    assert structure.paper_mapping()["exp04"]["rq"] == "RQ2-A"
    for folder in FOLDERS:  # Folder names never encode paper placement.
        assert not any(word in folder for word in ("main", "appendix", "paper"))


def test_internal_phase_identities_are_unchanged_and_homed_once():
    assert paths.PHASES == structure.PHASES
    assert {p: paths.phase_folder(p) for p in paths.PHASES} == {
        "exploration": "00_data", "question_preparation": "00_data",
        "feature_creation": "02_feature_creation", "experiment_0": "01_exp0a",
        "experiment_1": "exp04_decision_features", "experiment_2": "exp05_frozen_embeddings",
        "experiment_3": "exp06_tar", "experiment_4": "exp07_native_text", "synthesis": "synthesis",
        "rq1": "exp01_outputs", "rq2": "exp08_when_text_helps", "rq3": "exp10_adaptation",
        "regression_readout": "exp04_decision_features/regression_readout",
    }
    # Previous numbered folders and new folder names remain accepted phase aliases.
    assert paths.phase_name("3. experiment_0") == "experiment_0"
    assert paths.phase_name("03_exp0b") == "experiment_0"
    assert paths.phase_name("exp06_tar") == "experiment_3"
    assert paths.phase_name("exp09_adaptation") == "rq3"  # Previous folder name of Exp 10.
    with pytest.raises(ValueError):
        paths.phase_name("9. experiment_1")


def test_routes_split_experiment_0_and_rq1_by_subtree():
    assert paths.phase_path("experiment_0", "manifests", "after_extraction").parts[-3:] == (
        "03_exp0b", "manifests", "after_extraction")
    assert paths.phase_path("experiment_0", "manifests", "before_extraction").parts[-3:] == (
        "01_exp0a", "manifests", "before_extraction")
    assert paths.manifests_dir("experiment_0").parts[-2:] == ("01_exp0a", "manifests")
    assert paths.result_relative("rq1", "outputs") == Path("exp01_outputs/results")
    assert paths.result_relative("rq1", "robustness") == Path("exp02_robustness/results")
    assert paths.result_relative("rq1", "hosted_ledgers") == Path("exp02_robustness/results")
    assert paths.result_relative("rq1", "geometry_pilots") == Path("exp03_geometry/results")
    # 0B fixture fold records stay under one experiment_0 results root (relative records).
    assert paths.result_relative("experiment_0", "evaluation") == Path("01_exp0a/results")


@pytest.mark.parametrize("key, expected", [
    ("experiment_1/main", "exp04_decision_features/main.yaml"),
    ("4. experiment_1/main", "exp04_decision_features/main.yaml"),
    ("rq2/4. experiment_1/main", "exp04_decision_features/main.yaml"),
    ("exp04_decision_features/main", "exp04_decision_features/main.yaml"),
    ("synthesis/analysis", "synthesis/analysis.yaml"),
    ("rq2/8. synthesis/analysis", "synthesis/analysis.yaml"),
    ("experiment_0/debug", "01_exp0a/debug.yaml"),
    ("general/3. experiment_0/self_hosted", "01_exp0a/self_hosted.yaml"),
    ("feature_creation/embedding_models", "02_feature_creation/embedding_models.yaml"),
    ("feature_creation", "02_feature_creation/feature_creation.yaml"),
    ("audit", "00_data/audit.yaml"),
    ("rq1/behavior", "exp02_robustness/behavior.yaml"),
    ("rq1/geometry", "exp03_geometry/geometry.yaml"),
    ("rq3/adaptation", "exp10_adaptation/adaptation.yaml"),
    ("exp09_adaptation/adaptation", "exp10_adaptation/adaptation.yaml"),
    ("structure", "structure.yaml"),
])
def test_config_aliases_resolve_to_registry_files(key, expected):
    path = paths.config_path(key)
    assert path == (paths.REPO_ROOT / "config" / expected).resolve()
    assert path.is_file()


def test_every_registered_config_exists_and_old_keys_load_identically():
    for entry in structure.registry().entries:
        for name in entry.configs:
            assert (paths.REPO_ROOT / "config" / entry.folder / (
                name if "." in name else name + ".yaml")).is_file(), name
    assert load_yaml("experiment_1/main") == load_yaml("exp04_decision_features/main")
    assert load_yaml("rq2/4. experiment_1/main") == load_yaml("experiment_1/main")
    assert paths.config_file("question_versions.csv").is_file()
    with pytest.raises(ValueError):
        paths.config_path("../outside")
    with pytest.raises(ValueError):
        paths.config_file("../question_versions.csv")


def test_config_tree_and_notebooks_follow_registry():
    config = paths.REPO_ROOT / "config"
    assert {p.name for p in config.iterdir() if p.is_dir()} == {
        e.folder for e in structure.registry().entries if e.configs}
    notebooks = paths.notebooks_dir()
    assert not [p for p in notebooks.iterdir() if p.is_dir() and p.name != ".ipynb_checkpoints"]
    found = sorted(p.stem for p in notebooks.glob("*.ipynb"))
    assert found == sorted(n.id for n in structure.registry().notebooks)
    assert all(n.id == n.folder == n.name for n in structure.registry().notebooks)
    assert len(structure.registry().entry("exp10").notebooks) == 1


def test_planned_experiment_nine_is_reserved_without_a_folder():
    reg = structure.registry()
    planned = {p.id: p for p in reg.planned}
    assert planned["exp09"].folder == "exp09_representations"
    assert planned["exp09"].status == "planned_not_implemented"
    assert "exp09_representations" not in reg.folders
    for root in (paths.REPO_ROOT / "config", paths.notebooks_dir()):
        assert not list(root.glob("exp09*"))
    assert reg.previous_folders == {"exp09_adaptation": "exp10_adaptation"}


def test_notebook_ids_aliases_and_figure_folders():
    assert paths.notebook_id("rq2/01_decisions") == "exp04_decision_features"
    assert paths.notebook_id("exp04_decision_features/decisions") == "exp04_decision_features"
    assert paths.notebook_id("general/03_validation") == "03_exp0b"
    assert paths.notebook_id("exp09_adaptation/02_head") == "exp10_adaptation"
    assert paths.notebook_id("rq3/04_full") == "exp10_adaptation"
    assert paths.notebook_phase("exp04_decision_features") == "experiment_1"
    assert paths.notebook_phase("rq2/01_decisions") == "experiment_1"
    assert paths.figures_dir("rq2/01_decisions").parts[-2:] == ("figures", "exp04_decision_features")
    assert paths.figures_dir("exp10_adaptation").parts[-2:] == ("figures", "exp10_adaptation")
    assert paths.figures_dir("exp09_adaptation/02_head") == paths.figures_dir("exp10_adaptation")
    assert paths.notebook_artifacts("03_exp0b", kind="logs").parts[-3:] == (
        "logs", "notebooks", "03_exp0b")
    assert paths.captions_path("experiment_1").parts[-3:] == (
        "figures", "exp04_decision_features", "Captions.md")
    assert paths.all_results_path("03_exp0b").parts[-2:] == ("03_exp0b", "All Results.md")


@pytest.mark.parametrize("old, new", [
    ("3. experiment_0/manifests/before_extraction/a.json", "01_exp0a/manifests/before_extraction/a.json"),
    ("3. experiment_0/manifests/after_extraction/p.json", "03_exp0b/manifests/after_extraction/p.json"),
    ("3. experiment_0/results/token_audit/x.sqlite3", "01_exp0a/results/token_audit/x.sqlite3"),
    ("3. experiment_0/All Results.md", "03_exp0b/All Results.md"),
    ("1. question_preparation/logs/a.log", "00_data/logs/a.log"),
    ("2. feature_creation/manifests/f.json", "02_feature_creation/manifests/f.json"),
    ("4. experiment_1/manifests/reports.json", "exp04_decision_features/manifests/reports.json"),
    ("4. experiment_1/results/cv_folds/a.json", "exp04_decision_features/results/cv_folds/a.json"),
    ("rq2/5. experiment_2/results/pca/e5/x.sqlite3", "exp05_frozen_embeddings/results/pca/e5/x.sqlite3"),
    ("rq2/8. synthesis/results/paired_summary/a", "synthesis/results/paired_summary/a"),
    ("rq2/regression_readout/results/a", "exp04_decision_features/regression_readout/results/a"),
    ("rq2/results/row_benefit/A.json", "exp08_when_text_helps/results/row_benefit/A.json"),
    ("rq1/results/outputs/comparison.zip", "exp01_outputs/results/outputs/comparison.zip"),
    ("rq1/results/robustness/D/p/a.json", "exp02_robustness/results/robustness/D/p/a.json"),
    ("rq1/results/geometry/D.json", "exp03_geometry/results/geometry/D.json"),
    ("rq1/logs/x.log", "exp01_outputs/logs/x.log"),
    ("rq3/results/a", "exp10_adaptation/results/a"),
    ("figures/rq2/01_decisions/a.pdf", "figures/exp04_decision_features/a.pdf"),
    ("figures/4. experiment_1/Captions.md", "figures/exp04_decision_features/Captions.md"),
    ("logs/notebooks/general/01_data/stdout.log", "logs/notebooks/00_data/stdout.log"),
    # 19-11-2026: flat notebook records, Exp 9 -> Exp 10, merged adaptation notebooks archived.
    ("figures/exp04_decision_features/decisions/a.pdf", "figures/exp04_decision_features/a.pdf"),
    ("manifests/notebooks/exp01_outputs/outputs/_figures.json",
     "manifests/notebooks/exp01_outputs/_figures.json"),
    ("results/notebooks/exp01_outputs/outputs/tables/pairs.csv",
     "results/notebooks/exp01_outputs/tables/pairs.csv"),
    ("exp09_adaptation/All Results.md", "exp10_adaptation/All Results.md"),
    ("exp09_adaptation/manifests/reports.json", "exp10_adaptation/manifests/reports.json"),
    ("figures/exp09_adaptation/Captions.md", "figures/exp10_adaptation/Captions.md"),
    ("figures/exp09_adaptation/02_head/a.pdf",
     "archives/layout_20261119/figures/exp09_adaptation/02_head/a.pdf"),
    ("logs/notebooks/exp09_adaptation/01_calibration/stdout.log",
     "archives/layout_20261119/logs/notebooks/exp09_adaptation/01_calibration/stdout.log"),
    ("figures/rq1/rq1_behavior/a.pdf", "archives/layout_20261118/figures/rq1/rq1_behavior/a.pdf"),
    # Current paths are fixed points; a current path in a routed folder re-routes correctly.
    ("01_exp0a/manifests/a.json", "01_exp0a/manifests/a.json"),
    ("exp02_robustness/All Results.md", "exp02_robustness/All Results.md"),
    ("exp03_geometry/manifests/reports.json", "exp03_geometry/manifests/reports.json"),
    ("03_exp0b/manifests/before_extraction/a.json", "01_exp0a/manifests/before_extraction/a.json"),
    ("figures/exp04_decision_features/a.pdf", "figures/exp04_decision_features/a.pdf"),
    ("figures/exp10_adaptation/Captions.md", "figures/exp10_adaptation/Captions.md"),
    ("figures/Captions.md", "figures/Captions.md"),
    ("All Results.md", "All Results.md"),
])
def test_output_translation(old, new):
    assert structure.translate_output(PurePosixPath(old)) == PurePosixPath(new)
    if not new.startswith("archives/"):  # Archived records are outside the layout.
        assert structure.translate_output(PurePosixPath(new)) == PurePosixPath(new)


@pytest.mark.parametrize("unmapped", ["transfers/a.zip", "workspaces/w/x", "archives/a/b",
                                      "results/notebook_visual_review/x.png", "manifests/a.json"])
def test_paths_outside_the_layout_are_left_alone(unmapped):
    assert structure.translate_output(PurePosixPath(unmapped)) is None


def _write(root: Path, relative: str, content: bytes = b"x") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _legacy_tree(tmp_path):
    root = tmp_path / "output_JEVPFN"
    files = {
        "3. experiment_0/manifests/before_extraction/accepted.json": b"accept",
        "3. experiment_0/manifests/after_extraction/predictor.json": b"0b",
        "3. experiment_0/logs/run.log": b"log",
        "4. experiment_1/manifests/reports.json": b"{}",
        "4. experiment_1/All Results.md": b"# r",
        "rq2/4. experiment_1/results/cv_folds/a.json": b"fold",
        "rq2/5. experiment_2/results/pca/e5/D.txt": b"pca",
        "rq1/results/outputs/comparison.zip": b"zip",
        "rq1/results/geometry/D.json": b"geo",
        "figures/rq2/01_decisions/a.pdf": b"%PDF",
        "figures/general/01_data_exploration/old.pdf": b"%PDF-old",
        "exp09_adaptation/All Results.md": b"# rq3",
        "figures/exp09_adaptation/Captions.md": b"# captions",
        "figures/exp09_adaptation/01_calibration/c.pdf": b"%PDF-c",
        "figures/exp03_geometry/Captions.md": b"# geometry",
        "transfers/bundle.zip": b"bundle",
        "All Results.md": b"# all",
    }
    for relative, content in files.items():
        _write(root, relative, content)
    (root / "3. experiment_0/manifests/event_locks/builds").mkdir(parents=True)
    (root / "figures/exp03_geometry/geometry").mkdir(parents=True)  # Empty previous record dir.
    with closing(sqlite3.connect(root / "rq2/5. experiment_2/results/pca/e5/D.sqlite3")) as db:
        db.execute("create table t(x)")
        db.commit()
    return root, files


def test_migration_previews_by_default_then_moves_with_verified_receipt(tmp_path):
    root, files = _legacy_tree(tmp_path)
    before = {p.relative_to(root).as_posix(): file_sha256(p) for p in root.rglob("*") if p.is_file()}
    preview = layout_migration.plan(root)
    assert not preview["conflicts"]
    assert sorted(before) == sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    with pytest.raises(ValueError, match="writer"):
        layout_migration.migrate(root)

    receipt = layout_migration.migrate(root, writers_stopped=True)
    assert receipt["verified"] and receipt["deleted_files"] == 0
    # transfers/, All Results.md and a current folder's Captions.md stay put.
    assert receipt["file_count"] == len(before) - 3
    after = {p.relative_to(root).as_posix(): file_sha256(p) for p in root.rglob("*")
             if p.is_file() and not p.name.startswith("layout_migration_")}
    expected = {}
    for old, digest in before.items():
        new = structure.translate_output(PurePosixPath(old))
        expected[(new or PurePosixPath(old)).as_posix()] = digest
    assert after == expected
    for old, item in receipt["files"].items():
        assert item["sha256"] == before[old]
    assert (root / "03_exp0b/manifests/after_extraction/predictor.json").read_bytes() == b"0b"
    assert (root / "exp04_decision_features/results/cv_folds/a.json").is_file()
    assert (root / "exp04_decision_features/manifests/reports.json").is_file()
    assert (root / "exp03_geometry/results/geometry/D.json").is_file()
    assert (root / "archives/layout_20261118/figures/general/01_data_exploration/old.pdf").is_file()
    assert (root / "figures/exp04_decision_features/a.pdf").is_file()
    assert (root / "exp10_adaptation/All Results.md").read_bytes() == b"# rq3"
    assert (root / "archives/layout_20261119/figures/exp09_adaptation/01_calibration/c.pdf").is_file()
    for legacy in ("3. experiment_0", "4. experiment_1", "rq1", "rq2", "figures/general", "figures/rq2",
                   "exp09_adaptation", "figures/exp09_adaptation", "figures/exp03_geometry/geometry"):
        assert not (root / legacy).exists(), legacy
    saved = json.loads(next((root / "manifests").glob("layout_migration_*.json")).read_text())
    assert saved["files"]["rq1/results/outputs/comparison.zip"]["destination"] == (
        "exp01_outputs/results/outputs/comparison.zip")
    # Idempotent: a second preview finds nothing left to move.
    assert layout_migration.plan(root)["moves"] == []


def test_migration_refuses_existing_destinations_links_and_sidecars(tmp_path):
    root, _ = _legacy_tree(tmp_path)
    _write(root, "exp01_outputs/results/outputs/comparison.zip", b"other")
    conflicts = layout_migration.plan(root)["conflicts"]
    assert any("comparison.zip" in c for c in conflicts)
    with pytest.raises(ValueError, match="refused"):
        layout_migration.migrate(root, writers_stopped=True)
    assert (root / "rq1/results/outputs/comparison.zip").read_bytes() == b"zip"

    root2, _ = _legacy_tree(tmp_path / "second")
    _write(root2, "rq2/5. experiment_2/results/pca/e5/D.sqlite3-wal", b"")
    with pytest.raises(ValueError, match="sidecar"):
        layout_migration.migrate(root2, writers_stopped=True)
    assert (root2 / "rq2/5. experiment_2/results/pca/e5/D.txt").is_file()

    with pytest.raises(ValueError, match="output_JEVPFN"):
        layout_migration.plan(tmp_path)


def test_recorded_legacy_paths_resolve_after_migration(tmp_path):
    root, _ = _legacy_tree(tmp_path)
    old = root / "3. experiment_0/manifests/before_extraction/accepted.json"
    layout_migration.migrate(root, writers_stopped=True)
    moved = structure.recorded_output_path(old)
    assert moved == root / "01_exp0a/manifests/before_extraction/accepted.json"
    relative_across_route = (root / "03_exp0b/manifests/after_extraction"
                             / "../before_extraction/accepted.json")
    assert structure.recorded_output_path(relative_across_route).read_bytes() == b"accept"
    assert structure.recorded_output_path(tmp_path / "elsewhere.json") == tmp_path / "elsewhere.json"


def test_reviewed_acceptance_reads_evidence_recorded_before_migration(tmp_path):
    from src.utils.acceptance import reviewed_acceptance

    root = tmp_path / "output_JEVPFN"
    evidence = root / "3. experiment_0/manifests/preflight_x.json"
    _write(root, "3. experiment_0/manifests/preflight_x.json", b"{}")
    accepted = root / "3. experiment_0/manifests/before_extraction/accepted_x.json"
    accepted.parent.mkdir(parents=True)
    accepted.write_text(json.dumps({"accepted": True, "stage": "before_extraction", "reviewed_by": "owner",
                                    "evidence": [{"path": str(evidence), "sha256": file_sha256(evidence)}]}))
    layout_migration.migrate(root, writers_stopped=True)
    assert reviewed_acceptance(str(accepted), stage="before_extraction")["accepted"] is True


def test_layout_only_source_changes_explain_moves_and_folder_literals_only():
    old_files = {"config/rq2/4. experiment_1/main.yaml": "h1", "scripts/slurm/a.slurm": "h2",
                 "scripts/slurm/b.slurm": "h3", "config/general/3. experiment_0/scheduling.yaml": "h4"}
    current = {"config/exp04_decision_features/main.yaml": "h1", "scripts/slurm/a.slurm": "n2",
               "scripts/slurm/b.slurm": "n3", "config/01_exp0a/scheduling.yaml": "changed"}
    old_text = {"scripts/slurm/a.slurm": '#SBATCH --output="output_JEVPFN/3. experiment_0/logs/%j"\n',
                "scripts/slurm/b.slurm": "python -m science --seed 1\n"}
    new_text = {"scripts/slurm/a.slurm": '#SBATCH --output="output_JEVPFN/01_exp0a/logs/%j"\n',
                "scripts/slurm/b.slurm": "python -m science --seed 2\n"}
    explained = structure.layout_only_changes(old_files, current, old_text.get, new_text.get)
    assert explained == {"config/rq2/4. experiment_1/main.yaml",
                         "config/exp04_decision_features/main.yaml", "scripts/slurm/a.slurm"}


def test_slurm_scripts_use_registry_folders_only():
    import re

    for script in (paths.REPO_ROOT / "scripts").rglob("*.slurm"):
        text = script.read_text(encoding="utf-8")
        for folder in re.findall(r"output_JEVPFN/([^/\"]+)/", text):
            assert folder in FOLDERS, (script.name, folder)
