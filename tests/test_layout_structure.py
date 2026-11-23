"""The layout registry (18-11-2026, revised 19-11-2026; v3 letter groups 23-11-2026), its
aliases and the verified tree migration."""

import dataclasses
import json
import sqlite3
from contextlib import closing
from pathlib import Path, PurePosixPath

import pytest

from src.utils import layout_migration, paths, structure
from src.utils.config import load_yaml
from src.utils.serialization import file_sha256

FOLDERS = ("00_data", "01_exp0a", "02_feature_creation", "03_exp0b", "04_study_audit",
           "05_synthesis", "A/exp01_outputs", "A/exp02_robustness", "A/exp03_geometry",
           "A/A_synthesis", "B/exp04_decision_features", "B/exp05_frozen_embeddings",
           "B/exp06_tar", "B/exp07_native_text", "B/exp08_when_text_helps",
           "B/exp09_data_efficiency", "B/B_synthesis", "C/exp10_representations",
           "C/exp11_adaptation", "C/C_synthesis")
NOTEBOOKS = ("general/1_data_and_questions", "general/2_feature_creation",
             "general/3_model_and_feature_qualification", "general/4_study_audit",
             "general/5_synthesis", "A/exp01_outputs", "A/exp02_robustness", "A/exp03_geometry",
             "A/A_synthesis", "B/exp04_decision_features", "B/exp05_frozen_embeddings",
             "B/exp06_tar", "B/exp07_native_text", "B/exp08_when_text_helps",
             "B/exp09_data_efficiency", "B/B_synthesis", "C/exp10_representations",
             "C/exp11_adaptation", "C/C_synthesis")


def test_registry_order_folders_and_paper_mapping_default_undecided():
    reg = structure.registry()
    assert reg.folders == FOLDERS
    assert reg.top_level == ("00_data", "01_exp0a", "02_feature_creation", "03_exp0b",
                             "04_study_audit", "05_synthesis", "A", "B", "C")
    assert tuple(n.id for n in reg.notebooks) == NOTEBOOKS
    assert reg.notebook_groups == ("general", "A", "B", "C", "paper")
    assert {v["placement"] for v in structure.paper_mapping().values()} == {"undecided"}
    # Entry IDs are never renamed; the notebook-only entries were added as undecided.
    assert structure.paper_mapping()["exp04"] == {"placement": "undecided"}
    assert set(structure.paper_mapping()) == {e.id for e in reg.entries}
    assert set(reg.paper) == {e.id for e in reg.entries}
    for entry in reg.entries:  # Folder names never encode paper placement.
        assert not any(word in entry.folder for word in ("main", "appendix", "paper"))


def test_internal_phase_identities_are_unchanged_and_homed_once():
    assert paths.PHASES == structure.PHASES
    assert {p: paths.phase_folder(p) for p in paths.PHASES} == {
        "exploration": "00_data", "question_preparation": "00_data",
        "feature_creation": "02_feature_creation", "experiment_0": "01_exp0a",
        "experiment_1": "B/exp04_decision_features", "experiment_2": "B/exp05_frozen_embeddings",
        "experiment_3": "B/exp06_tar", "experiment_4": "B/exp07_native_text",
        "synthesis": "05_synthesis", "rq1": "A/exp01_outputs", "rq2": "B/exp08_when_text_helps",
        "rq3": "C/exp11_adaptation",
        "regression_readout": "B/exp04_decision_features/regression_readout",
        "data_efficiency": "B/exp09_data_efficiency", "representations": "C/exp10_representations",
    }
    # Previous numbered folders, flat and current folder names remain accepted phase aliases.
    assert paths.phase_name("3. experiment_0") == "experiment_0"
    assert paths.phase_name("03_exp0b") == "experiment_0"
    assert paths.phase_name("exp06_tar") == "experiment_3"
    assert paths.phase_name("B/exp06_tar") == "experiment_3"
    assert paths.phase_name("exp09_adaptation") == "rq3"  # Exp 9 name before 19-11-2026.
    assert paths.phase_name("exp10_adaptation") == "rq3"  # Exp 10 name 19-23-11-2026.
    assert paths.phase_name("exp11_adaptation") == "rq3"
    assert paths.phase_name("exp09_representations") == "representations"
    assert paths.phase_name("exp10_representations") == "representations"
    assert paths.phase_name("exp11_data_efficiency") == "data_efficiency"
    assert paths.phase_name("exp09_data_efficiency") == "data_efficiency"
    assert paths.phase_name("synthesis") == "synthesis"
    for invalid in ("9. experiment_1", "A_synthesis", "04_study_audit", "A"):
        with pytest.raises(ValueError):
            paths.phase_name(invalid)


def test_routes_split_experiment_0_and_rq1_by_subtree():
    assert paths.phase_path("experiment_0", "manifests", "after_extraction").parts[-3:] == (
        "03_exp0b", "manifests", "after_extraction")
    assert paths.phase_path("experiment_0", "manifests", "before_extraction").parts[-3:] == (
        "01_exp0a", "manifests", "before_extraction")
    assert paths.manifests_dir("experiment_0").parts[-2:] == ("01_exp0a", "manifests")
    assert paths.result_relative("rq1", "outputs") == Path("A/exp01_outputs/results")
    assert paths.result_relative("rq1", "robustness") == Path("A/exp02_robustness/results")
    assert paths.result_relative("rq1", "hosted_ledgers") == Path("A/exp02_robustness/results")
    assert paths.result_relative("rq1", "geometry_pilots") == Path("A/exp03_geometry/results")
    # 0B fixture fold records stay under one experiment_0 results root (relative records).
    assert paths.result_relative("experiment_0", "evaluation") == Path("01_exp0a/results")


@pytest.mark.parametrize("key, expected", [
    ("experiment_1/main", "B/main.yaml"),
    ("4. experiment_1/main", "B/main.yaml"),
    ("rq2/4. experiment_1/main", "B/main.yaml"),
    ("exp04_decision_features/main", "B/main.yaml"),
    ("B/main", "B/main.yaml"),
    ("main", "B/main.yaml"),
    ("synthesis/analysis", "05_synthesis/analysis.yaml"),
    ("05_synthesis/analysis", "05_synthesis/analysis.yaml"),
    ("rq2/8. synthesis/analysis", "05_synthesis/analysis.yaml"),
    ("experiment_0/debug", "01_exp0a/debug.yaml"),
    ("general/3. experiment_0/self_hosted", "01_exp0a/self_hosted.yaml"),
    ("feature_creation/embedding_models", "02_feature_creation/embedding_models.yaml"),
    ("feature_creation", "02_feature_creation/feature_creation.yaml"),
    ("audit", "00_data/audit.yaml"),
    ("rq1/behavior", "A/behavior.yaml"),
    ("exp02_robustness/behavior", "A/behavior.yaml"),
    ("A/behavior", "A/behavior.yaml"),
    ("rq1/geometry", "A/geometry.yaml"),
    ("exp08_when_text_helps/when_text_helps", "B/when_text_helps.yaml"),
    ("rq2/text_column_refits", "B/text_column_refits.yaml"),
    ("data_efficiency/data_efficiency", "B/data_efficiency.yaml"),
    ("exp11_data_efficiency/data_efficiency", "B/data_efficiency.yaml"),
    ("representations/representations", "C/representations.yaml"),
    ("exp09_representations/representations", "C/representations.yaml"),
    ("rq3/adaptation", "C/adaptation.yaml"),
    ("exp09_adaptation/adaptation", "C/adaptation.yaml"),
    ("exp10_adaptation/adaptation", "C/adaptation.yaml"),
    ("C/adaptation", "C/adaptation.yaml"),
    ("structure", "structure.yaml"),
])
def test_config_aliases_resolve_to_registry_files(key, expected):
    path = paths.config_path(key)
    assert path == (paths.REPO_ROOT / "config" / expected).resolve()
    assert path.is_file()


def test_every_registered_config_exists_and_old_keys_load_identically():
    for entry in structure.registry().entries:
        for name in entry.configs:
            assert (paths.REPO_ROOT / "config" / entry.config_dir / (
                name if "." in name else name + ".yaml")).is_file(), name
    assert load_yaml("experiment_1/main") == load_yaml("exp04_decision_features/main")
    assert load_yaml("rq2/4. experiment_1/main") == load_yaml("B/main")
    assert load_yaml("exp11_data_efficiency/data_efficiency") == load_yaml("B/data_efficiency")
    assert paths.config_file("question_versions.csv") == paths.REPO_ROOT / "config/A/question_versions.csv"
    assert paths.config_file("question_versions.csv").is_file()
    assert paths.CONFIG_GROUPS["experiment_1"] == "B" and paths.CONFIG_GROUPS["synthesis"] == "05_synthesis"
    assert paths.CONFIG_GROUPS["experiment_0"] == "01_exp0a" and paths.CONFIG_GROUPS["rq3"] == "C"
    with pytest.raises(ValueError):
        paths.config_path("../outside")
    with pytest.raises(ValueError):
        paths.config_file("../question_versions.csv")


def test_config_tree_and_notebooks_follow_registry():
    reg = structure.registry()
    config = paths.REPO_ROOT / "config"
    assert {p.name for p in config.iterdir() if p.is_dir()} == set(reg.config_dirs)
    assert set(reg.config_dirs) == {"00_data", "01_exp0a", "02_feature_creation", "05_synthesis",
                                    "A", "B", "C"}
    for group in ("A", "B", "C"):  # Grouped configs are flat in config/<group>/.
        assert not [p for p in (config / group).iterdir() if p.is_dir()]
    notebooks = paths.notebooks_dir()
    assert not list(notebooks.glob("*.ipynb"))  # No flat notebooks any more.
    assert {p.name for p in notebooks.iterdir() if p.is_dir() and p.name != ".ipynb_checkpoints"} == set(
        reg.notebook_groups)
    found = sorted(p.relative_to(notebooks).with_suffix("").as_posix() for p in notebooks.rglob("*.ipynb")
                   if ".ipynb_checkpoints" not in p.parts)
    assert found == sorted(NOTEBOOKS)
    assert [p.name for p in (notebooks / "paper").iterdir()] == [".gitkeep"]
    for n in reg.notebooks:
        entry = reg.entry(n.folder)
        assert n.id == f"{entry.notebook_group}/{n.name}" and n.folder == entry.path
    assert len(reg.entry("exp10").notebooks) == 1


def test_notebook_only_entries_own_records_but_no_phase():
    reg = structure.registry()
    only = {e.id: e for e in reg.entries if e.notebook_only}
    assert set(only) == {"study_audit", "A_synthesis", "B_synthesis", "C_synthesis"}
    assert {e.path for e in only.values()} == {"04_study_audit", "A/A_synthesis", "B/B_synthesis",
                                               "C/C_synthesis"}
    assert all(not e.phases and not e.routes and not e.configs for e in only.values())
    assert paths.notebook_artifacts("A/A_synthesis", kind="logs").parts[-5:] == (
        "output_JEVPFN", "A", "A_synthesis", "logs", "notebook")
    assert paths.figures_dir("general/4_study_audit").parts[-3:] == ("figures", "general", "4_study_audit")
    assert paths.notebook_artifacts("general/4_study_audit").parts[-3:] == (
        "04_study_audit", "manifests", "notebook")
    assert paths.notebook_phase("C/C_synthesis") == "rq3"
    for kept in ("A/A_synthesis/results/notebook/_report.txt", "A/A_synthesis/results/x.json",
                 "04_study_audit/logs/notebook/stdout.log"):
        assert structure.translate_output(PurePosixPath(kept)) == PurePosixPath(kept)


def test_group_b4_and_c_renumbering_keeps_ids_and_aliases_distinct():
    reg = structure.registry()
    # Entry IDs (paper keys) and internal phases stay; only folders are renumbered.
    assert reg.entry("exp11").path == "B/exp09_data_efficiency"
    assert reg.entry("exp09").path == "C/exp10_representations"
    assert reg.entry("exp10").path == "C/exp11_adaptation"
    assert reg.previous_folders == {"exp11_data_efficiency": "B/exp09_data_efficiency",
                                    "exp09_representations": "C/exp10_representations",
                                    "exp10_adaptation": "C/exp11_adaptation",
                                    "exp09_adaptation": "C/exp11_adaptation",
                                    "synthesis": "05_synthesis"}
    for old, new in [("exp11_data_efficiency", "B/exp09_data_efficiency"),
                     ("exp09_data_efficiency", "B/exp09_data_efficiency"),
                     ("exp09_representations", "C/exp10_representations"),
                     ("exp10_representations", "C/exp10_representations"),
                     ("exp10_adaptation", "C/exp11_adaptation"),
                     ("exp09_adaptation", "C/exp11_adaptation"),
                     ("exp11_adaptation", "C/exp11_adaptation"),
                     ("synthesis", "general/5_synthesis"), ("00_data", "general/1_data_and_questions"),
                     ("03_exp0b", "general/3_model_and_feature_qualification"),
                     ("exp04_decision_features", "B/exp04_decision_features")]:
        assert paths.notebook_id(old) == new, old
    # Old exp10_adaptation (C2) is never confused with new exp10_representations (C1), nor old
    # exp09_representations with new exp09_data_efficiency, nor old exp11 (B4) with new exp11 (C2).
    for old, new in [("exp10_adaptation/results/a", "C/exp11_adaptation/results/a"),
                     ("exp10_representations/results/a", "C/exp10_representations/results/a"),
                     ("exp09_representations/results/a", "C/exp10_representations/results/a"),
                     ("exp09_data_efficiency/results/a", "B/exp09_data_efficiency/results/a"),
                     ("exp11_data_efficiency/results/a", "B/exp09_data_efficiency/results/a"),
                     ("exp11_adaptation/results/a", "C/exp11_adaptation/results/a")]:
        assert structure.translate_output(PurePosixPath(old)) == PurePosixPath(new), old
    assert paths.notebook_phase("exp10_adaptation") == "rq3"
    assert paths.notebook_phase("exp10_representations") == "representations"
    assert paths.notebook_phase("exp11_data_efficiency") == "data_efficiency"
    assert paths.notebook_phase("exp11_adaptation") == "rq3"


def test_notebook_ids_aliases_and_figure_folders():
    assert paths.notebook_id("rq2/01_decisions") == "B/exp04_decision_features"
    assert paths.notebook_id("exp04_decision_features/decisions") == "B/exp04_decision_features"
    assert paths.notebook_id("general/03_validation") == "general/3_model_and_feature_qualification"
    assert paths.notebook_id("exp09_adaptation/02_head") == "C/exp11_adaptation"
    assert paths.notebook_id("rq3/04_full") == "C/exp11_adaptation"
    # The previous `general/01_data` ID and the current `general/1_...` ID stay distinct keys.
    assert paths.notebook_id("general/01_data") == "general/1_data_and_questions"
    assert paths.notebook_id("general/1_data_and_questions") == "general/1_data_and_questions"
    assert paths.notebook_id("general/02_features") == "general/2_feature_creation"
    assert paths.notebook_phase("B/exp04_decision_features") == "experiment_1"
    assert paths.notebook_phase("rq2/01_decisions") == "experiment_1"
    figures = paths.figures_dir()
    assert paths.figures_dir("rq2/01_decisions") == figures / "B" / "exp04_decision_features"
    assert paths.figures_dir("C/exp11_adaptation") == figures / "C" / "exp11_adaptation"
    assert paths.figures_dir("exp09_adaptation/02_head") == paths.figures_dir("C/exp11_adaptation")
    assert paths.figures_dir("exp11_data_efficiency") == figures / "B" / "exp09_data_efficiency"
    assert paths.notebook_artifacts("general/3_model_and_feature_qualification", kind="logs").parts[-3:] == (
        "03_exp0b", "logs", "notebook")
    assert paths.notebook_artifacts("B/exp09_data_efficiency", kind="results").parts[-4:] == (
        "B", "exp09_data_efficiency", "results", "notebook")
    assert paths.report_index_path("B/exp04_decision_features").parts[-4:] == (
        "B", "exp04_decision_features", "manifests", "reports.json")
    assert paths.report_index_path("rq3").parts[-4:-2] == ("C", "exp11_adaptation")
    assert paths.captions_path().parts[-2:] == ("figures", "CAPTIONS.md")
    assert paths.all_results_path().parts[-2:] == ("output_JEVPFN", "All_Results.md")
    # The figure grouping lives in one function: the folder notebook's ID.
    for folder in FOLDERS:
        group = structure.figure_group(folder)
        assert structure.figure_folder(group) == folder
    assert structure.figure_group("01_exp0a") == "general/01_exp0a"  # A folder without a notebook.
    assert structure.figure_group("exp04_decision_features/regression_readout") == (
        "B/exp04_decision_features/regression_readout")


_REPORTS = "../archive/folder_reports_20261119"
_QUALIFICATION = "figures/general/3_model_and_feature_qualification"


@pytest.mark.parametrize("old, new", [
    ("3. experiment_0/manifests/before_extraction/a.json", "01_exp0a/manifests/before_extraction/a.json"),
    ("3. experiment_0/manifests/after_extraction/p.json", "03_exp0b/manifests/after_extraction/p.json"),
    ("3. experiment_0/results/token_audit/x.sqlite3", "01_exp0a/results/token_audit/x.sqlite3"),
    ("3. experiment_0/All Results.md", f"{_REPORTS}/3. experiment_0/All Results.md"),
    ("1. question_preparation/logs/a.log", "00_data/logs/a.log"),
    ("2. feature_creation/manifests/f.json", "02_feature_creation/manifests/f.json"),
    ("4. experiment_1/manifests/reports.json", "B/exp04_decision_features/manifests/reports.json"),
    ("4. experiment_1/results/cv_folds/a.json", "B/exp04_decision_features/results/cv_folds/a.json"),
    ("rq2/5. experiment_2/results/pca/e5/x.sqlite3", "B/exp05_frozen_embeddings/results/pca/e5/x.sqlite3"),
    ("rq2/8. synthesis/results/paired_summary/a", "05_synthesis/results/paired_summary/a"),
    ("rq2/regression_readout/results/a", "B/exp04_decision_features/regression_readout/results/a"),
    ("rq2/results/row_benefit/A.json", "B/exp08_when_text_helps/results/row_benefit/A.json"),
    ("rq1/results/outputs/comparison.zip", "A/exp01_outputs/results/outputs/comparison.zip"),
    ("rq1/results/robustness/D/p/a.json", "A/exp02_robustness/results/robustness/D/p/a.json"),
    ("rq1/results/geometry/D.json", "A/exp03_geometry/results/geometry/D.json"),
    ("rq1/logs/x.log", "A/exp01_outputs/logs/x.log"),
    ("rq3/results/a", "C/exp11_adaptation/results/a"),
    ("figures/rq2/01_decisions/a.pdf", "figures/B/exp04_decision_features/a.pdf"),
    ("figures/4. experiment_1/Captions.md", f"{_REPORTS}/figures/4. experiment_1/Captions.md"),
    ("logs/notebooks/general/01_data/stdout.log", "00_data/logs/notebook/stdout.log"),
    # 19-11-2026: flat notebook records, Exp 9 -> Exp 10, merged adaptation notebooks archived.
    ("figures/exp04_decision_features/decisions/a.pdf", "figures/B/exp04_decision_features/a.pdf"),
    ("manifests/notebooks/exp01_outputs/outputs/_figures.json", "figures/A/exp01_outputs/_figures.json"),
    ("results/notebooks/exp01_outputs/outputs/tables/pairs.csv",
     "A/exp01_outputs/results/notebook/tables/pairs.csv"),
    ("exp09_adaptation/All Results.md", f"{_REPORTS}/exp09_adaptation/All Results.md"),
    ("exp09_adaptation/manifests/reports.json", "C/exp11_adaptation/manifests/reports.json"),
    ("exp09_adaptation/figures/01_a.pdf", "figures/C/exp11_adaptation/01_a.pdf"),
    ("figures/exp09_adaptation/Captions.md", f"{_REPORTS}/figures/exp09_adaptation/Captions.md"),
    ("figures/exp09_adaptation/02_head/a.pdf",
     "../archive/layout_20261119/figures/exp09_adaptation/02_head/a.pdf"),
    ("logs/notebooks/exp09_adaptation/01_calibration/stdout.log",
     "../archive/layout_20261119/logs/notebooks/exp09_adaptation/01_calibration/stdout.log"),
    ("figures/rq1/rq1_behavior/a.pdf", "../archive/layout_20261118/figures/rq1/rq1_behavior/a.pdf"),
    # The archive moved out of the output root (19-11-2026): output_JEVPFN/archives -> archive/.
    ("archives/layout_20261118/figures/x.pdf", "../archive/layout_20261118/figures/x.pdf"),
    ("archives/rq_layout_20261116", "../archive/rq_layout_20261116"),
    # The flat layout (18-23-11-2026) -> letter groups and renumbered B4/C (23-11-2026).
    ("01_exp0a/manifests/a.json", "01_exp0a/manifests/a.json"),
    ("exp02_robustness/results/robustness/x.json", "A/exp02_robustness/results/robustness/x.json"),
    ("exp01_outputs/results/robustness/x.json", "A/exp02_robustness/results/robustness/x.json"),
    ("exp11_data_efficiency/results/group_b/b4.zip", "B/exp09_data_efficiency/results/group_b/b4.zip"),
    ("exp09_representations/results/strands_hidden/D/x", "C/exp10_representations/results/strands_hidden/D/x"),
    ("exp10_adaptation/results/adaptation/D/fold_0/x", "C/exp11_adaptation/results/adaptation/D/fold_0/x"),
    ("exp10_adaptation/logs/notebook/stdout.log", "C/exp11_adaptation/logs/notebook/stdout.log"),
    ("synthesis/manifests/reports.json", "05_synthesis/manifests/reports.json"),
    ("synthesis/results/notebook/_report.txt", "05_synthesis/results/notebook/_report.txt"),
    ("figures/exp02_robustness/01_a.pdf", "figures/A/exp02_robustness/01_a.pdf"),
    ("figures/03_exp0b/_figures.json", f"{_QUALIFICATION}/_figures.json"),
    ("figures/00_data/check/01_a.pdf", "figures/general/1_data_and_questions/check/01_a.pdf"),
    ("figures/00_data", "figures/general/1_data_and_questions"),
    ("figures/exp11_data_efficiency/01_a.pdf", "figures/B/exp09_data_efficiency/01_a.pdf"),
    ("figures/exp09_representations/01_a.pdf", "figures/C/exp10_representations/01_a.pdf"),
    ("figures/exp10_adaptation/01_a.pdf", "figures/C/exp11_adaptation/01_a.pdf"),
    ("figures/synthesis/_figures.json", "figures/general/5_synthesis/_figures.json"),
    ("figures/01_exp0a/a.pdf", "figures/general/01_exp0a/a.pdf"),
    ("03_exp0b/logs/notebook/stdout.log", "03_exp0b/logs/notebook/stdout.log"),
    ("exp03_geometry/results/notebook/_report.txt", "A/exp03_geometry/results/notebook/_report.txt"),
    ("exp04_decision_features/regression_readout/figures/a.pdf",
     "B/exp04_decision_features/regression_readout/figures/a.pdf"),
    ("exp03_geometry/manifests/reports.json", "A/exp03_geometry/manifests/reports.json"),
    ("03_exp0b/manifests/reports.json", "03_exp0b/manifests/reports.json"),
    ("03_exp0b/manifests/before_extraction/a.json", "01_exp0a/manifests/before_extraction/a.json"),
    ("figures/CAPTIONS.md", "figures/CAPTIONS.md"),
    ("All_Results.md", "All_Results.md"),
    # The rejected experiment-folder-first figures and per-folder reports (19-11-2026).
    ("exp02_robustness/figures/01_a.pdf", "figures/A/exp02_robustness/01_a.pdf"),
    ("03_exp0b/figures/_figures.json", f"{_QUALIFICATION}/_figures.json"),
    ("00_data/figures/check/01_a.pdf", "figures/general/1_data_and_questions/check/01_a.pdf"),
    ("CAPTIONS.md", "figures/CAPTIONS.md"),
    ("exp02_robustness/All_Results.md", f"{_REPORTS}/exp02_robustness/All_Results.md"),
    ("exp02_robustness/CAPTIONS.md", f"{_REPORTS}/exp02_robustness/CAPTIONS.md"),
    # Kind-first notebook records (until 19-11-2026).
    ("figures/exp04_decision_features/a.pdf", "figures/B/exp04_decision_features/a.pdf"),
    ("figures/exp10_adaptation/Captions.md", f"{_REPORTS}/figures/exp10_adaptation/Captions.md"),
    ("figures/Captions.md", "figures/CAPTIONS.md"),
    ("All Results.md", "All_Results.md"),
    ("exp02_robustness/All Results.md", f"{_REPORTS}/exp02_robustness/All Results.md"),
    ("manifests/notebooks/03_exp0b/_figures.json", f"{_QUALIFICATION}/_figures.json"),
    ("manifests/notebooks/03_exp0b/_run.json", "03_exp0b/manifests/notebook/_run.json"),
    ("results/notebooks/exp01_outputs/_report.txt", "A/exp01_outputs/results/notebook/_report.txt"),
    ("logs/notebooks/exp02_robustness/stderr.log", "A/exp02_robustness/logs/notebook/stderr.log"),
    ("logs/notebooks/exp11_data_efficiency/stdout.log", "B/exp09_data_efficiency/logs/notebook/stdout.log"),
    # Current v3 paths are fixed points; the new `general/1_...` group never collides with the
    # retired `general/01_*` notebook IDs (moved or archived).
    ("figures/general/1_data_and_questions/a.pdf", "figures/general/1_data_and_questions/a.pdf"),
    ("figures/general/01_data/a.pdf", "figures/general/1_data_and_questions/a.pdf"),
    ("figures/general/01_data_exploration/old.pdf",
     "../archive/layout_20261118/figures/general/01_data_exploration/old.pdf"),
    ("figures/B/exp09_data_efficiency/a.pdf", "figures/B/exp09_data_efficiency/a.pdf"),
    ("figures/C/C_synthesis/_figures.json", "figures/C/C_synthesis/_figures.json"),
    ("B/exp04_decision_features/regression_readout/results/a",
     "B/exp04_decision_features/regression_readout/results/a"),
    ("A/exp01_outputs/results/robustness/x.json", "A/exp02_robustness/results/robustness/x.json"),
    ("C/exp10_representations/manifests/reports.json", "C/exp10_representations/manifests/reports.json"),
])
def test_output_translation(old, new):
    assert structure.translate_output(PurePosixPath(old)) == PurePosixPath(new)
    if not new.startswith("../"):  # Archived records are outside the output root.
        assert structure.translate_output(PurePosixPath(new)) == PurePosixPath(new)


@pytest.mark.parametrize("unmapped", ["transfers/a.zip", "workspaces/w/x", "archives",
                                      "results/notebook_visual_review/x.png", "manifests/a.json",
                                      "figures/A/x.pdf", "A/unknown/x.json", "B",
                                      "figures/general", "logs/notebooks/unknown/x.log"])
def test_paths_outside_the_layout_are_left_alone(unmapped):
    assert structure.translate_output(PurePosixPath(unmapped)) is None


def test_registry_validation_refuses_ambiguous_layouts():
    reg = structure.registry()
    exp04 = reg.entry("exp04")

    def with_entries(*entries):
        return dataclasses.replace(reg, entries=tuple(entries))

    # A folder name reused in another group would make flat paths ambiguous.
    clash = dataclasses.replace(reg.entry("exp01"), id="clash", group="C", folder="exp04_decision_features")
    with pytest.raises(ValueError, match="unique"):
        structure.validate(with_entries(*reg.entries, clash))
    with pytest.raises(ValueError, match="groups"):
        structure.validate(with_entries(*(dataclasses.replace(e, group="D") if e is exp04 else e
                                          for e in reg.entries)))
    with pytest.raises(ValueError, match="Notebook-only"):
        structure.validate(with_entries(*(dataclasses.replace(e, configs=("x",))
                                          if e.id == "A_synthesis" else e for e in reg.entries)))
    with pytest.raises(ValueError, match="Notebook groups"):
        structure.validate(dataclasses.replace(reg, notebook_groups=("general", "A", "B")))


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
        "archives/rq_layout_20261116/x.txt": b"archived",
        # The flat layout of 18-23-11-2026.
        "01_exp0a/manifests/receipt.json": b"0a",
        "exp11_data_efficiency/results/group_b/b4.zip": b"b4",
        "exp11_data_efficiency/manifests/reports.json": b"{b4}",
        "exp09_representations/logs/c1.log": b"c1",
        "exp10_adaptation/results/notebook/_report.txt": b"c2",
        "synthesis/results/notebook/_report.txt": b"synthesis",
        "figures/exp11_data_efficiency/01_curve.pdf": b"%PDF-b4",
        "figures/exp11_data_efficiency/_figures.json": b"[]",
        "figures/00_data/01_lengths.pdf": b"%PDF-data",
        "figures/03_exp0b/_figures.json": b"[0b]",
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
    assert receipt["archive"] == str(tmp_path / "archive")
    # transfers/ and the current 01_exp0a receipt stay put; everything else moves.
    assert receipt["file_count"] == len(before) - 2
    after = {p.relative_to(tmp_path).as_posix(): file_sha256(p) for p in tmp_path.rglob("*")
             if p.is_file() and not p.name.startswith("layout_migration_")}
    expected = {}
    for old, digest in before.items():
        new = structure.translate_output(PurePosixPath(old)) or PurePosixPath(old)
        moved = layout_migration.destination_path(root, new)
        expected[moved.relative_to(tmp_path).as_posix()] = digest
    assert after == expected
    for old, item in receipt["files"].items():
        assert item["sha256"] == before[old]
    assert (root / "03_exp0b/manifests/after_extraction/predictor.json").read_bytes() == b"0b"
    assert (root / "B/exp04_decision_features/results/cv_folds/a.json").is_file()
    assert (root / "B/exp04_decision_features/manifests/reports.json").is_file()
    assert (root / "A/exp03_geometry/results/geometry/D.json").is_file()
    assert (root / "B/exp09_data_efficiency/results/group_b/b4.zip").read_bytes() == b"b4"
    assert (root / "B/exp09_data_efficiency/manifests/reports.json").read_bytes() == b"{b4}"
    assert (root / "C/exp10_representations/logs/c1.log").read_bytes() == b"c1"
    assert (root / "C/exp11_adaptation/results/notebook/_report.txt").read_bytes() == b"c2"
    assert (root / "05_synthesis/results/notebook/_report.txt").read_bytes() == b"synthesis"
    assert (root / "figures/B/exp09_data_efficiency/01_curve.pdf").read_bytes() == b"%PDF-b4"
    assert (root / "figures/general/1_data_and_questions/01_lengths.pdf").is_file()
    assert (root / "figures/general/3_model_and_feature_qualification/_figures.json").is_file()
    archive = tmp_path / "archive"
    assert (archive / "layout_20261118/figures/general/01_data_exploration/old.pdf").is_file()
    assert (root / "figures/B/exp04_decision_features/a.pdf").is_file()
    reports = archive / "folder_reports_20261119"
    assert (reports / "exp09_adaptation/All Results.md").read_bytes() == b"# rq3"
    assert (reports / "figures/exp09_adaptation/Captions.md").read_bytes() == b"# captions"
    assert (reports / "figures/exp03_geometry/Captions.md").read_bytes() == b"# geometry"
    assert (reports / "4. experiment_1/All Results.md").read_bytes() == b"# r"
    assert (root / "All_Results.md").read_bytes() == b"# all"
    assert (archive / "layout_20261119/figures/exp09_adaptation/01_calibration/c.pdf").is_file()
    assert (archive / "rq_layout_20261116/x.txt").read_bytes() == b"archived"
    for legacy in ("3. experiment_0", "4. experiment_1", "rq1", "rq2", "figures/rq2",
                   "figures/exp09_adaptation", "figures/exp03_geometry", "exp09_adaptation",
                   "archives", "All Results.md", "exp11_data_efficiency", "exp09_representations",
                   "exp10_adaptation", "synthesis", "figures/exp11_data_efficiency", "figures/00_data",
                   "figures/03_exp0b", "figures/general/01_data_exploration"):
        assert not (root / legacy).exists(), legacy
    assert not list(root.glob("*/All_Results.md")) and not list(root.glob("*/CAPTIONS.md"))
    saved = json.loads(next((root / "manifests").glob("layout_migration_*.json")).read_text())
    assert saved["files"]["rq1/results/outputs/comparison.zip"]["destination"] == (
        "A/exp01_outputs/results/outputs/comparison.zip")
    assert saved["files"]["exp11_data_efficiency/results/group_b/b4.zip"]["destination"] == (
        "B/exp09_data_efficiency/results/group_b/b4.zip")
    # Idempotent: a second preview finds nothing left to move.
    assert layout_migration.plan(root)["moves"] == []


def test_migration_refuses_existing_destinations_links_and_sidecars(tmp_path):
    root, _ = _legacy_tree(tmp_path)
    _write(root, "A/exp01_outputs/results/outputs/comparison.zip", b"other")
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
    flat = root / "exp11_data_efficiency/results/group_b/b4.zip"
    layout_migration.migrate(root, writers_stopped=True)
    moved = structure.recorded_output_path(old)
    assert moved == root / "01_exp0a/manifests/before_extraction/accepted.json"
    relative_across_route = (root / "03_exp0b/manifests/after_extraction"
                             / "../before_extraction/accepted.json")
    assert structure.recorded_output_path(relative_across_route).read_bytes() == b"accept"
    assert structure.recorded_output_path(tmp_path / "elsewhere.json") == tmp_path / "elsewhere.json"
    # Archived records resolve into archive/ beside the output root.
    archived = root / "archives/rq_layout_20261116/x.txt"
    assert structure.recorded_output_path(archived) == tmp_path / "archive/rq_layout_20261116/x.txt"
    report = root / "4. experiment_1/All Results.md"
    assert structure.recorded_output_path(report).read_bytes() == b"# r"
    # Flat paths recorded 18-23-11-2026 (e.g. VSC manifests) resolve into the letter groups.
    assert structure.recorded_output_path(flat) == root / "B/exp09_data_efficiency/results/group_b/b4.zip"
    assert structure.recorded_output_path(root / "synthesis/results/notebook/_report.txt").read_bytes() == (
        b"synthesis")
    assert structure.recorded_output_path(root / "figures/00_data/01_lengths.pdf").is_file()


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
                 "scripts/slurm/b.slurm": "h3", "config/general/3. experiment_0/scheduling.yaml": "h4",
                 # The flat layout of 18-23-11-2026 -> letter groups (23-11-2026).
                 "config/exp11_data_efficiency/data_efficiency.yaml": "h5",
                 "config/exp08_when_text_helps/when_text_helps.yaml": "h6",
                 "config/exp10_adaptation/adaptation.yaml": "h7",
                 "notebooks/exp06_tar.ipynb": "n1", "notebooks/exp07_native_text.ipynb": "n2",
                 "scripts/slurm/c.slurm": "h8"}
    current = {"config/B/main.yaml": "h1", "scripts/slurm/a.slurm": "n2",
               "scripts/slurm/b.slurm": "n3", "config/01_exp0a/scheduling.yaml": "changed",
               "config/B/data_efficiency.yaml": "h5", "config/B/when_text_helps.yaml": "h6b",
               "config/C/adaptation.yaml": "changed", "notebooks/B/exp06_tar.ipynb": "n1",
               "notebooks/B/exp07_native_text.ipynb": "edited", "scripts/slurm/c.slurm": "n8"}
    old_text = {"scripts/slurm/a.slurm": '#SBATCH --output="output_JEVPFN/3. experiment_0/logs/%j"\n',
                "scripts/slurm/b.slurm": "python -m science --seed 1\n",
                "scripts/slurm/c.slurm": '#SBATCH --output="output_JEVPFN/exp10_adaptation/logs/%j"\n'
                                         "# config/exp09_representations/representations.yaml\n",
                "config/exp08_when_text_helps/when_text_helps.yaml": "# see config/synthesis/analysis.yaml\n",
                "config/exp10_adaptation/adaptation.yaml": "enabled: false\n"}
    new_text = {"scripts/slurm/a.slurm": '#SBATCH --output="output_JEVPFN/01_exp0a/logs/%j"\n',
                "scripts/slurm/b.slurm": "python -m science --seed 2\n",
                "scripts/slurm/c.slurm": '#SBATCH --output="output_JEVPFN/C/exp11_adaptation/logs/%j"\n'
                                         "# config/C/representations.yaml\n",
                "config/B/when_text_helps.yaml": "# see config/05_synthesis/analysis.yaml\n",
                "config/C/adaptation.yaml": "enabled: true\n"}
    explained = structure.layout_only_changes(old_files, current, old_text.get, new_text.get)
    assert explained == {"config/rq2/4. experiment_1/main.yaml", "config/B/main.yaml",
                         "scripts/slurm/a.slurm", "scripts/slurm/c.slurm",
                         "config/exp11_data_efficiency/data_efficiency.yaml", "config/B/data_efficiency.yaml",
                         "config/exp08_when_text_helps/when_text_helps.yaml", "config/B/when_text_helps.yaml",
                         "notebooks/exp06_tar.ipynb", "notebooks/B/exp06_tar.ipynb"}


def test_literal_rewrite_is_single_pass_and_never_chains():
    text = ("output_JEVPFN/exp10_adaptation/ output_JEVPFN/exp09_representations/ "
            "output_JEVPFN/exp11_data_efficiency/ output_JEVPFN/synthesis/ config/synthesis/ "
            "output_JEVPFN/figures/00_data/ output_JEVPFN/01_exp0a/")
    assert structure.rewrite_legacy_literals(text) == (
        "output_JEVPFN/C/exp11_adaptation/ output_JEVPFN/C/exp10_representations/ "
        "output_JEVPFN/B/exp09_data_efficiency/ output_JEVPFN/05_synthesis/ config/05_synthesis/ "
        "output_JEVPFN/figures/general/1_data_and_questions/ output_JEVPFN/01_exp0a/")
    current = structure.rewrite_legacy_literals(text)
    assert structure.rewrite_legacy_literals(current) == current


def test_slurm_scripts_use_registry_folders_only():
    import re

    for script in (paths.REPO_ROOT / "scripts").rglob("*.slurm"):
        text = script.read_text(encoding="utf-8")
        for folder in re.findall(r"output_JEVPFN/([^\"\s]+?)/logs/", text):
            assert folder in FOLDERS, (script.name, folder)
