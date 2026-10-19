import os

import pytest

from src.utils import output_cleanup as cleanup
from src.utils import paths


@pytest.fixture
def output(isolated_output, monkeypatch):
    for root in (paths.outputs_dir(), paths.large_outputs_dir()):
        root.mkdir(parents=True)
        for name in ("3. experiment_0/logs/run.log", "3. experiment_0/results/data.csv",
                     "4. experiment_1/logs/run.log", "transfers/bundle.zip"):
            file = root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("test")
        protected = root.parent / "data/jev_cache/paid.sqlite3"
        protected.parent.mkdir(parents=True)
        protected.write_text("paid")
    monkeypatch.setattr(cleanup, "require_idle_vsc", lambda: None)
    return isolated_output


def test_preview_and_phase_category_tier_selection(output):
    selection = dict(tier="project", phases=["experiment_0"], kinds=["logs"])
    preview = cleanup.run(**selection)
    assert preview["status"] == "preview" and preview["file_count"] == 1
    file = paths.large_outputs_dir() / "3. experiment_0/logs/run.log"
    assert file.exists()
    cleanup.run(**selection, apply=True, writers_stopped=True)
    assert not file.exists()
    assert (paths.outputs_dir() / "3. experiment_0/logs/run.log").exists()
    assert (paths.large_outputs_dir() / "4. experiment_1/logs/run.log").exists()
    assert paths.dataset_cache_path("paid").read_text() == "paid"


def test_age_pattern_and_no_implicit_deletion(output):
    old = paths.outputs_dir() / "3. experiment_0/logs/run.log"
    os.utime(old, (1, 1))
    _, _, files = cleanup.plan(tier="data", pattern="*/logs/*.log", older_than_days=1)
    assert [f["path"] for f in files] == [str(old)]
    with pytest.raises(ValueError, match="writers"):
        cleanup.run(apply=True)
    with pytest.raises(ValueError, match="Unknown research phase"):
        cleanup.plan(phases=["../../data"])


def test_all_tiers_validated_before_deleting_any(output, monkeypatch):
    original = cleanup.validate_deletion_tree
    def reject_project(root):
        if root == paths.large_outputs_dir():
            raise ValueError("synthetic linked subtree")
        return original(root)
    monkeypatch.setattr(cleanup, "validate_deletion_tree", reject_project)
    with pytest.raises(ValueError, match="linked"):
        cleanup.run(apply=True, writers_stopped=True)
    assert (paths.outputs_dir() / "3. experiment_0/logs/run.log").exists()


def test_full_cleanup_removes_archives_preserves_inputs(output):
    for root in (paths.outputs_dir(), paths.large_outputs_dir()):
        archive = root.parent / "archived_output_JEVPFN/old/run.log"
        archive.parent.mkdir(parents=True)
        archive.write_text("old")
    cleanup.run(include_archives=True, apply=True, writers_stopped=True)
    assert not list(output.rglob("run.log"))
    assert len(list(output.rglob("paid.sqlite3"))) == 2


def test_partial_figure_cleanup_invalidates_combined_reports(output):
    root = paths.outputs_dir()
    for name in ("figures/0. exploration/nb/01_plot.pdf", "figures/Captions.md", "All Results.md"):
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("old")
    cleanup.run(tier="data", phases=["exploration"], kinds=["figures"], apply=True, writers_stopped=True)
    assert not (root / "All Results.md").exists()
    assert not (root / "figures/Captions.md").exists()


def test_partial_pdf_cleanup_preserves_only_surviving_captions(output):
    import json
    root = paths.outputs_dir()
    figure_dir = root / "figures/0. exploration/nb"
    figure_dir.mkdir(parents=True)
    for stem in ("01_old", "02_keep"):
        (figure_dir / f"{stem}.pdf").write_text("pdf")
    manifest = root / "manifests/notebooks/0. exploration/nb/_figures.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps([{"stem": "01_old"}, {"stem": "02_keep"}]))
    preview = cleanup.run(tier="data", pattern="*/01_old.pdf")
    assert preview["caption_manifests_updated"] == [str(manifest)]
    cleanup.run(tier="data", pattern="*/01_old.pdf", apply=True, writers_stopped=True)
    assert json.loads(manifest.read_text()) == [{"stem": "02_keep"}]
    assert (figure_dir / "02_keep.pdf").exists()
