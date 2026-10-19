"""`src/utils/clean_run.py` â€” the wipe."""

from __future__ import annotations

from src.utils import clean_run


def test_lists_by_default_and_deletes_only_when_asked(isolated_output, capsys) -> None:
    """A listing you meant as a deletion costs one more command; the reverse costs the run."""
    from src.utils.paths import logs_dir

    logs_dir("experiment_1").mkdir(parents=True, exist_ok=True)
    victim = logs_dir("experiment_1") / "run.log"
    victim.write_text("x" * 100, encoding="utf-8")

    clean_run.main([])
    assert victim.exists(), "the default must not delete anything"
    assert "Nothing was deleted" in capsys.readouterr().out

    clean_run.main(["--clean"])
    assert victim.exists(), "operational logs are preserved"


def test_a_wipe_keeps_the_directory_skeleton(isolated_output) -> None:
    """`rmtree` would take `output/figures/.gitkeep` with it, and the next clone would have
    nodirectory markers."""
    from src.utils.paths import figures_dir, logs_dir

    for folder in (logs_dir("experiment_1"), figures_dir()):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / ".gitkeep").write_text("", encoding="utf-8")
    per_notebook = figures_dir("nb")
    per_notebook.mkdir(parents=True, exist_ok=True)
    (per_notebook / "01_x.pdf").write_bytes(b"%PDF")
    (logs_dir("experiment_1") / "run.log").write_text("x", encoding="utf-8")

    removed = sum(clean_run.wipe(root) for root in clean_run.roots())
    assert removed == 1  # the rebuildable figure only
    assert (logs_dir("experiment_1") / ".gitkeep").is_file()
    assert (figures_dir() / ".gitkeep").is_file()
    assert not (per_notebook / "01_x.pdf").exists()


def test_gitkeep_is_never_counted(isolated_output) -> None:
    """A directory holding only structure markers is already clean."""
    from src.utils.paths import logs_dir

    logs_dir("experiment_1").mkdir(parents=True, exist_ok=True)
    (logs_dir("experiment_1") / ".gitkeep").write_text("", encoding="utf-8")
    assert clean_run.measure(clean_run.roots()[0]) == (0, 0)


def test_both_storage_tiers_are_cleared_on_the_cluster(isolated_output) -> None:
    """Inspect both VSC output roots and the independent local notebook root."""
    roots = clean_run.roots()
    assert len(roots) == 3  # DATA, PROJECT, and independent local notebook output.
    assert any("staging" in str(r) for r in roots)
    assert any("local" in str(r) for r in roots)


def test_processed_is_protected(isolated_output) -> None:
    """Rebuilding the cache can cost far more than re-running the notebooks, so "clean the last
    run" must not silently throw it away."""
    from src.utils.paths import processed_dir

    assert processed_dir() not in clean_run.roots()


def test_processed_cleanup_never_includes_system_temp(isolated_output):
    import tempfile
    from pathlib import Path

    from src.utils.paths import notebook_cache_dir

    roots = clean_run.roots()
    assert notebook_cache_dir().parent not in roots
    assert all(root.resolve() != Path(tempfile.gettempdir()).resolve() for root in roots)


def test_linked_output_root_cannot_redirect_cleanup(isolated_output, monkeypatch):
    import pytest

    from src.utils.paths import outputs_dir

    root = outputs_dir()
    monkeypatch.setattr(clean_run, "_linked", lambda p: p == root)
    with pytest.raises(ValueError, match="linked output root"):
        clean_run.main(["--clean"])


def test_rebuildable_comparison_csv_is_removed_but_inputs_are_preserved(isolated_output):
    from src.utils.paths import figures_dir, raw_dir

    figure = figures_dir("comparison") / "pairs.csv"
    source = raw_dir("dataset", "data.csv")
    for file in (figure, source):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("value\n1")
    clean_run.main(["--clean"])
    assert not figure.exists() and source.is_file()


def test_cleanup_removes_notebook_transcripts_but_preserves_personal_notes(isolated_output):
    from src.utils.paths import figures_dir

    folder = figures_dir("nb")
    folder.mkdir(parents=True)
    for name in ("_report.txt", "_stdout.txt", "Captions.md", "notes.md", "notes.txt"):
        (folder / name).write_text("text", encoding="utf-8")
    clean_run.main(["--clean"])
    assert {p.name for p in folder.iterdir()} == {"notes.md", "notes.txt"}


def test_report_only_cleanup_updates_new_figure_manifest(isolated_output):
    import json

    from src.utils import paths
    from src.visualize.figures import manifest_path
    folder = paths.figures_dir("0. exploration/check")
    folder.mkdir(parents=True)
    (folder / "01_plot.pdf").write_text("pdf")
    manifest = manifest_path("0. exploration/check")
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps([{"stem": "01_plot"}]))
    report = paths.reports_dir("0. exploration/check")
    report.parent.mkdir(parents=True)
    report.write_text("old figure description")
    clean_run.main(["--clean"])
    assert json.loads(manifest.read_text()) == []
    assert not report.exists()
