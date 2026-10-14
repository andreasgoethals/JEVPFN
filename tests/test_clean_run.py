"""`src/utils/clean_run.py` — the wipe."""

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
    assert not victim.exists()


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
    assert removed == 2  # the pdf and the log, not the markers
    assert (logs_dir("experiment_1") / ".gitkeep").is_file()
    assert (figures_dir() / ".gitkeep").is_file()
    assert not per_notebook.exists()  # per-run, no marker, so it goes


def test_gitkeep_is_never_counted(isolated_output) -> None:
    """A directory holding only structure markers is already clean."""
    from src.utils.paths import logs_dir

    logs_dir("experiment_1").mkdir(parents=True, exist_ok=True)
    (logs_dir("experiment_1") / ".gitkeep").write_text("", encoding="utf-8")
    assert clean_run.measure(clean_run.roots()[0]) == (0, 0)


def test_both_storage_tiers_are_cleared_on_the_cluster(isolated_output) -> None:
    """`output/results/` lives on project storage there, so clearing only `$VSC_DATA` would leave
    the largest files behind."""
    roots = clean_run.roots()
    assert len(roots) == 3  # DATA, PROJECT, and independent local notebook output.
    assert any("staging" in str(r) for r in roots)
    assert any("local" in str(r) for r in roots)


def test_processed_is_opt_in(isolated_output) -> None:
    """Rebuilding the cache can cost far more than re-running the notebooks, so "clean the last
    run" must not silently throw it away."""
    from src.utils.paths import processed_dir

    assert processed_dir() not in clean_run.roots()
    assert processed_dir() in clean_run.roots(processed=True)


def test_processed_cleanup_never_includes_system_temp(isolated_output):
    import tempfile
    from pathlib import Path

    from src.utils.paths import notebook_cache_dir

    roots = clean_run.roots(processed=True)
    assert notebook_cache_dir().parent not in roots
    assert all(root.resolve() != Path(tempfile.gettempdir()).resolve() for root in roots)
