"""`src/utils/run_artifacts.py` — the cleaner. The protections are the point of the tests.

A cleaner that is 95% correct is a cleaner that eventually removes `data/raw`, so the
protected paths get more coverage here than the happy path does.
"""

from __future__ import annotations

import pytest

from src.utils import run_artifacts as ra
from src.utils.paths import REPO_ROOT


@pytest.mark.parametrize(
    "relative",
    ["data/raw", "data/raw/nested/file.csv", "checkpoints", "tfm-library",
     "src", "config", "tests", "docs", "notebooks", "scripts"],
)
def test_protected_paths_are_refused(relative: str) -> None:
    """These can never be removed, whatever arguments are passed."""
    assert ra.is_protected(REPO_ROOT / relative)


def test_a_parent_of_something_protected_is_refused() -> None:
    """Deleting `data/` to get at `data/processed` would take `data/raw` with it."""
    assert ra.is_protected(REPO_ROOT / "data")


def test_an_ordinary_output_path_is_not_protected(tmp_path) -> None:
    assert not ra.is_protected(tmp_path / "output" / "logs")


def test_clean_is_a_dry_run_by_default(tmp_path, monkeypatch) -> None:
    """The two mistakes are not equally expensive: a listing you meant as a deletion costs
    one more command, a deletion you meant as a listing costs the run."""
    monkeypatch.setenv("VSC_DATA", str(tmp_path))
    from src.utils.paths import logs_dir

    logs_dir().mkdir(parents=True, exist_ok=True)
    victim = logs_dir() / "run.log"
    victim.write_text("x" * 100, encoding="utf-8")

    report = ra.clean(("logs",))
    assert report["dry_run"] is True
    assert victim.exists(), "the default must not delete anything"

    ra.clean(("logs",), dry_run=False)
    assert not victim.exists()


def test_clean_keeps_the_directory_and_its_gitkeep(tmp_path, monkeypatch) -> None:
    """`rmtree` on `output/logs` removes a directory git expects to exist, and the next
    clone has nowhere to write."""
    monkeypatch.setenv("VSC_DATA", str(tmp_path))
    from src.utils.paths import logs_dir

    logs_dir().mkdir(parents=True, exist_ok=True)
    (logs_dir() / ".gitkeep").write_text("", encoding="utf-8")
    (logs_dir() / "run.log").write_text("x" * 50, encoding="utf-8")

    ra.clean(("logs",), dry_run=False)
    assert logs_dir().is_dir()
    assert (logs_dir() / ".gitkeep").exists()
    assert not (logs_dir() / "run.log").exists()


def test_expensive_categories_are_not_in_the_default_set() -> None:
    """"Clean up the logs" must not quietly delete a week of finished runs."""
    assert set(ra.CHEAP).isdisjoint(ra.EXPENSIVE)
    for category in ra.EXPENSIVE:
        assert category not in ra.CHEAP


def test_an_unknown_category_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown categories"):
        ra.clean(("not_a_category",), dry_run=True)


def test_gitkeep_is_neither_counted_nor_deleted(tmp_path, monkeypatch) -> None:
    """A directory holding only structure markers is already clean."""
    monkeypatch.setenv("VSC_DATA", str(tmp_path))
    from src.utils.paths import logs_dir

    logs_dir().mkdir(parents=True, exist_ok=True)
    (logs_dir() / ".gitkeep").write_text("", encoding="utf-8")
    assert not [a for a in ra.find_artifacts() if a.category == "logs"]


def test_summarise_names_what_it_will_never_touch() -> None:
    """The listing has to say what is out of scope, or the reader assumes nothing is."""
    text = ra.summarise([])
    assert "clean" in text.lower()
    populated = ra.summarise(
        [ra.Artifact(category="logs", path=REPO_ROOT / "output" / "logs", n_files=3, bytes=1234)]
    )
    for guard in ("data/raw", "checkpoints", "tfm-library"):
        assert guard in populated
