"""`src/utils/paths.py` — the resolver. Tested because every other module trusts it.

The interesting cases are the ones that only happen on the cluster, so they are forced with
environment variables rather than left untested until a job hits them.
"""

from __future__ import annotations

from pathlib import Path

from src.utils import paths


def test_repo_root_is_the_repository() -> None:
    """Resolved from `__file__`, so it is the same from any working directory."""
    assert (paths.REPO_ROOT / "pyproject.toml").is_file()
    assert (paths.REPO_ROOT / "src" / "utils" / "paths.py").is_file()


def test_off_cluster_everything_collapses_into_the_repo(monkeypatch) -> None:
    """No cluster, no overrides: one tree, inside the repository."""
    for var in ("VSC_DATA", "VSC_SCRATCH", "VSC_PROJECT_LUSTRE1", *paths.STAGING_ENV_VARS):
        monkeypatch.delenv(var, raising=False)
    assert paths.on_vsc() is False
    assert paths.outputs_dir() == paths.REPO_ROOT / "output"
    # results/ is a plain subdirectory locally; the tier split is invisible off-cluster.
    assert paths.results_dir() == paths.REPO_ROOT / "output" / "results"
    assert paths.checkpoints_dir() == paths.REPO_ROOT / "checkpoints"


def test_on_cluster_output_splits_across_two_tiers(tmp_path, monkeypatch) -> None:
    """The one rule that matters on VSC: `output/results/` is on project storage, the rest
    of `output/` is on the backed-up 75 GiB tier."""
    monkeypatch.setenv("VSC_DATA", str(tmp_path / "data"))
    monkeypatch.setenv(paths.STAGING_ENV_VARS[0], str(tmp_path / "staging"))

    assert paths.on_vsc() is True
    assert paths.outputs_dir() == tmp_path / "data" / paths.PROJECT_NAME / "output"
    assert paths.results_dir() == tmp_path / "staging" / paths.PROJECT_NAME / "output" / "results"
    # Everything else under output/ stays on $VSC_DATA — few inodes on staging.
    assert paths.logs_dir().is_relative_to(tmp_path / "data")
    assert paths.figures_dir().is_relative_to(tmp_path / "data")
    assert paths.manifests_dir().is_relative_to(tmp_path / "data")


def test_project_name_is_inserted_only_on_shared_tiers(tmp_path, monkeypatch) -> None:
    """A shared tier needs a `<Project>/` component; the repo root already IS the project. Without
    this the cluster path would be `<Project>/<Project>/output`."""
    monkeypatch.setenv("VSC_DATA", str(tmp_path))
    assert paths.outputs_dir() == tmp_path / paths.PROJECT_NAME / "output"

    monkeypatch.delenv("VSC_DATA")
    monkeypatch.delenv(paths.STAGING_ENV_VARS[0], raising=False)
    assert paths.outputs_dir() == paths.REPO_ROOT / "output"


def test_staging_override_works_off_cluster(tmp_path, monkeypatch) -> None:
    """The override has to work on a laptop — that is where it is used, to keep big files
    off the repository disk, and it is how these tests reach the staging branch at all."""
    monkeypatch.delenv("VSC_DATA", raising=False)
    monkeypatch.setenv(paths.STAGING_ENV_VARS[0], str(tmp_path / "external"))
    assert paths.on_vsc() is False
    assert paths.checkpoints_dir().is_relative_to(tmp_path / "external")


def test_figures_dir_is_per_notebook() -> None:
    root = paths.figures_dir()
    assert paths.figures_dir("some_notebook") == root / "some_notebook"
    assert paths.captions_path() == root / "CAPTIONS.md"
    assert paths.all_results_path() == paths.outputs_dir() / "All_Results.md"


def test_config_path_appends_the_suffix_once() -> None:
    assert paths.config_path("example").name == "example.yaml"
    assert paths.config_path("example.yaml").name == "example.yaml"


def test_data_search_paths_look_in_the_repo_first(tmp_path, monkeypatch) -> None:
    """Repo first so a laptop with the data checked out needs no configuration; staging
    second so the same code finds it on the cluster."""
    monkeypatch.setenv(paths.STAGING_ENV_VARS[0], str(tmp_path / "staging"))
    roots = paths.data_search_paths("raw")
    assert roots[0] == paths.REPO_ROOT / "data" / "raw"
    assert len(roots) == 2


def test_resolve_writable_probes_with_a_real_write(tmp_path) -> None:
    """`mkdir(exist_ok=True)` is not enough: a directory can exist and be unwritable."""
    good = tmp_path / "writable"
    assert paths.resolve_writable(good, fallback=tmp_path / "fallback") == good
    assert not (good / ".write_probe").exists()  # the probe cleans up after itself


def test_resolve_writable_falls_back_rather_than_raising(tmp_path, capsys) -> None:
    """A completed run in the wrong place beats a job that died at hour six."""
    blocked = tmp_path / "file_not_a_dir"
    blocked.write_text("in the way", encoding="utf-8")
    fallback = tmp_path / "fallback"
    assert paths.resolve_writable(blocked, fallback=fallback) == fallback
    assert "WARNING" in capsys.readouterr().out  # silent fallback would be worse than none


def test_ensure_creates_the_parent_for_a_file_path(tmp_path) -> None:
    target = paths.ensure(tmp_path / "a" / "b" / "thing.csv")
    assert target.parent.is_dir() and not target.exists()
    assert paths.ensure(tmp_path / "c" / "d").is_dir()


def test_describe_reports_every_root() -> None:
    """Logged at job start, so a run records where it actually wrote."""
    info = paths.describe()
    for key in ("project", "on_vsc", "repo_root", "staging_root", "outputs_dir", "results_dir"):
        assert key in info and info[key]


def test_touch_tree_refreshes_access_times(tmp_path) -> None:
    """Scratch is purged on ACCESS time, and `mv`/`rsync -a` do not count as an access."""
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    f = nested / "x.txt"
    f.write_text("x", encoding="utf-8")
    import os

    old = 10_000_000
    os.utime(f, (old, old))
    paths.touch_tree(tmp_path)
    assert Path(f).stat().st_atime > old
