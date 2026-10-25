"""B200 publication, cleanup and frozen identity use the correct storage tiers."""

from src.cluster import freeze
from src.utils import output_cleanup, paths


def test_output_reset_includes_registered_scratch_only_and_preserves_features(
    tmp_path, monkeypatch
):
    personal, project, scratch = [tmp_path / p for p in ("data", "lustre", "gpfs")]
    for root in (personal, project, scratch):
        root.mkdir()
    (personal / "JEVPFN").mkdir()
    (project / "JEVPFN").mkdir()
    monkeypatch.setenv("VSC_DATA", str(personal))
    monkeypatch.setenv("JEVPFN_STAGING_ROOT", str(project))
    monkeypatch.setenv("VSC_SCRATCH_GPFS1", str(scratch))
    monkeypatch.setattr(output_cleanup, "require_idle_vsc", lambda: None)
    work = scratch / "jevpfn-work/intern/JEVPFN"
    work.mkdir(parents=True)
    (work / "_JEVPFN_WORKSPACE.json").write_text("{}")
    output = work / "output_JEVPFN/3. experiment_0/results/probe.txt"
    output.parent.mkdir(parents=True)
    output.write_text("generated")
    cache = work / "data/jev_cache/paid.sqlite3"
    cache.parent.mkdir(parents=True)
    cache.write_text("permanent features")
    other = scratch / "another-project/output_JEVPFN/keep.log"
    other.parent.mkdir(parents=True)
    other.write_text("other project")
    assert "GPFS_intern" not in output_cleanup.selected_roots("data")
    assert "GPFS_intern" in output_cleanup.selected_roots("both")
    output_cleanup.run(tier="both", apply=True, writers_stopped=True)
    assert not output.exists()
    assert cache.read_text() == "permanent features"
    assert other.read_text() == "other project"


def test_frozen_b200_identity_uses_durable_pins_not_staged_subset(tmp_path, monkeypatch):
    durable = tmp_path / "lustre"
    native = tmp_path / "gpfs/jevpfn-work/intern"
    monkeypatch.setenv("JEVPFN_DURABLE_ROOT", str(durable))
    monkeypatch.setenv("JEVPFN_STORAGE_MODE", "scratch_mirror")
    monkeypatch.setenv("JEVPFN_STAGING_ROOT", str(native))
    monkeypatch.setattr(paths, "on_vsc", lambda: True)
    pin = durable / "JEVPFN/containers/blackwell-cu130/liquid/_JEVPFN_CONTAINER.json"
    pin.parent.mkdir(parents=True)
    pin.write_text('{"identity":"durable"}')
    staged = native / "JEVPFN/containers/blackwell-cu130/intern/_JEVPFN_CONTAINER.json"
    staged.parent.mkdir(parents=True)
    staged.write_text('{"identity":"relocated"}')
    pins = freeze.runtime_pins()
    assert list(pins) == ["containers/blackwell-cu130/liquid/_JEVPFN_CONTAINER.json"]
    pin.write_text('{"identity":"changed"}')
    assert freeze.runtime_pins() != pins
