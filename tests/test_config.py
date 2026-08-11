"""`src/utils/config.py` — the sweep expansion, which decides how many runs happen."""

from __future__ import annotations

import pytest
import yaml

from src.utils import config


def write(tmp_path, data: dict) -> str:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return str(path)


def test_the_shipped_example_config_loads_and_expands() -> None:
    """The template's own example must be valid — it is what every project copies."""
    cfg = config.load("example")
    assert config.n_points(cfg) == 2 * 2 * 5
    assert len(config.expand(cfg)) == config.n_points(cfg)


def test_expand_is_the_full_cartesian_product(tmp_path) -> None:
    cfg = config.load(write(tmp_path, {"name": "x", "sweep": {"a": [1, 2, 3], "b": ["p", "q"]}}))
    points = config.expand(cfg)
    assert len(points) == 6
    assert {(p["a"], p["b"]) for p in points} == {(a, b) for a in (1, 2, 3) for b in ("p", "q")}


def test_expand_flattens_swept_keys_to_the_top_level(tmp_path) -> None:
    """So downstream code never has to know whether a knob came from the sweep."""
    cfg = config.load(write(tmp_path, {"name": "x", "sweep": {"model": ["a"]}, "seed": 7}))
    (point,) = config.expand(cfg)
    assert point["model"] == "a" and point["seed"] == 7
    assert "sweep" not in point


def test_expand_order_is_deterministic(tmp_path) -> None:
    """The order names the runs, so a resumed sweep must not re-run points it finished."""
    path = write(tmp_path, {"name": "x", "sweep": {"a": [1, 2], "b": [3, 4]}})
    first = [p["run_id"] for p in config.expand(config.load(path))]
    second = [p["run_id"] for p in config.expand(config.load(path))]
    assert first == second


def test_a_config_with_no_sweep_is_one_run(tmp_path) -> None:
    (point,) = config.expand(config.load(write(tmp_path, {"name": "solo", "lr": 0.1})))
    assert point["run_id"] == "solo" and point["lr"] == 0.1


def test_a_single_value_in_the_sweep_block_is_rejected(tmp_path) -> None:
    """A knob with one value belongs below the sweep, not in it — otherwise the block stops
    being a reliable answer to "what is being varied here?"."""
    with pytest.raises(ValueError, match="non-empty list"):
        config.sweep_axes(config.load(write(tmp_path, {"sweep": {"a": 1}})))


def test_a_key_in_both_places_is_rejected(tmp_path) -> None:
    """One of the two would silently win, and which one is an implementation detail."""
    cfg = config.load(write(tmp_path, {"name": "x", "sweep": {"lr": [0.1, 0.2]}, "lr": 0.5}))
    with pytest.raises(ValueError, match="both"):
        config.expand(cfg)


def test_dotted_get_returns_the_default_for_a_missing_key(tmp_path) -> None:
    cfg = config.load(write(tmp_path, {"train": {"lr": 0.01}}))
    assert config.get(cfg, "train.lr") == 0.01
    assert config.get(cfg, "train.momentum", 0.9) == 0.9
    assert config.get(cfg, "nope.nope.nope") is None


def test_resolved_dump_records_what_the_run_used(tmp_path) -> None:
    """The YAML on disk may have been edited since, and a sweep point is not in it at all."""
    cfg = {"name": "x", "model": "a", "fold": 3}
    written = config.resolved_dump(cfg, tmp_path / "runs" / "x" / "config.yaml")
    assert yaml.safe_load(written.read_text(encoding="utf-8")) == cfg


def test_a_missing_config_says_where_it_looked() -> None:
    with pytest.raises(FileNotFoundError, match="no config at"):
        config.load("definitely_not_a_config")
