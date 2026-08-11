"""Read a `config/*.yaml`, expand its `sweep:` block, and record what was actually run.

THE SHAPE, fixed by the template:

    sweep:                  every knob with more than one value. Cartesian product.
      model: [a, b]
      fold: [0, 1, 2]
    <everything else>       single values, shared by every point in the sweep

`expand()` turns that into one flat dict per sweep point, with the swept keys folded in at
the top level so downstream code never has to know whether a value came from the sweep or
from the shared part. WHY flatten: the alternative is every consumer checking two places
for the same knob, and forgetting to in one of them.

NO INHERITANCE, NO INCLUDES, ON PURPOSE. A config file is read top to bottom and that is
the whole story. Layered configs make "what did this run actually use?" a question you
answer by simulating a merge — which is why `resolved_dump()` exists: the run writes the
fully expanded dict it used into its own output directory, so the answer is a file.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import config_path

SWEEP_KEY = "sweep"


def load(name: str | Path) -> dict[str, Any]:
    """Load `config/<name>.yaml` by name, or any path directly."""
    path = Path(name) if Path(name).is_file() else config_path(str(name))
    if not path.is_file():
        raise FileNotFoundError(f"no config at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level, got {type(data).__name__}")
    data.setdefault("name", path.stem)
    return data


def sweep_axes(config: dict[str, Any]) -> dict[str, list[Any]]:
    """The `sweep:` block, validated. `{}` when there is none."""
    axes = config.get(SWEEP_KEY) or {}
    if not isinstance(axes, dict):
        raise ValueError(f"`{SWEEP_KEY}:` must be a mapping of knob -> list of values")
    for key, values in axes.items():
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"`{SWEEP_KEY}.{key}` must be a non-empty list. A knob with one value "
                f"belongs below the sweep block, not in it."
            )
    return axes


def n_points(config: dict[str, Any]) -> int:
    """How many runs this config describes.

    Worth printing before submitting anything: a sweep grows multiplicatively and reads
    additively, so 4 x 3 x 5 looks like twelve lines of YAML and is sixty runs.
    """
    total = 1
    for values in sweep_axes(config).values():
        total *= len(values)
    return total


def expand(config: dict[str, Any]) -> list[dict[str, Any]]:
    """One flat config per sweep point, in a deterministic order.

    Deterministic because the order names the runs: `<name>__model=a__fold=0`. If the order
    changed between invocations, a resumed sweep would re-run points it had already done
    and skip others.
    """
    axes = sweep_axes(config)
    shared = {k: v for k, v in config.items() if k != SWEEP_KEY}
    if not axes:
        return [dict(shared, run_id=str(shared.get("name", "run")))]

    keys = list(axes)  # dict order = file order, which is the author's intended order
    points = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        assignment = dict(zip(keys, combo))
        collisions = set(assignment) & set(shared)
        if collisions:
            raise ValueError(
                f"{sorted(collisions)} appear both in `{SWEEP_KEY}:` and below it. One of "
                f"the two silently wins; remove the duplicate."
            )
        tag = "__".join(f"{k}={v}" for k, v in assignment.items())
        points.append(dict(shared, **assignment, run_id=f"{shared.get('name', 'run')}__{tag}"))
    return points


def get(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """`get(cfg, "train.learning_rate")`. Returns `default` for a missing key.

    Dotted access so a caller reads a nested knob without four `.get()` calls, each of
    which is a place to silently return None.
    """
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def resolved_dump(config: dict[str, Any], destination: Path) -> Path:
    """Write the fully expanded config a run actually used, next to that run's output.

    This is the record that makes a result reproducible: the YAML on disk may have been
    edited since, and a sweep point is not in the YAML at all.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config, sort_keys=False, default_flow_style=False), encoding="utf-8"
    )
    return destination
