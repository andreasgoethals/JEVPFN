"""Full feature scheduling covers exactly the accepted open-weight scope."""

import pytest

from src.cluster.feature_campaign import plan, run
from src.cluster.job_graph import validate
from src.jev.catalog import SELF_HOSTED
from src.utils.config import load_yaml


def test_complete_open_weight_grid_and_workspace_lifetimes():
    steps = plan()
    settings = load_yaml("experiment_0/scheduling")
    settings["test_pool_limits"].update(dict.fromkeys(SELF_HOSTED, 2))
    validate(steps, settings)
    for provider in SELF_HOSTED:
        jobs = [s for s in steps if s.get("pool") == provider]
        assert len(jobs) == 18
        assert len({s["command"][-1] for s in jobs}) == 18
        assert all("--partition=gpu_b200" in s["command"] for s in jobs)
        release = next(s for s in steps if s["name"] == provider + "_release")
        assert {s["name"] for s in jobs} <= set(release["terminal_dependencies"])
    assert sum(s["name"].startswith("embeddings_") for s in steps) == 3
    assert not any(s["name"].startswith("jev_") for s in steps)


def test_campaign_refuses_local_execution_before_any_submission():
    with pytest.raises(RuntimeError, match="VSC"):
        run(execute=True)


def test_dataset_partitioning_cannot_shrink_the_acceptance_pilot():
    from src.jev.self_hosted import run

    with pytest.raises(ValueError, match="pilots cover all datasets"):
        run("liquid", dataset_id="MUL_TEXT_WINE_REVIEW")
