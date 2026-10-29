"""Keep a bounded number of native working sets until their last consumer finishes."""

from src.cluster.job_graph import dependencies


def add_releases(steps, *, command_style=False):
    stages = [s for s in steps if s["name"].endswith("_scratch")]
    releases = []
    for stage in stages:
        runtime = stage["name"].removesuffix("_scratch")
        stage["workspace_stage"] = runtime
        consumers = {stage["name"]}
        while True:
            found = {s["name"] for s in steps if set(dependencies(s)) & consumers}
            if found <= consumers:
                break
            consumers.update(found)
        release = dict(
            name=runtime + "_release",
            cluster="wice",
            pool="transfer",
            workspace_release=runtime,
            terminal_dependencies=sorted(consumers),
        )
        if command_style:
            release.update(
                depends_on=[],
                command=[
                    "sbatch",
                    "--parsable",
                    "scripts/slurm/experiment_0_scratch_release.slurm",
                    runtime,
                    "--apply",
                ],
            )
        else:
            release.update(
                dependencies=[],
                script="experiment_0_scratch_release.slurm",
                args=[runtime, "--apply"],
                flags=[],
            )
        releases.append(release)
    return [*steps, *releases]
