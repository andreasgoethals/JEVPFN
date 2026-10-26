"""Reconcile a bounded Slurm DAG, collecting failures without abandoning other branches.

The caller owns the receipt lock and supplies persistence/submission/accounting.
Dependencies are success requirements, not concurrency limits. A lost submission
receipt is deliberately fatal: continuing could duplicate an already accepted job.
"""

from collections import Counter

TERMINAL = {"completed", "failed", "blocked"}


def dependencies(step):
    return step.get("dependencies", step.get("depends_on", []))


def pool(step):
    name = step["name"]
    if name.endswith("_contracts") or name.endswith("_pilot") or name.startswith("embeddings_"):
        return "gpu"
    if name.endswith("_build"):
        return "build"
    if name.endswith("_scratch"):
        return "transfer"
    if name.endswith("_stage"):
        return "download"
    return "cpu"


def validate(steps, settings):
    names = [s["name"] for s in steps]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate test names")
    limits = settings["test_pool_limits"]
    for value in [settings["max_active_jobs"], *limits.values()]:
        if type(value) is not int or value < 1:
            raise ValueError("Test concurrency limits must be positive integers")
    remaining = {s["name"]: set(dependencies(s)) for s in steps}
    if any(deps - set(names) for deps in remaining.values()):
        raise ValueError("Unknown test dependency")
    if {pool(s) for s in steps} - set(limits):
        raise ValueError("Missing test pool limit")
    done = set()
    while remaining:
        ready = {name for name, deps in remaining.items() if deps <= done}
        if not ready:
            raise ValueError("Cyclic test dependencies")
        done.update(ready)
        remaining = {name: deps for name, deps in remaining.items() if name not in ready}


def summary(steps):
    counts = Counter(s.get("status", "pending") for s in steps)
    return {
        "total": len(steps), "counts": dict(counts),
        "failures": [{k: s.get(k) for k in ("name", "cluster", "job_id", "scheduler_state", "exit_code", "command")}
                     for s in steps if s.get("status") == "failed"],
        "blocked": [{"name": s["name"], "blocked_by": s["blocked_by"]}
                    for s in steps if s.get("status") == "blocked"],
        "all_terminal": all(s.get("status") in TERMINAL for s in steps),
    }


def advance(steps, *, settings, state, submit, save):
    """One resumable pass. Unknown accounting remains active and never becomes a pass."""
    validate(steps, settings)
    for step in steps:
        if step.get("status") in TERMINAL:
            continue
        if step.get("job_id"):
            result = state(step["job_id"], cluster=step["cluster"])
            if result is not None:
                step.update(scheduler_state=result[0], exit_code=result[1],
                            status="completed" if result == ("COMPLETED", "0:0") else "failed")
                print(f"{step['name']}: {step['status']} ({step['job_id']}, {result[0]}, {result[1]})", flush=True)
                save()
        elif step.get("status") == "starting":
            raise ValueError("Interrupted submission has no job ID; inspect Slurm before retrying")
    # Propagate a failure to descendants only, including graphs not in topological order.
    while True:
        failed = {s["name"] for s in steps if s.get("status") in {"failed", "blocked"}}
        changed = False
        for step in steps:
            if step.get("job_id") or step.get("status") in TERMINAL:
                continue
            blockers = sorted(set(dependencies(step)) & failed)
            if blockers:
                step.update(status="blocked", blocked_by=blockers)
                changed = True
        if not changed:
            break
    save()
    completed = {s["name"] for s in steps if s.get("status") == "completed"}
    active = Counter(pool(s) for s in steps if s.get("job_id") and s.get("status") not in TERMINAL)
    # CPU token jobs cannot fill the GPU/build/transfer pools. Counts include queued jobs.
    for step in steps:
        if step.get("job_id") or step.get("status") in TERMINAL:
            continue
        kind = pool(step)
        if (sum(active.values()) >= settings["max_active_jobs"]
                or active[kind] >= settings["test_pool_limits"][kind]
                or not set(dependencies(step)) <= completed):
            continue
        submit(step)  # Persists 'starting' BEFORE calling sbatch; never retry ambiguous receipts.
        active[kind] += 1
    return summary(steps)
