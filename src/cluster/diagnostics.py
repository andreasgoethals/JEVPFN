"""Collect independent bounded diagnostics; never recover a poisoned CUDA context."""

import time


def check(report, name, function, *, save):
    started = time.perf_counter()
    item = {"name": name, "status": "running"}
    report.setdefault("checks", []).append(item)
    try:
        value = function()
        if isinstance(value, dict) and value.get("passed") is False:
            raise ValueError("Known-answer diagnostics require review; inspect saved raw answers")
        item["status"] = "passed"
        return value
    except Exception as exc:
        item.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        # Native device failures may invalidate all later CUDA operations. Aborting
        # this worker does not stop the other providers in the Slurm graph.
        fatal = ("device-side assert", "illegal memory access", "cuda error", "out of memory",
                 "request ceiling", "request budget")
        if any(term in str(exc).lower() for term in fatal) or isinstance(exc, MemoryError):
            item["runtime_unusable"] = True
            raise
        return None
    finally:
        item["seconds"] = time.perf_counter() - started
        save()
        print(f"{name}: {item['status']}", flush=True)
