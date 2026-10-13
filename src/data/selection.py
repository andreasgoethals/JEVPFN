"""Explicit cost/feasibility exclusions; exploration always retains the full catalog."""


def selected_entries(catalog: dict, cfg: dict) -> list[dict]:
    excluded = set(cfg.get("dataset_selection", {}).get("exclude", []))
    known = {entry["id"] for entry in catalog["datasets"]}
    if excluded - known:
        raise ValueError(f"Unknown excluded datasets: {sorted(excluded - known)}")
    return [entry for entry in catalog["datasets"] if entry["id"] not in excluded]
