"""Reviewed decision-model cohorts and frozen-embedding grid."""

import os
import re

from src.utils.config import load_yaml
from src.utils.serialization import digest


def decision_scopes(main=None):
    """Resolve immutable first-edition membership and declared, successive additions."""
    cfg = main if main is not None else load_yaml("experiment_1/main")
    members = tuple(cfg["candidate_extractors"])
    if len(set(members)) != len(members):
        raise ValueError("Duplicate decision model in the configured roster")
    candidates = load_yaml("extractors")["candidates"] if members else {}
    open_models, hosted_models = [], []
    for provider in members:
        delivery = candidates.get(provider, {}).get("delivery")
        if delivery == "self_hosted":
            open_models.append(provider)
        elif delivery == "hosted" or isinstance(delivery, str) and delivery.startswith("hosted_"):
            hosted_models.append(provider)
        else:
            raise ValueError(f"{provider} needs a registered open-weight or hosted delivery method")
    initial = tuple(cfg.get("initial_decision_providers", open_models))
    if len(initial) != len(set(initial)) or set(initial) - set(open_models):
        raise ValueError("The initial decision providers must be distinct registered open-weight models")
    scopes = {"open_weight": initial, "all": members, "paid": tuple(hosted_models)}
    for name, edition in cfg.get("decision_extensions", {}).items():
        if (not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
                or name in scopes or name.endswith("_added")):
            raise ValueError(f"Invalid decision extension name {name!r}")
        parent = edition["parent"]
        if parent not in scopes or parent in {"all", "paid"} or parent.endswith("_added"):
            raise ValueError(f"{name} needs an earlier declared full release as its parent")
        added = edition["add"]
        if (not isinstance(added, list) or not added or len(added) != len(set(added))
                or set(added) - set(members) or set(added) & set(scopes[parent])):
            raise ValueError(f"{name} must add distinct registered models outside its parent release")
        # Keep the parent's presentation order as the registered catalogue grows.
        scopes[name] = (*scopes[parent], *added)
        scopes[f"{name}_added"] = tuple(added)
    return scopes


def incremental_cohort(name):
    return name == "paid" or isinstance(name, str) and name.endswith("_added")


def decision_providers(cohort="all", main=None):
    """Select exactly the registered members, independently of completed fits."""
    scopes = decision_scopes(main)
    if cohort not in scopes:
        raise ValueError(f"Unknown decision-model cohort: {cohort}")
    return scopes[cohort]


def decision_versions(providers):
    """Pinned producer identities for a release, including adapter/base checkpoints."""
    candidates = load_yaml("extractors")["candidates"]
    versions = []
    for provider in providers:
        candidate = candidates.get(provider, {})
        version = candidate.get("revision") or candidate.get("version")
        model = candidate.get("model")
        if (not isinstance(model, str) or not model.strip() or not isinstance(version, str)
                or not version.strip() or
                re.search(r"(?:^|[-:@/])(latest|main|master|auto|default)$", version, re.IGNORECASE)):
            raise ValueError(f"{provider} needs an exact registered model and version")
        if candidate.get("delivery") == "self_hosted":
            if not re.fullmatch(r"[a-f0-9]{40}", version):
                raise ValueError(f"{provider} needs an immutable model checkpoint revision")
            for field in ("base_model_revision", "runtime_revision", "tokenizer_revision"):
                if field in candidate or (field == "base_model_revision" and candidate.get("base_model")):
                    pin = candidate.get(field)
                    if not isinstance(pin, str) or not re.fullmatch(r"[a-f0-9]{40}", pin):
                        raise ValueError(f"{provider} needs an immutable {field}")
        versions.append({"provider": provider, "model": model, "version": version,
                         **{key: candidate[key] for key in ("delivery", "base_model", "base_model_revision",
                                                           "runtime_revision", "tokenizer_revision")
                            if candidate.get(key)}})
    return versions


def cohort_name(cohort=None):
    return cohort or os.environ.get("JEVPFN_DECISION_COHORT") or "open_weight"


def cohort_record(cohort=None, *, main=None):
    """The same ordered, versioned roster travels with scheduler plans and reports."""
    name = cohort_name(cohort)
    members = list(decision_providers(name, main))
    if not members:
        raise ValueError(f"The configured decision-model cohort {name} is empty")
    versions = decision_versions(members)
    return {"decision_cohort": name, "decision_roster": members, "decision_models": len(members),
            "decision_model_versions": versions, "roster_sha256": digest(versions)}


def pca_grid():
    grid = load_yaml("experiment_2/embeddings")["pca_grid"]
    expected = {"per_column": [32], "joint": [64]}
    if grid != expected:
        raise ValueError("PCA grid differs from the reviewed per-column PCA32 and joint PCA64 design")
    return [(scope, k) for scope, dimensions in grid.items() for k in dimensions]
