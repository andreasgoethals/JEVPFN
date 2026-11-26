"""Group C (owner-approved 20 November 2026; Strands only): arms, features, comparisons, exports.

* C1, inside the decision model (C/C1, internal phase ``representations``):
  C1a = 3 positions x 2 layers x {per text column, joint} = 12 hidden-state arms (PCA 32 per
  column / 64 joint on outer-training rows); C1b = the ``<answer>``-last vector plus Strands'
  own probabilities, per column and joint (2 arms). Label-free.
* C2, learning from labels (C/C2): head-only, LoRA-only and
  LoRA + head cross-fitted probabilities for the joint question (3 arms); zero-shot is B1's
  Strands joint arm.

All arms use the seven Group B predictors, datasets and folds. Comparisons are fold-paired
differences of the fixed headline metric, per dataset (five-fold mean) and then equal-dataset
means with dataset-bootstrap intervals and win/tie/loss counts (``group_b.summarize``). No frozen
encoder enters C1; C2 adds TAR and the B2 label-trained classifier as context only.

Results: ``python -m src.evaluation.group_c --study C1|C2`` (Mindwell CPU,
``scripts/slurm/group_c_export.slurm``) writes ONE checksummed ZIP per experiment; the owner
copies only that ZIP to the same relative path under the local ``output_JEVPFN/`` and each
notebook reads nothing else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.features import (
    ADAPTED,
    HIDDEN,
    JOINT_PLUS_HIDDEN,
    PER_COLUMN_PLUS_HIDDEN,
)
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import digest, file_sha256

SCHEMA = "jevpfn.group_c.export.v1"
PROVIDER = "strands"
VIEWS = ("per_column", "joint")
STUDIES = {
    "C1": {"phase": "C1", "file": "c1_representations.zip",
           "phases": ("B1", "C1")},
    "C2": {"phase": "C2", "file": "c2_adaptation.zip", "phases": ("B1", "B3", "C2")},
}
TABLES = {
    "C1": ("coverage", "folds", "dataset_gains", "summary", "contrasts", "contrast_summary", "extraction"),
    "C2": ("coverage", "folds", "dataset_gains", "summary", "contrasts", "contrast_summary", "fits"),
}
METRICS = ("roc_auc", "roc_auc_ovr_macro", "r2", "log_loss", "rmse")
PLUS_HIDDEN = {"per_column": PER_COLUMN_PLUS_HIDDEN, "joint": JOINT_PLUS_HIDDEN}
ENCODERS = ("e5_small_v2", "e5_large_v2", "qwen3_embedding_0_6b")


# ------------------------------------------------------------------------------ arms


def arms(selected, models) -> list[dict]:
    """C1a (12), C1b (2) and C2 (3) arms per dataset and predictor."""
    c1 = load_yaml("C1")["arms"]
    from src.evaluation.strands_adaptation import CELLS

    rows = []
    for entry in selected:
        for model in models:
            for view in VIEWS:
                spec = c1["views"][view]
                for position in c1["positions"]:
                    for layer in c1["layers"]:
                        rows.append(dict(dataset_id=entry["id"], model=model, feature_set=HIDDEN,
                                         provider=None, phase="C1", hidden_view=view,
                                         hidden_position=position, hidden_layer=layer, **spec))
            both = c1["with_probabilities"]
            for view in both["views"]:
                rows.append(dict(dataset_id=entry["id"], model=model, feature_set=PLUS_HIDDEN[view],
                                 provider=PROVIDER, phase="C1", hidden_view=view,
                                 hidden_position=both["position"], hidden_layer=both["layer"],
                                 **c1["views"][view]))
            for cell in CELLS:
                rows.append(dict(dataset_id=entry["id"], model=model, feature_set=ADAPTED,
                                 provider=None, phase="C2", adaptation_cell=cell))
    return rows


def hidden_label(view, position, layer) -> str:
    return f"hidden/{view}/{position}/L{int(layer)}"


# ------------------------------------------------------------------ features (benchmark runner)


def add_hidden(features: pd.DataFrame, vectors: pd.DataFrame) -> pd.DataFrame:
    """Join hidden-state vectors by row ID; the shared text-presence flags stay unchanged."""
    if not features.index.equals(vectors.index) or set(features) & set(vectors):
        raise ValueError("Hidden-state rows or names do not align with the feature matrix")
    result = features.join(vectors, validate="one_to_one")
    result.attrs = dict(features.attrs)
    result.attrs["feature_artifact_ids"] = list(dict.fromkeys(
        features.attrs.get("feature_artifact_ids", []) + vectors.attrs["feature_artifact_ids"]))
    result.attrs["embedding_groups"] = vectors.attrs["embedding_groups"]
    return result


def hidden_control(estimator, groups, arm, seed):
    """Per-group PCA fitted only on the rows passed to ``fit`` (outer training)."""
    from src.evaluation.embedding_control import EmbeddingControl

    return EmbeddingControl(estimator, groups, components=arm["pca_components"], seed=seed,
                            scope="per_column", encoded_missing=True)


def decision_model_for_fit(sources):
    """Check the producer recorded by the hidden store or by every fold's adaptation bundle."""
    from src.evaluation.design import decision_versions

    registered, = decision_versions([PROVIDER])
    if not sources or any(source != registered for source in sources):
        raise ValueError("Group C features lack the registered Strands model and revision")
    return registered


def expected_artifacts(arm, dataset, specification=None) -> list[str]:
    """Group C artifact IDs: the verified hidden-state store or the five fold bundles."""
    if arm["feature_set"] == ADAPTED:
        from src.evaluation.strands_adaptation import read_bundle

        return [read_bundle(dataset, specification, i, arm["adaptation_cell"])["artifact_id"]
                for i in range(5)]
    from src.jev.strands_hidden import active_artifact_id

    return [active_artifact_id(dataset.metadata.dataset_id)]


# ------------------------------------------------------------------------- comparisons


def c1_contrasts() -> list[dict]:
    """(name, A, B): positive values favour A. Exactly the approved C1 comparisons."""
    c1 = load_yaml("C1")["arms"]
    out = []
    for view in VIEWS:
        answer = hidden_label(view, "answer", 24)
        probabilities = f"decision/{view}/{PROVIDER}"
        out.append(dict(comparison="answer_last_vs_strands_probabilities", view=view,
                        a=answer, b=probabilities))
        out.append(dict(comparison="answer_last_plus_probabilities_vs_probabilities", view=view,
                        a=f"hidden+decision/{view}/answer/L24/{PROVIDER}", b=probabilities))
        for layer in c1["layers"]:
            path = c1["positions"]
            for earlier, later in zip(path, path[1:]):
                out.append(dict(comparison=f"{later}_vs_{earlier}", view=view, layer=layer,
                                a=hidden_label(view, later, layer), b=hidden_label(view, earlier, layer)))
        for position in c1["positions"]:
            out.append(dict(comparison="last_vs_middle_layer", view=view, position=position,
                            a=hidden_label(view, position, 24), b=hidden_label(view, position, 12)))
    return out


def c2_contrasts() -> list[dict]:
    zero = f"decision/joint/{PROVIDER}"
    cells = {cell: f"adapted/joint/{cell}" for cell in ("head_only", "lora_only", "lora_head")}
    out = [
        dict(comparison="lora_vs_zero_shot", a=cells["lora_only"], b=zero),
        dict(comparison="head_only_vs_zero_shot", a=cells["head_only"], b=zero),
        dict(comparison="lora_vs_head_only", a=cells["lora_only"], b=cells["head_only"]),
        dict(comparison="lora_head_vs_lora_only", a=cells["lora_head"], b=cells["lora_only"]),
    ]
    context = [f"tar/e5_small_v2_tar/{scope}" for scope in ("per_column32", "joint64")]
    context += [f"probe/{encoder}/joint" for encoder in ENCODERS]
    for name, label in {"zero_shot": zero, **cells}.items():
        for other in context:
            out.append(dict(comparison=f"{name}_vs_{other.split('/')[0]}", context=other, a=label, b=other))
    return out


def contrast_gains(head: pd.DataFrame, contrasts) -> pd.DataFrame:
    """Per dataset x predictor x comparison: fold-paired five-fold mean of A - B (headline
    metric) and its share of B's headroom. Datasets lacking either arm are left out."""
    from src.evaluation import group_b

    rows = []
    keys = group_b.PAIR_KEYS
    group_b.validate_fold_identities(head)
    for spec in contrasts:
        a = head[head.representation.eq(spec["a"])][keys + ["value"]]
        b = head[head.representation.eq(spec["b"])][keys + ["value"]]
        if a.empty or b.empty:
            continue
        if a.duplicated(keys).any() or b.duplicated(keys).any():
            raise ValueError(f"Several fold records of one arm for contrast {spec}")
        paired = a.merge(b, on=keys, suffixes=("_a", "_b"))
        paired["difference"] = paired.value_a - paired.value_b
        identity = paired.groupby(["dataset_id", "model_id"])[["split_id", "checkpoint_id"]].nunique()
        if identity.gt(1).any(axis=None):
            raise ValueError(f"Contrast {spec} pairs folds of several splits or checkpoints")
        grouped = paired.groupby(["dataset_id", "task_type", "model_id"], as_index=False).agg(
            a=("value_a", "mean"), b=("value_b", "mean"), raw_gain=("difference", "mean"),
            folds=("outer_fold", "nunique"),
            finite=("difference", lambda v: int(np.isfinite(v.to_numpy(float)).sum())))
        # Five finite paired differences, or the dataset is left out (undefined fold).
        grouped = grouped[grouped.folds.eq(5) & grouped.finite.eq(5)].drop(columns="finite")
        name = "/".join(str(spec[k]) for k in ("comparison", "view", "layer", "position", "context")
                        if spec.get(k) is not None)
        grouped["representation"] = name
        grouped["headroom_share"] = grouped.raw_gain / (1 - grouped.b).where(grouped.b < 1)
        for k in ("comparison", "view", "layer", "position", "context", "a", "b"):
            grouped[f"contrast_{k}"] = spec.get(k)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ------------------------------------------------------------------------- export


def export_path(study) -> Path:
    spec = STUDIES[study]
    return paths.results_dir("group_c", spec["file"], phase=spec["phase"])


def analysis_identity(study) -> str:
    from src.evaluation.benchmark_export import scientific_configs
    from src.evaluation.design import decision_versions
    from src.utils.provenance import code_identity

    configs = scientific_configs(("B1", "B1/protocol", "C1",
                                  "B3", "C2"), roster=(PROVIDER,))
    modules = ["group_c", "group_b", "benchmark_export", "study_export", "records", "endpoints",
               "strands_adaptation", "campaign", "design"]
    return digest({"schema": SCHEMA, "study": study, "configs": configs,
                   "decision_model_versions": decision_versions([PROVIDER]),
                   "analysis": code_identity([f"src/evaluation/{name}.py" for name in modules]
                                             + ["src/jev/strands_hidden.py"])})


def _keep(head: pd.DataFrame, study: str) -> pd.DataFrame:
    families = {"C1": ("baseline", "hidden", "hidden+decision"),
                "C2": ("baseline", "adapted", "tar", "probe")}[study]
    labels = head.representation
    keep = labels.str.split("/").str[0].isin(families) | labels.eq(f"decision/per_column/{PROVIDER}") \
        | labels.eq(f"decision/joint/{PROVIDER}")
    if study == "C2":
        keep &= ~labels.str.startswith("probe/") | labels.str.endswith("/joint")
    return head[keep]


def extraction_table() -> pd.DataFrame:
    """C1 store manifests on PROJECT: coverage, reproduction contract and size per dataset."""
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries
    from src.jev import strands_hidden

    rows = []
    for entry in selected_entries(load_catalog(), load_yaml("B1")):
        try:
            body = strands_hidden.read_manifest(entry["id"], verify=False)
        except (FileNotFoundError, ValueError) as error:
            rows.append({"dataset_id": entry["id"], "status": type(error).__name__})
            continue
        folder = strands_hidden.store_dir(entry["id"])
        rows.append({"dataset_id": entry["id"], "status": "complete", "artifact_id": body["artifact_id"],
                     "unique_inputs": body["unique_inputs"], "joint_inputs": body["joint_inputs"],
                     "max_probability_deviation": body["max_probability_deviation"],
                     "engine_check_max_difference": body["engine_check"].get("max_abs_difference"),
                     "store_gb": (folder / strands_hidden.STORE).stat().st_size / 1e9})
    return pd.DataFrame(rows)


def fits_table() -> pd.DataFrame:
    """C2 fold bundles on PROJECT: epochs, validation loss, seconds and GPU memory per fit."""
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries
    from src.evaluation.strands_adaptation import CELLS, bundle_dir

    rows = []
    for entry in selected_entries(load_catalog(), load_yaml("B1")):
        for fold in range(5):
            for cell in CELLS:
                target = bundle_dir(entry["id"], fold, cell) / "manifest.json"
                if not target.exists():
                    continue
                body = json.loads(target.read_text(encoding="utf-8"))
                for fit in body["fits"]:
                    rows.append({"dataset_id": entry["id"], "fold": fold, "cell": cell,
                                 "part": fit["part"], "epochs": len(fit["history"]),
                                 "best_validation_loss": fit["best_validation_loss"],
                                 "gpu_hours": fit["seconds"] / 3600, "peak_gpu_gb": fit["peak_gpu_bytes"] / 1e9})
    return pd.DataFrame(rows)


def build_tables(study, folds) -> dict[str, pd.DataFrame]:
    from src.evaluation import benchmark_export, group_b

    head = _keep(group_b.headline(folds), study)
    paired = group_b.fold_gains(head)
    gains = group_b.dataset_gains(paired)
    contrasts = contrast_gains(head, c1_contrasts() if study == "C1" else c2_contrasts())
    group_c_phase = STUDIES[study]["phase"]
    tables = {
        "coverage": benchmark_export.coverage(folds[folds.phase.eq(group_c_phase)], (group_c_phase,), cohort="all"),
        "folds": folds, "dataset_gains": gains,
        "summary": group_b.summarize(gains) if not gains.empty else pd.DataFrame(),
        "contrasts": contrasts,
        "contrast_summary": group_b.summarize(contrasts) if not contrasts.empty else pd.DataFrame(),
    }
    tables["extraction" if study == "C1" else "fits"] = extraction_table() if study == "C1" else fits_table()
    return tables


def read(study, target=None):
    """Verified tables and receipt, or an explicit waiting state (absent/superseded ZIP)."""
    from src.evaluation import study_export

    return study_export.read_export(target or export_path(study), schema=SCHEMA, tables=TABLES[study],
                                    identity=analysis_identity(study))


def build(study):
    from src.evaluation import benchmark_export, study_export
    from src.evaluation.design import decision_versions
    from src.evaluation.endpoints import HEADLINES

    study_export.require_mindwell_cpu(f"Group C {study}")
    folds = benchmark_export.collect_folds(STUDIES[study]["phases"], metrics=METRICS, providers=[PROVIDER])
    if folds.empty or not folds.phase.eq(STUDIES[study]["phase"]).any():
        raise ValueError(f"No complete five-fold arm exists for {study}")
    tables = {name: benchmark_export._parquet_safe(frame) for name, frame in build_tables(study, folds).items()}
    target = study_export.write_export(
        export_path(study), {name: tables[name] for name in TABLES[study]}, schema=SCHEMA,
        identity=analysis_identity(study), origin=study_export.job_origin(),
        extra={"study": study, "headlines": HEADLINES, "decision_model_versions": decision_versions([PROVIDER])})
    print(f"{target}  SHA256 {file_sha256(target)}", flush=True)


def design_counts() -> dict:
    """Arm, configuration and training-fit counts from the plan (no results)."""
    from src.evaluation.campaign import benchmark_plan
    from src.evaluation.strands_adaptation import training_plan

    plan = [a for a in benchmark_plan() if a["phase"] in ("C1", "C2")]
    frame = pd.DataFrame(plan)
    fits = pd.DataFrame(training_plan())
    return {
        "c1a_arms": 12, "c1b_arms": 2, "c2_arms": 3,
        "configurations": {f"{phase}/{name}": int(n) for (phase, name), n
                           in frame.groupby(["phase", "feature_set"]).size().items()},
        "unsupported_width": int(frame.width_status.eq("unsupported_width").sum()),
        "c2_trained_fits": len(fits), "c2_lora_fits": int(fits.cell.ne("head_only").sum()),
        "c2_gpu_jobs": len(fits[["dataset_id", "fold", "cell"]].drop_duplicates()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, choices=tuple(STUDIES))
    build(parser.parse_args().study)


if __name__ == "__main__":
    main()
