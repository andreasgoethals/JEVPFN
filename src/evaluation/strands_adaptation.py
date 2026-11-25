"""Group C: C0 training qualification and C2 learning from labels with Strands v1 (C/exp11_adaptation, internal phase rq3).

A 2 x 2 design on the JOINT question with probability features only:

| torso \\ head | released head | trained head |
|---|---|---|
| released torso | zero-shot (= B1 Strands joint, no new run) | head-only (cached C1 vectors) |
| LoRA-trained torso | LoRA only (head frozen) | LoRA + head |

Every trained cell starts from Strands' released adapter and head and uses Strands' own
published recipe (`config/C/adaptation.yaml`, each value with its source line
in the pinned snapshot). Targets are the true class; regression uses nine equal-frequency bins
from the fit's optimizer rows, mapped lowest-first to Strands' nine ordered levels.

Cross-fitting: inside each outer training fold the rows are split into three parts by a hash
of (seed, dataset, split, fold, row) only, never by label. Fit k trains on the two other parts
(minus a label-independent early-stopping validation split: 10%, at most 1,000 rows, at most
three epochs) and scores part k; test rows get the mean of the three fits' probabilities. The
same parts serve all trained cells. Empty joint text has no request: its features stay missing.
Training rows thus never receive a probability from a model that saw their label.

GPU entry points (B200; ``scripts/slurm/c2_adaptation.slurm``): ``--dataset D --fold F --cell C
--execute`` trains the three cross-fitted fits of one cell and publishes the fold bundle;
``--pilot`` exercises one bounded cross-fitted fold, including scoring, publication and
readback. Pilot bundles have their own directory and never enter benchmark results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.utils import paths
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import canonical_json, digest, file_sha256, scalar, write_json

SCHEMA = "jevpfn.strands_adaptation.v2"
CONFIG = "rq3/adaptation"
CELLS = ("head_only", "lora_only", "lora_head")
TASK_TYPES = ("binary_classification", "multiclass_classification", "regression")
SNAPSHOT = Path("dm-library/repositories/StrandsDecider.txt")


def config():
    cfg = load_yaml(CONFIG)
    if tuple(cfg["trained_cells"]) != CELLS or cfg["crossfit"]["parts"] != 3 or cfg["question"] != "joint":
        raise ValueError("C2 differs from the approved 2 x 2 joint-question design")
    return cfg


def recipe_values(cfg=None) -> dict:
    """{name: value} of the published Strands recipe, as configured with its sources."""
    return {name: item["value"] for name, item in (cfg or config())["recipe"].items()}


def recipe_sources(cfg=None) -> dict:
    return {name: item["source"] for name, item in (cfg or config())["recipe"].items()}


# ------------------------------------------------------------------ cross-fitting (pure)


def crossfit_parts(train_rows, *, dataset_id: str, split_id: str, fold: int, seed: int,
                   parts: int = 3) -> np.ndarray:
    """Part (0..parts-1) of every outer-training row: hash rank only, never labels or text."""
    rows = [int(r) for r in train_rows]
    if len(rows) != len(set(rows)) or len(rows) < parts:
        raise ValueError("Cross-fitting needs unique training rows, at least one per part")
    keys = [hashlib.sha256(canonical_json([seed, dataset_id, split_id, fold, r]).encode()).hexdigest()
            for r in rows]
    order = np.argsort(np.array(keys), kind="stable")
    assignment = np.empty(len(rows), dtype=np.int64)
    assignment[order] = np.arange(len(rows)) * parts // len(rows)
    return assignment


def validation_rows(rows, *, seed: int, key: str, fraction=0.10, cap=1000) -> np.ndarray:
    """Label-independent early-stopping rows inside one fit's training part."""
    rows = np.asarray(sorted(int(r) for r in rows))
    count = min(cap, max(1, int(math.floor(fraction * len(rows)))))
    if count >= len(rows):
        raise ValueError("A fit needs training rows besides its validation rows")
    scores = [hashlib.sha256(canonical_json([seed, key, int(r)]).encode()).hexdigest() for r in rows]
    return np.sort(rows[np.argsort(np.array(scores), kind="stable")[:count]])


def fit_partition(train_rows, assignment, part: int):
    """(rows the fit trains on, rows it scores) for fit ``part``; disjoint by construction."""
    train_rows = np.asarray(train_rows)
    fit_rows, held_out = train_rows[assignment != part], train_rows[assignment == part]
    if set(fit_rows.tolist()) & set(held_out.tolist()):
        raise ValueError("A cross-fit model would score a row it trained on")
    return fit_rows, held_out


def _fit_rows(train_rows, assignment, joint, part, *, dataset_id, fold, cfg):
    fit_rows, held = fit_partition(train_rows, assignment, part)
    fit_rows = fit_rows[joint[fit_rows] >= 0]
    validation = validation_rows(fit_rows, seed=cfg["crossfit"]["seed"],
                                 key=f"{dataset_id}|{fold}|{part}",
                                 fraction=cfg["early_stopping"]["validation_fraction"],
                                 cap=cfg["early_stopping"]["validation_cap"])
    return np.setdiff1d(fit_rows, validation), validation, held


def regression_levels(training_targets, values, bins: int = 9):
    """Equal-frequency bins of the fit's training targets -> Strands levels 0..8 (lowest first).

    Tied cut-offs are kept (some levels may stay empty); bins are never renumbered.
    """
    training = np.asarray(training_targets, dtype=float)
    if not np.isfinite(training).all() or len(training) < bins:
        raise ValueError("Regression bins need finite training targets")
    edges = np.quantile(training, np.arange(1, bins) / bins)
    levels = np.searchsorted(edges, np.asarray(values, dtype=float), side="right")
    return levels.astype(np.int64), edges


def class_slots(values, classes, task_type) -> np.ndarray:
    """Option index of the true class: binary -> noul slot (1 = true = last declared class)."""
    wanted = [canonical_json(c) for c in classes]
    codes = np.array([wanted.index(canonical_json(scalar(v))) for v in values], dtype=np.int64)
    if task_type == "binary_classification":
        if len(wanted) != 2:
            raise ValueError("Binary tasks need two declared classes")
        return (codes == 1).astype(np.int64)
    return codes


def ordinal_target(level: int, n_options: int, smoothing: float, order=None) -> np.ndarray:
    """Strands' collator rule (collate.py _target_distribution): 1 - eps on the gold level,
    eps shared by its neighbours, mapped through the rendered option order."""
    dist = np.zeros(n_options)
    neighbours = [lv for lv in (level - 1, level + 1) if 0 <= lv < n_options]
    mass = {level: 1.0 - smoothing} | {lv: smoothing / len(neighbours) for lv in neighbours}
    if not neighbours:
        mass = {level: 1.0}
    order = list(range(n_options)) if order is None else list(order)
    for lv, value in mass.items():
        dist[order.index(lv)] += value
    return dist


def option_order(kind: str, n_options: int, rng: random.Random, reverse_probability: float):
    """Strands' training augmentation (collate.py _option_order): shuffle choice/noul options;
    reverse a score rubric with probability 0.5 (levels are never permuted)."""
    if kind == "score":
        return list(reversed(range(n_options))) if rng.random() < reverse_probability else None
    order = list(range(n_options))
    rng.shuffle(order)
    return order


def _verify_fit_scores(values, width, held, test_rows, nonempty):
    wanted = np.zeros(len(nonempty), dtype=bool)
    scored = np.concatenate([held, test_rows])
    wanted[scored] = nonempty[scored]
    if values.shape != (len(nonempty), width) or not np.isfinite(values[wanted]).all():
        raise ValueError("Every fit must score every held-out row with text with finite probabilities")
    if not np.isnan(values[~wanted]).all():
        raise ValueError("Unscored rows must stay missing")
    if ((values[wanted] < 0).any() or (values[wanted] > 1).any()
            or not np.allclose(values[wanted].sum(axis=1), 1, atol=1e-4, rtol=0)):
        raise ValueError("Each fit's adapted probabilities must be normalised")


def assemble_fold(n_rows: int, train_rows, test_rows, assignment, fit_probabilities,
                  fit_training_rows, *, nonempty) -> np.ndarray:
    """Training rows: the fit that held out their part. Test rows: mean of all fits.

    ``fit_probabilities[k]`` is an ``n_rows x K`` array (NaN where fit k produced no score);
    ``fit_training_rows[k]`` the rows fit k trained or validated on. Refuses any training row
    scored by a fit that saw it, and any test row a fit trained on. ``nonempty`` is the
    independent per-row mask of rows with joint text (from the request plan, not the scores):
    those rows of the fold need finite probabilities summing to one; every other row stays NaN.
    """
    parts = len(fit_probabilities)
    width = fit_probabilities[0].shape[1]
    out = np.full((n_rows, width), np.nan)
    train_rows, test_rows = np.asarray(train_rows), np.asarray(test_rows)
    nonempty = np.asarray(nonempty, dtype=bool)
    if nonempty.shape != (n_rows,):
        raise ValueError("The non-empty text mask must cover every dataset row")
    for k in range(parts):
        seen = set(np.asarray(fit_training_rows[k]).tolist())
        held = train_rows[assignment == k]
        if seen & set(held.tolist()) or seen & set(test_rows.tolist()):
            raise ValueError("A cross-fit model saw a row it scores")
        _verify_fit_scores(fit_probabilities[k], width, held, test_rows, nonempty)
        out[held] = fit_probabilities[k][held]
    scored_test = test_rows[nonempty[test_rows]]
    out[scored_test] = np.mean([p[scored_test] for p in fit_probabilities], axis=0)
    return out


def feature_columns(task_type: str, probabilities: np.ndarray, cell: str) -> dict:
    """B1's joint widths: binary P(true), multiclass C, regression nine level probabilities."""
    if task_type == "binary_classification":
        return {f"adapted__{cell}__joint__p_true": probabilities[:, 1]}
    return {f"adapted__{cell}__joint__p_{j:02d}": probabilities[:, j] for j in range(probabilities.shape[1])}


# ------------------------------------------------------------------ plan


def smallest_dataset(task_type=None) -> str:
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries

    entries = selected_entries(load_catalog(), load_yaml("experiment_1/main"))
    if task_type is not None:
        tasks = load_yaml("dataset_semantics")["datasets"]
        entries = [e for e in entries if tasks[e["id"]]["task_type"] == task_type]
    return min(entries, key=lambda e: (e["paper_summary"]["rows"], e["id"]))["id"]


def training_plan() -> list[dict]:
    """Every trained fit: dataset x outer fold x cross-fit part x trained cell (810)."""
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries

    cfg = config()
    return [dict(dataset_id=e["id"], fold=fold, part=part, cell=cell,
                 device="b200", execution_enabled=cfg["training_enabled"] is True)
            for e in selected_entries(load_catalog(), load_yaml("experiment_1/main"))
            for fold in range(5) for part in range(cfg["crossfit"]["parts"]) for cell in CELLS]


# ------------------------------------------------------------------ bundles


def bundle_dir(dataset_id: str, fold: int, cell: str) -> Path:
    if cell not in CELLS:
        raise ValueError(f"Unknown trained cell {cell}")
    return paths.group_c_dir("rq3", "adaptation", paths.output_component(dataset_id), f"fold_{fold}", cell)


def _source_assets(dataset, cell):
    """Resolve the inputs, checkpoint and runtime used by the native producer."""
    from src.cluster import containers, runtime_profiles
    from src.evaluation.design import decision_versions
    from src.jev import checkpoints, strands_hidden
    from src.jev.inputs import _read_set

    dataset_id = dataset.metadata.dataset_id
    questions = paths.dataset_question_path(dataset_id)
    with closing(sqlite3.connect(questions.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        input_id, prepared, _, _ = _read_set(db, None)
    if (prepared["dataset_id"] != dataset_id or prepared["data_identity"] != dataset.source["data_identity"]
            or prepared["task_schema_sha256"] != dataset.metadata.fingerprint
            or prepared["row_count"] != len(dataset.frame)):
        raise ValueError("C2 prepared questions belong to different rows or task definitions")
    checkpoint = json.loads((checkpoints.checkpoint_path("strands") /
                             "_JEVPFN_CHECKPOINT.json").read_text(encoding="utf-8"))
    image = containers.container_dir("strands", profile=runtime_profiles.BLACKWELL) / containers.MARKER
    seal = json.loads(image.read_text(encoding="utf-8"))
    registered, = decision_versions(["strands"])
    if (checkpoint["provider"] != "strands" or checkpoint["revision"] != checkpoints.PINS["strands"]
            or (checkpoint.get("model"), checkpoint["revision"]) != (registered["model"], registered["version"])
            or (checkpoint.get("base") or {}).get("revision") != registered.get("base_model_revision")
            or seal["provider"] != "strands" or seal["runtime_profile"] != runtime_profiles.BLACKWELL
            or seal["container_id"] != digest({k: v for k, v in seal.items() if k != "container_id"})):
        raise ValueError("C2 requires the pinned Strands checkpoint and sealed B200 runtime")
    sources = {"input_set_id": input_id, "checkpoint": digest(checkpoint), "container_id": seal["container_id"],
               "decision_model": registered}
    if cell == "head_only":
        hidden = strands_hidden.read_manifest(dataset_id, verify=False)
        if (hidden["recipe"]["input_set_id"] != input_id
                or hidden["recipe"]["checkpoint"] != sources["checkpoint"]
                or hidden["recipe"].get("decision_model") != registered):
            raise ValueError("C2 head-only vectors differ from the current inputs or Strands checkpoint")
        sources["hidden_artifact"] = hidden["artifact_id"]
    return sources


def _fold_definition(dataset, split, fold, cell):
    cfg = config()
    train = split["folds"][fold]["train"]
    return {
        "schema": SCHEMA, "dataset_id": dataset.metadata.dataset_id,
        "data_identity": dataset.source["data_identity"], "split_id": split["split_id"],
        "fold": fold, "cell": cell, "train": train, "test": split["folds"][fold]["test"],
        "training_labels": digest([scalar(v) for v in dataset.frame[dataset.metadata.target].iloc[train]]),
        "task_type": dataset.metadata.task_type, "crossfit": cfg["crossfit"],
        "early_stopping": cfg["early_stopping"], "recipe": cfg["recipe"],
        "regression_bins": cfg["regression_bins"],
    }


def fold_recipe(dataset, split, fold: int, cell: str) -> dict:
    return {**_fold_definition(dataset, split, fold, cell), "code_sha256": file_sha256(Path(__file__)),
            "sources": _source_assets(dataset, cell)}


def _verify_bundle_files(folder, body):
    if body.get("artifact_id") != digest({k: v for k, v in body.items() if k != "artifact_id"}):
        raise ValueError("Adapted bundle manifest checksum differs")
    for name, checksum in body["files"].items():
        file = (folder / name).resolve()
        if not file.is_relative_to(folder.resolve()) or not file.is_file() or file_sha256(file) != checksum:
            raise ValueError("Adapted bundle file missing or changed")


def sealed_bundle(folder: Path, recipe: dict) -> dict | None:
    """The published result of a complete, verified bundle with this exact recipe, or None."""
    target = Path(folder) / "manifest.json"
    if not target.exists():
        return None
    body = json.loads(target.read_text(encoding="utf-8"))
    if body.get("recipe") != recipe or body.get("status") != "complete":
        return None
    _verify_bundle_files(Path(folder), body)
    return {"dataset_id": recipe.get("dataset_id"), "fold": recipe.get("fold"),
            "cell": recipe.get("cell"), "artifact_id": body["artifact_id"]}


def seal_bundle(folder: Path, recipe: dict, names, reports) -> dict:
    """Publish the fold manifest once; a repeated call verifies and returns the sealed result,
    so identical predictions never receive a new artifact identity."""
    sealed = sealed_bundle(folder, recipe)
    if sealed is not None:
        return sealed
    body = {"schema": SCHEMA, "status": "complete", "recipe": recipe,
            "files": {name: file_sha256(Path(folder) / name) for name in names},
            "fits": [{k: r[k] for k in ("part", "best_validation_loss", "seconds", "peak_gpu_bytes", "history")}
                     for r in reports], "created_utc": datetime.now(UTC).isoformat()}
    body["artifact_id"] = digest(body)
    write_json(Path(folder) / "manifest.json", body)
    return {"dataset_id": recipe.get("dataset_id"), "fold": recipe.get("fold"),
            "cell": recipe.get("cell"), "artifact_id": body["artifact_id"]}


def read_bundle(dataset, split, fold: int, cell: str) -> dict:
    """Verify the published fold on a consumer using only its dataset, split and sealed files."""
    folder = bundle_dir(dataset.metadata.dataset_id, fold, cell)
    return _read_bundle(folder, _fold_definition(dataset, split, fold, cell), dataset, published_sources=True)


def _read_bundle(folder, recipe, dataset, *, published_sources=False):
    body = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if published_sources:
        pinned = body.get("recipe", {})
        sources = pinned.get("sources")
        required = {"input_set_id", "checkpoint", "container_id"}
        if recipe["cell"] == "head_only":
            required.add("hidden_artifact")
        trainer = pinned.get("code_sha256")
        if (not isinstance(sources, dict) or set(sources) - {"decision_model"} != required
                or not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                           for value in [trainer, *(sources[key] for key in required)])):
            raise ValueError("Adapted bundle lacks its sealed producer identities")
        if "decision_model" in sources:
            model = sources["decision_model"]
            if (not isinstance(model, dict) or model.get("provider") != "strands"
                    or not isinstance(model.get("model"), str) or not model["model"].strip()
                    or not re.fullmatch(r"[0-9a-f]{40}", str(model.get("version", "")))):
                raise ValueError("Adapted bundle has an invalid Strands model identity")
        recipe = {**recipe, "code_sha256": trainer, "sources": sources}
    if body.get("schema") != SCHEMA or body["recipe"] != recipe or body["status"] != "complete":
        raise ValueError("Adapted bundle does not match this outer partition and recipe")
    parts = recipe["crossfit"]["parts"]
    expected = {"probabilities.npy"} | {f"fit_{k}.{ext}" for k in range(parts) for ext in ("json", "npy")}
    if set(body["files"]) != expected or [r["part"] for r in body["fits"]] != list(range(parts)):
        raise ValueError("Adapted bundle lacks its three fits or predictions")
    _verify_bundle_files(folder, body)
    for part in range(parts):
        fit = json.loads((folder / f"fit_{part}.json").read_text(encoding="utf-8"))
        if (fit.get("recipe") != recipe or fit.get("cell") != recipe["cell"]
                or fit.get("checkpoint") != recipe["sources"]["checkpoint"]
                or fit.get("sha256") != body["files"][f"fit_{part}.npy"]
                or any(fit.get(key) != value for key, value in body["fits"][part].items())):
            raise ValueError("Adapted bundle contains a fit from another recipe or prediction file")
    values = np.load(folder / "probabilities.npy", allow_pickle=False)
    width = (recipe["regression_bins"] if dataset.metadata.task_type == "regression"
             else len(dataset.metadata.classes))
    if values.shape != (len(dataset.frame), width):
        raise ValueError("Adapted probabilities have another row count or option width")
    return {**body, "probabilities": values}


class AdaptedControl:
    """Adds one fold's cross-fitted adapted probabilities; refuses inner refits."""

    def __init__(self, estimator, dataset, split, bundles, cell):
        self.estimator, self.dataset, self.split = estimator, dataset, split
        self.bundles, self.cell = bundles, cell

    def _features(self, x):
        rows = [int(i.removeprefix("csv:")) for i in x.index]
        allowed = self.split["folds"][self.fold]
        if not set(rows) <= set(allowed["train"]) | set(allowed["test"]):
            raise ValueError("Adapted features requested outside their outer partition")
        values = self.bundles[self.fold]["probabilities"][rows]
        columns = feature_columns(self.dataset.metadata.task_type, values, self.cell)
        import pandas as pd

        return x.join(pd.DataFrame(columns, index=x.index), validate="one_to_one")

    def fit(self, x, y):
        matches = [i for i, f in enumerate(self.split["folds"])
                   if x.index.tolist() == [f"csv:{r}" for r in f["train"]]]
        if len(matches) != 1:
            raise ValueError("Adapted features use the exact outer training rows, with no inner refits")
        self.fold = matches[0]
        augmented = self._features(x)
        self.output_feature_count_ = augmented.shape[1]
        self.estimator.fit(augmented, y)
        if hasattr(self.estimator, "classes_"):
            self.classes_ = self.estimator.classes_
        return self

    def predict(self, x):
        return self.estimator.predict(self._features(x))

    def predict_proba(self, x):
        return self.estimator.predict_proba(self._features(x))


# ------------------------------------------------------------------ training (B200, torch)


def _joint_inputs(dataset_id):
    from src.jev.strands_hidden import input_plan

    identity, recipe, _, matrix, _ = input_plan(dataset_id)
    slot = next(i for i, s in enumerate(recipe["slots"]) if s["representation"] == "joint")
    return identity, matrix[:, slot]


def _targets(dataset, rows, fit_rows, cfg):
    """(per-row option targets for ``rows``, kind, n_options, bin edges or None)."""
    meta = dataset.metadata
    values = dataset.frame[meta.target].to_numpy()
    if meta.task_type == "regression":
        levels, edges = regression_levels(values[fit_rows].astype(float), values[rows].astype(float),
                                          cfg["regression_bins"])
        return levels, "score", cfg["regression_bins"], edges.tolist()
    slots = class_slots(values[rows], meta.classes, meta.task_type)
    if meta.task_type == "binary_classification":
        return slots, "noul", 2, None
    return slots, "choice", len(meta.classes), None


class _Fit:
    """One cross-fitted training run of one cell; holds the GPU objects for its lifetime."""

    def __init__(self, cell, dataset, cfg, *, seed):
        import torch

        self.torch, self.cell, self.dataset, self.cfg = torch, cell, dataset, cfg
        self.recipe = recipe_values(cfg)
        self.seed = seed
        torch.manual_seed(seed)
        torch.backends.cuda.enable_cudnn_sdp(self.recipe["cudnn_sdp"])
        from src.jev.self_hosted import inference_environment

        inference_environment()
        from src.jev.checkpoints import verify_checkpoint

        self.folder, self.checkpoint = verify_checkpoint("strands")
        from strands_decider import modeling

        self.modeling = modeling
        if cell == "head_only":
            config_ = modeling.StrandsDeciderConfig.from_json(modeling.config_path(str(self.folder)))
            self.model = None
            from src.jev.strands_hidden import read_manifest

            hidden = read_manifest(dataset.metadata.dataset_id, verify=False)["recipe"]["hidden_size"]
            self.head = modeling.build_head(config_, hidden)
            self.head.load_state_dict(modeling.load_head_state(str(self.folder)))
        else:
            self.model = modeling.StrandsDeciderModel.load(str(self.folder), trainable=True).to("cuda")
            self.head = self.model.head
            self._check_adapter()
        # init_from (train.py 52401): fitted temperatures are reset; train and predict at T = 1.
        self.head.dropout = (torch.nn.Dropout(self.recipe["head_dropout"])
                             if self.recipe["head_dropout"] > 0 else torch.nn.Identity())
        self.head.to("cuda").float()
        for p in self.head.parameters():
            p.requires_grad_(cell in {"head_only", "lora_head"})
        if self.model is not None:
            self.model.config.temperature, self.model.config.temperature_by_kind = self.recipe["temperature"], {}

    def _check_adapter(self):
        cfg = self.model.config
        if sorted(cfg.lora_targets) != sorted(self.recipe["lora_targets"]):
            raise ValueError("Released adapter targets differ from the published recipe")
        if cfg.lora_alpha / cfg.lora_r != self.recipe["lora_alpha"] / self.recipe["lora_r"]:
            raise ValueError("Released adapter scale differs from the recipe's alpha / r")
        adapters = self.model.torso.peft_config.values()
        if not self.model.torso.peft_config or any(a.lora_dropout != self.recipe["lora_dropout"] for a in adapters):
            raise ValueError("Loaded LoRA dropout differs from the published training recipe")
        base = getattr(self.model.torso, "base_model", self.model.torso)
        core = getattr(base, "model", base)
        if self.recipe["gradient_checkpointing"]:
            core.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        elif getattr(core, "is_gradient_checkpointing", False):
            core.gradient_checkpointing_disable()
        trainable = [n for n, p in self.model.torso.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in n for n in trainable):
            raise ValueError("Only LoRA parameters may train in the torso")

    def parameters(self):
        found = [p for p in self.head.parameters() if p.requires_grad]
        if self.model is not None:
            found += [p for p in self.model.torso.parameters() if p.requires_grad]
        return found

    def optimizer(self, total_steps):
        """Strands' _build_optimizer (train.py): head matrices decay, norms/biases and LoRA not."""
        torch, r = self.torch, self.recipe
        decay = [p for p in self.head.parameters() if p.requires_grad and p.ndim > 1]
        plain = [p for p in self.head.parameters() if p.requires_grad and p.ndim <= 1]
        groups = [{"params": decay, "lr": r["head_lr"], "weight_decay": r["weight_decay"]},
                  {"params": plain, "lr": r["head_lr"], "weight_decay": 0.0}]
        if self.model is not None:
            groups.append({"params": [p for p in self.model.torso.parameters() if p.requires_grad],
                           "lr": r["lr"], "weight_decay": 0.0})
        groups = [g for g in groups if g["params"]]
        optim = torch.optim.AdamW(groups, betas=tuple(r["adam_betas"]), eps=1e-8)
        warmup = max(1, int(total_steps * r["warmup_ratio"]))

        def factor(step):
            step += 1
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        return optim, torch.optim.lr_scheduler.LambdaLR(optim, factor)

    # -- batches -------------------------------------------------------------------------
    def head_batch(self, items, targets, kind):
        torch = self.torch
        width = max(len(keys) for _, keys in items)
        decide = torch.tensor(np.stack([a for a, _ in items]), device="cuda")
        options = np.zeros((len(items), width, decide.shape[1]), dtype=np.float32)
        for i, (_, keys) in enumerate(items):
            options[i, :len(keys)] = keys
        n_slots = torch.tensor([len(k) for _, k in items], device="cuda")
        logits = self.head(decide, torch.tensor(options, device="cuda")) / self.recipe["temperature"]
        log_probs = self.modeling.masked_log_softmax(logits, n_slots)
        return log_probs, self._loss(log_probs, targets, kind, width, [None] * len(items))

    def prompt_batch(self, payloads, targets, kind, *, train, rng):
        from strands_decider import prompting

        from src.jev.provider import QUESTION_ID, semantic_question
        from src.jev.strands_hidden import option_token_index

        torch = self.torch
        backend = self.model.tokenizer.backend_tokenizer
        backend.no_padding()
        backend.no_truncation()
        rows, orders = [], []
        for payload in payloads:
            from strands_decider.schema import SystemOneRequest

            request = SystemOneRequest(state=payload["state"],
                                       questions={QUESTION_ID: semantic_question(payload["question"])})
            question = request.questions[QUESTION_ID]
            n_options = 2 if kind == "noul" else len(question.criteria)
            order = (option_order(kind, n_options, rng, self.recipe["reverse_score_prob"])
                     if train and self.recipe["shuffle_options"] else None)
            rendered = prompting.render_question(question, option_order=order)
            state = backend.encode(prompting.render_state(request.state), add_special_tokens=True)
            asked = backend.encode(rendered.text, add_special_tokens=False)
            if len(state.ids) + len(asked.ids) > self.model.config.max_length:
                raise ValueError("C2 rendered prompt exceeds the released context window")
            index = [len(state.ids) + i for i in option_token_index(asked.offsets, rendered.option_spans)]
            rows.append((state.ids + asked.ids, index))
            orders.append(order)
        pad = self.model.tokenizer.pad_token_id or 0
        length = max(len(ids) for ids, _ in rows)
        width = max(len(index) for _, index in rows)
        ids = torch.tensor([ids + [pad] * (length - len(ids)) for ids, _ in rows], device="cuda")
        mask = torch.tensor([[1] * len(ids) + [0] * (length - len(ids)) for ids, _ in rows], device="cuda")
        opt = torch.tensor([index + [-1] * (width - len(index)) for _, index in rows], device="cuda")
        n_slots = torch.tensor([len(index) for _, index in rows], device="cuda")
        out = self.model(input_ids=ids, attention_mask=mask, n_slots=n_slots, opt_idx=opt,
                         temperature=self.recipe["temperature"])
        return out["log_probs"], orders, self._loss(out["log_probs"], targets, kind, width, orders)

    def _loss(self, log_probs, targets, kind, width, orders):
        torch = self.torch
        if targets is None:
            return None
        if kind == "score":
            dist = np.stack([np.pad(ordinal_target(int(t), self.cfg["regression_bins"],
                                                   self.recipe["ordinal_smoothing"], order),
                                    (0, width - self.cfg["regression_bins"]))
                             for t, order in zip(targets, orders)])
            dist = torch.tensor(dist, device=log_probs.device, dtype=torch.float32)
            safe = torch.where(torch.isinf(log_probs), torch.zeros_like(log_probs), log_probs)
            return -(dist * safe).sum(dim=-1).mean()
        labels = torch.tensor([order.index(int(t)) if order is not None else int(t)
                               for t, order in zip(targets, orders)], device=log_probs.device)
        return torch.nn.functional.nll_loss(log_probs, labels)


def canonical_probabilities(values, orders, n_options: int) -> list[np.ndarray]:
    """Rendered-slot probabilities back to canonical option order (slot k shows order[k])."""
    out = []
    for row, order in zip(np.asarray(values, dtype=float), orders):
        row = row[:n_options]
        out.append(row if order is None else row[np.argsort(order)])
    return out


def train_fit(dataset, split, fold: int, part: int, cell: str, *, seed: int, max_epochs=None,
              input_set_id=None) -> dict:
    """Train fit ``part`` of one cell; returns per-row probabilities for held-out and test rows."""
    started = time.perf_counter()
    cfg = config()
    recipe = recipe_values(cfg)
    torch = __import__("torch")
    torch.cuda.reset_peak_memory_stats()
    meta = dataset.metadata
    identity, joint = _joint_inputs(meta.dataset_id)
    if input_set_id is not None and identity != input_set_id:
        raise ValueError("C2 fit received different prepared inputs")
    folds = split["folds"][fold]
    train_rows, test_rows = np.asarray(folds["train"]), np.asarray(folds["test"])
    assignment = crossfit_parts(train_rows, dataset_id=meta.dataset_id, split_id=split["split_id"],
                                fold=fold, seed=cfg["crossfit"]["seed"])
    optimizer_rows, validation, held_out = _fit_rows(
        train_rows, assignment, joint, part, dataset_id=meta.dataset_id, fold=fold, cfg=cfg)
    fit_rows = np.concatenate([optimizer_rows, validation])
    targets, kind, n_options, edges = _targets(dataset, fit_rows, optimizer_rows, cfg)
    target_of = dict(zip(fit_rows.tolist(), targets.tolist()))
    scored = np.concatenate([held_out, test_rows])
    scored = scored[joint[scored] >= 0]
    fit = _Fit(cell, dataset, cfg, seed=seed)
    rng = random.Random(seed)
    if cell == "head_only":
        from src.jev.strands_hidden import read_manifest, read_training_vectors, store_dir

        hidden = read_manifest(meta.dataset_id, verify=False)["recipe"]["hidden_size"]
        needed = {int(joint[r]) for r in np.concatenate([fit_rows, scored])}
        cache = read_training_vectors(store_dir(meta.dataset_id) / "hidden.sqlite3", needed, hidden)

        def batch_forward(rows, train):
            items = [cache[int(joint[r])] for r in rows]
            wanted = [target_of[int(r)] for r in rows] if train is not None else None
            log_probs, loss = fit.head_batch(items, wanted, kind)
            return log_probs, [None] * len(rows), loss
    else:
        from src.jev.inputs import iter_prepared_inputs

        needed = {int(joint[r]) for r in np.concatenate([fit_rows, scored])}
        payloads = {rec["input_no"]: rec["payload"] for rec in iter_prepared_inputs(
            paths.dataset_question_path(meta.dataset_id), input_set_id=identity,
            input_numbers=needed if len(needed) <= 900 else None) if rec["input_no"] in needed}

        def batch_forward(rows, train):
            wanted = [target_of[int(r)] for r in rows] if train is not None else None
            log_probs, orders, loss = fit.prompt_batch([payloads[int(joint[r])] for r in rows], wanted,
                                                       kind, train=bool(train), rng=rng)
            return log_probs, orders, loss

    def forward(rows, train):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=recipe["bf16_autocast"]):
            return batch_forward(rows, train)

    micro, accum = recipe["micro_batch_size"], recipe["grad_accum"]
    epochs = int(max_epochs or cfg["early_stopping"]["max_epochs"])
    steps_per_epoch = math.ceil(len(optimizer_rows) / (micro * accum))
    if steps_per_epoch < 1 or epochs < 1:
        raise ValueError("A C2 fit requires an optimizer update")
    optim, schedule = fit.optimizer(steps_per_epoch * epochs)
    best, best_loss, history = None, math.inf, []
    order_rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        if fit.model is not None:
            fit.model.train()
        fit.head.train()
        permutation = order_rng.permutation(optimizer_rows)
        epoch_started = time.perf_counter()
        for start in range(0, len(permutation), micro * accum):
            group = permutation[start:start + micro * accum]
            for offset in range(0, len(group), micro):
                batch = group[offset:offset + micro]
                _, _, loss = forward(batch, True)
                (loss * (len(batch) / len(group))).backward()
            torch.nn.utils.clip_grad_norm_(fit.parameters(), recipe["max_grad_norm"], error_if_nonfinite=True)
            optim.step()
            schedule.step()
            optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - epoch_started
        if fit.model is not None:
            fit.model.eval()
        fit.head.eval()
        with torch.no_grad():
            losses = [float(forward(validation[i:i + micro], False)[2]) * len(validation[i:i + micro])
                      for i in range(0, len(validation), micro)]
        loss = sum(losses) / len(validation)
        if not math.isfinite(loss):
            raise ValueError("C2 produced a non-finite validation loss; no checkpoint published")
        history.append({"epoch": epoch + 1, "validation_loss": loss, "train_seconds": train_seconds,
                        "examples": len(permutation), "optimizer_steps": steps_per_epoch})
        if loss < best_loss:
            best_loss = loss
            best = {id(p): p.detach().clone() for p in fit.parameters()}
        elif cfg["early_stopping"]["patience_epochs"] == 0:
            break
    with torch.no_grad():
        for p in fit.parameters():
            p.copy_(best[id(p)])
    probabilities = np.full((len(dataset.frame), n_options), np.nan)
    with torch.no_grad():
        unique = sorted({int(joint[r]) for r in scored})
        by_input = {}
        representative = {int(joint[r]): int(r) for r in scored}
        for i in range(0, len(unique), micro):
            chunk = unique[i:i + micro]
            log_probs, orders, _ = forward([representative[n] for n in chunk], None)
            values = log_probs.exp().detach().float().cpu().numpy()
            for number, row in zip(chunk, canonical_probabilities(values, orders, n_options)):
                by_input[number] = row
        for r in scored:
            probabilities[r] = by_input[int(joint[r])]
    return {
        "part": part, "cell": cell, "kind": kind, "options": n_options, "bin_edges": edges,
        "optimizer_rows": optimizer_rows.tolist(), "validation_rows": validation.tolist(),
        "held_out_rows": held_out.tolist(), "history": history, "best_validation_loss": best_loss,
        "seconds": time.perf_counter() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "checkpoint": fit.checkpoint, "probabilities": probabilities,
    }


def _dataset_split(dataset_id):
    from src.data.loaders import load_catalog, load_dataset
    from src.data.selection import selected_entries
    from src.evaluation.protocol import shared_splits

    main = load_yaml("experiment_1/main")
    selected = {e["id"]: e for e in selected_entries(load_catalog(), main)}
    if dataset_id not in selected:
        raise ValueError("C2 requires a selected benchmark dataset")
    entry = selected[dataset_id]
    dataset = load_dataset(entry, allow_download=False)
    return dataset, shared_splits(dataset, seed=main["random_seed"], folds=5)


def _require_native_runtime():
    from src.cluster import runtime_profiles
    from src.cluster.containers import active_identity
    from src.jev.checkpoints import verify_checkpoint
    from src.jev.self_hosted import inference_environment

    if (not paths.on_vsc() or not os.environ.get("SLURM_JOB_ID")
            or (os.environ.get("SLURM_CLUSTER_NAME"), os.environ.get("SLURM_JOB_PARTITION"))
            != ("mindwell", "gpu_b200")):
        raise RuntimeError("C2 training requires a Mindwell B200 allocation")
    inference_environment()
    image = active_identity()
    if (not image or image["provider"] != "strands"
            or image["runtime_profile"] != runtime_profiles.BLACKWELL):
        raise ValueError("C2 requires the verified native Strands container")
    return {"checkpoint": verify_checkpoint("strands")[1], "container_id": image["container_id"]}


def _train_bundle(dataset, split, fold, cell, folder, recipe, runtime, *, max_epochs=None):
    """Both production and pilot use the same writer, resume checks, fold assembly and sealer."""
    if any(recipe["sources"].get(k) != v for k, v in runtime.items()):
        raise ValueError("C2 training assets differ from the running model or image")
    with file_lock(folder / ".fit.lock"):
        return _train_locked(dataset, split, fold, cell, folder, recipe, max_epochs=max_epochs)


def _train_locked(dataset, split, fold, cell, folder, recipe, *, max_epochs):
    cfg = config()
    dataset_id = dataset.metadata.dataset_id
    if cell == "head_only":
        from src.jev.strands_hidden import read_manifest

        if read_manifest(dataset_id)["artifact_id"] != recipe["sources"]["hidden_artifact"]:
            raise ValueError("C1 vectors changed before C2 training")
    if (folder / "manifest.json").exists():
        manifest = _read_bundle(folder, recipe, dataset)
        return {"dataset_id": dataset_id, "fold": fold, "cell": cell, "artifact_id": manifest["artifact_id"]}
    train_rows = np.asarray(split["folds"][fold]["train"])
    assignment = crossfit_parts(train_rows, dataset_id=dataset_id, split_id=split["split_id"], fold=fold,
                                seed=cfg["crossfit"]["seed"])
    input_set_id, joint = _joint_inputs(dataset_id)
    if input_set_id != recipe["sources"]["input_set_id"]:
        raise ValueError("Prepared inputs changed before C2 training")
    fits = []
    reports = []
    settings = recipe_values(cfg)
    width = cfg["regression_bins"] if dataset.metadata.task_type == "regression" else len(dataset.metadata.classes)
    for part in range(cfg["crossfit"]["parts"]):
        target = folder / f"fit_{part}.npy"
        report = folder / f"fit_{part}.json"
        reusing = report.is_file()
        if reusing:
            stored = json.loads(report.read_text(encoding="utf-8"))
            if (stored["recipe"] != recipe or stored["part"] != part or stored["cell"] != cell
                    or stored["checkpoint"] != recipe["sources"]["checkpoint"]
                    or not target.is_file() or stored["sha256"] != file_sha256(target)):
                raise ValueError("A stored fit belongs to another recipe or changed")
        else:
            stored = train_fit(dataset, split, fold, part, cell, seed=settings["seed"],
                               max_epochs=max_epochs, input_set_id=input_set_id)
            if stored["checkpoint"] != recipe["sources"]["checkpoint"]:
                raise ValueError("The C2 fit loaded another checkpoint")
        expected_rows = _fit_rows(train_rows, assignment, joint, part, dataset_id=dataset_id, fold=fold, cfg=cfg)
        for key, expected in zip(("optimizer_rows", "validation_rows", "held_out_rows"), expected_rows):
            if stored[key] != expected.tolist():
                raise ValueError(f"C2 part {part} {key} differ from the planned cross-fit partition")
        n = len(expected_rows[0])
        steps = math.ceil(n / (settings["micro_batch_size"] * settings["grad_accum"]))
        history = stored["history"]
        if (not history or any(h["examples"] != n or h["optimizer_steps"] != steps
                               or not math.isfinite(h["validation_loss"]) for h in history)
                or stored["best_validation_loss"] != min(h["validation_loss"] for h in history)):
            raise ValueError(f"C2 part {part} lacks valid optimization and validation evidence")
        if not reusing:
            _verify_fit_scores(stored["probabilities"], width, expected_rows[2],
                               split["folds"][fold]["test"], joint >= 0)
            partial = target.with_suffix(".partial.npy")
            np.save(partial, stored.pop("probabilities"), allow_pickle=False)
            os.replace(partial, target)
            stored.update(recipe=recipe, sha256=file_sha256(target), job_id=os.environ.get("SLURM_JOB_ID"),
                          created_utc=datetime.now(UTC).isoformat())
            write_json(report, stored)
        reports.append(stored)
        fits.append(target)
    probabilities = [np.load(f, allow_pickle=False) for f in fits]
    values = assemble_fold(len(dataset.frame), train_rows, split["folds"][fold]["test"], assignment,
                           probabilities, [r["optimizer_rows"] + r["validation_rows"] for r in reports],
                           nonempty=joint >= 0)
    np.save(folder / "probabilities.partial.npy", values, allow_pickle=False)
    os.replace(folder / "probabilities.partial.npy", folder / "probabilities.npy")
    names = ["probabilities.npy", *(f.name for f in fits), *(f"fit_{k}.json" for k in range(len(fits)))]
    return seal_bundle(folder, recipe, names, reports)


def run_fold(dataset_id: str, fold: int, cell: str, *, execute=False) -> dict:
    """Train a production fold after a matching, published native pilot for its cell and task."""
    if execute is not True or config()["training_enabled"] is not True:
        raise RuntimeError("C2 training is disabled; enable rq3/adaptation training and pass --execute")
    if cell not in CELLS or fold not in range(5):
        raise ValueError("Choose a trained C2 cell and an outer fold from 0 to 4")
    runtime = _require_native_runtime()
    dataset, split = _dataset_split(dataset_id)
    _qualified_pilot(dataset.metadata.task_type, cell)
    recipe = fold_recipe(dataset, split, fold, cell)
    return _train_bundle(dataset, split, fold, cell, bundle_dir(dataset_id, fold, cell), recipe, runtime)


def _pilot_token_lengths(dataset_id):
    """Use the existing exact Strands audit to exercise the pilot fold's longest joint requests."""
    identity, joint = _joint_inputs(dataset_id)
    evidence = paths.manifests_dir("experiment_0") / "before_extraction" / "token_audit.json"
    combined = json.loads(evidence.read_text(encoding="utf-8"))
    report, = [r for r in combined["self_hosted"] if r["provider"] == "strands"]
    item, = [d for d in report["datasets"] if d["dataset_id"] == dataset_id]
    if (report["status"] != "counted_all_inputs" or report["overflow_unique_inputs"]
            or item["input_set_id"] != identity):
        raise ValueError("C2 pilot needs the complete Strands token audit for these prepared inputs")
    counts = paths.token_audit_dir() / f"strands_{report['audit_id']}.sqlite3"
    with closing(sqlite3.connect(counts.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        recipe = db.execute("SELECT recipe_json FROM identity WHERE id=1").fetchone()
        if not recipe or digest(json.loads(recipe[0])) != report["audit_id"]:
            raise ValueError("C2 pilot token counts differ from the audited recipe")
        lengths = dict(db.execute("SELECT input_no,input_tokens FROM counts WHERE dataset_id=?", (dataset_id,)))
    return np.array([lengths[int(n)] if n >= 0 else 0 for n in joint])


def _pilot_partition(dataset, split, spec, lengths):
    """Bound one existing outer fold, retaining its longest joint requests and class vocabulary."""
    fold = spec["fold"]
    rng = np.random.default_rng(spec["seed"])
    sampled = {}
    labels = dataset.frame[dataset.metadata.target].to_numpy()
    for role in ("train", "test"):
        shuffled = rng.permutation(split["folds"][fold][role]).tolist()
        limit = spec[f"{role}_rows"]
        nonempty = [r for r in shuffled if lengths[r] > 0]
        if not nonempty:
            raise ValueError(f"C2 pilot {role} partition has no joint requests")
        required = [max(nonempty, key=lambda r: lengths[r])]
        if dataset.metadata.task_type != "regression":
            first_by_class = {}
            for row in required + nonempty:
                first_by_class.setdefault(canonical_json(scalar(labels[row])), row)
            if len(first_by_class) != len(dataset.metadata.classes):
                raise ValueError("Pilot partition cannot exercise every class with joint text")
            required.extend(first_by_class.values())
        empty = next((r for r in shuffled if lengths[r] == 0), None)
        if empty is not None:
            required.append(empty)
        if len(set(required)) > limit:
            raise ValueError("Pilot row limit cannot cover the required long input and classes")
        sampled[role] = sorted(list(dict.fromkeys(required + shuffled))[:limit])
    folds = list(split["folds"])
    folds[fold] = sampled
    return {**split, "folds": folds, "split_id": digest({"parent": split["split_id"], "fold": fold,
                                                         "pilot_rows": sampled})}


def _pilot_inputs(dataset_id, cell):
    spec = config()["pilot"]
    dataset, parent = _dataset_split(dataset_id)
    lengths = _pilot_token_lengths(dataset_id)
    split = _pilot_partition(dataset, parent, spec, lengths)
    longest = {role: int(max(lengths[rows])) for role, rows in split["folds"][spec["fold"]].items()}
    recipe = dict(fold_recipe(dataset, split, spec["fold"], cell), pilot=spec,
                  parent_split_id=parent["split_id"], pilot_max_joint_tokens=longest)
    folder = paths.group_c_dir("rq3", "adaptation_pilots", dataset_id, cell)
    receipt = paths.manifests_dir("rq3") / "adaptation_pilot" / f"{dataset_id}_{cell}.json"
    return dataset, split, recipe, folder, receipt


def _qualified_pilot(task_type, cell):
    from src.utils.acceptance import evidence_path, reviewed_acceptance

    dataset_id = smallest_dataset(task_type)
    dataset, _, recipe, folder, receipt = _pilot_inputs(dataset_id, cell)
    report = json.loads(receipt.read_text(encoding="utf-8"))
    if (report.get("status") != "passed" or report.get("task_type") != task_type
            or report.get("cell") != cell or report.get("recipe_id") != digest(recipe)
            or report.get("artifact_id") != _read_bundle(folder, recipe, dataset)["artifact_id"]):
        raise ValueError(f"C2 requires the current passed {task_type}/{cell} producer pilot")
    filename = config()["pilot_acceptance"]
    approval = reviewed_acceptance(filename, stage="c2_adaptation_pilot")
    if not any(evidence_path(filename, item).resolve() == receipt.resolve() for item in approval["evidence"]):
        raise ValueError(f"C2 pilot review lacks the {task_type}/{cell} receipt")
    return report


def pilot(*, execute=False, dataset_id=None, cell=None, task_type=None) -> dict:
    """Run one bounded, end-to-end qualification fold in an isolated, resumable pilot directory."""
    cfg = config()
    spec = cfg["pilot"]
    dataset_id = dataset_id or smallest_dataset(task_type)
    cell = cell or spec["cell"]
    family = load_yaml("dataset_semantics")["datasets"][dataset_id]["task_type"]
    if (task_type is not None and family != task_type) or dataset_id != smallest_dataset(family) or cell not in CELLS:
        raise ValueError("Use the smallest selected dataset for the pilot task type and a trained C2 cell")
    if execute is not True:
        return {"dataset_id": dataset_id, "fold": spec["fold"], "parts": cfg["crossfit"]["parts"], "cell": cell,
                "task_type": family, "train_rows": spec["train_rows"], "test_rows": spec["test_rows"],
                "execution": False}
    if cfg.get("pilot_enabled") is not True:
        raise RuntimeError("C0 pilot is disabled. After approval to use B200 time, set pilot_enabled and pass --execute.")
    runtime = _require_native_runtime()
    dataset, split, recipe, folder, receipt = _pilot_inputs(dataset_id, cell)
    _train_bundle(dataset, split, spec["fold"], cell, folder, recipe, runtime, max_epochs=1)
    published = _read_bundle(folder, recipe, dataset)
    fits = published["fits"]
    histories = [fit["history"][0] for fit in fits]
    seconds = sum(h["train_seconds"] for h in histories)
    examples = sum(h["examples"] for h in histories)
    if not math.isfinite(seconds) or seconds <= 0 or examples < 1:
        raise ValueError("C2 pilot has no finite positive training measurement")
    report = {"status": "passed", "dataset_id": dataset_id, "cell": cell, "task_type": family,
              "recipe_id": digest(recipe), "artifact_id": published["artifact_id"],
              "train_rows": len(recipe["train"]), "test_rows": len(recipe["test"]),
              "max_joint_tokens": recipe["pilot_max_joint_tokens"],
              "parts": len(fits), "optimizer_examples": examples, "training_seconds": seconds,
              "fit_seconds": sum(f["seconds"] for f in fits), "examples_per_second": examples / seconds,
              "peak_gpu_bytes": max(f["peak_gpu_bytes"] for f in fits)}
    with file_lock(folder / ".fit.lock"):
        if receipt.is_file():
            previous = json.loads(receipt.read_text(encoding="utf-8"))
            if all(previous.get(k) == v for k, v in report.items()):
                return previous
        report.update(job_id=os.environ["SLURM_JOB_ID"], created_utc=datetime.now(UTC).isoformat())
        write_json(receipt, report)
    return report


def size_jobs(pilot_report: dict) -> list[dict]:
    """Approximate same-cell, same-task hours from measured training throughput (3 fits, <= 3 epochs)."""
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries

    rate = pilot_report["examples_per_second"]
    task_type = pilot_report["task_type"]
    cell = pilot_report["cell"]
    if (pilot_report.get("status") != "passed" or pilot_report.get("parts") != 3
            or cell not in CELLS or not np.isfinite(rate) or rate <= 0):
        raise ValueError("C2 sizing requires a passing end-to-end pilot with positive, finite throughput")
    out = []
    semantics = load_yaml("dataset_semantics")["datasets"]
    for e in selected_entries(load_catalog(), load_yaml("experiment_1/main")):
        if semantics[e["id"]]["task_type"] != task_type:
            continue
        rows = e["paper_summary"]["rows"]
        examples = 0.8 * rows * 2 / 3  # outer-training rows in two parts
        # training epochs + validation + scoring held-out and test rows (forward ~ 1/3 of a step)
        seconds = 3 * (3 * examples / rate + (0.8 * rows / 3 + 0.2 * rows) / (3 * rate))
        out.append({"dataset_id": e["id"], "cell": cell, "task_type": task_type, "rows": rows,
                    "hours_per_fold_cell": seconds / 3600,
                    "slurm_time_hours": math.ceil(1.5 * seconds / 3600 + 0.5)})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--cell", choices=CELLS)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="Run one C0 Strands v1 qualification case with --execute")
    parser.add_argument("--task-type", choices=TASK_TYPES)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--size-from", type=Path, help="pilot report JSON for job sizing")
    args = parser.parse_args()
    if args.plan:
        plan = training_plan()
        print(json.dumps({"trained_fits": len(plan),
                          "lora_fits": sum(p["cell"] != "head_only" for p in plan),
                          "jobs_dataset_fold_cell": len({(p["dataset_id"], p["fold"], p["cell"]) for p in plan}),
                          "pilots": [pilot(task_type=t, cell=c) for t in TASK_TYPES for c in CELLS]}, indent=2))
    elif args.size_from:
        print(json.dumps(size_jobs(json.loads(args.size_from.read_text(encoding="utf-8"))), indent=2))
    elif args.pilot:
        print(json.dumps(pilot(execute=args.execute, dataset_id=args.dataset, cell=args.cell,
                               task_type=args.task_type), indent=2))
    else:
        print(json.dumps(run_fold(args.dataset, args.fold, args.cell, execute=args.execute), indent=2))


if __name__ == "__main__":
    main()
