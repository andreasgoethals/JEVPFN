"""Group C, C2: learning from labels with Strands 2B (exp10_adaptation, internal phase rq3).

A 2 x 2 design on the JOINT question with probability features only:

| torso \\ head | released head | trained head |
|---|---|---|
| released torso | zero-shot (= B1 Strands joint, no new run) | head-only (cached C1 vectors) |
| LoRA-trained torso | LoRA only (head frozen) | LoRA + head |

Every trained cell starts from Strands' released adapter and head and uses Strands' own
published recipe (`config/exp10_adaptation/adaptation.yaml`, each value with its source line
in the pinned snapshot). Targets are the true class; regression uses nine equal-frequency bins
of the fit's own training rows, mapped lowest-first to Strands' nine ordered levels.

Cross-fitting: inside each outer training fold the rows are split into three parts by a hash
of (seed, dataset, split, fold, row) only, never by label. Fit k trains on the two other parts
(minus a label-independent early-stopping validation split: 10%, at most 1,000 rows, at most
three epochs) and scores part k; test rows get the mean of the three fits' probabilities. The
same parts serve all trained cells. Empty joint text has no request: its features stay missing.
Training rows thus never receive a probability from a model that saw their label.

GPU entry points (B200; ``scripts/slurm/c2_adaptation.slurm``): ``--dataset D --fold F --cell C
--execute`` trains the three cross-fitted fits of one cell and publishes the fold bundle;
``--pilot`` times one LoRA fit on the smallest dataset. ``--plan`` prints fits and jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import canonical_json, digest, file_sha256, scalar, write_json

SCHEMA = "jevpfn.strands_adaptation.v1"
CONFIG = "rq3/adaptation"
CELLS = ("head_only", "lora_only", "lora_head")
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


def assemble_fold(n_rows: int, train_rows, test_rows, assignment, fit_probabilities,
                  fit_training_rows) -> np.ndarray:
    """Training rows: the fit that held out their part. Test rows: mean of all fits.

    ``fit_probabilities[k]`` is an ``n_rows x K`` array (NaN where fit k produced no score);
    ``fit_training_rows[k]`` the rows fit k trained or validated on. Refuses any training row
    scored by a fit that saw it, and any test row a fit trained on.
    """
    parts = len(fit_probabilities)
    width = fit_probabilities[0].shape[1]
    out = np.full((n_rows, width), np.nan)
    train_rows, test_rows = np.asarray(train_rows), np.asarray(test_rows)
    for k in range(parts):
        seen = set(np.asarray(fit_training_rows[k]).tolist())
        held = train_rows[assignment == k]
        if seen & set(held.tolist()) or seen & set(test_rows.tolist()):
            raise ValueError("A cross-fit model saw a row it scores")
        out[held] = fit_probabilities[k][held]
    stacked = np.stack([p[test_rows] for p in fit_probabilities])
    present = np.isfinite(stacked).all(axis=2)
    if not (present.all(axis=0) | ~present.any(axis=0)).all():
        raise ValueError("Test rows must be scored by every fit or by none (empty text)")
    with np.errstate(invalid="ignore"):
        out[test_rows] = stacked.mean(axis=0)
    return out


def feature_columns(task_type: str, probabilities: np.ndarray, cell: str) -> dict:
    """B1's joint widths: binary P(true), multiclass C, regression nine level probabilities."""
    if task_type == "binary_classification":
        return {f"adapted__{cell}__joint__p_true": probabilities[:, 1]}
    return {f"adapted__{cell}__joint__p_{j:02d}": probabilities[:, j] for j in range(probabilities.shape[1])}


# ------------------------------------------------------------------ plan


def smallest_dataset() -> str:
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries

    entries = selected_entries(load_catalog(), load_yaml("experiment_1/main"))
    return min(entries, key=lambda e: (e["paper_summary"]["rows"], e["id"]))["id"]


def training_plan() -> list[dict]:
    """Every trained fit: dataset x outer fold x cross-fit part x trained cell (810)."""
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries

    cfg = config()
    return [dict(dataset_id=e["id"], fold=fold, part=part, cell=cell,
                 device="b200", execution_enabled=bool(cfg["training_enabled"]))
            for e in selected_entries(load_catalog(), load_yaml("experiment_1/main"))
            for fold in range(5) for part in range(cfg["crossfit"]["parts"]) for cell in CELLS]


# ------------------------------------------------------------------ bundles


def bundle_dir(dataset_id: str, fold: int, cell: str) -> Path:
    if cell not in CELLS:
        raise ValueError(f"Unknown trained cell {cell}")
    return paths.group_c_dir("rq3", "adaptation", paths.output_component(dataset_id), f"fold_{fold}", cell)


def fold_recipe(dataset, split, fold: int, cell: str) -> dict:
    cfg = config()
    train = split["folds"][fold]["train"]
    return {
        "schema": SCHEMA, "dataset_id": dataset.metadata.dataset_id,
        "data_identity": dataset.source["data_identity"], "split_id": split["split_id"],
        "fold": fold, "cell": cell, "train": train, "test": split["folds"][fold]["test"],
        "training_labels": digest([scalar(v) for v in dataset.frame[dataset.metadata.target].iloc[train]]),
        "task_type": dataset.metadata.task_type, "crossfit": cfg["crossfit"],
        "early_stopping": cfg["early_stopping"], "recipe": cfg["recipe"],
        "regression_bins": cfg["regression_bins"], "code_sha256": file_sha256(Path(__file__)),
    }


def read_bundle(dataset, split, fold: int, cell: str) -> dict:
    """The verified fold bundle: probabilities for every row (NaN = empty joint text)."""
    folder = bundle_dir(dataset.metadata.dataset_id, fold, cell)
    body = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if body["recipe"] != fold_recipe(dataset, split, fold, cell) or body["status"] != "complete":
        raise ValueError("Adapted bundle does not match this outer partition and recipe")
    if body["artifact_id"] != digest({k: v for k, v in body.items() if k != "artifact_id"}):
        raise ValueError("Adapted bundle manifest checksum differs")
    for name, checksum in body["files"].items():
        file = (folder / name).resolve()
        if not file.is_relative_to(folder.resolve()) or file_sha256(file) != checksum:
            raise ValueError("Adapted bundle file missing or changed")
    values = np.load(folder / "probabilities.npy", allow_pickle=False)
    if values.shape[0] != len(dataset.frame):
        raise ValueError("Adapted probabilities have another row count")
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
            self.model.config.temperature, self.model.config.temperature_by_kind = 1.0, {}

    def _check_adapter(self):
        cfg = self.model.config
        if sorted(cfg.lora_targets) != sorted(self.recipe["lora_targets"]):
            raise ValueError("Released adapter targets differ from the published recipe")
        if cfg.lora_alpha / cfg.lora_r != self.recipe["lora_alpha"] / self.recipe["lora_r"]:
            raise ValueError("Released adapter scale differs from the recipe's alpha / r")
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
        logits = self.head(decide, torch.tensor(options, device="cuda"))
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
            n_options = {"noul": 2}.get(kind) or len(question.criteria)
            order = (option_order(kind, n_options, rng, self.recipe["reverse_score_prob"])
                     if train and self.recipe["shuffle_options"] else None)
            rendered = prompting.render_question(question, option_order=order)
            state = backend.encode(prompting.render_state(request.state), add_special_tokens=True)
            asked = backend.encode(rendered.text, add_special_tokens=False)
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
                         temperature=1.0)
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
              timing_only=False) -> dict:
    """Train fit ``part`` of one cell; returns per-row probabilities for held-out and test rows."""
    cfg = config()
    recipe = recipe_values(cfg)
    torch = __import__("torch")
    meta = dataset.metadata
    _, joint = _joint_inputs(meta.dataset_id)
    folds = split["folds"][fold]
    train_rows, test_rows = np.asarray(folds["train"]), np.asarray(folds["test"])
    assignment = crossfit_parts(train_rows, dataset_id=meta.dataset_id, split_id=split["split_id"],
                                fold=fold, seed=cfg["crossfit"]["seed"])
    fit_rows, held_out = fit_partition(train_rows, assignment, part)
    fit_rows = fit_rows[joint[fit_rows] >= 0]  # empty joint text: no request, no example
    validation = validation_rows(fit_rows, seed=cfg["crossfit"]["seed"],
                                 key=f"{meta.dataset_id}|{fold}|{part}",
                                 fraction=cfg["early_stopping"]["validation_fraction"],
                                 cap=cfg["early_stopping"]["validation_cap"])
    optimizer_rows = np.setdiff1d(fit_rows, validation)
    targets, kind, n_options, edges = _targets(dataset, fit_rows, fit_rows, cfg)
    target_of = dict(zip(fit_rows.tolist(), targets.tolist()))
    scored = np.concatenate([held_out, test_rows])
    scored = scored[joint[scored] >= 0]
    fit = _Fit(cell, dataset, cfg, seed=seed)
    rng = random.Random(seed)
    if cell == "head_only":
        from src.jev.strands_hidden import read_manifest, read_training_vectors, store_dir

        hidden = read_manifest(meta.dataset_id)["recipe"]["hidden_size"]
        needed = {int(joint[r]) for r in np.concatenate([fit_rows, scored])}
        cache = read_training_vectors(store_dir(meta.dataset_id) / "hidden.sqlite3", needed, hidden)

        def forward(rows, train):
            items = [cache[int(joint[r])] for r in rows]
            wanted = [target_of[int(r)] for r in rows] if train is not None else None
            log_probs, loss = fit.head_batch(items, wanted, kind)
            return log_probs, [None] * len(rows), loss
    else:
        from src.jev.inputs import iter_prepared_inputs

        needed = {int(joint[r]) for r in np.concatenate([fit_rows, scored])}
        payloads = {rec["input_no"]: rec["payload"] for rec in iter_prepared_inputs(
            paths.dataset_question_path(meta.dataset_id)) if rec["input_no"] in needed}

        def forward(rows, train):
            wanted = [target_of[int(r)] for r in rows] if train is not None else None
            log_probs, orders, loss = fit.prompt_batch([payloads[int(joint[r])] for r in rows], wanted,
                                                       kind, train=bool(train), rng=rng)
            return log_probs, orders, loss

    micro, accum = recipe["micro_batch_size"], recipe["grad_accum"]
    epochs = int(max_epochs or cfg["early_stopping"]["max_epochs"])
    steps_per_epoch = max(1, (len(optimizer_rows) // micro) // accum)
    optim, schedule = fit.optimizer(steps_per_epoch * epochs)
    best, best_loss, history, started = None, math.inf, [], time.perf_counter()
    order_rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        if fit.model is not None:
            fit.model.train()
        fit.head.train()
        permutation = order_rng.permutation(optimizer_rows)
        batches = [permutation[i:i + micro] for i in range(0, len(permutation) - micro + 1, micro)]
        epoch_started = time.perf_counter()
        for step, batch in enumerate(batches[:steps_per_epoch * accum]):
            _, _, loss = forward(batch, True)
            (loss / accum).backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(fit.parameters(), recipe["max_grad_norm"])
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
        history.append({"epoch": epoch + 1, "validation_loss": loss, "train_seconds": train_seconds,
                        "examples": len(batches[:steps_per_epoch * accum]) * micro})
        if loss < best_loss:
            best_loss = loss
            best = {id(p): p.detach().clone() for p in fit.parameters()}
        elif cfg["early_stopping"]["patience_epochs"] == 0:
            break
        if timing_only:
            break
    with torch.no_grad():
        for p in fit.parameters():
            p.copy_(best[id(p)])
    probabilities = np.full((len(dataset.frame), n_options), np.nan)
    if not timing_only:
        if fit.model is not None:
            fit.model.eval()
        fit.head.eval()
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


def run_fold(dataset_id: str, fold: int, cell: str, *, execute=False) -> dict:
    """Train (or resume) the three cross-fitted fits of one cell and publish the fold bundle."""
    cfg = config()
    if not execute or cfg["training_enabled"] is not True:
        raise RuntimeError("C2 training is disabled; enable rq3/adaptation training and pass --execute")
    if not paths.on_vsc() or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("C2 training runs only inside a VSC B200 allocation")
    from src.cluster.scheduling import reject_a100_allocation
    from src.data.loaders import load_catalog, load_dataset
    from src.evaluation.protocol import shared_splits

    reject_a100_allocation()
    entry = next(e for e in load_catalog()["datasets"] if e["id"] == dataset_id)
    dataset = load_dataset(entry, allow_download=False)
    split = shared_splits(dataset, seed=load_yaml("experiment_1/main")["random_seed"], folds=5)
    folder = bundle_dir(dataset_id, fold, cell)
    folder.mkdir(parents=True, exist_ok=True)
    recipe = fold_recipe(dataset, split, fold, cell)
    fits = []
    for part in range(cfg["crossfit"]["parts"]):
        target = folder / f"fit_{part}.npy"
        report = folder / f"fit_{part}.json"
        if target.exists() and report.exists():
            stored = json.loads(report.read_text(encoding="utf-8"))
            if stored["recipe"] != recipe or stored["sha256"] != file_sha256(target):
                raise ValueError("A stored fit belongs to another recipe or changed")
        else:
            result = train_fit(dataset, split, fold, part, cell, seed=recipe_values(cfg)["seed"])
            np.save(folder / f"fit_{part}.partial.npy", result.pop("probabilities"), allow_pickle=False)
            os.replace(folder / f"fit_{part}.partial.npy", target)
            write_json(report, {"recipe": recipe, **result, "sha256": file_sha256(target),
                                "job_id": os.environ.get("SLURM_JOB_ID"),
                                "created_utc": datetime.now(UTC).isoformat()})
        fits.append(target)
    reports = [json.loads((folder / f"fit_{k}.json").read_text(encoding="utf-8")) for k in range(len(fits))]
    probabilities = [np.load(f, allow_pickle=False) for f in fits]
    train_rows = np.asarray(split["folds"][fold]["train"])
    assignment = crossfit_parts(train_rows, dataset_id=dataset_id, split_id=split["split_id"], fold=fold,
                                seed=cfg["crossfit"]["seed"])
    values = assemble_fold(len(dataset.frame), train_rows, split["folds"][fold]["test"], assignment,
                           probabilities, [r["optimizer_rows"] + r["validation_rows"] for r in reports])
    np.save(folder / "probabilities.npy", values, allow_pickle=False)
    names = ["probabilities.npy", *(f.name for f in fits), *(f"fit_{k}.json" for k in range(len(fits)))]
    body = {"schema": SCHEMA, "status": "complete", "recipe": recipe,
            "files": {name: file_sha256(folder / name) for name in names},
            "fits": [{k: r[k] for k in ("part", "best_validation_loss", "seconds", "peak_gpu_bytes", "history")}
                     for r in reports], "created_utc": datetime.now(UTC).isoformat()}
    body["artifact_id"] = digest(body)
    write_json(folder / "manifest.json", body)
    return {"dataset_id": dataset_id, "fold": fold, "cell": cell, "artifact_id": body["artifact_id"]}


def pilot(*, execute=False, dataset_id=None, cell=None) -> dict:
    """Time ONE LoRA fit (fold 0, part 0) on the smallest dataset; size the Slurm jobs from it."""
    cfg = config()
    spec = cfg["pilot"]
    dataset_id = dataset_id or smallest_dataset()
    cell = cell or spec["cell"]
    if not execute:
        return {"dataset_id": dataset_id, "fold": spec["fold"], "part": spec["part"], "cell": cell,
                "execution": False}
    if not paths.on_vsc() or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("The C2 pilot runs only inside a VSC B200 allocation")
    from src.cluster.scheduling import reject_a100_allocation
    from src.data.loaders import load_catalog, load_dataset
    from src.evaluation.protocol import shared_splits

    reject_a100_allocation()
    entry = next(e for e in load_catalog()["datasets"] if e["id"] == dataset_id)
    dataset = load_dataset(entry, allow_download=False)
    split = shared_splits(dataset, seed=load_yaml("experiment_1/main")["random_seed"], folds=5)
    result = train_fit(dataset, split, spec["fold"], spec["part"], cell, seed=recipe_values(cfg)["seed"],
                       max_epochs=1, timing_only=True)
    history = result["history"][0]
    report = {"dataset_id": dataset_id, "cell": cell, "rows": len(dataset.frame),
              "optimizer_examples": len(result["optimizer_rows"]),
              "seconds_per_epoch": history["train_seconds"],
              "examples_per_second": history["examples"] / max(history["train_seconds"], 1e-9),
              "peak_gpu_bytes": result["peak_gpu_bytes"], "job_id": os.environ.get("SLURM_JOB_ID"),
              "created_utc": datetime.now(UTC).isoformat()}
    write_json(paths.manifests_dir("rq3") / "adaptation_pilot" / f"{dataset_id}_{cell}.json", report)
    return report


def size_jobs(pilot_report: dict) -> list[dict]:
    """Hours per (dataset, fold, cell) job from one pilot's examples/second (3 fits, <= 3 epochs)."""
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries

    rate = pilot_report["examples_per_second"]
    out = []
    for e in selected_entries(load_catalog(), load_yaml("experiment_1/main")):
        rows = e["paper_summary"]["rows"]
        examples = 0.8 * rows * 2 / 3  # outer-training rows in two parts
        # training epochs + validation + scoring held-out and test rows (forward ~ 1/3 of a step)
        seconds = 3 * (3 * examples / rate + (0.8 * rows / 3 + 0.2 * rows) / (3 * rate))
        out.append({"dataset_id": e["id"], "rows": rows, "hours_per_fold_cell": seconds / 3600,
                    "slurm_time_hours": math.ceil(1.5 * seconds / 3600 + 0.5)})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--cell", choices=CELLS)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--size-from", type=Path, help="pilot report JSON for job sizing")
    args = parser.parse_args()
    if args.plan:
        plan = training_plan()
        print(json.dumps({"trained_fits": len(plan),
                          "lora_fits": sum(p["cell"] != "head_only" for p in plan),
                          "jobs_dataset_fold_cell": len({(p["dataset_id"], p["fold"], p["cell"]) for p in plan}),
                          "pilot": pilot()}, indent=2))
    elif args.size_from:
        print(json.dumps(size_jobs(json.loads(args.size_from.read_text(encoding="utf-8"))), indent=2))
    elif args.pilot:
        print(json.dumps(pilot(execute=args.execute, dataset_id=args.dataset, cell=args.cell), indent=2))
    else:
        print(json.dumps(run_fold(args.dataset, args.fold, args.cell, execute=args.execute), indent=2))


if __name__ == "__main__":
    main()
