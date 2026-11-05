"""Fold-specific MulTaBench TAR assets. Preparation is explicit and independently cached.

Upstream training/encoding files are unmodified. This adapter supplies the reviewed
JEVPFN rows, offline checkpoint and artifact/publication boundaries.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.data.text_embeddings import checkpoint_dir, encoder_config
from src.evaluation.embedding_control import EmbeddingControl
from src.evaluation.protocol import shared_splits
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import digest, file_sha256, scalar, write_json

ENCODER = "e5_small_v2_tar"
PCA_SEED = 42  # Pinned multabench.constants.SEED; outer partitions keep the project seed.
VENDOR = Path(__file__).parent / "vendor"


def upstream():
    """Check every copied byte before importing the isolated upstream namespace."""
    receipt = json.loads((VENDOR / "SOURCE.json").read_text())
    if receipt["commit"] != load_yaml("experiment_3/tar")["upstream_commit"]:
        raise ValueError("TAR source pin differs from its approved configuration")
    for name, checksum in receipt["files"].items():
        if file_sha256(VENDOR / name) != checksum:
            raise ValueError(f"Modified MulTaBench source: {name}")
    if "multabench" in sys.modules:
        package = sys.modules["multabench"]
        if list(package.__path__) != [str(VENDOR / "multabench")]:
            raise RuntimeError("Another MulTaBench installation is already imported")
    elif str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    return receipt


def recipe(dataset, split, fold):
    cfg = load_yaml("experiment_3/tar")
    upstream()
    train = split["folds"][fold]["train"]
    return {
        "schema": "jevpfn.tar.v1",
        "dataset_id": dataset.metadata.dataset_id,
        "data_identity": dataset.source["data_identity"],
        "split_id": split["split_id"], "fold": fold,
        "train": train, "test": split["folds"][fold]["test"],
        "training_labels": digest([scalar(v) for v in dataset.frame[dataset.metadata.target].iloc[train]]),
        "text_columns": list(dataset.metadata.text_columns),
        "training_column_order": sorted(dataset.metadata.text_columns),
        "task_type": dataset.metadata.task_type,
        "upstream": json.loads((VENDOR / "SOURCE.json").read_text()),
        "checkpoint": {k: encoder_config(cfg["base_encoder"])[k] for k in ("model", "revision")},
        "adapter_source": file_sha256(Path(__file__)),
    }


def directory(dataset_id, fold):
    return paths.tar_results_dir("tar", dataset_id, f"fold_{fold}")


def read_bundle(dataset, split, fold):
    folder = directory(dataset.metadata.dataset_id, fold)
    body = json.loads((folder / "manifest.json").read_text())
    if body["recipe"] != recipe(dataset, split, fold) or body["status"] != "complete":
        raise ValueError("TAR bundle does not match this training partition")
    if body["artifact_id"] != digest({k: v for k, v in body.items() if k != "artifact_id"}):
        raise ValueError("TAR manifest checksum differs")
    required = {f"text_{i}.npy" for i in range(len(dataset.metadata.text_columns))}
    required.update({"adapter/adapter_config.json", "adapter/adapter_model.safetensors"})
    if not required.issubset(body["files"]):
        raise ValueError("TAR bundle lacks its vectors or saved adapter")
    for name, checksum in body["files"].items():
        file = (folder / name).resolve()
        if not file.is_relative_to(folder.resolve()) or file_sha256(file) != checksum:
            raise ValueError("TAR file missing, changed or outside its bundle")
    for number, _ in enumerate(dataset.metadata.text_columns):
        values = np.load(folder / f"text_{number}.npy", mmap_mode="r", allow_pickle=False)
        if (values.shape != (len(dataset.frame), 384) or not np.isfinite(values).all()
                or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4)):
            raise ValueError("TAR vectors have invalid row coverage, shape or values")
    return body


def replay_indices(raw, texts, token_lengths):
    """Bounded, label-free coverage of every column's difficult inputs and row mapping."""
    selected = set(range(min(8, len(texts)))) | {len(texts) - 1}
    masks = [raw.isna(), texts.str.strip().eq(""), texts.duplicated(keep=False),
             texts.map(lambda value: not value.isascii())]
    for mask in masks:
        selected.update(np.flatnonzero(mask.to_numpy())[:2].tolist())
    selected.add(int(np.argmax([token_lengths[value] for value in texts])))
    return sorted(selected)


def check_adapter_replay(raw, texts, folder, token_lengths, encode):
    """Compare saved row vectors against a reloaded adapter, reordered and one at a time."""
    report = {}
    for number, col in enumerate(texts):
        rows = replay_indices(raw[col], texts[col], token_lengths[col])
        probes = texts[col].iloc[rows].tolist()
        expected = np.load(folder / f"text_{number}.npy", mmap_mode="r", allow_pickle=False)[rows]
        batch = encode(probes, col, 32)
        reversed_single = encode(list(reversed(probes)), col, 1)[::-1]
        if any(values.shape != expected.shape or not np.allclose(values, expected, atol=1e-5, rtol=1e-4)
               for values in (batch, reversed_single)):
            raise ValueError(f"Reloaded TAR adapter does not reproduce row/batch order for {col}")
        report[col] = {"rows": rows, "batch_order_checked": True,
                       "max_absolute_difference": float(max(np.max(np.abs(batch - expected)),
                                                            np.max(np.abs(reversed_single - expected))))}
    return report


def prepare(dataset, fold, *, execute=False):
    cfg = load_yaml("experiment_3/tar")
    if not execute or cfg["prepare_enabled"] is not True:
        raise RuntimeError("Explicit TAR preparation authorization/configuration is required")
    from src.cluster.freeze import check_if_present
    from src.evaluation.qualification import execution_runtime

    check_if_present()
    runtime = execution_runtime()
    if (not runtime.get("container_id") or os.environ.get("JEVPFN_CONTAINER_PROVIDER") != "e5_tar"
            or not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CLUSTER_NAME") != "mindwell"):
        raise RuntimeError("TAR training requires the reviewed Mindwell allocation")
    runtime["provider"] = "e5_tar"
    split = shared_splits(dataset, seed=cfg["random_seed"], folds=5)
    folder = paths.require_project_output(directory(dataset.metadata.dataset_id, fold))
    folder.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(folder.with_suffix(".lock"), timeout=1):
        if (folder / "manifest.json").exists():
            return read_bundle(dataset, split, fold)
        if folder.exists():
            raise ValueError("Incomplete TAR directory requires inspection")
        upstream()
        import torch
        from multabench.baselines.preprocessing.feature_types import transform_feature_types
        from multabench.encoders.e5.constants import E5_LORA, format_e5_passage
        from multabench.encoders.e5.e5_finetune import encode_texts_with_e5, finetune_e5
        from multabench.finetune.lora import finetuning_split

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one CUDA GPU to the upstream Trainer")
        checkpoint = checkpoint_dir(encoder_config(cfg["base_encoder"]))
        seal = json.loads((checkpoint / "_JEVPFN_EMBEDDING.json").read_text())
        from src.data.text_embeddings import _files
        if (seal["files"] != _files(checkpoint) or any(
            seal[k] != encoder_config(cfg["base_encoder"])[k] for k in ("model", "revision")
        )):
            raise ValueError("TAR base checkpoint seal differs")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        temporary = folder.with_name(folder.name + ".incomplete_" + uuid4().hex)
        temporary.mkdir()
        columns = list(dataset.metadata.text_columns)
        texts = transform_feature_types(dataset.frame[columns], set(), set())
        train = split["folds"][fold]["train"]
        y = dataset.frame[dataset.metadata.target].iloc[train]
        classification = dataset.metadata.task_type != "regression"
        fit_x, val_x, _, _ = finetuning_split(texts.iloc[train], y, classification)
        torch.cuda.synchronize()
        start = time.perf_counter()
        model, tokenizer, stats = finetune_e5(
            texts.iloc[train], y, sorted(columns), classification, "cuda", model_name=str(checkpoint)
        )
        torch.cuda.synchronize()
        fit_seconds = time.perf_counter() - start
        model.save_pretrained(temporary / "adapter")
        encoding, truncation, token_lengths = {}, {}, {}
        for number, col in enumerate(columns):
            unique = list(dict.fromkeys(texts[col].tolist()))
            lengths = [len(tokenizer(format_e5_passage(col, v), truncation=False)["input_ids"]) for v in unique]
            token_lengths[col] = dict(zip(unique, lengths))
            truncation[col] = {"unique_inputs": len(unique), "over_512_tokens": sum(n > 512 for n in lengths),
                               "maximum_tokens": max(lengths, default=0)}
            start = time.perf_counter()
            vectors = encode_texts_with_e5(unique, col, model, tokenizer, "cuda")
            torch.cuda.synchronize()
            encoding[col] = time.perf_counter() - start
            positions = pd.Index(unique).get_indexer(texts[col])
            np.save(temporary / f"text_{number}.npy", vectors[positions], allow_pickle=False)
        # A real 0B round trip proves the saved adapter reproduces its vectors.
        from peft import PeftModel
        from transformers import AutoModel
        restored = PeftModel.from_pretrained(AutoModel.from_pretrained(str(checkpoint)), temporary / "adapter")
        restored.to("cuda").eval()
        replay = check_adapter_replay(
            dataset.frame[columns], texts, temporary, token_lengths,
            lambda values, col, batch: encode_texts_with_e5(values, col, restored, tokenizer, "cuda", batch_size=batch),
        )
        body = {
            "adapter_roundtrip_checked": True,
            "adapter_replay": replay,
            "status": "complete", "recipe": recipe(dataset, split, fold),
            "runtime": runtime, "job_id": os.environ["SLURM_JOB_ID"],
            "optimizer_rows": fit_x.index.tolist(), "early_stopping_rows": val_x.index.tolist(),
            "lora": E5_LORA.to_dict(), "stats": stats,
            "fit_seconds": fit_seconds, "encode_seconds": encoding, "truncation": truncation,
            "gpu": torch.cuda.get_device_name(),
            "files": {p.relative_to(temporary).as_posix(): file_sha256(p)
                      for p in temporary.rglob("*") if p.is_file()},
        }
        body["artifact_id"] = digest(body)
        write_json(temporary / "manifest.json", body)
        temporary.rename(folder)
        return read_bundle(dataset, split, fold)


def vector_frame(dataset, fold, indices):
    folder = directory(dataset.metadata.dataset_id, fold)
    groups, arrays = [], []
    for i, _ in enumerate(dataset.metadata.text_columns):
        groups.append([f"tar__{i:03d}__{j:03d}" for j in range(384)])
        arrays.append(np.load(folder / f"text_{i}.npy", mmap_mode="r", allow_pickle=False)[indices])
    frame = pd.DataFrame(np.concatenate(arrays, axis=1), index=[f"csv:{i}" for i in indices],
                         columns=[c for group in groups for c in group])
    return frame, groups


def pca_path(dataset_id):
    return paths.tar_results_dir("tar_pca", dataset_id + ".sqlite3")


def prepare_pca(dataset, fold):
    from src.evaluation.design import pca_grid
    split = shared_splits(dataset, seed=load_yaml("experiment_3/tar")["random_seed"], folds=5)
    body = read_bundle(dataset, split, fold)
    vectors, groups = vector_frame(dataset, fold, split["folds"][fold]["train"])
    for scope, components in pca_grid():
        EmbeddingControl(None, groups, components=components, scope=scope, seed=PCA_SEED,
                         prepared_cache=pca_path(dataset.metadata.dataset_id), prepare=True,
                         embedding_artifacts=[body["artifact_id"]], encoded_missing=True, standardize=True).fit(vectors, None)


def validate_dataset(dataset):
    """0B requires all outer adapters, their real runtime evidence and both PCA recipes."""
    from src.evaluation import pca_cache
    from src.evaluation.design import pca_grid

    split = shared_splits(dataset, seed=load_yaml("experiment_3/tar")["random_seed"], folds=5)
    artifacts, keys = [], []
    for fold in range(5):
        body = read_bundle(dataset, split, fold)
        if not body.get("job_id") or not body.get("runtime", {}).get("container_id") or body.get("adapter_roundtrip_checked") is not True:
            raise ValueError("TAR requires sealed runtime and real training evidence")
        if body["runtime"].get("provider") != "e5_tar":
            raise ValueError("TAR requires its dedicated e5_tar runtime")
        replay = body.get("adapter_replay", {})
        if set(replay) != set(dataset.metadata.text_columns) or any(
            not v.get("rows") or v.get("batch_order_checked") is not True
            or not np.isfinite(v.get("max_absolute_difference", float("nan")))
            or not 0 <= v["max_absolute_difference"] <= 1.1e-4
            or len(set(v["rows"])) != len(v["rows"])
            or any(type(i) is not int or not 0 <= i < len(dataset.frame) for i in v["rows"])
            for v in replay.values()
        ):
            raise ValueError("TAR requires per-column row/batch replay evidence")
        train = split["folds"][fold]["train"]
        optimizer, validation = body["optimizer_rows"], body["early_stopping_rows"]
        if set(optimizer) & set(validation) or sorted(optimizer + validation) != sorted(train):
            raise ValueError("TAR early stopping rows escape or overlap the training partition")
        groups = [[f"tar__{i:03d}__{j:03d}" for j in range(384)]
                  for i in range(len(dataset.metadata.text_columns))]
        rows = pd.DataFrame(index=[f"csv:{i}" for i in train])
        for scope, components in pca_grid():
            for group in groups if scope == "per_column" else [[c for g in groups for c in g]]:
                key = pca_cache.identity(rows, group, components, scope, PCA_SEED, [body["artifact_id"]], standardize=True)
                if pca_cache.read(pca_path(dataset.metadata.dataset_id), key) is None:
                    raise ValueError("Missing TAR training-only PCA parameters")
                keys.append(key)
        artifacts.append(body["artifact_id"])
    return {"dataset_id": dataset.metadata.dataset_id, "encoder_id": ENCODER,
            "artifact_ids": artifacts, "prepared_pca_keys": keys, "status": "passed"}


class TarControl:
    """Reuse one outer-fold adapter across predictors without fitting it on inner labels."""

    def __init__(self, estimator, dataset, split, bundles, arm):
        self.estimator, self.dataset, self.split = estimator, dataset, split
        self.bundles, self.arm = bundles, arm

    def _features(self, x):
        indices = [int(i.removeprefix("csv:")) for i in x.index]
        allowed = self.split["folds"][self.fold]
        if not set(indices).issubset(set(allowed["train"]) | set(allowed["test"])):
            raise ValueError("TAR request outside its outer partition")
        vectors, groups = vector_frame(self.dataset, self.fold, indices)
        return x.join(vectors, validate="one_to_one"), groups

    def fit(self, x, y):
        matches = [i for i, f in enumerate(self.split["folds"])
                   if x.index.tolist() == [f"csv:{r}" for r in f["train"]]]
        if len(matches) != 1:
            raise ValueError("TAR must use the exact outer training rows, with no inner refits")
        self.fold = matches[0]
        vectors, groups = self._features(x)
        self.control = EmbeddingControl(
            self.estimator, groups, components=self.arm["pca_components"], scope=self.arm["pca_scope"],
            seed=PCA_SEED, prepared_cache=pca_path(self.dataset.metadata.dataset_id),
            embedding_artifacts=[self.bundles[self.fold]["artifact_id"]], encoded_missing=True, standardize=True,
        ).fit(vectors, y)
        self.output_feature_count_ = self.control.output_feature_count_
        self.pca_diagnostics_ = self.control.pca_diagnostics_
        if hasattr(self.control, "classes_"):
            self.classes_ = self.control.classes_
        return self

    def predict(self, x):
        return self.control.predict(self._features(x)[0])

    def predict_proba(self, x):
        return self.control.predict_proba(self._features(x)[0])


def main():
    from src.data.loaders import load_catalog, load_dataset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"dataset": args.dataset, "fold": args.fold, "execution": False}))
        return
    dataset = load_dataset(next(e for e in load_catalog()["datasets"] if e["id"] == args.dataset),
                           allow_download=False)
    if args.pca:
        prepare_pca(dataset, args.fold)
    else:
        report = prepare(dataset, args.fold, execute=True)
        print(json.dumps({"status": "tar_prepared", "dataset": args.dataset,
                          "fold": args.fold, "artifact_id": report["artifact_id"]}))


if __name__ == "__main__":
    main()
