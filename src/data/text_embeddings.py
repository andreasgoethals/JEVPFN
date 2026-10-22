"""Frozen text embeddings, deduplicated and resumable in one SQLite file per dataset.

Encoding is label-free. PCA is deliberately absent here: it belongs inside CV fits.
No weights are downloaded or loaded without an explicit CLI action.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sqlite3
import zlib
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loaders import load_catalog, load_dataset
from src.data.selection import selected_entries
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.events import append_event
from src.utils.locking import file_lock
from src.utils.provenance import code_identity
from src.utils.serialization import canonical_json, digest, file_sha256, write_json
from src.utils.timing import TimingJournal

SCHEMA = "jevpfn.frozen-text.v1"
# Stable storage namespaces must also work when importing into a new checkout.
MODEL_NAMESPACES = {
    "intfloat/e5-small-v2": "e5_small_v2",
    "intfloat/e5-large-v2": "e5_large_v2",
    "Qwen/Qwen3-Embedding-0.6B": "qwen3_embedding_0_6b",
}


def encoder_config(encoder_id="e5_small_v2"):
    catalog = load_yaml("feature_creation/embedding_models")["models"]
    if encoder_id not in catalog:
        raise ValueError("Unknown embedding encoder")
    return (
        load_yaml("feature_creation/text_embeddings")
        | catalog[encoder_id]
        | {"encoder_id": encoder_id}
    )


def encoder_id_for_model(model):
    if model not in MODEL_NAMESPACES:
        raise ValueError("Unknown embedding model in cache")
    return MODEL_NAMESPACES[model]


def cache_path(dataset_id, cfg=None):
    encoder_id = (cfg or {}).get("encoder_id", "e5_small_v2")
    root = paths.jev_cache_path("text_embeddings")
    # Preserve the existing small-E5 location; other encoders cannot overwrite it.
    if encoder_id != "e5_small_v2":
        if encoder_id not in MODEL_NAMESPACES.values():
            raise ValueError("Unknown embedding cache namespace")
        root = root / encoder_id
    return root / f"{paths.output_component(dataset_id)}.sqlite3"


def checkpoint_dir(cfg=None):
    return paths.checkpoints_dir(
        "text_embeddings", (cfg or {"model": "intfloat/e5-small-v2"})["model"].split("/")[-1]
    )


def _files(folder):
    return {
        p.relative_to(folder).as_posix(): file_sha256(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file()
        and ".cache" not in p.relative_to(folder).parts
        and p.name != "_JEVPFN_EMBEDDING.json"
    }


def stage(cfg):
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Stage embeddings on a connected setup host before acquiring a GPU")
    # Hub/Xet constants are fixed on first import; keep download caches on PROJECT too.
    os.environ["HF_HOME"] = str(paths.checkpoints_dir("huggingface"))
    os.environ["HF_HUB_CACHE"] = str(paths.checkpoints_dir("huggingface", "hub"))
    os.environ["HF_XET_CACHE"] = str(paths.checkpoints_dir("huggingface", "xet"))
    from huggingface_hub import snapshot_download

    folder = paths.require_project_output(checkpoint_dir(cfg))
    marker = folder / "_JEVPFN_EMBEDDING.json"
    if marker.exists():
        manifest = json.loads(marker.read_text(encoding="utf-8"))
        if manifest != {
            "model": cfg["model"],
            "revision": cfg["revision"],
            "files": _files(folder),
        }:
            raise ValueError(
                "Existing embedding checkpoint differs; preserve it before reviewing a new pin"
            )
        return manifest
    snapshot_download(
        cfg["model"],
        revision=cfg["revision"],
        local_dir=folder,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer*",
            "vocab.txt",
            "merges.txt",
            "vocab.json",
            "special_tokens_map.json",
        ],
    )
    manifest = {"model": cfg["model"], "revision": cfg["revision"], "files": _files(folder)}
    if not (folder / "model.safetensors").is_file():
        raise ValueError("Embedding checkpoint weights were not downloaded")
    write_json(folder / "_JEVPFN_EMBEDDING.json", manifest)
    return manifest


def token_chunks(tokenizer, text, *, prefix, max_tokens):
    """Cover every content token exactly once; repeat only the required encoder prefix."""
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    content = tokenizer.encode(text, add_special_tokens=False)
    capacity = max_tokens - len(prefix_ids) - tokenizer.num_special_tokens_to_add(False)
    if capacity < 1:
        raise ValueError("No room for embedding text")
    return prefix_ids, [content[i : i + capacity] for i in range(0, len(content), capacity)] or [[]]


def model_input_chunks(tokenizer, text, *, prefix, max_tokens):
    """Chunk token encodings, then apply the publisher's Rust special-token processor.

    Transformers 5 tokenizers do not expose the legacy prepare_for_model API. Never
    decode/re-encode chunks: a chunk can end inside a multi-token Unicode character.
    """
    from tokenizers import Encoding

    backend = tokenizer.backend_tokenizer
    backend.no_padding()
    backend.no_truncation()
    prefix_encoding = backend.encode(prefix, add_special_tokens=False)
    content = backend.encode(text, add_special_tokens=False)
    capacity = max_tokens - len(prefix_encoding.ids) - backend.num_special_tokens_to_add(False)
    if capacity < 1:
        raise ValueError("No room for embedding text")
    if len(content.ids) > capacity:
        # All overflow pieces are retained, with no overlap or discarded content.
        content.truncate(capacity, stride=0, direction="right")
    for chunk in [content, *content.overflowing]:
        combined = Encoding.merge([prefix_encoding, chunk], growing_offsets=False)
        prepared = backend.post_process(combined, add_special_tokens=True)
        if len(prepared.ids) > max_tokens:
            raise ValueError("Embedding input exceeds its context; never truncate")
        inputs = {"input_ids": prepared.ids, "attention_mask": prepared.attention_mask}
        if "token_type_ids" in tokenizer.model_input_names:
            inputs["token_type_ids"] = prepared.type_ids
        yield inputs, max(1, len(chunk.ids))


def tokenizer_contract(tokenizer, *, prefix, max_tokens):
    """Check actual input IDs/masks against the publisher's ordinary tokenizer call."""
    samples = ("x", "A short sentence.", "Unicode: café 中文.", "<|endoftext|> literal")
    for text in samples:
        direct = tokenizer(prefix + text, truncation=False, padding=False)
        chunks = list(model_input_chunks(tokenizer, text, prefix=prefix, max_tokens=max_tokens))
        if len(chunks) != 1 or any(direct[key] != values for key, values in chunks[0][0].items()):
            raise ValueError("Chunk preparation differs from the publisher's tokenizer")
    return {
        "publisher_input_parity": True,
        "probe_count": len(samples),
        "method": "backend_post_process_v1",
    }


class E5Encoder:
    """Publisher-specific E5 mean/Qwen last-token pooling; all text covered in chunks."""

    def __init__(self, cfg):
        from src.cluster.freeze import check_if_present
        from src.cluster.scheduling import reject_a100_allocation

        reject_a100_allocation()
        check_if_present()
        expected = {
            "intfloat/e5-small-v2": (384, "attention_mask_mean", "query: "),
            "intfloat/e5-large-v2": (1024, "attention_mask_mean", "query: "),
            "Qwen/Qwen3-Embedding-0.6B": (1024, "last_nonpadding_token", ""),
        }
        if (cfg.get("dimensions"), cfg.get("pooling"), cfg.get("prefix")) != expected.get(
            cfg.get("model")
        ):
            raise ValueError(
                "Encoder dimensions, pooling and prefix must match its publisher recipe"
            )
        if (
            cfg["dimensions"] not in (384, 1024)
            or not 4
            <= cfg["max_tokens"]
            <= (32768 if cfg["pooling"] == "last_nonpadding_token" else 512)
            or cfg["batch_size"] < 1
            or cfg["pooling"] not in ("attention_mask_mean", "last_nonpadding_token")
            or cfg["long_text"] != "nonoverlapping_token_chunks_weighted_by_content_tokens"
            or cfg["normalization"] != "l2_per_chunk_then_l2_weighted_mean"
        ):
            raise ValueError(
                "Unsupported frozen encoding recipe; configuration must match implemented behavior"
            )
        import torch
        from transformers import AutoModel, AutoTokenizer

        folder = checkpoint_dir(cfg)
        manifest = json.loads((folder / "_JEVPFN_EMBEDDING.json").read_text())
        if manifest != {
            "model": cfg["model"],
            "revision": cfg["revision"],
            "files": _files(folder),
        }:
            raise ValueError("Embedding checkpoint differs from its pinned manifest")
        if paths.on_vsc() and not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Embedding inference on VSC requires a compute allocation")
        self.cfg, self.torch = cfg, torch
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        if cfg["pooling"] == "last_nonpadding_token":
            self.tokenizer.padding_side = "left"
        self.tokenizer_contract = tokenizer_contract(
            self.tokenizer, prefix=cfg["prefix"], max_tokens=cfg["max_tokens"]
        )
        self.timings = TimingJournal(
            "embeddings", phase=os.environ.get("JEVPFN_PHASE", "feature_creation"),
            context={"model": cfg["model"], "revision": cfg["revision"],
                     "device": cfg["device"], "dimensions": cfg["dimensions"]},
        )
        self.synchronize = torch.cuda.synchronize if cfg["device"].startswith("cuda") else None
        try:
            with self.timings.measure("model_loading", synchronize=self.synchronize, items=0):
                self.model = (
                    AutoModel.from_pretrained(folder, local_files_only=True).to(cfg["device"]).eval()
                )
        finally:
            self.timings.write_summary()
        self.load_seconds = self.timings.summary()["stages"][0]["wall_seconds"]
        self.identity = {
            "batching": "cross_text_chunks_v2_publisher_postprocessor",
            "tokenizer_contract": self.tokenizer_contract,
            "checkpoint_sha256": digest(manifest),
            "runtime": {
                p: importlib.metadata.version(p) for p in ("torch", "transformers", "tokenizers")
            },
            "device": cfg["device"],
            "dtype": str(next(self.model.parameters()).dtype),
        }
        from src.cluster.containers import active_identity

        image = active_identity()
        if image is not None:
            self.identity["execution_container"] = image

    def __call__(self, texts):
        torch, cfg = self.torch, self.cfg
        totals = np.zeros((len(texts), cfg["dimensions"]), dtype=np.float64)
        pending = []

        def flush():
            inputs = self.tokenizer.pad(
                [prepared for _, prepared, _ in pending],
                padding=True,
                return_tensors="pt",
            ).to(cfg["device"])
            if inputs["input_ids"].shape[1] > cfg["max_tokens"]:
                raise ValueError("Embedding input exceeds its context; never truncate")
            with torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                if cfg["pooling"] == "last_nonpadding_token":
                    positions = torch.arange(hidden.shape[1], device=hidden.device).expand(
                        hidden.shape[0], -1
                    )
                    last = (
                        positions.masked_fill(inputs["attention_mask"] == 0, -1).max(dim=1).values
                    )
                    pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
                else:
                    pooled = (hidden * mask).sum(1) / mask.sum(1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
            for (row, _, content_tokens), vector in zip(pending, pooled):
                totals[row] += vector * content_tokens
            pending.clear()

        for row, text in enumerate(texts):
            chunks = model_input_chunks(
                self.tokenizer, text, prefix=cfg["prefix"], max_tokens=cfg["max_tokens"]
            )
            for prepared, content_tokens in chunks:
                pending.append((row, prepared, content_tokens))
                if len(pending) == cfg["batch_size"]:
                    flush()
        if pending:
            flush()
        lengths = np.linalg.norm(totals, axis=1)
        if not np.isfinite(lengths).all() or (lengths == 0).any():
            raise ValueError("Invalid embedding norm")
        return (totals / lengths[:, None]).astype("<f4")


def input_rows(dataset):
    """Only official text columns are read; identical text shares one encoder result."""
    unique, columns = {}, []
    for column in dataset.metadata.text_columns:
        keys = []
        for value in dataset.frame[column]:
            text = None if pd.isna(value) else str(value)
            key = None if text is None or not text.strip() else digest(text)
            if key is not None:
                unique[key] = text
            keys.append(key)
        encoded = zlib.compress(canonical_json(keys).encode("utf-8"))
        columns.append((column, encoded, hashlib.sha256(encoded).hexdigest()))
    return unique, columns


def _connect(path, *, readonly=False):
    db = sqlite3.connect(
        Path(path).resolve().as_uri() + ("?mode=ro" if readonly else "?mode=rwc"), uri=True
    )
    db.execute("PRAGMA trusted_schema=OFF")
    return db


def build(dataset, cfg, encoder, *, destination=None):
    """Explicit caller owns inference permission; partial completed batches survive interruption."""
    destination = Path(destination or cache_path(dataset.metadata.dataset_id, cfg))
    unique, columns = input_rows(dataset)
    timings = getattr(encoder, "timings", None) or TimingJournal(
        "embeddings", phase=os.environ.get("JEVPFN_PHASE", "feature_creation")
    )
    dataset_id = dataset.metadata.dataset_id
    recipe = {
        "schema": SCHEMA,
        "dataset_id": dataset.metadata.dataset_id,
        "data_identity": dataset.source["data_identity"],
        "rows": len(dataset.frame),
        "text_columns": list(dataset.metadata.text_columns),
        "column_hashes": [v[2] for v in columns],
        "encoder": {
            k: cfg[k]
            for k in (
                "model",
                "revision",
                "dimensions",
                "prefix",
                "max_tokens",
                "batch_size",
                "pooling",
                "long_text",
                "normalization",
            )
        },
        "runtime": encoder.identity,
        "code_sha256": code_identity(["src/data/text_embeddings.py"]),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    log = paths.logs_dir("feature_creation") / "text_embeddings.log"
    with file_lock(destination.with_suffix(".lock")), closing(_connect(destination)) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS columns (position INTEGER PRIMARY KEY, name TEXT NOT NULL, row_keys BLOB NOT NULL, sha256 TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, text TEXT NOT NULL, vector BLOB NOT NULL, sha256 TEXT NOT NULL);
        """)
        saved = db.execute("SELECT value FROM metadata WHERE key='recipe'").fetchone()
        if saved and saved[0] != canonical_json(recipe):
            raise ValueError(
                "Embedding cache recipe changed; preserve it and review before replacement"
            )
        if not saved:
            with db:
                db.execute("INSERT INTO metadata VALUES ('recipe', ?)", (canonical_json(recipe),))
                db.executemany(
                    "INSERT INTO columns VALUES (?,?,?,?)", [(i, *v) for i, v in enumerate(columns)]
                )
        validate(destination)
        existing = {v[0] for v in db.execute("SELECT key FROM embeddings")}
        pending = [(k, v) for k, v in unique.items() if k not in existing]
        if existing:
            timings.record("cache_reuse", dataset_id=dataset_id, items=len(existing))
        append_event(
            log,
            "embedding_started",
            dataset_id=recipe["dataset_id"],
            unique=len(unique),
            pending=len(pending),
        )
        for start in range(0, len(pending), cfg["batch_size"]):
            batch = pending[start : start + cfg["batch_size"]]
            with timings.measure(
                "embedding_batch", dataset_id=dataset_id, item_keys=[k for k, _ in batch],
                items=len(batch), synchronize=getattr(encoder, "synchronize", None),
            ):
                vectors = np.asarray(encoder([v for _, v in batch]), dtype="<f4")
                if vectors.shape != (len(batch), cfg["dimensions"]) or not np.isfinite(vectors).all():
                    raise ValueError("Invalid embedding matrix")
            with db:
                for (key, text), vector in zip(batch, vectors):
                    blob = vector.tobytes()
                    db.execute(
                        "INSERT INTO embeddings VALUES (?,?,?,?)",
                        (key, text, blob, hashlib.sha256(blob).hexdigest()),
                    )
            append_event(
                log,
                "embedding_progress",
                dataset_id=recipe["dataset_id"],
                completed=len(existing) + start + len(batch),
                total=len(unique),
            )
        identity = digest(
            {
                "recipe": recipe,
                "vectors": list(db.execute("SELECT key,sha256 FROM embeddings ORDER BY key")),
            }
        )
        with db:
            db.execute("INSERT OR REPLACE INTO metadata VALUES ('artifact_id', ?)", (identity,))
    validate(destination, require_complete=True)
    append_event(log, "embedding_completed", dataset_id=recipe["dataset_id"], artifact_id=identity,
                 timing=timings.write_summary())
    return identity


def validate(path, *, require_complete=False):
    """Reject corrupt vectors, row links, recipes and falsely completed caches."""
    with closing(_connect(path, readonly=True)) as db:
        objects = db.execute(
            "SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
        if set(objects) != {("table", "metadata"), ("table", "columns"), ("table", "embeddings")}:
            raise ValueError("Unexpected objects in embedding database")
        if db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("Embedding SQLite integrity failure")
        meta = dict(db.execute("SELECT key,value FROM metadata"))
        recipe = json.loads(meta["recipe"])
        if recipe["schema"] != SCHEMA or recipe["rows"] < 1:
            raise ValueError("Unknown embedding recipe")
        linked = set()
        columns = list(
            db.execute("SELECT position,name,row_keys,sha256 FROM columns ORDER BY position")
        )
        if [v[0] for v in columns] != list(range(len(recipe["text_columns"]))):
            raise ValueError("Missing embedding text column")
        for i, name, blob, sha in columns:
            keys = json.loads(zlib.decompress(blob))
            if (
                name != recipe["text_columns"][i]
                or len(keys) != recipe["rows"]
                or hashlib.sha256(blob).hexdigest() != sha
                or sha != recipe["column_hashes"][i]
            ):
                raise ValueError("Embedding row mapping mismatch")
            linked.update(k for k in keys if k is not None)
        observed, hashes = set(), []
        for key, text, vector, sha in db.execute(
            "SELECT key,text,vector,sha256 FROM embeddings ORDER BY key"
        ):
            values = np.frombuffer(vector, dtype="<f4")
            if (
                key != digest(text)
                or key not in linked
                or not text.strip()
                or values.shape != (recipe["encoder"]["dimensions"],)
                or not np.isfinite(values).all()
                or hashlib.sha256(vector).hexdigest() != sha
            ):
                raise ValueError("Embedding content mismatch")
            observed.add(key)
            hashes.append((key, sha))
        if (require_complete or "artifact_id" in meta) and (
            observed != linked
            or meta.get("artifact_id") != digest({"recipe": recipe, "vectors": hashes})
        ):
            raise ValueError("Incomplete or corrupt embedding artifact")
        return recipe


def _validated_dataset_rows(dataset, cfg, source):
    source = Path(source or cache_path(dataset.metadata.dataset_id, cfg))
    recipe = validate(source, require_complete=True)
    _, columns = input_rows(dataset)
    expected_encoder = {k: cfg[k] for k in recipe["encoder"]}
    if (
        recipe["dataset_id"] != dataset.metadata.dataset_id
        or recipe["data_identity"] != dataset.source["data_identity"]
        or recipe["column_hashes"] != [v[2] for v in columns]
        or recipe["encoder"] != expected_encoder
    ):
        raise ValueError("Embedding cache does not match the current dataset/encoder")
    return recipe, columns


def validate_dataset(dataset, cfg, *, source=None):
    """Validate the complete artifact against raw rows without expanding dense row vectors."""
    return _validated_dataset_rows(dataset, cfg, source)[0]


def load(dataset, cfg, *, source=None):
    source = Path(source or cache_path(dataset.metadata.dataset_id, cfg))
    _, columns = _validated_dataset_rows(dataset, cfg, source)
    matrices, groups = [], []
    with closing(_connect(source, readonly=True)) as db:
        vectors = {
            k: np.frombuffer(v, dtype="<f4")
            for k, v in db.execute("SELECT key,vector FROM embeddings")
        }
        for i, (_, blob, _) in enumerate(columns):
            names = [f"frozen_e5__text_{i:03d}__{j:03d}" for j in range(cfg["dimensions"])]
            keys = json.loads(zlib.decompress(blob))
            matrix = np.full((len(keys), cfg["dimensions"]), np.nan, dtype="float32")
            for row, key in enumerate(keys):
                if key is not None:
                    matrix[row] = vectors[key]
            matrices.append(
                pd.DataFrame(matrix, columns=names, index=[f"csv:{i}" for i in range(len(keys))])
            )
            groups.append(names)
        artifact = db.execute("SELECT value FROM metadata WHERE key='artifact_id'").fetchone()[0]
    result = pd.concat(matrices, axis=1)
    result.attrs.update(feature_artifact_ids=[artifact], embedding_groups=groups)
    return result


def merge(source, destination):
    """Merge compatible partial stores transactionally; conflicting vectors are never replaced."""
    recipe = validate(source)
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("Embedding import destination must not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(destination.with_suffix(".lock")):
        if not destination.exists():
            temporary = destination.with_suffix(".partial")
            with (
                closing(_connect(source, readonly=True)) as incoming,
                closing(_connect(temporary)) as target,
            ):
                incoming.backup(target)
            temporary.replace(destination)
            return {"dataset_id": recipe["dataset_id"], "status": "copied"}
        if validate(destination) != recipe:
            raise ValueError("Embedding recipes conflict")
        with (
            closing(_connect(source, readonly=True)) as incoming,
            closing(_connect(destination)) as target,
            target,
        ):
            for row in incoming.execute("SELECT key,text,vector,sha256 FROM embeddings"):
                prior = target.execute(
                    "SELECT key,text,vector,sha256 FROM embeddings WHERE key=?", (row[0],)
                ).fetchone()
                if prior is not None and prior != row:
                    raise ValueError("Conflicting embedding vector; transaction rolled back")
                target.execute("INSERT OR IGNORE INTO embeddings VALUES (?,?,?,?)", row)
            marker = incoming.execute(
                "SELECT value FROM metadata WHERE key='artifact_id'"
            ).fetchone()
            if marker:
                target.execute("INSERT OR REPLACE INTO metadata VALUES ('artifact_id',?)", marker)
    validate(destination)
    return {"dataset_id": recipe["dataset_id"], "status": "merged"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument(
        "--encoder",
        default="e5_small_v2",
        choices=tuple(load_yaml("feature_creation/embedding_models")["models"]),
    )
    args = parser.parse_args()
    cfg = encoder_config(args.encoder)
    if args.stage:
        print(json.dumps(stage(cfg), indent=2))
    elif args.execute:
        if cfg["enabled"] is not True:
            raise RuntimeError("Frozen embedding creation remains disabled in configuration")
        entries = [
            e
            for e in selected_entries(
                load_catalog(), load_yaml("feature_creation/feature_creation")
            )
            if args.dataset is None or args.dataset == e["id"]
        ]
        if not entries:
            raise ValueError("No selected dataset matches")
        encoder = E5Encoder(cfg)
        try:
            for entry in entries:
                print(
                    entry["id"],
                    build(load_dataset(entry, allow_download=False), cfg, encoder),
                    flush=True,
                )
        finally:
            encoder.timings.write_summary()
    else:
        print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
