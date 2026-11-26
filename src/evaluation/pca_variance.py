"""Read-only training explained-variance fractions of the prepared frozen-embedding PCA fits.

Kept outside `pca_cache` on purpose: `pca_cache.py` and `embedding_control.py` enter every
prepared PCA key (`pca_cache.identity`), so they must not change for a reader.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pandas as pd


def outer_variance_table(encoders=None) -> pd.DataFrame:
    """One row per dataset x encoder x outer fold x scope x text group of the OUTER-training fits.

    Keys are rebuilt exactly as `prepare_pca` builds them (shared split, text groups, encoder
    artifact), so the inner binary partitions are never mixed in. Absent stores or keys are
    skipped; the caller reports coverage. Nothing is fitted or written.
    """
    from src.data.loaders import load_catalog, load_dataset
    from src.data.selection import selected_entries
    from src.data.text_embeddings import cache_path as embedding_path
    from src.data.text_embeddings import encoder_config
    from src.evaluation.design import pca_grid
    from src.evaluation.pca_cache import cache_path, identity, read
    from src.evaluation.protocol import shared_splits
    from src.utils.config import load_yaml

    cfg = load_yaml("B2")
    encoders = list(encoders or cfg["encoders"])
    rows = []
    for entry in selected_entries(load_catalog(), cfg):
        stores = {e: cache_path(entry["id"], e) for e in encoders}
        if not any(p.is_file() for p in stores.values()):
            continue
        dataset = load_dataset(entry, allow_download=False)
        split = shared_splits(dataset, seed=cfg["random_seed"], folds=5)
        n_text = len(dataset.metadata.text_columns)
        for encoder, store in stores.items():
            config = encoder_config(encoder)
            source = embedding_path(entry["id"], config)
            if not store.is_file() or not source.is_file():
                continue
            with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                found = db.execute("SELECT value FROM metadata WHERE key='artifact_id'").fetchone()
            if found is None:
                continue
            groups = [[f"frozen_e5__text_{i:03d}__{j:03d}" for j in range(config["dimensions"])]
                      for i in range(n_text)]
            for number, fold in enumerate(split["folds"]):
                index = pd.DataFrame(index=[f"csv:{i}" for i in fold["train"]])
                for scope, components in pca_grid():
                    targets = groups if scope == "per_column" else [[c for g in groups for c in g]]
                    for group_number, group in enumerate(targets):
                        saved = read(store, identity(index, group, components, scope,
                                                     cfg["random_seed"], [found[0]]))
                        if saved is not None:
                            rows.append({"dataset_id": entry["id"], "task_type": dataset.metadata.task_type,
                                         "encoder": encoder, "outer_fold": number, "pca_scope": scope,
                                         "text_group": group_number, **saved[2]})
    return pd.DataFrame(rows)
