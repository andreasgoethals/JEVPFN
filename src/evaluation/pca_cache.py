"""CPU-prepared PCA parameters, shared across predictors; never fit on held-out rows.

One rebuildable SQLite store per dataset/encoder on PROJECT. Numeric arrays only,
no pickle. Identities contain the exact training row IDs and immutable encoder artifact.
"""

import io
import json
import sqlite3
from contextlib import closing

import numpy as np
from sklearn.decomposition import PCA

from src.utils import paths
from src.utils.serialization import digest


def cache_path(dataset_id, encoder):
    return paths.results_dir("pca", phase="experiment_2") / encoder / f"{dataset_id}.sqlite3"


def identity(x, group, components, scope, seed, artifacts):
    import sklearn

    from src.utils.provenance import code_identity

    if not artifacts or not x.index.is_unique:
        raise ValueError("PCA reuse requires immutable embedding artifacts and unique row IDs")
    return digest(
        {
            "version": 1,
            "sklearn": sklearn.__version__,
            "implementation": code_identity(
                ["src/evaluation/embedding_control.py", "src/evaluation/pca_cache.py"]
            ),
            "training_rows": list(x.index),
            "columns": group,
            "components": components,
            "scope": scope,
            "seed": seed,
            "embedding_artifacts": artifacts,
        }
    )


def read(path, key):
    if not path.is_file():
        return None
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        row = db.execute("SELECT payload, sha256 FROM parameters WHERE key=?", (key,)).fetchone()
    if row is None:
        return None
    import hashlib

    if hashlib.sha256(row[0]).hexdigest() != row[1]:
        raise ValueError("Corrupt prepared PCA parameters")
    with np.load(io.BytesIO(row[0]), allow_pickle=False) as saved:
        info = json.loads(str(saved["info"]))
        imputation = saved["imputation"] if info["joint"] else None
        pca = None
        if info["fitted_components"]:
            pca = PCA(n_components=info["fitted_components"])
            pca.components_, pca.mean_ = saved["components"], saved["mean"]
            pca.n_components_ = info["fitted_components"]
            pca.n_features_in_ = pca.components_.shape[1]
            pca.explained_variance_ = saved["variance"]
        return pca, imputation, info["diagnostics"]


def write(path, key, pca, imputation, diagnostics):
    import hashlib

    payload = io.BytesIO()
    np.savez_compressed(
        payload,
        info=json.dumps(
            {
                "joint": imputation is not None,
                "fitted_components": pca.n_components_ if pca else 0,
                "diagnostics": diagnostics,
            }
        ),
        imputation=imputation if imputation is not None else np.array([]),
        components=pca.components_ if pca else np.array([]),
        mean=pca.mean_ if pca else np.array([]),
        variance=pca.explained_variance_ if pca else np.array([]),
    )
    data = payload.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=60)) as db, db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS parameters (key TEXT PRIMARY KEY, payload BLOB, sha256 TEXT)"
        )
        db.execute(
            "INSERT OR IGNORE INTO parameters VALUES (?,?,?)",
            (key, data, hashlib.sha256(data).hexdigest()),
        )
