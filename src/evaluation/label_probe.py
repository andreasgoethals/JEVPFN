"""B2b label-trained classifier on frozen embeddings ("supervised probe"), cross-fitted.

For one encoder and one view (each text column separately, or all text columns jointly) an
L2-regularised logistic regression is trained on the full standardized frozen embedding to
predict the target. Its output width equals the decision features: binary one probability
(the canonical-last positive class), multiclass C probabilities, regression nine
probabilities over equal-frequency target bins defined on the training rows.

Everything uses training rows only:

* inner CV selects regularisation strength by log loss, with stratification when feasible
  for classification and training-only quantiles in each regression split;
* each training row receives the prediction of a model fitted on the other four of five
  cross-fitting parts (unstratified, so neither the model nor the partition depends on the
  row's own label); that model's regression bin edges and C use only its own four parts;
* every other row receives the prediction of one model fitted on all training rows (bins and
  C from all training rows).

Missing text stays missing (all probe outputs NaN for that block); the joint view imputes an
absent column with its training mean when at least one text column of the row is present.
Blocks too small for cross-fitting receive a fixed uniform distribution on present text.
Within a cross-fit part with a single training class, its training prior is sufficient.

Full-size arms read features prepared once per dataset/encoder/outer fold on a CPU allocation
(`python -m src.evaluation.label_probe --dataset D --encoder E`); B4 small-sample arms fit the
probe inline on the sampled rows. Numeric arrays only, no pickle.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.serialization import canonical_json, digest, scalar

VERSION = "supervised-probe-v3"  # Label-independent sparse fallback and fold-local tuning bins.
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
INNER_FOLDS = 3
CROSSFIT_FOLDS = 5
REGRESSION_BINS = 9
MAX_ITER = 1000  # lbfgs; the library default (100) often stops before convergence at 1,024 dims.
VIEWS = ("per_column", "joint")


def output_width(task_type, classes):
    if task_type == "binary_classification":
        return 1
    if task_type == "regression":
        return REGRESSION_BINS
    return len(classes)


def regression_bins(y_train):
    """Equal-frequency cut points of the training targets (8 interior quantiles)."""
    y = np.asarray(y_train, dtype=float)
    return np.quantile(y, np.arange(1, REGRESSION_BINS) / REGRESSION_BINS)


def encode_target(y, task_type, classes, edges=None):
    if task_type == "regression":
        return np.searchsorted(edges, np.asarray(y, dtype=float), side="right")
    mapping = {canonical_json(c): i for i, c in enumerate(classes)}
    return np.array([mapping[canonical_json(scalar(v))] for v in y])


def _splitter(labels, folds, seed):
    counts = np.bincount(labels)
    if counts[counts > 0].min() >= folds:
        return StratifiedKFold(folds, shuffle=True, random_state=seed)
    return KFold(folds, shuffle=True, random_state=seed)


def _pipeline(c, seed):
    return make_pipeline(
        SimpleImputer(strategy="mean", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=c, max_iter=MAX_ITER, random_state=seed),
    )


class _Prior:
    """Training class frequencies for a partition that cannot support a classifier."""

    def __init__(self, labels, n_outputs):
        counts = np.bincount(labels, minlength=n_outputs).astype(float)
        self.p = counts / counts.sum()

    def predict(self, x):
        return np.tile(self.p, (len(x), 1))


def _fit(x, labels, n_outputs, c, seed):
    if len(np.unique(labels)) < 2:
        return _Prior(labels, n_outputs)
    return _pipeline(c, seed).fit(x, labels)


def _predict(model, x, n_outputs):
    if isinstance(model, _Prior):
        return model.predict(x)
    full = np.zeros((len(x), n_outputs))
    full[:, model.classes_] = model.predict_proba(x)
    return full


def _log_loss(labels, p):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(labels)), labels], 1e-15, 1))))


class SupervisedProbe:
    """Cross-fitted training features and a full-training model for every other row."""

    def __init__(self, groups, *, view, task_type, classes, seed, c_grid=C_GRID,
                 inner_folds=INNER_FOLDS, crossfit_folds=CROSSFIT_FOLDS):
        if view not in VIEWS:
            raise ValueError("Unknown probe view")
        self.groups = [list(g) for g in groups]
        self.view, self.task_type, self.classes = view, task_type, list(classes)
        self.seed, self.c_grid = seed, tuple(c_grid)
        self.inner_folds, self.crossfit_folds = inner_folds, crossfit_folds
        self.n_outputs = REGRESSION_BINS if task_type == "regression" else len(self.classes)

    def _blocks(self):
        if self.view == "per_column":
            return [(f"text_{i:03d}", group) for i, group in enumerate(self.groups)]
        return [("all_text", [c for g in self.groups for c in g])]

    def _present(self, x, columns):
        finite = np.isfinite(x[columns].to_numpy(dtype=float))
        return finite.any(axis=1) if self.view == "joint" else finite.all(axis=1)

    def _output(self, name, p):
        if self.task_type == "binary_classification":
            p = p[:, -1:]
        return {f"probe__{name}__p_{j:02d}": p[:, j] for j in range(p.shape[1])}

    def _choose_c(self, matrix, labels, targets, stream):
        """C with the lowest mean inner log loss on these rows; a tie keeps the stronger
        regularisation."""
        losses = {}
        regression = self.task_type == "regression"
        inner = (KFold(self.inner_folds, shuffle=True, random_state=stream) if regression
                 else _splitter(labels, self.inner_folds, stream))
        partitions = []
        for a, t in inner.split(matrix, labels):
            if regression:
                edges = regression_bins(targets[a])
                fit_labels = encode_target(targets[a], self.task_type, self.classes, edges)
                validation_labels = encode_target(targets[t], self.task_type, self.classes, edges)
            else:
                fit_labels, validation_labels = labels[a], labels[t]
            partitions.append((a, t, fit_labels, validation_labels))
        for c in self.c_grid:
            scores = []
            for a, t, fit_labels, validation_labels in partitions:
                model = _fit(matrix[a], fit_labels, self.n_outputs, c, self.seed)
                scores.append(_log_loss(validation_labels, _predict(model, matrix[t], self.n_outputs)))
            losses[c] = float(np.mean(scores))
        return min(self.c_grid, key=lambda c: (losses[c], c)), losses

    def fit_transform(self, x, y):
        """Cross-fitted features for the training rows; keeps full-training models."""
        if self.task_type == "regression":
            self.edges_ = regression_bins(y)
        labels_all = encode_target(y, self.task_type, self.classes, getattr(self, "edges_", None))
        self.models_, self.diagnostics_ = [], []
        out = {}
        for b, (name, columns) in enumerate(self._blocks()):
            present = self._present(x, columns)
            matrix = x.loc[present, columns].to_numpy(dtype=float)
            labels = labels_all[present]
            values = np.full((len(x), self.n_outputs), np.nan)
            n = len(labels)
            diagnostic = {"block": name, "training_rows": int(n), "fallback": None}
            if n < max(self.crossfit_folds, self.inner_folds) * 2:
                model = _Prior(np.arange(self.n_outputs), self.n_outputs) if n else None
                diagnostic["fallback"] = "uniform_label_independent" if n else "no_training_text"
                if n:
                    values[present] = model.predict(matrix)
                self.models_.append(model)
                diagnostic["selected_C"] = None
            else:
                stream = self.seed + 7919 * b
                targets = np.asarray(y, dtype=object)[present]
                chosen, losses = self._choose_c(matrix, labels, targets, stream)
                crossfit = np.full((n, self.n_outputs), np.nan)
                part_c = []
                # Unstratified parts: which rows share a part never depends on a row's own label.
                # Fully cross-fitted: each part's regression bins and C use only the rows that
                # part is trained on, never the rows it scores.
                parts = KFold(self.crossfit_folds, shuffle=True, random_state=stream + 1)
                for k, (a, t) in enumerate(parts.split(matrix)):
                    edges = regression_bins(targets[a]) if self.task_type == "regression" else None
                    part_labels = encode_target(targets[a], self.task_type, self.classes, edges)
                    if len(np.unique(part_labels)) < 2:
                        model, c = _Prior(part_labels, self.n_outputs), None
                    else:
                        c, _ = self._choose_c(matrix[a], part_labels, targets[a], stream + 101 * (k + 1))
                        model = _fit(matrix[a], part_labels, self.n_outputs, c, self.seed)
                    part_c.append(c)
                    crossfit[t] = _predict(model, matrix[t], self.n_outputs)
                values[present] = crossfit
                self.models_.append(_fit(matrix, labels, self.n_outputs, chosen, self.seed))
                diagnostic.update(selected_C=chosen, inner_log_loss={str(k): v for k, v in losses.items()},
                                  crossfit_selected_C=part_c)
            self.diagnostics_.append(diagnostic)
            out.update(self._output(name, values))
        self.training_rows_ = set(x.index)
        return pd.DataFrame(out, index=x.index)

    def transform(self, x):
        """Full-training-model features for rows outside the training partition."""
        if set(x.index) & self.training_rows_:
            raise ValueError("Training rows receive cross-fitted probe features only")
        out = {}
        for (name, columns), model in zip(self._blocks(), self.models_, strict=True):
            present = self._present(x, columns)
            values = np.full((len(x), self.n_outputs), np.nan)
            if model is not None and present.any():
                values[present] = _predict(model, x.loc[present, columns].to_numpy(dtype=float),
                                           self.n_outputs)
            out.update(self._output(name, values))
        return pd.DataFrame(out, index=x.index)


def identity(train_index, y, groups, view, task_type, seed, artifacts):
    import sklearn

    from src.utils.provenance import code_identity

    if not artifacts:
        raise ValueError("Probe reuse requires immutable embedding artifacts")
    return digest({
        "version": VERSION,
        "sklearn": sklearn.__version__,
        "implementation": code_identity(["src/evaluation/label_probe.py"]),
        "training_rows": list(train_index),
        "targets_sha256": digest([canonical_json(scalar(v)) for v in y]),
        "columns": groups, "view": view, "task_type": task_type, "seed": seed,
        "c_grid": list(C_GRID), "inner_folds": INNER_FOLDS, "crossfit_folds": CROSSFIT_FOLDS,
        "embedding_artifacts": artifacts,
    })


def cache_path(dataset_id, encoder):
    from src.utils import paths

    return paths.results_dir("supervised_probe", phase="experiment_3") / encoder / f"{dataset_id}.sqlite3"


def write(path, key, table, diagnostics):
    payload = io.BytesIO()
    np.savez_compressed(payload, rows=np.array(table.index, dtype=str),
                        columns=np.array(table.columns, dtype=str), values=table.to_numpy(float),
                        diagnostics=json.dumps(diagnostics))
    data = payload.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=60)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS features (key TEXT PRIMARY KEY, payload BLOB, sha256 TEXT)")
        db.execute("INSERT OR IGNORE INTO features VALUES (?,?,?)",
                   (key, data, hashlib.sha256(data).hexdigest()))


def read(path, key):
    if not path.is_file():
        return None
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        row = db.execute("SELECT payload, sha256 FROM features WHERE key=?", (key,)).fetchone()
    if row is None:
        return None
    if hashlib.sha256(row[0]).hexdigest() != row[1]:
        raise ValueError("Corrupt prepared probe features")
    with np.load(io.BytesIO(row[0]), allow_pickle=False) as saved:
        table = pd.DataFrame(saved["values"], index=list(saved["rows"]), columns=list(saved["columns"]))
        return table, json.loads(str(saved["diagnostics"]))


class ProbeControl:
    """Replace raw embedding columns by probe features, then fit the fixed predictor.

    With `prepared_cache`, features prepared on CPU for this exact training partition are read
    (training rows cross-fitted, other rows from the full-training model); otherwise the probe
    is fitted inline on the supplied training rows (B4 small samples).
    """

    def __init__(self, estimator, groups, *, view, task_type, classes, seed,
                 prepared_cache=None, embedding_artifacts=None, prepare=False):
        self.estimator, self.groups = estimator, groups
        self.probe = SupervisedProbe(groups, view=view, task_type=task_type, classes=classes,
                                     seed=seed)
        self.prepared_cache, self.embedding_artifacts = prepared_cache, embedding_artifacts
        self.prepare = prepare
        self.view, self.task_type, self.seed = view, task_type, seed

    def _replace(self, x, probe):
        result = x.drop(columns=[c for g in self.groups for c in g])
        return result.join(probe.loc[x.index], validate="one_to_one")

    def fit(self, x, y, *, all_rows=None):
        self.schema = list(x.columns)
        self.table_ = None
        if self.prepared_cache is not None:
            key = identity(x.index, y, self.groups, self.view, self.task_type, self.seed,
                           self.embedding_artifacts)
            cached = read(self.prepared_cache, key)
            if cached is None:
                if not self.prepare:
                    raise ValueError("Missing CPU-prepared probe features; run src.evaluation.label_probe")
                train = self.probe.fit_transform(x, y)
                rest = all_rows.drop(index=x.index)
                table = pd.concat([train, self.probe.transform(rest)])
                write(self.prepared_cache, key, table, self.probe.diagnostics_)
                return self
            if self.prepare:  # Resumed CPU preparation: this partition is already stored.
                return self
            self.table_, self.probe_diagnostics_ = cached
            training = self.table_.loc[x.index]
        else:
            training = self.probe.fit_transform(x, y)
            self.probe_diagnostics_ = self.probe.diagnostics_
        self.training_rows_ = set(x.index)
        transformed = self._replace(x, training)
        self.output_feature_count_ = transformed.shape[1]
        self.estimator.fit(transformed, y)
        if hasattr(self.estimator, "classes_"):
            self.classes_ = self.estimator.classes_
        return self

    def transform(self, x):
        if list(x.columns) != self.schema:
            raise ValueError("Probe control schema changed")
        if set(x.index) & self.training_rows_:
            raise ValueError("Probe features for training rows are cross-fitted at fit time only")
        probe = self.table_.loc[x.index] if self.table_ is not None else self.probe.transform(x)
        return self._replace(x, probe)

    def predict(self, x):
        return self.estimator.predict(self.transform(x))

    def predict_proba(self, x):
        return self.estimator.predict_proba(self.transform(x))


def run(dataset_id, encoder):
    """Prepare all five outer-training partitions of one dataset/encoder for both views."""
    from src.cluster.freeze import check_if_present
    from src.data.loaders import load_catalog, load_dataset
    from src.data.selection import selected_entries
    from src.data.text_embeddings import encoder_config, load
    from src.evaluation.protocol import shared_splits
    from src.utils import paths
    from src.utils.config import load_yaml

    if paths.on_vsc() and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Probe preparation needs a CPU allocation")
    check_if_present()
    cfg = load_yaml("experiment_3/tar")
    entry = next((e for e in selected_entries(load_catalog(), cfg) if e["id"] == dataset_id), None)
    if entry is None:
        raise ValueError("Unselected dataset")
    dataset = load_dataset(entry, allow_download=False)
    vectors = load(dataset, encoder_config(encoder))
    target = pd.Series(dataset.frame[dataset.metadata.target].to_numpy(), index=vectors.index)
    split = shared_splits(dataset, seed=cfg["random_seed"], folds=5)
    for number, fold in enumerate(split["folds"]):
        train = vectors.iloc[fold["train"]]
        for view in VIEWS:
            ProbeControl(None, vectors.attrs["embedding_groups"], view=view,
                         task_type=dataset.metadata.task_type, classes=dataset.metadata.classes,
                         seed=cfg["random_seed"], prepared_cache=cache_path(dataset_id, encoder),
                         embedding_artifacts=vectors.attrs["feature_artifact_ids"], prepare=True,
                         ).fit(train, target.iloc[fold["train"]], all_rows=vectors)
        print(f"{dataset_id}/{encoder}: outer fold {number} prepared", flush=True)


def verify_prepared(dataset, encoder, artifact_id):
    """Read both probe views for every exact outer fold and verify their rows and probabilities."""
    from src.evaluation.protocol import shared_splits
    from src.utils.config import load_yaml
    from src.utils.serialization import text_value

    seed = load_yaml("experiment_3/tar")["random_seed"]
    split = shared_splits(dataset, seed=seed)
    rows = pd.Index([f"csv:{i}" for i in range(len(dataset.frame))])
    task = dataset.metadata.task_type
    targets = dataset.frame[dataset.metadata.target]
    columns = list(dataset.metadata.text_columns)
    groups = [[f"frozen_e5__text_{i:03d}__{j:03d}" for j in range(encoder["dimensions"])]
              for i in range(len(columns))]
    present = [dataset.frame[col].map(text_value).notna().to_numpy() for col in columns]
    width = output_width(task, dataset.metadata.classes)
    path = cache_path(dataset.metadata.dataset_id, encoder["encoder_id"])
    keys = []
    for fold in split["folds"]:
        train = fold["train"]
        for view in VIEWS:
            key = identity(rows[train], targets.iloc[train], groups, view, task, seed, [artifact_id])
            cached = read(path, key)
            if cached is None:
                raise ValueError(f"Missing prepared {view} supervised probe partition")
            table, _ = cached
            blocks = ([(f"text_{i:03d}", mask) for i, mask in enumerate(present)]
                      if view == "per_column" else [("all_text", np.any(present, axis=0))])
            names = [f"probe__{name}__p_{j:02d}" for name, _ in blocks for j in range(width)]
            if (not table.index.is_unique or set(table.index) != set(rows)
                    or list(table.columns) != names):
                raise ValueError("Prepared probe rows or probability columns differ")
            values = table.loc[rows].to_numpy(dtype=float)
            for i, (_, mask) in enumerate(blocks):
                block = values[:, i * width:(i + 1) * width]
                if not mask[train].any():
                    # A field absent from every training row has no fitted probe.
                    if not np.isnan(block).all():
                        raise ValueError("Prepared probe supplies predictions without training text")
                    continue
                observed = block[mask]
                if (not np.isnan(block[~mask]).all() or not np.isfinite(observed).all()
                        or np.any((observed < 0) | (observed > 1))
                        or (width > 1 and not np.allclose(observed.sum(axis=1), 1, atol=1e-6, rtol=0))):
                    raise ValueError("Prepared probe probabilities or missingness differ")
            keys.append(key)
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--encoder", required=True)
    args = parser.parse_args()
    run(args.dataset, args.encoder)


if __name__ == "__main__":
    main()
