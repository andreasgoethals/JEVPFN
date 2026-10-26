"""Fit per-column or joint PCA exclusively on the partition passed to estimator.fit."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class EmbeddingControl:
    def __init__(
        self,
        estimator,
        groups,
        *,
        components=32,
        seed=20260929,
        scope="per_column",
        prepared_cache=None,
        embedding_artifacts=None,
        prepare=False,
    ):
        self.estimator, self.groups = estimator, groups
        self.components, self.seed = components, seed
        if scope not in {"per_column", "joint"} or type(components) is not int or components < 1:
            raise ValueError("Invalid PCA scope or component count")
        self.scope = scope
        self.prepared_cache, self.embedding_artifacts = prepared_cache, embedding_artifacts
        self.prepare = prepare

    def fit(self, x, y):
        self.schema = list(x.columns)
        for group in self.groups:
            finite = np.isfinite(x[group].to_numpy(dtype=float))
            if not np.array_equal(finite.any(axis=1), finite.all(axis=1)):
                raise ValueError("Partially missing embedding vector")
        self.projected_groups = (
            self.groups
            if self.scope == "per_column"
            else [[c for group in self.groups for c in group]]
        )
        self.imputation_ = []
        self.pcas = []
        self.pca_diagnostics_ = []
        for group in self.projected_groups:
            key = None
            if self.prepared_cache is not None:
                from src.evaluation import pca_cache

                key = pca_cache.identity(
                    x, group, self.components, self.scope, self.seed, self.embedding_artifacts
                )
                cached = pca_cache.read(self.prepared_cache, key)
                if cached is not None:
                    pca, imputation, diagnostics = cached
                    self.pcas.append(pca)
                    self.imputation_.append(imputation)
                    self.pca_diagnostics_.append(diagnostics)
                    continue
                if not self.prepare:
                    raise ValueError(
                        "Missing CPU-prepared PCA; run src.evaluation.prepare_pca first"
                    )
            matrix = x[group].to_numpy(dtype=float)
            if self.scope == "joint":
                # A missing text block is imputed using TRAINING coordinates only.
                # Keep text-presence indicators outside PCA; no non-text feature enters it.
                count_finite = np.isfinite(matrix).sum(axis=0)
                mean = np.divide(
                    np.nansum(matrix, axis=0),
                    count_finite,
                    out=np.zeros(matrix.shape[1]),
                    where=count_finite > 0,
                )
                self.imputation_.append(mean)
                present = np.isfinite(matrix).any(axis=1)
                matrix = np.where(np.isfinite(matrix), matrix, mean)
            else:
                self.imputation_.append(None)
                present = np.isfinite(matrix).all(axis=1)
            if self.scope == "per_column" and ((np.isfinite(matrix).any(axis=1)) != present).any():
                raise ValueError("Partially missing embedding vector")
            count = min(self.components, int(present.sum()) - 1, len(group))
            pca = None
            if count > 0:
                pca = PCA(n_components=count, svd_solver="randomized", random_state=self.seed).fit(
                    matrix[present]
                )
            self.pcas.append(pca)
            ratio = float(pca.explained_variance_ratio_.sum()) if pca is not None else float("nan")
            self.pca_diagnostics_.append(
                {
                    "requested_components": self.components,
                    "fitted_components": count if pca is not None else 0,
                    "training_nonempty_rows": int(present.sum()),
                    "training_explained_variance_fraction": ratio if np.isfinite(ratio) else None,
                }
            )
            if key is not None:
                pca_cache.write(
                    self.prepared_cache, key, pca, self.imputation_[-1], self.pca_diagnostics_[-1]
                )
        if self.prepare:
            return self  # CPU preparation does not fit a predictor or transform the entire dataset.
        transformed = self.transform(x)
        self.output_feature_count_ = transformed.shape[1]
        self.estimator.fit(transformed, y)
        if hasattr(self.estimator, "classes_"):
            self.classes_ = self.estimator.classes_
        return self

    def transform(self, x):
        if list(x.columns) != self.schema:
            raise ValueError("Embedding control schema changed")
        result = x.drop(columns=[c for group in self.groups for c in group]).copy()
        for i, group in enumerate(self.groups):
            values = x[group].to_numpy(dtype=float)
            if not np.array_equal(np.isfinite(values).all(axis=1), np.isfinite(values).any(axis=1)):
                raise ValueError("Partially missing embedding vector")
            from src.evaluation.features import missing_column

            name = missing_column(i)
            missing = (~np.isfinite(values).all(axis=1)).astype(int)
            if name in result and not np.array_equal(result[name].to_numpy(), missing):
                raise ValueError("Text presence and embedding availability disagree")
            result[name] = missing
        for i, (group, pca) in enumerate(zip(self.projected_groups, self.pcas)):
            matrix = x[group].to_numpy(dtype=float)
            if self.scope == "joint":
                present = np.isfinite(matrix).any(axis=1)
                matrix = np.where(np.isfinite(matrix), matrix, self.imputation_[i])
            else:
                present = np.isfinite(matrix).all(axis=1)
            if self.scope == "per_column" and ((np.isfinite(matrix).any(axis=1)) != present).any():
                raise ValueError("Partially missing embedding vector")
            names = [f"e5_pca__{i:03d}__{j:03d}" for j in range(self.components)]
            values = np.full((len(x), self.components), np.nan)
            if pca is not None and present.any():
                values[present, : pca.n_components_] = pca.transform(matrix[present])
            addition = pd.DataFrame(values, index=x.index, columns=names)
            if set(addition) & set(result):
                raise ValueError("Embedding PCA names collide with non-text columns")
            result = result.join(addition, validate="one_to_one")
        return result

    def predict(self, x):
        return self.estimator.predict(self.transform(x))

    def predict_proba(self, x):
        return self.estimator.predict_proba(self.transform(x))
