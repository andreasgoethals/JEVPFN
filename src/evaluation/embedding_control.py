"""Fit per-column PCA exclusively on the partition passed to estimator.fit."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class EmbeddingControl:
    def __init__(self, estimator, groups, *, components=32, seed=20260929):
        self.estimator, self.groups = estimator, groups
        self.components, self.seed = components, seed

    def fit(self, x, y):
        self.schema = list(x.columns)
        self.pcas = []
        for group in self.groups:
            matrix = x[group].to_numpy()
            present = np.isfinite(matrix).all(axis=1)
            if ((np.isfinite(matrix).any(axis=1)) != present).any():
                raise ValueError("Partially missing embedding vector")
            count = min(self.components, int(present.sum()) - 1, len(group))
            pca = None
            if count > 0:
                pca = PCA(n_components=count, svd_solver="randomized", random_state=self.seed).fit(
                    matrix[present]
                )
            self.pcas.append(pca)
        self.estimator.fit(self.transform(x), y)
        if hasattr(self.estimator, "classes_"):
            self.classes_ = self.estimator.classes_
        return self

    def transform(self, x):
        if list(x.columns) != self.schema:
            raise ValueError("Embedding control schema changed")
        result = x.drop(columns=[c for group in self.groups for c in group]).copy()
        for i, (group, pca) in enumerate(zip(self.groups, self.pcas)):
            matrix = x[group].to_numpy()
            present = np.isfinite(matrix).all(axis=1)
            if ((np.isfinite(matrix).any(axis=1)) != present).any():
                raise ValueError("Partially missing embedding vector")
            names = [f"e5_pca__{i:03d}__{j:03d}" for j in range(self.components)]
            values = np.full((len(x), self.components), np.nan)
            if pca is not None and present.any():
                values[present, : pca.n_components_] = pca.transform(matrix[present])
            addition = pd.DataFrame(values, index=x.index, columns=names)
            from src.evaluation.features import missing_column

            name = missing_column(i)
            if name in result:
                if not np.array_equal(result[name].to_numpy(), (~present).astype(int)):
                    raise ValueError("Text presence and embedding availability disagree")
            else:
                addition[name] = (~present).astype(int)
            if set(addition) & set(result):
                raise ValueError("Embedding PCA names collide with non-text columns")
            result = result.join(addition, validate="one_to_one")
        return result

    def predict(self, x):
        return self.estimator.predict(self.transform(x))

    def predict_proba(self, x):
        return self.estimator.predict_proba(self.transform(x))
