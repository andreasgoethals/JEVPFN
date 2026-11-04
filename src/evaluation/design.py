"""Reviewed PCA grid for the frozen-embedding comparison."""

from src.utils.config import load_yaml


def pca_grid():
    grid = load_yaml("experiment_2/embeddings")["pca_grid"]
    expected = {"per_column": [32], "joint": [64]}
    if grid != expected:
        raise ValueError("PCA grid differs from the reviewed per-column PCA32 and joint PCA64 design")
    return [(scope, k) for scope, dimensions in grid.items() for k in dimensions]
