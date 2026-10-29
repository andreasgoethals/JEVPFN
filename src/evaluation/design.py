"""Reviewed PCA grid for the frozen-embedding comparison."""

from src.utils.config import load_yaml


def pca_grid():
    grid = load_yaml("experiment_2/embeddings")["pca_grid"]
    expected = {"per_column": [32], "joint": [256, 128, 64, 32, 16, 8]}
    if grid != expected:
        raise ValueError("PCA grid differs from the reviewed per-column32/joint256-to8 design")
    return [(scope, k) for scope, dimensions in grid.items() for k in dimensions]
