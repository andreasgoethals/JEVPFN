"""Bounded, label-free geometry of aligned texts; no model loading or inference.

Pairwise correlations are descriptive: shared endpoints make pairs dependent.
Any future uncertainty calculation must resample texts/datasets, not pairs.
"""

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


def encoder_distances(vectors):
    """Cosine distances in raw encoder space, in scipy condensed pair order."""
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("At least two aligned encoder vectors are required")
    if not np.isfinite(x).all() or (np.linalg.norm(x, axis=1) == 0).any():
        raise ValueError("Encoder vectors must be finite and nonzero")
    return np.clip(pdist(x, metric="cosine"), 0, 2)


def decision_distances(probabilities):
    """Total variation for aligned class/level vectors; binary accepts p-positive."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim == 1:
        p = np.column_stack((1 - p, p))
    if p.ndim != 2 or len(p) < 2 or p.shape[1] < 2:
        raise ValueError("At least two probability vectors are required")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(
            p.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Invalid probability vectors")
    return pdist(p, metric="cityblock") / 2


def distance_agreement(encoder, decision):
    """Describe paired distances, without an invalid independent-pair p-value.

    Caller must enforce identical ordered input identities before constructing both
    vectors. Missing outputs are a coverage error, not permission to resample texts.
    """
    a, b = np.asarray(encoder), np.asarray(decision)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or len(a) < 1:
        raise ValueError("Matched condensed distance vectors are required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Distances must be finite")
    constant = len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0
    return dict(pairs=len(a), spearman=None if constant else float(spearmanr(a, b).statistic),
                status="undefined_constant_distances" if constant else "descriptive",
                independent_pairs=False)
