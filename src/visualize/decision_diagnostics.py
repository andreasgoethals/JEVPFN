"""Shared helpers for the decision-model figures of Group A (labels, roster order, cell ink).

The earlier five label-free diagnostic views (similarity, model profile, peer majority,
all-model consensus, disputed cases) were retired on 23 November 2026: A1 now shows one model
profile strip (`outputs.model_profile_strip`) and clustered pairwise matrices from the
compact export. The evaluation tables they read remain in `src.evaluation.decision_diagnostics`.
"""

from __future__ import annotations

import numpy as np

from src.visualize import layout, style

TASKS = layout.TASK_LABELS


def short(name):
    return style.DECISION_SHORT.get(name, name)


def _models(names):
    """Canonical roster order, so a model keeps its row and colour in every figure."""
    names = set(names)
    return [n for n in style.DECISION_SHORT if n in names] + sorted(names - set(style.DECISION_SHORT))


def _ink(image, value):
    rgba = image.cmap(image.norm(value))
    return "white" if np.dot(rgba[:3], [0.2126, 0.7152, 0.0722]) < 0.5 else style.TEXT_COLOR
