"""Row-aligned non-text, decision and embedding features plus cheap controls."""

from __future__ import annotations

import numpy as np
import pandas as pd


def text_statistics(dataset) -> pd.DataFrame:
    """No encoder, vocabulary fitting, target values or API calls are involved.

    A missing/blank cell has zero lengths and missing=1. Names map to the official
    text-column order, retained in dataset.metadata.text_columns.
    """
    result = pd.DataFrame(index=[f"csv:{i}" for i in range(len(dataset.frame))])
    for i, column in enumerate(dataset.metadata.text_columns):
        cells = dataset.frame[column].reset_index(drop=True)
        text = cells.map(lambda x: "" if pd.isna(x) else str(x))
        present = text.str.strip().ne("")
        prefix = f"text_stats__text_{i:03d}"
        result[f"{prefix}__characters"] = text.str.len().where(present, 0).to_numpy()
        result[f"{prefix}__words"] = text.str.split().str.len().where(present, 0).to_numpy()
        result[f"{prefix}__missing"] = (~present).astype(int).to_numpy()
    result["text_stats__total_characters"] = result.filter(like="__characters").sum(axis=1)
    result["text_stats__total_words"] = result.filter(like="__words").sum(axis=1)
    result["text_stats__nonempty_columns"] = len(dataset.metadata.text_columns) - result.filter(
        like="__missing"
    ).sum(axis=1)
    return result


SELECTOR_FIELDS = {
    "provider",
    "model",
    "version",
    "question_bank_id",
    "question_variant",
    "settings_sha256",
}

MAIN_FEATURE_SETS = (
    "non_text",
    "non_text_plus_joint",
    "non_text_plus_per_column",
    "non_text_plus_both",
    "non_text_plus_frozen_e5",
    "non_text_plus_frozen_e5_and_joint",
    "non_text_plus_frozen_e5_and_per_column",
    "non_text_plus_frozen_e5_and_both",
)

EMBEDDING_BASELINE = "non_text_plus_frozen_e5"
PROVIDER_INDEPENDENT_SETS = ("non_text", EMBEDDING_BASELINE)


def uses_embeddings(name):
    return name in MAIN_FEATURE_SETS[4:]


def decision_feature_set(name):
    """Select the decision block independently of the optional embedding addition."""
    return dict(zip(MAIN_FEATURE_SETS[4:], MAIN_FEATURE_SETS[:4])).get(name, name)


def benchmark_feature_table():
    """The eight reviewed configurations, shared by the notebook and campaign."""
    labels = (
        "Non-text",
        "+ Joint",
        "+ Per-column",
        "+ Both",
        "+ E5",
        "+ E5 + Joint",
        "+ E5 + Per-column",
        "+ E5 + Both",
    )
    return pd.DataFrame(
        [
            {
                "configuration": label,
                "feature_set": name,
                "Non-text": True,
                "E5 / PCA": uses_embeddings(name),
                "Joint": decision_feature_set(name)
                in ("non_text_plus_joint", "non_text_plus_both"),
                "Per-column": decision_feature_set(name)
                in ("non_text_plus_per_column", "non_text_plus_both"),
            }
            for name, label in zip(MAIN_FEATURE_SETS, labels)
        ]
    )


def add_embeddings(features, embeddings):
    """Require identical row identities and preserve both sources' artifact identities."""
    if not features.index.is_unique or not embeddings.index.is_unique:
        raise ValueError("Embedding join requires unique row IDs")
    if not features.index.equals(embeddings.index):
        raise ValueError("Embedding row IDs/order differ from the selected feature matrix")
    if set(features) & set(embeddings):
        raise ValueError("Embedding names collide with existing features")
    groups = embeddings.attrs.get("embedding_groups", [])
    grouped = [column for group in groups for column in group]
    if not groups or len(grouped) != len(set(grouped)) or set(grouped) != set(embeddings):
        raise ValueError("Embedding groups must cover each embedding column exactly once")
    result = features.join(embeddings, validate="one_to_one")
    result.attrs = dict(features.attrs)
    result.attrs["feature_artifact_ids"] = list(
        dict.fromkeys(
            features.attrs["feature_artifact_ids"] + embeddings.attrs["feature_artifact_ids"]
        )
    )
    result.attrs["embedding_groups"] = groups
    return result


def original_features(dataset):
    """One original-text comparator arm, all original features and original row IDs."""
    matrix = dataset.features.copy()
    matrix.index = [f"csv:{i}" for i in range(len(matrix))]
    matrix.attrs["feature_artifact_ids"] = []
    return matrix


def select_feature_blocks(blocks, selector):
    """Select one extractor recipe explicitly; never silently mix models or question versions."""
    if selector is not None and (not selector or set(selector) - SELECTOR_FIELDS):
        raise ValueError("Feature selector must specify provider/model/version/question identity")
    selected = [
        b
        for b in blocks
        if selector is None
        or all(b.manifest["identity"].get(key) == value for key, value in selector.items())
    ]
    from src.utils.serialization import digest

    recipes = {
        digest({k: v for k, v in b.manifest["identity"].items() if k != "representation"})
        for b in selected
    }
    if len(recipes) > 1:
        raise ValueError(
            "Ambiguous feature selection: specify one provider, version and question bank"
        )
    return selected


def load_feature_selection(path, selector, representations=("per_column", "joint")):
    """Load only selected artifacts from one dataset file, rejecting missing/ambiguous matches."""
    from src.jev.features import feature_inventory, load_block

    if not selector or set(selector) - SELECTOR_FIELDS:
        raise ValueError("An explicit feature selector is required")
    if (
        not representations
        or len(set(representations)) != len(representations)
        or not set(representations) <= {"per_column", "joint"}
    ):
        raise ValueError("Invalid requested representations")
    inventory = feature_inventory(path)
    for key, value in selector.items():
        inventory = inventory.loc[
            inventory[key].isna() if value is None else inventory[key].eq(value)
        ]
    inventory = inventory.loc[inventory.representation.isin(representations)]
    if sorted(inventory.representation) != sorted(representations):
        raise ValueError(
            "Each requested representation needs exactly one matching feature artifact"
        )
    blocks = [load_block(path, artifact_id) for artifact_id in inventory.artifact_id]
    return select_feature_blocks(blocks, selector)


def assemble_feature_set(dataset, blocks, name: str, *, view="main", selector=None):
    """Build one of the configured primary arms; retain stable row IDs and exclude raw text."""
    from src.jev.features import assemble_features

    modes = {
        "non_text": set(),
        "non_text_plus_text_statistics": set(),
        "non_text_plus_per_column": {"per_column"},
        "non_text_plus_joint": {"joint"},
        "non_text_plus_both": {"per_column", "joint"},
    }
    if name not in modes:
        raise ValueError("Unknown primary feature set")
    selected = [b for b in blocks if b.manifest["identity"]["representation"] in modes[name]]
    selected = select_feature_blocks(selected, selector)
    observed = [b.manifest["identity"]["representation"] for b in selected]
    if len(observed) != len(set(observed)) or set(observed) != modes[name]:
        raise ValueError("Each requested representation needs exactly one feature artifact")
    matrix = assemble_features(dataset, selected, view=view)
    if len(dataset.metadata.text_columns) == 1 and modes[name] == {"per_column", "joint"}:
        from src.jev.features import feature_view

        per_column = next(
            b for b in selected if b.manifest["identity"]["representation"] == "per_column"
        )
        joint = next(b for b in selected if b.manifest["identity"]["representation"] == "joint")
        keys = [
            b.requests.set_index("row_id").request_key.reindex(matrix.index)
            for b in (per_column, joint)
        ]
        if keys[0].equals(keys[1]):
            left, right = [
                feature_view(b, dataset.metadata.task_type, view)
                .set_index("row_id")
                .reindex(matrix.index)
                for b in (per_column, joint)
            ]
            if left.shape != right.shape or not np.array_equal(
                left.to_numpy(), right.to_numpy(), equal_nan=True
            ):
                raise ValueError("Identical requests have inconsistent feature values")
            matrix = matrix.drop(columns=right.columns)
    if name == "non_text_plus_text_statistics":
        stats = text_statistics(dataset)
        if set(stats.columns) & set(matrix.columns):
            raise ValueError("Text-statistics names collide with non-text feature names")
        matrix = matrix.join(stats, validate="one_to_one")
    # pandas joins can discard attrs, including the baseline's explicit empty list.
    matrix.attrs["feature_artifact_ids"] = [b.manifest["artifact_id"] for b in selected]
    return matrix


def shuffle_feature_block(train: pd.DataFrame, test: pd.DataFrame, *, seed: int):
    """Shuffle whole probability vectors independently inside training and test partitions.

    Break the association with the row, preserve the feature distribution and
    dependencies inside each vector. Never exchange features across CV partitions.
    No targets are accepted. Apply to Jev columns only, not non-text features.
    """
    if list(train.columns) != list(test.columns):
        raise ValueError("Train/test feature schemas differ")
    rng = np.random.default_rng(seed)
    return tuple(
        pd.DataFrame(
            frame.to_numpy()[rng.permutation(len(frame))],
            index=frame.index,
            columns=frame.columns,
        )
        for frame in (train, test)
    )
