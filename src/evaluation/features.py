"""Row-aligned non-text, decision and frozen-embedding features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.serialization import text_value

MISSINGNESS_VERSION = "shared_text_missing_v2"


def missing_column(index):
    return f"text_presence__{index:03d}__missing"


def add_text_presence(matrix, dataset):
    """Identical label-free text-availability information for every representation arm."""
    result = matrix.copy()
    for i, column in enumerate(dataset.metadata.text_columns):
        values = dataset.frame[column].map(lambda x: text_value(x) is None)
        name = missing_column(i)
        if name in result:
            raise ValueError("Text-presence name collides with existing features")
        result[name] = values.to_numpy(dtype=int)
    result.attrs = dict(matrix.attrs)
    result.attrs["missingness_policy"] = MISSINGNESS_VERSION
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
)

EMBEDDING_BASELINE = "non_text_plus_frozen_e5"
PROVIDER_INDEPENDENT_SETS = ("non_text", EMBEDDING_BASELINE)
DECISION_FEATURE_SETS = MAIN_FEATURE_SETS[:4]
EMBEDDING_FEATURE_SETS = MAIN_FEATURE_SETS[4:]
NATIVE_FEATURE_SETS = (
    "original_features",
)
#: Group B additions (approved 20 November 2026).
#: B2: one decision model's "both" block plus frozen E5-small per-column PCA32.
DECISION_PLUS_EMBEDDING = "non_text_plus_both_plus_frozen_e5"
#: B3: label-trained L2-logistic classifier on frozen embeddings (cross-fitted).
SUPERVISED_PROBE = "non_text_plus_supervised_probe"
#: B4: the per-column decision arm refitted without one text column's decision block.
PER_COLUMN_WITHOUT_ONE = "non_text_plus_per_column_without_one_text"
GROUP_B_FEATURE_SETS = (DECISION_PLUS_EMBEDDING, SUPERVISED_PROBE, PER_COLUMN_WITHOUT_ONE)
#: Group C additions (approved 20 November 2026), Strands only.
#: C1a: one Strands hidden-state position/layer, PCA fitted on outer-training rows only.
HIDDEN = "non_text_plus_hidden"
#: C1b: the <answer>-last vector together with Strands' own probabilities of the same view.
PER_COLUMN_PLUS_HIDDEN = "non_text_plus_per_column_plus_hidden"
JOINT_PLUS_HIDDEN = "non_text_plus_joint_plus_hidden"
#: C2: cross-fitted, label-trained Strands probabilities for the joint question.
ADAPTED = "non_text_plus_adapted_joint"
HIDDEN_FEATURE_SETS = (HIDDEN, PER_COLUMN_PLUS_HIDDEN, JOINT_PLUS_HIDDEN)
GROUP_C_FEATURE_SETS = (*HIDDEN_FEATURE_SETS, ADAPTED)


def uses_embeddings(name):
    return name in (*MAIN_FEATURE_SETS[4:], "non_text_plus_tar", DECISION_PLUS_EMBEDDING,
                    SUPERVISED_PROBE)


def decision_feature_set(name):
    """Select the decision block independently of the optional embedding addition."""
    return (
        dict(zip(EMBEDDING_FEATURE_SETS, DECISION_FEATURE_SETS))
        | dict(zip(NATIVE_FEATURE_SETS, DECISION_FEATURE_SETS))
        | {
            DECISION_PLUS_EMBEDDING: "non_text_plus_both",
            SUPERVISED_PROBE: "non_text",
            PER_COLUMN_WITHOUT_ONE: "non_text_plus_per_column",
            HIDDEN: "non_text",
            PER_COLUMN_PLUS_HIDDEN: "non_text_plus_per_column",
            JOINT_PLUS_HIDDEN: "non_text_plus_joint",
            ADAPTED: "non_text",
        }
    ).get(name, name)


def without_text_column_block(features, text_index):
    """Drop every per-column decision column of one text field; keep its availability flag."""
    marker = f"__per_column__text_only__text_{int(text_index):03d}__"
    dropped = [c for c in features.columns if marker in str(c)]
    if not dropped:
        raise ValueError("No per-column decision block exists for this text column")
    result = features.drop(columns=dropped)
    result.attrs = dict(features.attrs)
    return result


def combine_representations(base, additions):
    """Concatenate extractor outputs, retaining non-text/presence columns exactly once."""
    result = base.copy()
    artifacts = list(base.attrs.get("feature_artifact_ids", []))
    for namespace, matrix in additions:
        if not result.index.equals(matrix.index) or not matrix.index.is_unique:
            raise ValueError("Decision feature row identities differ")
        new = {}
        for column in matrix:
            if column in base:
                if not matrix[column].equals(base[column]):
                    raise ValueError("Shared non-text or presence values differ")
            elif column.startswith("text_presence__") and column in result:
                if not matrix[column].equals(result[column]):
                    raise ValueError("Shared text presence differs")
            else:
                name = column if column.startswith("text_presence__") else f"{namespace}__{column}"
                if name in result:
                    raise ValueError("Duplicate decision feature namespace")
                new[name] = matrix[column]
        result = result.join(pd.DataFrame(new, index=result.index), validate="one_to_one")
        artifacts.extend(matrix.attrs.get("feature_artifact_ids", []))
    result.attrs = dict(base.attrs)
    result.attrs["feature_artifact_ids"] = list(dict.fromkeys(artifacts))
    if any(c.startswith("text_presence__") for c in result):
        result.attrs["missingness_policy"] = MISSINGNESS_VERSION
    return result


def benchmark_feature_table():
    """Feature families in Experiments 1 and 2, shared by notebook and campaign."""
    labels = (
        "Non-text",
        "+ Joint",
        "+ Per-column",
        "+ Both",
        "+ Embedding",
    )
    return pd.DataFrame(
        [
            {
                "configuration": label,
                "feature_set": name,
                "Non-text": True,
                "Embedding / PCA": uses_embeddings(name),
                "Joint": decision_feature_set(name)
                in ("non_text_plus_joint", "non_text_plus_both"),
                "Per-column": decision_feature_set(name)
                in ("non_text_plus_per_column", "non_text_plus_both"),
            }
            for name, label in zip(MAIN_FEATURE_SETS[:5], labels)
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
    for i, group in enumerate(groups):
        values = embeddings[group].isna()
        if not values.all(axis=1).equals(values.any(axis=1)):
            raise ValueError("Partially missing embedding vector")
        name = missing_column(i)
        missing = values.all(axis=1).astype(int)
        if name in result and not result[name].equals(missing):
            raise ValueError("Embedding and decision text-presence information differs")
        result[name] = missing
    result.attrs["missingness_policy"] = MISSINGNESS_VERSION
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


def match_artifact_pins(inventory, pins):
    """An explicit representation pin can distinguish repeated publications."""
    if pins is None:
        return inventory
    return inventory.loc[inventory.artifact_id.eq(inventory.representation.map(pins))]


def load_feature_selection(path, selector, representations=("per_column", "joint"), *, artifact_pins=None):
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
    inventory = match_artifact_pins(feature_inventory(path), artifact_pins)
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
    # pandas joins can discard attrs, including the baseline's explicit empty list.
    matrix.attrs["feature_artifact_ids"] = [b.manifest["artifact_id"] for b in selected]
    return add_text_presence(matrix, dataset)
