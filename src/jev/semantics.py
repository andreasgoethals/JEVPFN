"""Source-backed dataset meanings, without changing labels or learning from outcomes."""

import copy

from src.jev.design import BINARY, COMMON, MULTICLASS, REGRESSION, reviewed_definition
from src.jev.templates import make_question
from src.utils.serialization import canonical_json

DOCUMENTED_VERSION = "documented-task-v3"


def dataset_semantics(metadata, catalog):
    """Validate typed raw class identity and every official text field against the catalog."""
    entry = copy.deepcopy(catalog["datasets"][metadata.dataset_id])
    if entry["raw_target"] != metadata.target or entry["task_type"] != metadata.task_type:
        raise ValueError("Documentary target/task disagrees with the loaded dataset")
    values = [canonical_json(c["value"]) for c in entry["classes"]]
    if values != [canonical_json(c) for c in metadata.classes]:
        raise ValueError("Documentary class vocabulary/order differs from the loaded dataset")
    if set(entry["text_column_definitions"]) != set(metadata.text_columns):
        raise ValueError("Documentary text columns differ from the official columns")
    if not entry["sources"] or any(
        not s["url"].startswith("https://") or not s["evidence"].strip() for s in entry["sources"]
    ):
        raise ValueError("Documentary meanings require source references")
    definition = {
        "source_reference": entry["sources"][0]["url"],
        "dataset_display_name": entry["dataset_display_name"],
        "observation_unit": entry["observation_unit"],
        "target_definition": entry["target_definition"],
        "text_column_definitions": entry["text_column_definitions"],
        "class_definitions": {
            f"c{i:03d}": c["description"] for i, c in enumerate(entry["classes"])
        },
    }
    if "target_unit" in entry:
        definition["target_unit"] = entry["target_unit"]
    entry["documentary_definition"] = reviewed_definition(metadata, definition)
    entry["semantics_version"] = catalog["version"]
    return entry


def documented_question(metadata, entry):
    """Three shared sentence templates; per-dataset facts come only from documentary YAML.

    Raw allowed answers stay typed and ordered. Descriptions change what providers read,
    while c000/c001/... still map to the original targets used for evaluation.
    """
    question = make_question(metadata)
    target = entry["target_display_name"]
    if metadata.task_type == "binary_classification":
        # Treat descriptions as quoted answer labels, not bare noun phrases. This
        # also handles labels such as "funding goal achieved" without changing meaning.
        other, positive = [canonical_json(c["description"]) for c in entry["classes"]]
        text = BINARY.format(positive=positive, other=other, target=target)
    elif metadata.task_type == "multiclass_classification":
        text = MULTICLASS.format(target=target)
        question["answer_descriptions"] = [c["description"] for c in entry["classes"]]
    else:
        text = REGRESSION.format(target=target)
    question["text"] = text + " " + COMMON
    if metadata.task_type != "regression":
        question["text"] += " Judge the target value, not merely whether a class name is mentioned."
    return question
