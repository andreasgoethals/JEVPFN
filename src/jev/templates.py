"""Central deterministic wording for review before any Jev API integration.

The kinds below describe intended semantics, NOT verified Jev API field names.
Changing wording changes the content hash even if the human version is not bumped.
"""

from __future__ import annotations

from src.data.metadata import TaskMetadata
from src.utils.serialization import canonical_json

TEMPLATE_VERSION = "directional-v1"
BINARY = (
    "Based on the supplied text and context, does this text provide evidence in favour "
    "of target class {positive} rather than {other}, for target {target}?"
)
MULTICLASS = "Which value of target {target} is most supported by the supplied text and context?"
REGRESSION = (
    "Based on the supplied text and context, does the information suggest that {target} "
    "should be relatively lower or higher?"
)
REGRESSION_LABELS = {
    -4: "strongly suggests a lower target value",
    -3: "suggests a lower target value (level -3)",
    -2: "suggests a lower target value (level -2)",
    -1: "suggests a lower target value (level -1)",
    0: "provides no meaningful directional evidence",
    1: "suggests a higher target value (level +1)",
    2: "suggests a higher target value (level +2)",
    3: "suggests a higher target value (level +3)",
    4: "strongly suggests a higher target value",
}


def make_question(metadata: TaskMetadata, *, scores: list[int] | None = None) -> dict:
    target = canonical_json(metadata.target)
    if metadata.task_type == "binary_classification":
        other, positive = metadata.classes
        return {
            "kind": "binary_probability",
            "text": BINARY.format(
                positive=canonical_json(positive), other=canonical_json(other), target=target
            ),
            "positive_class": positive,
            "other_class": other,
            "intended_output": {"probability_range": [0, 1]},
        }
    if metadata.task_type == "multiclass_classification":
        return {
            "kind": "choice",
            "text": MULTICLASS.format(target=target),
            "allowed_answers": list(metadata.classes),
            "intended_output": "complete class probability vector in allowed_answers order",
        }
    scores = list(range(-4, 5)) if scores is None else scores
    if scores != list(range(-4, 5)):
        raise ValueError("The agreed regression scale is exactly -4 through +4.")
    return {
        "kind": "ordered_choice",
        "text": REGRESSION.format(target=target),
        "allowed_answers": [{"score": v, "meaning": REGRESSION_LABELS[v]} for v in scores],
        "intended_output": "preserve raw response; final feature reduction remains open",
    }
