"""Central deterministic wording for review before any Jev API integration.

The kinds below describe intended semantics, NOT verified Jev API field names.
Changing wording changes the content hash even if the human version is not bumped.
"""

from __future__ import annotations

from src.data.metadata import TaskMetadata
from src.utils.serialization import canonical_json

TEMPLATE_VERSION = "directional-v2"
BINARY = (
    "Based on the supplied text and context, does this text provide evidence in favour "
    "of target class {positive} rather than {other}, for target {target}?"
)
MULTICLASS = "Which value of target {target} is most supported by the supplied text and context?"
REGRESSION = (
    "Based on the supplied text and context, does the information suggest that {target} "
    "should be relatively lower or higher than would ordinarily be expected for this kind "
    "of case? Judge the direction and strength of the evidence in the text, not a numeric "
    "difference in target units. No dataset average or target statistics are supplied."
)
REGRESSION_LABELS = {
    -4: "Strong lower evidence: an explicit lower-value comparison with supporting detail and no material contradiction.",
    -3: "Clear lower evidence: concrete, directly relevant information supports a lower value, without material opposing evidence.",
    -2: "Moderate lower evidence: relevant information supports a lower value, but qualifications or conflicting details weaken it.",
    -1: "Weak lower evidence: an indirect hint supports a lower value only if unstated assumptions hold.",
    0: "No directional evidence: the text provides no relevant information, or neither lower nor higher is supported over the other.",
    1: "Weak higher evidence: an indirect hint supports a higher value only if unstated assumptions hold.",
    2: "Moderate higher evidence: relevant information supports a higher value, but qualifications or conflicting details weaken it.",
    3: "Clear higher evidence: concrete, directly relevant information supports a higher value, without material opposing evidence.",
    4: "Strong higher evidence: an explicit higher-value comparison with supporting detail and no material contradiction.",
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
        "intended_output": "preserve all nine probabilities; main numeric feature is expected direction: sum(score * probability)",
    }
