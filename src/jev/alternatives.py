"""Offline adapter previews. No credentials, network or model loading.

These document request mapping, not verified live response adapters. Each provider
still needs an approved contract pilot, immutable revision and context checks.
"""

from __future__ import annotations

import json

from src.jev.provider import QUESTION_ID, api_preview, semantic_question
from src.utils.serialization import canonical_json

PERPLEXITY_SOURCE = (
    "https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/blob/"
    "14c9d90137962d300f3de95633f7b6535e80c5ce/source/src/autojev/model.py"
)


def preview(request, provider, *, deployment_name=None):
    """Return documentation-verified call shapes, never execute their transports.

    Offline placeholders name an already staged, reviewed checkpoint; constructing
    a preview never imports a publisher runtime or downloads a checkpoint.
    """
    state = json.loads(request.payload_json)["state"]
    question = semantic_question(json.loads(request.payload_json)["question"])
    questions = {QUESTION_ID: question}
    if provider == "jev":
        return {
            "transport": "TypeSafeClient.system_one",
            "method": "POST",
            "endpoint": "https://api.typesafe.ai/v1/systemone",
            "headers": {
                "Authorization": "Bearer <TYPESAFE_API_KEY>",
                "Content-Type": "application/json",
            },
            "body": api_preview(request),
            "reference": "https://docs.typesafe.ai/api",
            "live_verified": False,
        }
    if provider == "microsoft":
        if not deployment_name:
            raise ValueError("Microsoft requires an explicit Azure deployment name")
        return {
            "transport": "https_post",
            "method": "POST",
            "endpoint": "<foundry-resource-endpoint>/providers/microsoft/v1/systemone",
            "headers": {"api-key": "<AZURE_API_KEY>", "Content-Type": "application/json"},
            "body": {"model": deployment_name, "state": state, "questions": questions},
            "pin_requirement": "underlying Microsoft-Decision-1 version 1, separate from deployment alias",
            "response_note": "The returned model identifies the underlying model, not the deployment alias.",
            "reference": "https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision",
            "live_verified": False,
        }
    if provider == "cloudflare":
        return {
            "transport": "https_post",
            "method": "POST",
            "endpoint": "https://api.cloudflare.com/client/v4/accounts/<account-id>/ai/run/@cf/cloudflare/clef-flash",
            "headers": {
                "Authorization": "Bearer <CLOUDFLARE_AUTH_TOKEN>",
                "Content-Type": "application/json",
            },
            "body": {"model": "clef-flash", "state": state, "questions": questions},
            "pin_requirement": "immutable hosted revision; prevent implicit truncation",
            "runtime_requirements": [
                "Prevalidate actual token length: the service otherwise truncates text state."
            ],
            "reference": "https://developers.cloudflare.com/workers-ai/models/clef-flash/",
            "live_verified": False,
        }
    if provider == "strands":
        return {
            "transport": "self_hosted_post",
            "method": "POST",
            "endpoint": "<local-server>/v1/systemone",
            "headers": {"Content-Type": "application/json"},
            "body": {"state": state, "questions": questions},
            "pin_requirement": "server checkpoint revision and inference package hash",
            "server_command": (
                "strands-decider serve <staged-pinned-checkpoint> "
                "--device cuda --port 8000 --strict-window"
            ),
            "runtime_requirements": [
                "Stage and pin both the adapter checkpoint and its separate base model.",
                "Use --strict-window; default serving truncates long text.",
                "Serialize calls per engine; publisher reports shared mutable option offsets.",
                "Keep the unauthenticated server on its default 127.0.0.1 binding.",
            ],
            "reference": "https://github.com/strands-labs/strands-decider/blob/main/docs/inference.md",
            "live_verified": False,
        }
    if provider == "liquid":
        return {
            "transport": "self_hosted_python_model.system_one",
            "call": "model.system_one(**arguments)",
            "arguments": {"state": canonical_json(state), "questions": questions},
            "pin_requirement": "checkpoint revision and custom inference code hash",
            "runtime_requirements": [
                "Load the staged pinned checkpoint with the publisher's custom decision implementation."
            ],
            "reference": "https://huggingface.co/LiquidAI/d1-3B",
            "live_verified": False,
        }
    if provider == "perplexity":
        return {
            "transport": "self_hosted_publisher_inference",
            "imports": "from autojev.model import DecisionModel, answer",
            "module_path": "<staged-pinned-checkpoint>/source/src",
            "initialization": "model = DecisionModel(checkpoint, device='cuda')",
            "call": "probabilities = model.predict(**arguments)[0]",
            "arguments": {
                "rows": [{"state": canonical_json(state), "question": question}],
                "batch_size": 1,
            },
            "decode": "answer(arguments['rows'][0]['question'], probabilities)",
            "runtime_requirements": [
                "Use the pinned publisher runtime and checkpoint, in an isolated environment.",
                "Preserve saved noncausal attention and calibration; do not override temperature.",
                "prepare rejects inputs exceeding 8192 tokens without truncation.",
            ],
            "reference": PERPLEXITY_SOURCE,
            "live_verified": False,
        }
    raise ValueError("Unknown extractor")
