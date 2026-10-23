"""Offline adapter previews. No credentials, network or model loading.

These match the implemented transports without loading them. Actual provider/GPU
compatibility still requires an approved pilot and context checks.
"""

from __future__ import annotations

import json

from src.jev.provider import QUESTION_ID, api_preview, semantic_question
from src.jev.requests import require_nonempty_text
from src.utils.serialization import canonical_json

PERPLEXITY_SOURCE = (
    "https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/blob/"
    "14c9d90137962d300f3de95633f7b6535e80c5ce/source/src/autojev/model.py"
)


def preview(request, provider):
    """Return documentation-verified call shapes, never execute their transports.

    Offline placeholders name an already staged, reviewed checkpoint; constructing
    a preview never imports a publisher runtime or downloads a checkpoint.
    """
    require_nonempty_text(json.loads(request.payload_json))
    state = json.loads(request.payload_json)["state"]
    question = semantic_question(json.loads(request.payload_json)["question"])
    questions = {QUESTION_ID: question}
    from src.jev.registry import ADDED

    if provider in ADDED:
        from src.utils.config import load_yaml

        return {
            "transport": "self_hosted_pinned_open_weights",
            "initialization": "runtime = ExtendedRuntime(provider, staged_checkpoint, audited_counter)",
            "call": "runtime.predict_payload(arguments)",
            "arguments": json.loads(request.payload_json),
            "native_question_semantics": questions,
            "reference": load_yaml("extractors")["candidates"][provider]["reference"],
            "pin_requirement": "checkpoint revision, source files, tokenizer, container and CPU-prepared-token hashes",
            "runtime_requirements": [
                "Reuse the exact publisher renderer and complete answer descriptions.",
                "Reject overflow; retain every class probability and publisher calibration.",
                "Use offline staged weights on an allocated H100; pass Experiment 0A before extraction.",
            ],
            "live_verified": False,
        }
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
    if provider == "cloudflare":
        return {
            "transport": "self_hosted_publisher_forward",
            "call": "ClefRuntime(checkpoint, limit=16384).predict(arguments)",
            "arguments": {"state": state, "questions": questions},
            "pin_requirement": "Cloudflare/clef-flash checkpoint and joint_schema_model.py hashes",
            "runtime_requirements": [
                "Reject complete input above 16384 tokens before the publisher's default truncation.",
                "Retain full softmax precision through encode_record/collate_records/model forward.",
            ],
            "reference": "https://huggingface.co/Cloudflare/clef-flash",
            "live_verified": False,
        }
    if provider == "intern":
        return {
            "transport": "self_hosted_publisher_inference",
            "initialization": "DecisionEngine(checkpoint, device='cuda', max_length=8192, temperature=1.99241824)",
            "call": "engine.predict(arguments)",
            "arguments": {"state": state, "questions": questions},
            "pin_requirement": "internlm/Intern-Decision-4B checkpoint and inference.py hashes",
            "runtime_requirements": [
                "At most 62 options; 8192 tokens; no truncation; retain publisher calibration."
            ],
            "reference": "https://huggingface.co/internlm/Intern-Decision-4B",
            "live_verified": False,
        }
    if provider == "strands":
        return {
            "transport": "self_hosted_SystemOneEngine.evaluate",
            "initialization": "SystemOneEngine(StrandsDeciderModel.load(checkpoint), EngineConfig(device='cuda', strict_window=True, max_batch=1, model_name=pinned_model))",
            "call": "engine.evaluate(SystemOneRequest(**arguments)).model_dump(mode='json')",
            "arguments": {"state": state, "questions": questions},
            "pin_requirement": "adapter, base checkpoint and installed inference package hashes",
            "runtime_requirements": [
                "Stage and pin both the adapter checkpoint and its separate base model.",
                "Use strict_window=True; reject overlong input without truncation.",
                "Serialize calls per engine; publisher reports shared mutable option offsets.",
            ],
            "reference": "https://github.com/strands-labs/strands-decider/blob/b9fc83626bcca822d00f822d6c2e151b8d244ee9/src/strands_decider/infer.py",
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
