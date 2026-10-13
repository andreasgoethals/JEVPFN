"""Provider-specific cache identities for offline preparation, with shared semantics."""

import json

from src.jev.requests import IntendedRequest
from src.utils.serialization import canonical_json


def bind_provider(request, provider, catalog):
    """Keep the exact text/question; isolate each provider/checkpoint's response cache.

    This is not a response decoder or an inference client. A missing immutable
    hosted revision remains missing, never substituted with an invented pin.
    """
    if catalog["enabled"] is not False:
        raise ValueError("Only disabled-candidate preparation is supported")
    candidate = catalog["candidates"][provider]
    if candidate["enabled"] is not False:
        raise ValueError("Candidate inference must remain disabled")
    original = json.loads(request.settings_json)
    if provider == "jev":
        if (candidate["model"], candidate["version"]) != (original["model"], original["version"]):
            raise ValueError("Jev catalog and request model pins disagree")
        return request
    settings = {
        "provider": provider,
        "model": candidate["model"],
        "version": candidate.get("revision") or candidate.get("version"),
        "base_model_revision": candidate.get("base_model_revision"),
        "template_version": original["template_version"],
        "request_configuration": {},
        "api_mapping_version": f"{provider}-preview-v1",
    }
    return IntendedRequest(request.payload_json, canonical_json(settings), request.provenance_json)
