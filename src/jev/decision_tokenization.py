"""All five open-weight counters; retain the original three audit identities unchanged.

tokenization.py remains byte-for-byte stable so an already-running VSC audit can be
imported. Clef and Intern use their pinned publisher code's pure rendering functions.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import sys
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from src.jev import tokenization as legacy
from src.jev.provider import QUESTION_ID, semantic_question
from src.jev.requests import require_nonempty_text
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import canonical_json, digest, file_sha256, write_json

SELF_HOSTED = (*legacy.SELF_HOSTED, "cloudflare", "intern")
HOSTED_UNAVAILABLE = {
    k: v for k, v in legacy.HOSTED_UNAVAILABLE.items() if k in {"jev", "microsoft"}
}
PINS = {
    **legacy.PINS,
    "cloudflare": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
    "intern": "0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd",
}


def asset_urls(provider):
    if provider in legacy.SELF_HOSTED:
        return legacy.asset_urls(provider)
    candidate = load_yaml("extractors")["candidates"][provider]
    if candidate["revision"] != PINS[provider]:
        raise ValueError("Unreviewed tokenizer revision")
    files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "config.json",
        "LICENSE",
    ]
    files += (
        ["joint_schema_model.py", "joint_head_config.json"]
        if provider == "cloudflare"
        else ["inference.py"]
    )
    root = f"https://huggingface.co/{candidate['model']}/resolve/{PINS[provider]}/"
    return {name: root + name for name in files}


def stage_tokenizers(providers=SELF_HOSTED):
    from src.cluster.preflight import storage_policy_failures

    if failures := storage_policy_failures():
        raise ValueError("; ".join(failures))
    for provider in providers:
        if provider in legacy.SELF_HOSTED:
            legacy.stage_tokenizers([provider])
            continue
        folder = paths.tokenizer_dir(provider)
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, url in asset_urls(provider).items():
            destination = folder / name
            if not destination.exists():
                with urllib.request.urlopen(url, timeout=120) as response:
                    content = response.read(40 * 1024 * 1024 + 1)
                if len(content) > 40 * 1024 * 1024:
                    raise ValueError("Unexpectedly large tokenizer/source file")
                temporary = destination.with_suffix(".partial")
                temporary.write_bytes(content)
                temporary.replace(destination)
            files[name] = {
                "url": url,
                "sha256": file_sha256(destination),
                "bytes": destination.stat().st_size,
            }
        identity = {"provider": provider, "revision": PINS[provider], "files": files}
        manifest = folder / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError("Tokenizer/source assets changed")
        write_json(manifest, identity)
        print(f"{provider}: staged tokenizer/source only; no weights", flush=True)


def publisher_renderer(provider, folder):
    """No upstream imports/model code executed. Constants come from reviewed AST literals."""
    name = "joint_schema_model.py" if provider == "cloudflare" else "inference.py"
    namespace = {
        "__name__": __name__,
        "json": json,
        "math": math,
        "copy": copy,
        "dataclass": dataclass,
        "dataclass_field": field,
        "Mapping": Mapping,
        "Any": Any,
    }
    wanted = (
        {
            "SYSTEM_PROMPT",
            "QUESTION_TYPES",
            "MEDIA_BATCH_KEYS",
            "MEDIA_TOKEN_KEYS",
            "IMAGE_PLACEHOLDER",
            "VIDEO_PLACEHOLDER",
        }
        if provider == "cloudflare"
        else {"SYSTEM_PROMPT", "DECISION_TOKEN", "ANSWER_SYMBOLS"}
    )
    for node in ast.parse((folder / name).read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in wanted
        ):
            namespace[node.targets[0].id] = ast.literal_eval(node.value)
    symbols = (
        (
            "render",
            "question_options",
            "EncodedQuestion",
            "EncodedRecord",
            "_tokens",
            "_encode_media",
            "encode_record",
        )
        if provider == "cloudflare"
        else (
            "CompiledExample",
            "_json",
            "_options",
            "_message_content",
            "compile_row",
            "validate_request",
        )
    )
    return legacy._pure_symbols(folder / name, symbols, namespace)


class AdditionalCounter:
    def __init__(self, provider):
        import transformers
        from transformers import AutoTokenizer

        self.provider = provider
        folder = paths.tokenizer_dir(provider)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        urls = asset_urls(provider)
        if manifest["revision"] != PINS[provider] or set(manifest["files"]) != set(urls):
            raise ValueError("Tokenizer manifest differs from reviewed assets")
        for name, record in manifest["files"].items():
            if record["url"] != urls[name] or record["sha256"] != file_sha256(folder / name):
                raise ValueError("Tokenizer/source checksum mismatch")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(folder), local_files_only=True, trust_remote_code=False
        )
        self.tokenizer.model_max_length = sys.maxsize
        self.renderer = publisher_renderer(provider, folder)
        self.limit = 16384 if provider == "cloudflare" else 8192
        self.identity = {
            "assets": manifest,
            "transformers": transformers.__version__,
            "renderer_version": "publisher-text-rendering-v1",
            "context_limit": self.limit,
            "implementation_sha256": file_sha256(
                paths.REPO_ROOT / "src/jev/decision_tokenization.py"
            ),
        }
        self.identity["counter_id"] = digest(self.identity)

    @lru_cache(maxsize=256)
    def _clef_fixed(self, question):
        # Publisher encodes the state and schema separately, then concatenates IDs.
        # Empty string contributes zero state tokens. Unlimited counting avoids its truncation.
        encoded = self.renderer["encode_record"](
            self.tokenizer,
            {"state": "", "questions": {QUESTION_ID: json.loads(question)}},
            max_length=sys.maxsize,
        )
        return len(encoded.input_ids)

    def count_many(self, payloads):
        texts, fixed = [], []
        for payload in payloads:
            require_nonempty_text(payload)
            question = semantic_question(payload["question"])
            if self.provider == "cloudflare":
                texts.append(self.renderer["render"](payload["state"]))
                fixed.append(self._clef_fixed(canonical_json(question)))
            else:
                row = self.renderer["validate_request"](
                    {"state": payload["state"], "questions": {QUESTION_ID: question}}
                )
                compiled = self.renderer["compile_row"](row)
                texts.append(
                    self.tokenizer.apply_chat_template(
                        compiled.messages,
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=False,
                        add_vision_id=True,
                    )
                )
                fixed.append(0)
        if not texts:
            return []
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_length=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return [n + base for n, base in zip(encoded["length"], fixed, strict=True)]

    def __call__(self, payload):
        return self.count_many([payload])[0]


def ExactCounter(provider):
    if provider in legacy.SELF_HOSTED:
        return legacy.ExactCounter(provider)
    if provider not in SELF_HOSTED:
        raise ValueError(HOSTED_UNAVAILABLE.get(provider, "Unknown provider"))
    return AdditionalCounter(provider)
