"""Pinned Fastino GLiDE text renderer and native, non-generative runtime.

The publisher's prompt and recurrent-depth implementation are used unchanged.
Its 40,000-token serving limit reserves eight positions internally; our exact
audit therefore permits at most 39,992 actual prompt tokens. Probabilities and
native confidence are retained. Only the score summary is mapped from the
publisher's modal level to JEVPFN's probability-weighted level, with the original
response retained alongside it.
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import sys
import types
import urllib.request

from src.jev.provider import semantic_question
from src.jev.registry import source_module
from src.jev.requests import require_nonempty_text
from src.jev.tokenization import ExactCounter as HistoricalCounter
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import digest, file_sha256, write_json

PROVIDER = "glide"
REVISION = "4d32743ae96b74f0116f050d37742bbd43c04e7d"
MODEL = "fastino/GLiDE-27B"
SERVING_LIMIT = 40_000
RESERVED_POSITIONS = 8
PROMPT_LIMIT = SERVING_LIMIT - RESERVED_POSITIONS
VERSIONS = {"transformers": "5.17.0", "tokenizers": "0.23.2"}
FILES = (
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json",
    "decision_config.json", "requirements.txt", "LICENSE", "NOTICE",
    "licenses/Qwen3.8-27B-LICENSE", "glide/__init__.py", "glide/prompt.py",
    "glide/images.py", "glide/model.py", "glide/recurrent.py",
)
MIDLOOP = {
    "conv": "A", "strategy": "euler", "gamma": 1.0, "a": 24, "b": 47,
    "blocks": [[24, 35], [36, 47]], "n_gate": 0, "n_lora": 0, "passes": 2,
}


def asset_urls():
    candidate = load_yaml("extractors")["candidates"][PROVIDER]
    if candidate["revision"] != REVISION or candidate["model"] != MODEL:
        raise ValueError("Unreviewed GLiDE checkpoint revision")
    root = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/"
    return {name: root + name for name in FILES}


def stage_tokenizer():
    """Stage only small, pinned source/tokenizer assets on a connected CPU host."""
    from src.cluster.preflight import storage_policy_failures

    if failures := storage_policy_failures():
        raise ValueError("; ".join(failures))
    folder = paths.tokenizer_dir(PROVIDER)
    folder.mkdir(parents=True, exist_ok=True)
    with file_lock(folder / ".staging.lock", timeout=1):
        records = {}
        for name, url in asset_urls().items():
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with urllib.request.urlopen(url, timeout=120) as response:
                    content = response.read(40 * 1024 * 1024 + 1)
                if len(content) > 40 * 1024 * 1024:
                    raise ValueError("Unexpectedly large GLiDE tokenizer/source asset")
                tmp = target.with_suffix(target.suffix + ".partial")
                tmp.write_bytes(content)
                tmp.replace(target)
            records[name] = dict(url=url, sha256=file_sha256(target), bytes=target.stat().st_size)
        manifest = dict(provider=PROVIDER, revision=REVISION, files=records)
        destination = folder / "manifest.json"
        if destination.exists() and json.loads(destination.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Staged GLiDE assets changed; preserve and review them")
        write_json(destination, manifest)
    print("glide: staged pinned tokenizer/source only; no weights", flush=True)


def verify_assets(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    urls = asset_urls()
    if (manifest.get("provider") != PROVIDER or manifest.get("revision") != REVISION
            or set(manifest.get("files", {})) != set(urls)):
        raise ValueError("GLiDE tokenizer manifest differs from reviewed assets")
    for name, record in manifest["files"].items():
        if record["url"] != urls[name] or file_sha256(folder / name) != record["sha256"]:
            raise ValueError("GLiDE tokenizer/source checksum mismatch")
    return manifest


def validate_config(config, tokenizer):
    """Check the publisher's label readout and recurrence before any weight load."""
    if (config.get("format_version") != 1 or config.get("attention_mode") != "noncausal_full_attention"
            or config.get("pooling") != "last" or config.get("prompt_format") != "glide-v1"
            or config.get("temperature") != 1.0 or config.get("midloop") != MIDLOOP
            or config.get("loop")):
        raise ValueError("GLiDE attention, recurrence or calibration differs from the pinned release")
    codes, ids = config.get("codes", []), config.get("token_ids", [])
    if (len(codes) != 255 or len(ids) != 255 or len(set(codes)) != 255
            or len(set(ids)) != 255):
        raise ValueError("GLiDE requires 255 distinct answer labels and token IDs")
    for code, token_id in zip(codes, ids, strict=True):
        if tokenizer.encode(code, add_special_tokens=False) != [token_id]:
            raise ValueError("GLiDE answer label does not match its tokenizer")


class Counter(HistoricalCounter):
    """Exact text-only prompt counts; imports neither Torch nor model code."""

    def __init__(self):
        versions = {name: importlib.metadata.version(name) for name in VERSIONS}
        if versions != VERSIONS:
            raise ValueError("GLiDE token audit requires its isolated pinned container")
        from transformers import AutoTokenizer

        self.folder = paths.tokenizer_dir(PROVIDER)
        manifest = verify_assets(self.folder)
        self.provider = PROVIDER
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.folder, local_files_only=True, trust_remote_code=False,
        )
        self.tokenizer.model_max_length = sys.maxsize
        self.config = json.loads((self.folder / "decision_config.json").read_text(encoding="utf-8"))
        validate_config(self.config, self.tokenizer)
        self.renderer = source_module(self.folder, "glide/prompt.py", "jevpfn_glide_cpu_prompt")
        self.prompter = self.renderer.Prompter(self.tokenizer, self.config["codes"])
        candidate = load_yaml("extractors")["candidates"][PROVIDER]
        self.limit = candidate["context_tokens"]
        if (self.limit != PROMPT_LIMIT or candidate["publisher_context_tokens"] != SERVING_LIMIT
                or candidate["reserved_context_positions"] != RESERVED_POSITIONS):
            raise ValueError("GLiDE context policy differs from the publisher's eight-position guard")
        self.identity = dict(
            assets=manifest, packages=versions, renderer_version="glide-native-text-v1",
            context_limit=self.limit, publisher_context_limit=SERVING_LIMIT,
            reserved_positions=RESERVED_POSITIONS,
            implementation_sha256=file_sha256(paths.REPO_ROOT / "src/jev/glide.py"),
            inherited_counter_sha256=file_sha256(paths.REPO_ROOT / "src/jev/tokenization.py"),
        )
        self.identity["counter_id"] = digest(self.identity)

    def render_parts(self, payload):
        require_nonempty_text(payload)
        state, images = self.renderer.split_state(payload["state"])
        if images:
            raise ValueError("JEVPFN GLiDE accepts text only")
        question = semantic_question(payload["question"])
        options = self.renderer.options_for(question)
        if not 2 <= len(options) <= 255:
            raise ValueError("GLiDE question exceeds the pinned option contract")
        return [(self.prompter.text(state, question, options), False)]


def canonical_response(raw, questions):
    """Map score units without altering the native probability vector or confidence."""
    if set(raw.get("answers", {})) != set(questions):
        raise ValueError("GLiDE did not return exactly the requested questions")
    result = copy.deepcopy(raw)
    result["native_response"] = raw
    for key, question in questions.items():
        answer = result["answers"][key]
        if answer.get("type") != question["type"]:
            raise ValueError("GLiDE answer type differs from its question")
        if question["type"] == "score":
            keys = [str(i) for i in range(len(question["criteria"]))]
            probabilities = answer.get("probabilities", {})
            if set(probabilities) != set(keys):
                raise ValueError("GLiDE score probability vector is incomplete")
            answer["score"] = sum(i * probabilities[k] for i, k in enumerate(keys))
            answer["legend"] = dict(zip(keys, question["criteria"], strict=True))
    return result


class Runtime:
    """Use the sealed publisher network, including its two-pass recurrent blocks."""

    def __init__(self, folder, counter):
        if not folder.is_dir():
            raise ValueError("GLiDE requires an already staged local checkpoint")
        verify_assets(counter.folder)
        for name in asset_urls():
            if file_sha256(folder / name) != file_sha256(counter.folder / name):
                raise ValueError("GLiDE inference source differs from the audited source")
        package_name = "jevpfn_pinned_glide"
        package = types.ModuleType(package_name)
        package.__path__ = [str(folder / "glide")]
        sys.modules[package_name] = package
        for name in ("prompt", "images", "model"):
            module = source_module(folder, f"glide/{name}.py", f"{package_name}.{name}")
        self.publisher = module
        self.counter = counter
        self.model = module.Glide.from_pretrained(
            str(folder), device="cuda", vision=False, max_len=SERVING_LIMIT,
            token_budget=load_yaml("extractors")["candidates"][PROVIDER]["batch_token_budget"],
            name=MODEL,
        )

    def predict(self, request):
        return self.predict_many([request])[0]

    def predict_many(self, requests):
        # The outer client counts all semantic inputs before inference. Recheck the
        # actual publisher serialization too, before a possible batch runs at all.
        for request in requests:
            if request.get("images"):
                raise ValueError("JEVPFN GLiDE accepts text only")
            entries, images = self.model._prepare(request)
            if images:
                raise ValueError("JEVPFN GLiDE accepts text only")
            ids = self.model.tokenizer(
                [entry[3] for entry in entries], add_special_tokens=False, truncation=False,
            )["input_ids"]
            if any(len(value) > self.counter.limit for value in ids):
                raise ValueError("GLiDE context overflow; truncation forbidden")
        responses = self.model.decide_batch(requests)
        if len(responses) != len(requests):
            raise ValueError("GLiDE batch response count differs from its requests")
        result = []
        for request, raw in zip(requests, responses, strict=True):
            if isinstance(raw, Exception):
                raise raw
            result.append(canonical_response(raw, request["questions"]))
        return result
