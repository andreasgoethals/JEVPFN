"""Pinned, weight-free tokenizers and publisher prompt rendering for text-only decisions.

Read-only copies of publisher source are downloaded with the tokenizer, never imported as
an inference runtime. Only the reviewed pure rendering symbols below are compiled. This
uses the actual publisher format, including separate state/question encoding for Strands.
No requests to inference services, model tensors, truncation or target labels are involved.
"""

from __future__ import annotations

import ast
import json
import sys
import types
import urllib.request
from functools import lru_cache

from src.jev.provider import semantic_question
from src.jev.requests import require_nonempty_text
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import canonical_json, digest, file_sha256, write_json

SELF_HOSTED = ("liquid", "strands", "perplexity")
HOSTED_UNAVAILABLE = {
    "jev": "Publisher tokenizer and complete server prompt are not public; exact counting unavailable.",
    "cloudflare": "Hosted tokenizer/server wrapper is unverified; no substitute tokenizer is exact.",
    "microsoft": "Foundry deployment tokenizer/server wrapper is unverified; base Qwen is not proof.",
}
RENDERER_VERSION = "publisher-text-rendering-v1"
# Immutable, reviewed source versions. A changed candidate requires reviewing the renderer again.
PINS = {
    "liquid": "051bcc464b01b9f92942b364d9586b0ef5912432",
    "strands": "216d1ccb46a9e9b036eb9187b248db98480f77d4",
    "perplexity": "14c9d90137962d300f3de95633f7b6535e80c5ce",
}
STRANDS_RUNTIME = "b9fc83626bcca822d00f822d6c2e151b8d244ee9"


def asset_urls(provider):
    candidate = load_yaml("extractors")["candidates"][provider]
    if candidate["revision"] != PINS[provider]:
        raise ValueError("Tokenizer renderer has not been reviewed for this checkpoint revision")
    root = f"https://huggingface.co/{candidate['model']}/resolve/{candidate['revision']}/"
    files = {
        name: root + name
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    }
    if provider == "liquid":
        files.update(
            {name: root + name for name in ("prompt.py", "runner.py", "config.json", "LICENSE")}
        )
    elif provider == "perplexity":
        files.update(
            {
                name: root + name
                for name in ("decision_config.json", "processor_config.json", "LICENSE")
            }
        )
        files["publisher_model.py"] = root + "source/src/autojev/model.py"
    else:
        if candidate["runtime_revision"] != STRANDS_RUNTIME:
            raise ValueError("Strands renderer runtime differs from the reviewed source")
        files.update({name: root + name for name in ("strands_decider_config.json", "LICENSE.md")})
        runtime = f"https://raw.githubusercontent.com/strands-labs/strands-decider/{STRANDS_RUNTIME}/src/strands_decider/"
        files.update({name: runtime + name for name in ("prompting.py", "infer.py")})
    return files


def stage_tokenizers(providers=SELF_HOSTED):
    """Explicit small-file download. Never fetch a model shard or run remote code."""
    from src.cluster.preflight import storage_policy_failures

    failures = storage_policy_failures()
    if failures:
        raise ValueError("; ".join(failures))
    for provider in providers:
        folder = paths.tokenizer_dir(provider)
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, url in asset_urls(provider).items():
            destination = folder / name
            if not destination.is_file():
                request = urllib.request.Request(url, headers={"User-Agent": "JEVPFN-token-audit"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    content = response.read(40 * 1024 * 1024 + 1)
                if len(content) > 40 * 1024 * 1024:
                    raise ValueError("Unexpectedly large tokenizer asset; refusing model weights")
                temporary = destination.with_suffix(destination.suffix + ".partial")
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
            raise ValueError(
                "Staged tokenizer assets changed; refusing to overwrite their identity"
            )
        write_json(manifest, identity)
        print(
            f"{provider}: staged tokenizer/source files ({sum(f['bytes'] for f in files.values()):,} bytes), no weights",
            flush=True,
        )


def _pure_symbols(path, names, namespace):
    """Compile only named, reviewed definitions; exclude imports and all model-loading code."""
    source = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        n for n in source.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    if {n.name for n in nodes} != set(names):
        raise ValueError("Publisher rendering source no longer matches the reviewed symbols")
    tree = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace


def _liquid_renderer(folder):
    # prompt.py was reviewed in full: pure stdlib rendering/readout helpers, no model imports.
    module = types.ModuleType("jevpfn_pinned_liquid_prompt")
    sys.modules[module.__name__] = module  # dataclass inspects its module
    exec(
        compile(
            (folder / "prompt.py").read_text(encoding="utf-8"), str(folder / "prompt.py"), "exec"
        ),
        module.__dict__,
    )
    return module


def _strands_renderer(folder):
    from dataclasses import dataclass

    class Question:
        def __init__(self, instructions, criteria=None):
            self.instructions, self.criteria = instructions, criteria

    namespace = {
        "__name__": __name__,
        "json": json,
        "dataclass": dataclass,
        "NoulQuestion": type("NoulQuestion", (Question,), {}),
        "ChoiceQuestion": type("ChoiceQuestion", (Question,), {}),
        "ScoreQuestion": type("ScoreQuestion", (Question,), {}),
        "NOUL_SLOT_LABELS": ("false", "true"),
        "NOUL_DEFAULT_CRITERIA": {
            "false": "the statement does not hold for this state",
            "true": "the statement holds for this state",
        },
    }
    return _pure_symbols(
        folder / "prompting.py",
        ("RenderedQuestion", "render_content", "_option_block", "render_question", "render_state"),
        namespace,
    )


class ExactCounter:
    """Actual fast tokenizer plus exact single-question publisher input formatting."""

    def __init__(self, provider):
        if provider not in SELF_HOSTED:
            raise ValueError(HOSTED_UNAVAILABLE.get(provider, "Unknown provider"))
        import transformers
        from transformers import AutoTokenizer

        self.provider = provider
        folder = paths.tokenizer_dir(provider)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        urls = asset_urls(provider)
        if manifest["revision"] != PINS[provider] or set(manifest["files"]) != set(urls):
            raise ValueError("Tokenizer manifest does not match the pinned asset set")
        for name, record in manifest["files"].items():
            if record["url"] != urls[name] or file_sha256(folder / name) != record["sha256"]:
                raise ValueError("Tokenizer/source checksum mismatch")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(folder), local_files_only=True, trust_remote_code=False
        )
        self.tokenizer.model_max_length = (
            sys.maxsize
        )  # count all tokens; enforce our explicit limit later
        self.tokenizer.padding_side = "left"
        self.identity = {
            "assets": manifest,
            "transformers": transformers.__version__,
            "renderer_version": RENDERER_VERSION,
            "implementation_sha256": file_sha256(paths.REPO_ROOT / "src/jev/tokenization.py"),
        }
        if provider == "liquid":
            self.renderer = _liquid_renderer(folder)
            self.limit = 32768
        elif provider == "strands":
            self.renderer = _strands_renderer(folder)
            self.limit = json.loads((folder / "strands_decider_config.json").read_text())[
                "max_length"
            ]
        else:
            self.renderer = _pure_symbols(
                folder / "publisher_model.py",
                ("describe", "options", "decision_messages"),
                {"json": json, "MAX_OPTIONS": 255},
            )
            self.codes = json.loads((folder / "decision_config.json").read_text())["codes"]
            self.limit = 8192
        self.identity["context_limit"] = self.limit
        self.identity["counter_id"] = digest(self.identity)

    @lru_cache(maxsize=256)
    def _question(self, encoded):
        question = json.loads(encoded)
        if self.provider == "liquid":
            return self.renderer.as_question(question)
        if self.provider == "strands":
            cls = self.renderer[
                {"noul": "NoulQuestion", "choice": "ChoiceQuestion", "score": "ScoreQuestion"}[
                    question["type"]
                ]
            ]
            return self.renderer["render_question"](
                cls(question["instructions"], question.get("criteria"))
            ).text
        return question

    def render_parts(self, payload):
        require_nonempty_text(payload)
        question = self._question(canonical_json(semantic_question(payload["question"])))
        if self.provider == "strands":
            # Matches SystemOneEngine._fit(strict_window=True), including state special tokens.
            return [(self.renderer["render_state"](payload["state"]), True), (question, False)]
        state = canonical_json(payload["state"])
        if self.provider == "liquid":
            text = self.renderer.render(
                self.tokenizer, state, question, bos=self.tokenizer.bos_token or ""
            )
            return [(text, False)]  # SystemOne.run single-question/no-image route
        messages = self.renderer["decision_messages"](
            {"state": state, "question": question}, self.codes
        )
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return [(text, True)]  # Processor's text-only tokenizer call, no padding added to length

    def count_many(self, payloads):
        parts = [self.render_parts(p) for p in payloads]
        if not parts:
            return []
        counts = [0] * len(parts)
        for part in range(len(parts[0])):
            encoded = self.tokenizer(
                [p[part][0] for p in parts],
                add_special_tokens=parts[0][part][1],
                truncation=False,
                padding=False,
                return_length=True,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            counts = [a + b for a, b in zip(counts, encoded["length"], strict=True)]
        return counts

    def __call__(self, payload):
        return self.count_many([payload])[0]
