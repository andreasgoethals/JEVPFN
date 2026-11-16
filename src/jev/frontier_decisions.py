"""Pinned Mammoth/OneJev and Drex native text-only decision adapters.

Publisher attention, option ordering, readout and calibration are retained.
Mammoth's reviewed serving patch is vendored verbatim with provenance; Drex's
native source is staged from the exact checkpoint. No generated answers,
quantization, truncation, or replacement of the native probability readout.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import shutil
import sys
import types
import urllib.request

from src.jev.provider import semantic_question
from src.jev.registry import pure_functions, source_module
from src.jev.requests import require_nonempty_text
from src.jev.tokenization import ExactCounter as HistoricalCounter
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import digest, file_sha256, write_json

MODELS = {
    "mammoth": ("baselabs/mammoth-1", "61cec44c281c693992e3d6ebeb44dd3402dfa186"),
    "drex_max": ("nace-ai/drex-max-v1", "005d4eb46cd2f2f56ae0e82c6444fba8363c9c17"),
}
VERSIONS = {
    "mammoth": {"transformers": "5.19.0", "tokenizers": "0.23.3"},
    "drex_max": {"transformers": "5.17.0", "tokenizers": "0.23.3"},
}
COMMON = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json")
EXTRA = {
    "mammoth": ("readout.json", "constraints.txt", "serve.patch", "setup.sh"),
    "drex_max": ("decision_config.json", "processor_config.json", "requirements.txt",
                 "LICENSE", "code/drex_max/model.py", "code/drex_max/types.py"),
}
LIMIT = 16384


def vendor_files():
    root = paths.REPO_ROOT / "src/jev/upstream/mammoth"
    manifest = json.loads((root / "PROVENANCE.json").read_text())
    for name, expected in manifest["files"].items():
        if file_sha256(root / name) != expected:
            raise ValueError("Vendored Mammoth runtime differs from its reviewed publisher patch")
    return root, manifest


def asset_urls(provider):
    model, revision = MODELS[provider]
    cfg = load_yaml("extractors")["candidates"][provider]
    if (cfg["model"], cfg["revision"]) != (model, revision):
        raise ValueError("Unreviewed decision checkpoint")
    urls = {n: f"https://huggingface.co/{model}/resolve/{revision}/{n}" for n in (*COMMON, *EXTRA[provider])}
    if provider == "mammoth":
        _, manifest = vendor_files()
        urls.update({"source/" + n: f"vendored:{manifest['revision']}+serve.patch/{n}"
                     for n in manifest["files"]})
    return urls


def stage_tokenizer(provider):
    from src.cluster.preflight import storage_policy_failures

    if failures := storage_policy_failures():
        raise ValueError("; ".join(failures))
    folder = paths.tokenizer_dir(provider)
    folder.mkdir(parents=True, exist_ok=True)
    with file_lock(folder / ".staging.lock", timeout=1):
        records = {}
        for name, url in asset_urls(provider).items():
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                tmp = target.with_suffix(target.suffix + ".partial")
                if url.startswith("vendored:"):
                    root, _ = vendor_files()
                    shutil.copyfile(root / name.removeprefix("source/"), tmp)
                else:
                    with urllib.request.urlopen(url, timeout=120) as response:
                        content = response.read(40 * 1024 * 1024 + 1)
                    if len(content) > 40 * 1024 * 1024:
                        raise ValueError("Unexpectedly large tokenizer/source file")
                    tmp.write_bytes(content)
                tmp.replace(target)
            records[name] = dict(url=url, sha256=file_sha256(target), bytes=target.stat().st_size)
        manifest = dict(provider=provider, revision=MODELS[provider][1], files=records)
        target = folder / "manifest.json"
        if target.exists() and json.loads(target.read_text()) != manifest:
            raise ValueError("Staged assets changed; preserve the previous identity")
        write_json(target, manifest)
    print(f"{provider}: pinned tokenizer and native runtime staged", flush=True)


def verify_assets(provider, folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    urls = asset_urls(provider)
    if (manifest.get("provider") != provider or manifest.get("revision") != MODELS[provider][1]
            or set(manifest.get("files", {})) != set(urls)):
        raise ValueError("Native asset inventory mismatch")
    for name, record in manifest["files"].items():
        if record["url"] != urls[name] or file_sha256(folder / name) != record["sha256"]:
            raise ValueError("Native asset checksum mismatch")
        if name.startswith("source/"):
            root, _ = vendor_files()
            if file_sha256(root / name.removeprefix("source/")) != record["sha256"]:
                raise ValueError("Staged Mammoth serving patch differs from reviewed source")
    return manifest


def package(name, folder):
    existing = sys.modules.get(name)
    if existing is not None and list(existing.__path__) != [str(folder)]:
        raise ValueError("Conflicting publisher package already loaded")
    if existing is None:
        module = types.ModuleType(name)
        module.__path__ = [str(folder)]
        sys.modules[name] = module


class Counter(HistoricalCounter):
    def __init__(self, provider):
        versions = {n: importlib.metadata.version(n) for n in VERSIONS[provider]}
        if versions != VERSIONS[provider]:
            raise ValueError("Exact counter requires its pinned provider image")
        from transformers import AutoTokenizer

        self.provider = provider
        self.folder = paths.tokenizer_dir(provider)
        manifest = verify_assets(provider, self.folder)
        self.tokenizer = AutoTokenizer.from_pretrained(self.folder, local_files_only=True, trust_remote_code=False)
        self.tokenizer.model_max_length = sys.maxsize
        self.limit = load_yaml("extractors")["candidates"][provider]["context_tokens"]
        if self.limit != LIMIT:
            raise ValueError("Unreviewed context policy")
        if provider == "mammoth":
            package("jevpfn_mammoth", self.folder / "source/qev")
            self.schema = source_module(self.folder, "source/qev/schema.py", "jevpfn_mammoth.schema")
            self.renderer = source_module(self.folder, "source/qev/prompt.py", "jevpfn_mammoth.prompt")
        else:
            self.config = json.loads((self.folder / "decision_config.json").read_text())
            self.codes = self.config["codes"]
            if (self.config.get("attention_mode") != "noncausal_full_attention"
                    or len(self.codes) != 255 or len(set(self.codes)) != 255
                    or [self.tokenizer.encode(c, add_special_tokens=False) for c in self.codes]
                    != [[i] for i in self.config["token_ids"]]):
                raise ValueError("Drex native attention or label mapping differs from release")
            self.renderer = pure_functions(self.folder / "code/drex_max/model.py",
                {"_text", "options", "_prompt", "answer"}, {"MAX_OPTIONS": 255, "math": math})
        self.identity = dict(assets=manifest, packages=versions, context_limit=LIMIT,
            renderer_version=f"{provider}-native-v1",
            implementation_sha256=file_sha256(paths.REPO_ROOT / "src/jev/frontier_decisions.py"),
            inherited_counter_sha256=file_sha256(paths.REPO_ROOT / "src/jev/tokenization.py"))
        self.identity["counter_id"] = digest(self.identity)

    def render_parts(self, payload):
        require_nonempty_text(payload)
        question = semantic_question(payload["question"])
        if self.provider == "mammoth":
            request = self.schema.SystemOneRequest(state=payload["state"], questions={"q": question})
            suffix = self.renderer.render_question("q", request.questions["q"]).suffix
            messages = self.renderer.build_messages(payload["state"], suffix)
        else:
            messages = self.renderer._prompt({"state": payload["state"], "question": question}, self.codes)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        return [(text, self.provider == "drex_max")]


class Runtime:
    def __init__(self, provider, folder, counter):
        self.provider, self.counter = provider, counter
        for name in asset_urls(provider):
            if file_sha256(folder / name) != file_sha256(counter.folder / name):
                raise ValueError("Inference source differs from audited source")
        if provider == "mammoth":
            # The package already points at the verified identical tokenizer-side source.
            for name in ("answers", "calibrate", "readout", "engine"):
                module = source_module(counter.folder, f"source/qev/{name}.py", f"jevpfn_mammoth.{name}")
            self.model = module.DecisionEngine(str(folder), device="cuda:0",
                max_branch_tokens=LIMIT, max_request_tokens=LIMIT, fork_mode="sequential")
            if self.model.info.extra["attention"] != "noncausal" or self.model.info.extra["readout"] != "slot_lm":
                raise ValueError("Mammoth native attention/readout not active")
        else:
            package("drex_max", folder / "code/drex_max")
            source_module(folder, "code/drex_max/types.py", "drex_max.types")
            module = source_module(folder, "code/drex_max/model.py", "drex_max.model")
            self.model = module.DecisionModel(folder, device="cuda:0", load_in_8bit=False)
            self.decode = module.answer

    def predict(self, request):
        if request.get("images") or len(request["questions"]) != 1:
            raise ValueError("JEVPFN native contract requires one text-only question")
        if self.provider == "mammoth":
            response, meta = self.model.decide(self.counter.schema.SystemOneRequest(**request))
            raw = response.model_dump(mode="json")
            raw["native_metadata"] = meta
            return raw
        key, question = next(iter(request["questions"].items()))
        rows = [dict(state=request["state"], question=question)]
        batch = self.model.prepare(rows, max_length=LIMIT)
        count = int(batch.inputs["attention_mask"].sum().item())
        probabilities = self.model.predict(rows, batch_size=1, max_length=LIMIT)[0]
        return {"answers": {key: self.decode(question, probabilities)},
                "usage": {"input_tokens": count, "output_tokens": 0},
                "raw_probabilities": probabilities}
