"""Pinned d1-2B text-only renderer; no weights, Torch or inference on CPU.

Use the publisher's prompt/media modules, config and tokenizer unchanged. CPU
audits run in the same sealed Transformers 5.19 image as GPU inference. Native
inference must report exactly the audited token count before a response is saved.
"""

import ast
import json
import sys
import types
import urllib.request
from functools import lru_cache

from src.jev.provider import semantic_question
from src.jev.registry import source_module
from src.jev.requests import require_nonempty_text
from src.jev.tokenization import ExactCounter as HistoricalCounter
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import canonical_json, digest, file_sha256, write_json

PROVIDER = "liquid_lfm3"
FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "config.json",
    "prompt.py",
    "media.py",
    "lfm3.py",
    "calibration.json",
    "LICENSE",
)


def publisher_lead(folder, model_type):
    """Read the literal registration made by lfm3.py without importing its Torch code."""
    tree = ast.parse((folder / "lfm3.py").read_text(encoding="utf-8"))
    modules = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "MODULES"
    )
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if ast.unparse(call.func) == "LEADS.update" and len(call.args) == 1:
                inner = call.args[0]
                if (
                    isinstance(inner, ast.Call)
                    and ast.unparse(inner.func) == "dict.fromkeys"
                    and len(inner.args) == 2
                    and ast.unparse(inner.args[0]) == "MODULES"
                    and model_type in modules
                ):
                    lead = ast.literal_eval(inner.args[1])
                    if isinstance(lead, str):
                        return lead
    raise ValueError("Publisher LFM3 answer-lead registration changed")


def asset_urls():
    from src.jev.catalog import PINS

    candidate = load_yaml("extractors")["candidates"][PROVIDER]
    if candidate["revision"] != PINS[PROVIDER]:
        raise ValueError("Unreviewed Liquid d1-2B revision")
    base = f"https://huggingface.co/{candidate['model']}/resolve/{PINS[PROVIDER]}/"
    return {name: base + name for name in FILES}


def stage_tokenizer():
    from src.cluster.preflight import storage_policy_failures
    from src.jev.catalog import PINS

    if failures := storage_policy_failures():
        raise ValueError("; ".join(failures))
    folder = paths.tokenizer_dir(PROVIDER)
    folder.mkdir(parents=True, exist_ok=True)
    with file_lock(folder / ".staging.lock", timeout=1):
        records = {}
        for name, url in asset_urls().items():
            target = folder / name
            if not target.exists():
                with urllib.request.urlopen(url, timeout=120) as response:
                    content = response.read(40 * 1024 * 1024 + 1)
                if len(content) > 40 * 1024 * 1024:
                    raise ValueError("Unexpectedly large tokenizer/source asset")
                tmp = target.with_suffix(target.suffix + ".partial")
                tmp.write_bytes(content)
                tmp.replace(target)
            records[name] = dict(url=url, sha256=file_sha256(target), bytes=target.stat().st_size)
        manifest = dict(provider=PROVIDER, revision=PINS[PROVIDER], files=records)
        destination = folder / "manifest.json"
        if destination.exists() and json.loads(destination.read_text()) != manifest:
            raise ValueError("Staged Liquid assets changed; preserve and review them")
        write_json(destination, manifest)
    print("liquid_lfm3: staged pinned tokenizer/source only; no weights", flush=True)


class Counter(HistoricalCounter):
    def __init__(self):
        import transformers
        from transformers import AutoTokenizer

        from src.jev.catalog import PINS

        folder = paths.tokenizer_dir(PROVIDER)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        urls = asset_urls()
        if manifest["revision"] != PINS[PROVIDER] or set(manifest["files"]) != set(urls):
            raise ValueError("Liquid tokenizer manifest differs from reviewed assets")
        for name, record in manifest["files"].items():
            if record["url"] != urls[name] or file_sha256(folder / name) != record["sha256"]:
                raise ValueError("Liquid tokenizer/source checksum mismatch")
        self.provider = PROVIDER
        self.tokenizer = AutoTokenizer.from_pretrained(
            folder, local_files_only=True, trust_remote_code=False
        )
        self.tokenizer.model_max_length = sys.maxsize
        config = json.loads((folder / "config.json").read_text())
        for name, default in (
            ("d1_codes", 1),
            ("d1_numbers", False),
            ("d1_prompt", 1),
            ("d1_box_edges", "median"),
        ):
            setattr(self.tokenizer, name, config.get(name, default))
        name = "jevpfn_liquid_lfm3_prompt"
        package = types.ModuleType(name)
        package.__path__ = [str(folder)]
        sys.modules[name] = package
        source_module(folder, "media.py", name + ".media")
        self.renderer = source_module(folder, "prompt.py", name + ".prompt")
        # Importing lfm3.py registers an exact answer lead in the real GPU runner.
        # The chat-template fallback can differ (e.g. an extra newline).
        self.lead = publisher_lead(folder, config["model_type"])
        self.limit = load_yaml("extractors")["candidates"][PROVIDER]["context_tokens"]
        if self.limit != config["text_config"]["max_position_embeddings"]:
            raise ValueError("Liquid context policy differs from the reviewed checkpoint")
        self.identity = dict(
            assets=manifest,
            transformers=transformers.__version__,
            renderer_version="liquid-lfm3-single-text-v1",
            context_limit=self.limit,
            implementation_sha256=file_sha256(paths.REPO_ROOT / "src/jev/liquid_lfm3.py"),
            inherited_counter_sha256=file_sha256(paths.REPO_ROOT / "src/jev/tokenization.py"),
        )
        self.identity["counter_id"] = digest(self.identity)
        self._question = lru_cache(maxsize=256)(self._question)

    def _question(self, encoded):
        question = json.loads(encoded)
        self.renderer.check_question(question, self.tokenizer)
        q = self.renderer.as_question(question)
        groups = self.renderer.readout_ids(self.tokenizer, q)
        if not groups or any(not group for group in groups):
            raise ValueError("Liquid has no answer token for an option")
        return q

    def render_parts(self, payload):
        require_nonempty_text(payload)
        q = self._question(canonical_json(semantic_question(payload["question"])))
        return [
            (
                self.renderer.render(
                    self.tokenizer,
                    canonical_json(payload["state"]),
                    q,
                    bos=self.tokenizer.bos_token or "",
                    lead=self.lead,
                ),
                False,
            )
        ]
