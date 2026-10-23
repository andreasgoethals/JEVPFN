"""Active eleven-extractor registry; preserve the five historical counter identities.

New tokenizers/renderers are staged explicitly, without importing Torch or loading
weights. The two older fingerprinted tokenizer modules are deliberately unchanged.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
import types
import urllib.request
import zlib
from functools import lru_cache
from typing import Any

from src.jev import decision_tokenization as historical
from src.jev.provider import QUESTION_ID, semantic_question
from src.jev.requests import require_nonempty_text
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import digest, file_sha256, write_json

ADDED = ("h2o", "sii", "kev", "decision3", "nimble")
SELF_HOSTED = (*historical.SELF_HOSTED, *ADDED)
HOSTED_UNAVAILABLE = historical.HOSTED_UNAVAILABLE
PINS = {
    **historical.PINS,
    "h2o": "e7b32490c9d2a994bd75a4d1ea7e45716080175e",
    "sii": "8f5a51dcde7b0fd6a2eea3ab423bad5e3b11dfda",
    "kev": "af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf",
    "decision3": "a0f3e9e59ebd1202fe1d4de1c4c0f9a986758cbb",
    "nimble": "bd792f44ec8e265be861bfcdf4e05967ffe0e858",
}
EXTRA_FILES = {
    "h2o": (
        "h2o_lightning_shim.py",
        "pooled_readout.py",
        "serve_config.json",
        "label_rows.lightning",
    ),
    "sii": (),
    "kev": (),
    "decision3": ("d3_format.py", "decision_config.json"),
    "nimble": (
        "parallel_schema.py",
        "extended_schema.py",
        "serving_schema.py",
        "candidate_schema.py",
        "inference.py",
        "schema_config.json",
        "serving_config.json",
    ),
}
SOURCES = {
    "sii": (
        "amazingljy1206/SII-Decision",
        "1723d83cbda69f48641c273280b5a2d06d6b3510",
        ("sii_decision/prompts.py", "LICENSE"),
    ),
    "kev": (
        "jaredpalmer/kev",
        "2074cc44256200bb37540103fa4572901faa10f3",
        ("kev/__init__.py", "kev/model.py", "kev/checkpoint.py", "kev/api.py", "LICENSE"),
    ),
}


def asset_urls(provider):
    if provider not in ADDED:
        return historical.asset_urls(provider)
    c = load_yaml("extractors")["candidates"][provider]
    if c["revision"] != PINS[provider]:
        raise ValueError("Unreviewed checkpoint revision")
    base = f"https://huggingface.co/{c['model']}/resolve/{PINS[provider]}/"
    files = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json")
    urls = {name: base + name for name in (*files, *EXTRA_FILES[provider])}
    if provider == "nimble":
        urls["config.json"] = (
            f"https://huggingface.co/{c['base_model']}/resolve/{c['base_model_revision']}/config.json"
        )
    if provider in SOURCES:
        repo, revision, files = SOURCES[provider]
        if c["runtime_revision"] != revision:
            raise ValueError("Unreviewed runtime revision")
        urls.update(
            {
                "source/" + f: f"https://raw.githubusercontent.com/{repo}/{revision}/{f}"
                for f in files
            }
        )
    return urls


def stage_tokenizers(providers=SELF_HOSTED):
    from src.cluster.preflight import storage_policy_failures

    if failures := storage_policy_failures():
        raise ValueError("; ".join(failures))
    for provider in providers:
        if provider not in ADDED:
            historical.stage_tokenizers([provider])
            continue
        folder = paths.tokenizer_dir(provider)
        folder.mkdir(parents=True, exist_ok=True)
        with file_lock(folder / ".staging.lock", timeout=1):
            records = {}
            for name, url in asset_urls(provider).items():
                path = folder / name
                if not path.exists():
                    with urllib.request.urlopen(url, timeout=120) as response:
                        data = response.read(40 * 1024 * 1024 + 1)
                    if len(data) > 40 * 1024 * 1024:
                        raise ValueError("Unexpectedly large tokenizer/source asset")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(path.suffix + ".partial")
                    tmp.write_bytes(data)
                    tmp.replace(path)
                records[name] = dict(url=url, sha256=file_sha256(path), bytes=path.stat().st_size)
            identity = dict(provider=provider, revision=PINS[provider], files=records)
            manifest = folder / "manifest.json"
            if manifest.exists() and json.loads(manifest.read_text()) != identity:
                raise ValueError("Staged assets changed; preserve and review the previous assets")
            write_json(manifest, identity)
        print(f"{provider}: staged pinned tokenizer/source assets; no model weights", flush=True)


def source_module(folder, filename, name):
    """Load an explicitly staged publisher module, never a library snapshot."""
    spec = importlib.util.spec_from_file_location(name, folder / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


def pure_functions(path, names, namespace=None):
    """Extract a reviewed set of pure functions/constants, excluding model imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    ns = {"json": json, "re": re, "Any": Any, **(namespace or {})}
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError("Publisher pure renderer changed")
    code = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *selected,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), ns)
    return types.SimpleNamespace(**ns)


def option_data(question):
    kind, c = question["type"], question.get("criteria")
    if kind == "noul":
        c = c or {}
        return ["false", "true"], [c.get("false", "no"), c.get("true", "yes")]
    if kind == "score":
        return [str(i) for i in range(len(c))], list(c)
    return list(c), list(c.values())


def require_prepared_coverage(db):
    """Fail before GPU loading if any audited input lost its CPU-prepared tokens."""
    if not db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='prepared_tokens'"
    ).fetchone():
        raise ValueError("CPU-prepared token table is missing; repeat the CPU token audit")
    if db.execute(
        "SELECT 1 FROM counts AS c LEFT JOIN prepared_tokens AS p "
        "ON c.payload_sha256=p.payload_sha256 WHERE p.payload_sha256 IS NULL LIMIT 1"
    ).fetchone():
        raise ValueError(
            "CPU-prepared tokens are incomplete; repair/recount on CPU before GPU loading"
        )


class ExtendedCounter:
    def __init__(self, provider):
        import transformers
        from transformers import AutoTokenizer

        self.provider = provider
        self.folder = paths.tokenizer_dir(provider)
        manifest = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
        urls = asset_urls(provider)
        if manifest["revision"] != PINS[provider] or set(manifest["files"]) != set(urls):
            raise ValueError("Tokenizer manifest differs from the active pinned assets")
        for name, record in manifest["files"].items():
            if record["url"] != urls[name] or file_sha256(self.folder / name) != record["sha256"]:
                raise ValueError("Publisher tokenizer/source checksum mismatch")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.folder, local_files_only=True, trust_remote_code=False
        )
        self.tokenizer.model_max_length = sys.maxsize
        self.candidate = load_yaml("extractors")["candidates"][provider]
        self.limit = self.candidate["context_tokens"]
        self._load_renderer()
        self.cache_db = None
        self.cache_writable = False
        self._prepared = lru_cache(maxsize=64)(self._prepared)
        self._question = lru_cache(maxsize=256)(self._question)
        self.identity = dict(
            assets=manifest,
            transformers=transformers.__version__,
            renderer_version="extended-publisher-v1",
            context_limit=self.limit,
            implementation_sha256=file_sha256(paths.REPO_ROOT / "src/jev/registry.py"),
        )
        self.identity["counter_id"] = digest(self.identity)

    def _load_renderer(self):
        f, p = self.folder, self.provider
        if p == "sii":
            self.renderer = source_module(f, "source/sii_decision/prompts.py", "jevpfn_sii_prompts")
            self.labels = self.renderer.option_labels(self.tokenizer)
        elif p == "decision3":
            self.renderer = source_module(f, "d3_format.py", "jevpfn_d3_format")
            self.labels = json.loads((f / "decision_config.json").read_text())["codes"]
        elif p == "nimble":
            name = "jevpfn_nimble_schema"
            package = types.ModuleType(name)
            package.__path__ = [str(f)]
            sys.modules[name] = package
            for file in ("parallel_schema", "extended_schema", "serving_schema"):
                self.renderer = source_module(f, file + ".py", name + "." + file)
        elif p == "h2o":
            tree = ast.parse((f / "h2o_lightning_shim.py").read_text())
            constants = {}
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in {"PRIMITIVES", "CHATML"}:
                            constants[target.id] = ast.literal_eval(node.value)
            self.renderer = pure_functions(
                f / "h2o_lightning_shim.py",
                {
                    "render_state",
                    "desc_text",
                    "render_instr",
                    "options_of",
                    "option_lines",
                    "user_turn",
                    "render_chat",
                },
                {**constants, "Unprocessable": ValueError},
            )
            self.config = json.loads((f / "serve_config.json").read_text())
        else:
            self.api = pure_functions(
                f / "source/kev/api.py", {"render", "option_text", "question_keys", "to_record"}
            )
            self.renderer = pure_functions(
                f / "source/kev/model.py",
                {"user_tokens", "encode"},
                {
                    "MAX_STATE": sys.maxsize,
                    "MAX_BRANCH": sys.maxsize,
                    "OPT_NONE": -1,
                    "OPT_DECIDE": -2,
                    "SPECIAL": [
                        "<|fim_prefix|>",
                        "<|fim_middle|>",
                        "<|box_start|>",
                        "<|box_end|>",
                        "<|fim_suffix|>",
                    ],
                    "_SPECIAL_RE": re.compile(r"<\|([A-Za-z0-9_]+)\|>"),
                    "ContextOverflow": ValueError,
                },
            )

    def _question(self, serialized):
        q = json.loads(serialized)
        keys, descriptions = option_data(q)
        if not 2 <= len(keys) <= self.candidate["max_options"]:
            raise ValueError("Question option count exceeds the pinned provider contract")
        return q, keys, descriptions

    def attach_cache(self, db, *, writable=False):
        """One compressed prepared-token table per provider on PROJECT, not per input files."""
        self.cache_db, self.cache_writable = db, writable
        self._prepared.cache_clear()
        if writable:
            db.execute(
                "CREATE TABLE IF NOT EXISTS prepared_tokens (payload_sha256 TEXT PRIMARY KEY, prepared_sha256 TEXT NOT NULL, content BLOB NOT NULL) WITHOUT ROWID"
            )
            db.commit()

    def prepare(self, payload):
        from src.utils.serialization import canonical_json

        key = canonical_json(payload)
        return self._prepared(key)

    def _prepared(self, serialized):
        key = digest(json.loads(serialized))
        if self.cache_db is not None:
            row = self.cache_db.execute(
                "SELECT prepared_sha256,content FROM prepared_tokens WHERE payload_sha256=?", (key,)
            ).fetchone()
            if row:
                value = json.loads(zlib.decompress(row[1]))
                if digest(value) != row[0]:
                    raise ValueError("Prepared-token cache checksum mismatch")
                return value
        value = self._prepare(json.loads(serialized))
        if self.cache_db is not None and self.cache_writable:
            self.cache_db.execute(
                "INSERT INTO prepared_tokens VALUES (?,?,?)",
                (
                    key,
                    digest(value),
                    zlib.compress(json.dumps(value, ensure_ascii=False).encode(), level=1),
                ),
            )
        return value

    def _prepare(self, payload):
        require_nonempty_text(payload)
        question = semantic_question(payload["question"])
        q, keys, descriptions = self._question(json.dumps(question, ensure_ascii=False))
        state, p = payload["state"], self.provider
        extra = {}
        if p == "sii":
            row = dict(
                state=state,
                question=q["instructions"],
                options=[dict(key=k, desc=d) for k, d in zip(keys, descriptions, strict=True)],
            )
            text = self.renderer.prompt(self.tokenizer, row, self.labels)
        elif p == "decision3":
            rendered = (
                q
                if q["type"] != "score"
                else dict(
                    type="choice",
                    instructions=q["instructions"],
                    criteria={d: None for d in descriptions},
                )
            )
            text = self.renderer.render(self.tokenizer, state, rendered, self.labels)
            extra["state_tokens"] = self.renderer.encode_split(self.tokenizer, text, state)[1]
        elif p == "nimble":
            # Enum fields retain both binary class descriptions, unlike a bare boolean.
            schema = {
                QUESTION_ID: dict(
                    type="enum",
                    description=q["instructions"],
                    choices=keys,
                    choice_descriptions=dict(zip(keys, descriptions, strict=True)),
                )
            }
            prepared = self.renderer.prepare_prompts(
                self.tokenizer,
                json.dumps(state, ensure_ascii=False, sort_keys=True),
                schema,
                sys.maxsize,
            )
            return dict(
                ids=prepared.full_ids[0],
                keys=keys,
                question=q,
                schema=schema,
                candidate_ids=prepared.candidate_ids[0],
            )
        elif p == "h2o":
            c, r = self.config, self.renderer
            keys, descriptions = r.options_of(q)
            labels = c["labels"][: len(keys)]
            state_text = r.render_state(state, c["prompt"]["state_compact"])
            extra["state_tokens"] = len(self.tokenizer.encode(state_text, add_special_tokens=False))
            user = r.user_turn(
                state_text,
                r.render_instr(q["instructions"]),
                keys,
                descriptions,
                labels,
                c["prompt"]["layout"],
            )
            text = r.render_chat(c["prompt"]["system"], user, c["prompt"]["prefill"], True)
            extra["labels"] = labels
        else:
            req = types.SimpleNamespace(
                state=state,
                questions={QUESTION_ID: types.SimpleNamespace(**{"criteria": None, **q})},
            )
            record, _ = self.api.to_record(req)
            encoded = self.renderer.encode(self.tokenizer, record)
            return dict(ids=encoded["ids"], keys=keys, question=q, encoded=encoded)
        return dict(
            ids=self.tokenizer.encode(text, add_special_tokens=False),
            keys=keys,
            question=q,
            **extra,
        )

    def count_many(self, payloads):
        return [len(self.prepare(payload)["ids"]) for payload in payloads]

    def __call__(self, payload):
        return len(self.prepare(payload)["ids"])


def ExactCounter(provider):
    return ExtendedCounter(provider) if provider in ADDED else historical.ExactCounter(provider)
