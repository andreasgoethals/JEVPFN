"""Text-only KnowLine and Sieve with the publishers' pinned renderers and readouts."""
import importlib
import importlib.metadata
import json
import sys
import urllib.request
from types import SimpleNamespace

from src.jev.frontier_decisions import COMMON, LIMIT, package
from src.jev.provider import semantic_question
from src.jev.registry import source_module
from src.jev.requests import require_nonempty_text
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import digest, file_sha256, write_json

MODELS = {
    "knowline": ("PelaAI/KnowLine-31B-Gen1", "a45b64060e1d6928e7e20bdffd2c482e4ebf7de6"),
    "sieve": ("sthanika-ai/Sieve-31B-v3", "ed0a046788205f0457918af8dbae3c6e8c3ee9f5"),
}
SIEVE_SOURCE = "1e64657c930f91db7119b5638774c0f11a77e441"
SIEVE_FILES = ("_kit.py", "api.py", "calibration.py", "letters/format.py", "letters/layout.py",
               "letters/attention.py", "letters/model.py", "letters/engine.py")


def asset_urls(provider):
    model, revision = MODELS[provider]
    cfg = load_yaml("extractors")["candidates"][provider]
    if (cfg["model"], cfg["revision"], cfg["context_tokens"]) != (model, revision, LIMIT):
        raise ValueError("Unreviewed checkpoint or context policy")
    extras = (("knowline_layout.py", "knowline_server.py", "temperatures.json")
              if provider == "knowline" else ("head.pt",))
    urls = {n: f"https://huggingface.co/{model}/resolve/{revision}/{n}" for n in (*COMMON, *extras)}
    if provider == "sieve":
        urls.update({"source/sieve/" + n: f"https://raw.githubusercontent.com/sthanika-ai/Sieve/{SIEVE_SOURCE}/sieve/{n}"
                     for n in SIEVE_FILES})
    return urls


def stage_tokenizer(provider):
    from src.cluster.preflight import storage_policy_failures

    if failures := storage_policy_failures():
        raise ValueError(" ".join(failures))
    directory = paths.tokenizer_dir(provider)
    directory.mkdir(parents=True, exist_ok=True)
    with file_lock(directory / ".staging.lock", timeout=2):
        files = {}
        for name, url in asset_urls(provider).items():
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with urllib.request.urlopen(url, timeout=120) as response:
                    data = response.read(40 * 1024**2 + 1)
                if len(data) > 40 * 1024**2:
                    raise ValueError("Unexpected asset size")
                temp = target.with_suffix(target.suffix + ".partial")
                temp.write_bytes(data)
                temp.replace(target)
            files[name] = {"url": url, "sha256": file_sha256(target), "bytes": target.stat().st_size}
        manifest = {"provider": provider, "revision": MODELS[provider][1], "files": files}
        marker = directory / "manifest.json"
        if marker.exists() and json.loads(marker.read_text()) != manifest:
            raise ValueError("Stored tokenizer/source identity differs")
        write_json(marker, manifest)


class Counter:
    def __init__(self, provider):
        from transformers import AutoTokenizer

        self.provider, self.limit = provider, LIMIT
        self.folder = paths.tokenizer_dir(provider)
        manifest = json.loads((self.folder / "manifest.json").read_text())
        urls = asset_urls(provider)
        if (manifest["revision"] != MODELS[provider][1] or manifest["provider"] != provider
                or set(manifest["files"]) != set(urls)):
            raise ValueError("Unexpected tokenizer/source inventory")
        for n, url in urls.items():
            record = manifest["files"][n]
            if record["url"] != url or file_sha256(self.folder / n) != record["sha256"]:
                raise ValueError("Publisher source or tokenizer differs from its staged receipt")
        tok = AutoTokenizer.from_pretrained(self.folder, local_files_only=True)
        tok.model_max_length = sys.maxsize
        self.tokenizer = tok
        if provider == "knowline":
            self.server = source_module(self.folder, "knowline_server.py", "knowline_server")
            layout = source_module(self.folder, "knowline_layout.py", "knowline_layout")
            self.layout = layout.Layout(read_twice=True, twice_max_tokens=3000)
            self.engine = self.server.KnowLine(tok, None, workers=1, layout=self.layout)
        else:
            import torch

            root = self.folder / "source/sieve"
            package("jevpfn_sieve", root)
            package("jevpfn_sieve.letters", root / "letters")
            from jevpfn_sieve.letters.model import LetterModel, escape_of, letter_format

            self.model_type = LetterModel
            self.renderer = SimpleNamespace(tok=tok, esc=escape_of(tok), fmt=letter_format(tok))
            meta = torch.load(self.folder / "head.pt", weights_only=True, map_location="cpu")
            if meta["readout"] != "letters" or meta["global_attention"] != "bidirectional":
                raise ValueError("Unexpected Sieve readout/attention")
            self.policy = {"reads": 1, "reread_max": 3000, "orders": 1, **meta["serving"]}
        packages = {k: importlib.metadata.version(k) for k in ("transformers", "tokenizers")}
        self.identity = dict(assets=manifest, packages=packages, context_limit=LIMIT,
                             implementation_sha256=file_sha256(paths.REPO_ROOT / "src/jev/arena_decisions.py"),
                             count_unit="longest_native_forward_per_request")
        self.identity["counter_id"] = digest(self.identity)

    def __call__(self, payload):
        require_nonempty_text(payload)
        from src.jev.provider import QUESTION_ID

        state, q = payload["state"], semantic_question(payload["question"])
        questions = {QUESTION_ID: q}
        if self.provider == "knowline":
            _, prompts, _ = self.engine._render(state, questions, False)
            return len(self.tokenizer.encode(prompts[QUESTION_ID][0], add_special_tokens=False))
        st, ins, opts, _ = self.model_type.encode(self.renderer, state, questions)
        lengths = []
        orders = [opts]
        if self.policy["orders"] == 2 and q["type"] == "choice" and q.get("permute") is not False:
            n = len(opts[0]) // 2
            orders.append([opts[0][n:] + opts[0][:n]])
        for order in orders:
            units = self.model_type.units(self.renderer, st, ins, order,
                                          self.policy["reads"], self.policy["reread_max"])
            lengths.extend(len(u[0]) for u in units)
        return max(lengths)

    def count_many(self, payloads):
        return [self(p) for p in payloads]


class Runtime:
    def __init__(self, provider, folder, counter):
        self.provider = provider
        for n in asset_urls(provider):
            if file_sha256(folder / n) != file_sha256(counter.folder / n):
                raise ValueError("Inference source differs from the audited source")
        if provider == "knowline":
            self.engine = counter.server.KnowLine(counter.tokenizer,
                counter.server.HF(str(folder), device="cuda:0"), workers=1,
                temperatures=json.loads((folder / "temperatures.json").read_text()), layout=counter.layout)
        else:
            module = importlib.import_module("jevpfn_sieve.letters.engine")
            self.engine = module.LettersEngine(str(folder), device="cuda:0", max_tokens=LIMIT)
            check = self.engine.m.check_length
            self.lengths = []

            def measure(units):
                check(units)
                self.lengths.extend(len(u[0]) for u in units)

            self.engine.m.check_length = measure

    def predict(self, request):
        if request.get("images") or len(request["questions"]) != 1:
            raise ValueError("Expected a single text-only decision")
        if self.provider == "knowline":
            answers, usage = self.engine.run(request["state"], request["questions"])
            return {"answers": answers, "usage": usage}
        self.lengths.clear()
        result, _ = self.engine(request["state"], request["questions"])
        result["usage"] = {"input_tokens": max(self.lengths), "output_tokens": 0,
                           "total_forward_input_tokens": sum(self.lengths)}
        return result
