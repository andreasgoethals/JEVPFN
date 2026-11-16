"""Readout variants, shared by training (train.common / train.sft) and serving (qev.engine).

attention:
  causal     the base model's mask.
  noncausal  in the full-attention layers every token attends bidirectionally within its block and causally to
             earlier blocks. A rendered prompt has two blocks: the shared prefix (system turn and state block) and
             the question suffix (question, options, answer line, generation prompt). The prefix never sees a
             suffix, so its cache is still shared exactly by all questions of a request, and one forward of a
             suffix scores all of its options. The GatedDeltaNet (linear-attention) layers are recurrent and stay
             causal.

readout:
  slot_lm    softmax over the LM-head logits of the slot letters at the answer position.
  head       a learned head over the final (normed) hidden states:
                 score_i = <w_i, h_ans> + <W_q h_ans, W_k mean(h over option i's line)> / sqrt(d)
             w has one row per slot and starts as the LM-head rows of the slot letters, W_k starts at zero, so the
             untrained head reproduces slot_lm exactly; the pointer term scores each option by its content.

Both settings are recorded next to the weights, so a checkpoint is always served the way it was trained. The
attention mask is declared per layer type under `modeling_overrides` in `config.json`, where serving stacks
read it; `readout.json` holds the rest. Checkpoints without the overrides block carry `attention` in `readout.json`.
"""
from __future__ import annotations

import bisect
import contextlib
import contextvars
import importlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

from .prompt import MAX_SLOTS

CONFIG_FILE = "readout.json"
MODEL_CONFIG_FILE = "config.json"
OVERRIDES_KEY = "modeling_overrides"
FULL_ATTENTION_MASK = {"causal": "causal", "noncausal": "block_bidirectional"}
HEAD_FILE = "readout_head.pt"
ATTENTION = ("causal", "noncausal")
READOUT = ("slot_lm", "head")


@dataclass
class ReadoutConfig:
    attention: str = "causal"
    readout: str = "slot_lm"
    head_dim: int = 256
    n_slots: int = MAX_SLOTS

    def __post_init__(self) -> None:
        if self.attention not in ATTENTION:
            raise ValueError(f"attention must be one of {ATTENTION}, got {self.attention!r}")
        if self.readout not in READOUT:
            raise ValueError(f"readout must be one of {READOUT}, got {self.readout!r}")

    @property
    def noncausal(self) -> bool:
        return self.attention == "noncausal"

    @property
    def head(self) -> bool:
        return self.readout == "head"

    @property
    def needs_layout(self) -> bool:
        return self.noncausal or self.head

    @classmethod
    def load(cls, model_dir: str | Path) -> "ReadoutConfig":
        path, model_path = Path(model_dir) / CONFIG_FILE, Path(model_dir) / MODEL_CONFIG_FILE
        fields = json.loads(path.read_text()) if path.exists() else {}
        overrides = json.loads(model_path.read_text()).get(OVERRIDES_KEY) if model_path.exists() else None
        if overrides:
            mask = overrides["attention"]["full_attention"]
            attention = {v: k for k, v in FULL_ATTENTION_MASK.items()}.get(mask)
            if attention is None:
                raise ValueError(f"unsupported full_attention mask {mask!r} in {model_path}")
            if fields.get("attention", attention) != attention:
                raise ValueError(f"{path} says attention {fields['attention']!r} but {model_path} says {mask!r}")
            fields["attention"] = attention
        return cls(**fields)

    def save(self, model_dir: str | Path) -> None:
        fields = asdict(self)
        model_path = Path(model_dir) / MODEL_CONFIG_FILE
        if model_path.exists():  # LoRA adapters have no config.json; their readout.json keeps the attention
            attention = fields.pop("attention")
            config = json.loads(model_path.read_text())
            config[OVERRIDES_KEY] = {"attention": {"full_attention": FULL_ATTENTION_MASK[attention],
                                                   "linear_attention": "causal"}, **fields}
            model_path.unlink()  # may be a symlink to the source checkpoint (model_v5/mkvariant.py)
            model_path.write_text(json.dumps(config, indent=2) + "\n")
        (Path(model_dir) / CONFIG_FILE).write_text(json.dumps(fields, indent=2))


_BLOCKS: contextvars.ContextVar = contextvars.ContextVar("qev_readout_blocks", default=None)


def install_noncausal(model) -> None:
    """Route block ids (set with `blocks`) into the full-attention mask of the model's modeling module. Outside a
    `blocks` context the model is unchanged. Needs a mask-based attention implementation (sdpa, eager, flex)."""
    impl = getattr(model.config.get_text_config(), "_attn_implementation", None) or model.config._attn_implementation
    if impl not in ("sdpa", "eager", "flex_attention"):
        raise ValueError(f"noncausal attention needs sdpa, eager or flex_attention, not {impl}")
    mod = importlib.import_module(type(model).__module__)
    if getattr(mod, "_qev_noncausal", False):
        return
    if not hasattr(mod, "create_causal_mask"):
        raise ValueError(f"{mod.__name__} builds no causal mask to extend")
    original = mod.create_causal_mask

    def create_causal_mask(*args, **kwargs):
        ids = _BLOCKS.get()
        if ids is not None and kwargs.get("block_sequence_ids") is None:
            kwargs["block_sequence_ids"] = ids
        return original(*args, **kwargs)

    mod.create_causal_mask = create_causal_mask
    mod._qev_noncausal = True


@contextlib.contextmanager
def blocks(ids):
    """Block ids [B, kv_length] over absolute positions (cached prefix included); None leaves the mask causal."""
    token = _BLOCKS.set(ids)
    try:
        yield
    finally:
        _BLOCKS.reset(token)


def block_ids(prefix_lens: Sequence[int], lengths: Sequence[int], width: int, device) -> torch.Tensor:
    """0 on the prefix, 1 on the suffix, -1 on right padding (causal, and masked as a key anyway)."""
    pos = torch.arange(width, device=device)[None]
    p = torch.tensor(list(prefix_lens), device=device)[:, None]
    n = torch.tensor(list(lengths), device=device)[:, None]
    return torch.where(pos < p, 0, torch.where(pos < n, 1, -1))


def token_layout(tokenizer, prompt: str, suffix_start: int, option_spans: Sequence[tuple[int, int]] | None):
    """Token ids of `prompt`, the prefix length (tokens that end at or before the suffix) and each option line's
    token span [start, end), from the tokenizer's character offsets. Option spans are characters of the prompt."""
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    ends = [e for _, e in offs]
    starts = [s for s, _ in offs]
    prefix = bisect.bisect_right(ends, suffix_start)
    spans = []
    for cs, ce in option_spans or []:
        a = bisect.bisect_right(ends, cs)
        b = bisect.bisect_left(starts, ce)
        if b <= a:
            raise ValueError(f"option span {cs}:{ce} covers no token")
        spans.append((a, b))
    return ids, prefix, spans


class ReadoutHead(torch.nn.Module):
    def __init__(self, hidden: int, n_slots: int = MAX_SLOTS, head_dim: int = 256):
        super().__init__()
        self.slot = torch.nn.Parameter(torch.zeros(n_slots, hidden))
        self.q = torch.nn.Linear(hidden, head_dim, bias=False)
        self.k = torch.nn.Linear(hidden, head_dim, bias=False)
        torch.nn.init.normal_(self.q.weight, std=hidden ** -0.5)
        torch.nn.init.zeros_(self.k.weight)

    @classmethod
    def from_lm_head(cls, lm_head_weight, slot_ids: Sequence[int], head_dim: int) -> "ReadoutHead":
        rows = lm_head_weight.detach()[list(slot_ids)].float()
        head = cls(rows.shape[1], rows.shape[0], head_dim)
        with torch.no_grad():
            head.slot.copy_(rows.to(head.slot.device))
        return head

    def forward(self, h_ans, h_opt):
        """h_ans [B, H], h_opt [B, K, H] -> scores [B, K] (fp32)."""
        h_ans, h_opt = h_ans.float(), h_opt.float()
        k = h_opt.shape[1]
        slot = h_ans @ self.slot[:k].T
        ptr = torch.einsum("bd,bkd->bk", self.q(h_ans), self.k(h_opt)) / math.sqrt(self.q.out_features)
        return slot + ptr


def pool_options(hidden, spans: Sequence[Sequence[tuple[int, int]]], kmax: int):
    """Mean of hidden [B, T, H] over each option's token span -> [B, kmax, H] fp32 (zeros past a row's options)."""
    bsz, _, dim = hidden.shape
    dev = hidden.device
    rows, toks, segs = [], [], []
    for b, sp in enumerate(spans):
        for k, (s, e) in enumerate(sp):
            n = e - s
            rows.append(torch.full((n,), b, dtype=torch.long))
            toks.append(torch.arange(s, e, dtype=torch.long))
            segs.append(torch.full((n,), b * kmax + k, dtype=torch.long))
    rows, toks, segs = (torch.cat(x).to(dev) for x in (rows, toks, segs))
    flat = hidden[rows, toks].float()
    total = torch.zeros(bsz * kmax, dim, dtype=torch.float32, device=dev).index_add(0, segs, flat)
    count = torch.zeros(bsz * kmax, dtype=torch.float32, device=dev).index_add(
        0, segs, torch.ones_like(segs, dtype=torch.float32))
    return (total / count.clamp(min=1)[:, None]).view(bsz, kmax, dim)


def final_norm(model):
    """The final norm of a (possibly multimodal) Qwen-style decoder: its output is the readout hidden state."""
    inner = getattr(model, "model", model)
    inner = getattr(inner, "language_model", inner)
    norm = getattr(inner, "norm", None)
    if norm is None:
        raise ValueError(f"cannot find the final norm of {type(model).__name__}")
    return norm


class HiddenCapture:
    """Keeps the final-norm output of the latest forward (with its graph, so the head trains the backbone)."""

    def __init__(self, model):
        self.value = None
        self.handle = final_norm(model).register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        self.value = output

    def pop(self):
        out, self.value = self.value, None
        if out is None:
            raise RuntimeError("no hidden state captured; the forward did not reach the final norm")
        return out


@dataclass
class Readout:
    """A model's readout at runtime: its config, and for `head` the head module and the hidden-state hook."""

    config: ReadoutConfig
    head: ReadoutHead | None = None
    capture: HiddenCapture | None = None

    @classmethod
    def attach(cls, model, config: ReadoutConfig, slot_ids: Sequence[int] | None = None,
               head_path: str | Path | None = None, device=None) -> "Readout":
        """Install the variant on `model`. A head is loaded from `head_path` when given, else initialised from the
        model's LM-head rows of `slot_ids` (call before the model is sharded)."""
        if config.noncausal:
            install_noncausal(model)
        if not config.head:
            return cls(config)
        if head_path is not None:
            state = torch.load(head_path, map_location="cpu", weights_only=True)
            head = ReadoutHead(state["slot"].shape[1], state["slot"].shape[0], state["q.weight"].shape[0])
            head.load_state_dict(state)
        else:
            head = ReadoutHead.from_lm_head(model.get_output_embeddings().weight, slot_ids[: config.n_slots],
                                            config.head_dim)
        head = head.to(device or next(model.parameters()).device)
        return cls(config, head, HiddenCapture(model))

    def save(self, out_dir: str | Path) -> None:
        self.config.save(out_dir)
        if self.head is not None:
            torch.save({k: v.detach().cpu() for k, v in self.head.state_dict().items()}, Path(out_dir) / HEAD_FILE)

    @property
    def max_slots(self) -> int:
        return self.config.n_slots
