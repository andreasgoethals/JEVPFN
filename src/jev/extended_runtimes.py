"""Pinned open-weight inference for the five owner-approved additions.

Only SelfHostedClient constructs these in an explicit GPU allocation. Text is
never generated. CPU-prepared token IDs are reused from the exact-token audit.
"""

import ast
import importlib
import math
import sys
from pathlib import Path

from src.jev.open_runtimes import probability_answer
from src.jev.provider import QUESTION_ID
from src.jev.registry import pure_functions, source_module


def bidirectional_hidden(model, ids, torch):
    """H2O pooling semantics: full attention is bidirectional, DeltaNet remains causal.

    This mirrors the documented pooling engine, with no generation, padding,
    truncation, state shortening or added attention-normalization step.
    """
    from transformers.masking_utils import create_recurrent_attention_mask

    hidden = model.embed_tokens(ids)
    positions = torch.arange(ids.shape[1], device=ids.device)[None, None, :].expand(4, 1, -1)
    rope = model.rotary_emb(hidden, positions[1:])
    present = torch.ones_like(ids)
    recurrent = create_recurrent_attention_mask(
        config=model.config,
        inputs_embeds=hidden,
        attention_mask=present,
        past_key_values=None,
        position_ids=positions[0],
    )
    mask = present.bool()[:, None, None, :]
    for layer, kind in zip(model.layers, model.config.layer_types, strict=True):
        hidden = layer(
            hidden,
            position_embeddings=rope,
            attention_mask=mask if kind == "full_attention" else recurrent,
            position_ids=positions[0],
            past_key_values=None,
            use_cache=False,
        )
    return model.norm(hidden)[:, -1, :]


class ExtendedRuntime:
    def __init__(self, provider, folder, counter):
        import torch
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

        self.provider, self.counter, self.torch = provider, counter, torch
        self.release_check = {"status": "not_required", "provider": provider}
        if provider == "decision3":
            # Native prefix-bidirectional readout; ordinary vLLM serving is not interchangeable.
            self.model = AutoModel.from_pretrained(
                str(folder),
                local_files_only=True,
                trust_remote_code=True,
                device="cuda",
                # verify_checkpoint already checks the complete immutable inventory;
                # do not rehash 50+ GB on an allocated GPU. Native fast verification
                # still checks the publisher runtime's source/shape manifest.
                verify="fast",
                batch_size=1,
            )
        elif provider == "kev":
            root = folder / "source"
            sys.path.insert(0, str(root))
            module = importlib.import_module("kev.checkpoint")
            if not Path(module.__file__).resolve().is_relative_to(root.resolve()):
                raise ValueError("A different Kev package is already imported")
            checkpoint = module.Checkpoint(folder)
            options = module.LoadOptions(
                dtype=torch.bfloat16, fused=False, cuda_graphs=False, backend="torch"
            )
            self.model = checkpoint._load_torch(counter.tokenizer, "cuda", options)
            self.model.head.load_state_dict(checkpoint.meta.head)
            self.model.head.temperature = checkpoint.meta.temperature
            self.model.eval()
        elif provider == "h2o":
            self.model = AutoModel.from_pretrained(
                str(folder),
                local_files_only=True,
                dtype=torch.bfloat16,
                device_map={"": "cuda"},
                attn_implementation="sdpa",
            ).eval()
            self.language = getattr(self.model, "language_model", self.model)
            self.readout = source_module(folder, "pooled_readout.py", "jevpfn_h2o_pooled")
            self.rows = self.readout.read_label_rows(folder / "label_rows.lightning")
            self.formatter = pure_functions(
                folder / "h2o_lightning_shim.py",
                {"confidence", "probabilities", "commit_noul", "format_answer"},
                {"math": math, "UpstreamError": ValueError},
            )
        else:
            location = folder if provider == "sii" else folder.parent / "nimble_base"
            config = AutoConfig.from_pretrained(str(location), local_files_only=True)
            if config.model_type == "qwen3_5":
                from transformers import Qwen3_5ForConditionalGeneration

                cls = Qwen3_5ForConditionalGeneration
            else:
                cls = AutoModelForCausalLM
            self.model, information = cls.from_pretrained(
                str(location),
                local_files_only=True,
                dtype=torch.bfloat16,
                device_map={"": "cuda"},
                attn_implementation="sdpa",
                output_loading_info=True,
            )
            if information.get("missing_keys") or information.get("mismatched_keys"):
                raise ValueError("Incomplete open-weight model loading")
            if provider == "nimble":
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(
                    self.model, str(folder), local_files_only=True
                )
            self.model.eval()
            unwrapped = self.model.get_base_model() if provider == "nimble" else self.model
            self.language = getattr(unwrapped.model, "language_model", unwrapped.model)
            self.head = self.model.get_output_embeddings()
        if provider == "h2o":
            self._release_probe()

    def _scores(self, prepared):
        t = self.torch
        ids = t.tensor([prepared["ids"]], device="cuda", dtype=t.long)
        n = len(prepared["keys"])
        if self.provider == "decision3":
            runtime = self.model.runtime
            return runtime.probabilities([prepared["ids"]], [n], [prepared["state_tokens"]])[0]
        if self.provider == "kev":
            return self.model.forward(prepared["encoded"])[0].float().softmax(-1).cpu().tolist()
        if self.provider == "h2o":
            hidden = bidirectional_hidden(self.language, ids, t)[0].float().cpu().tolist()
            groups = [self.rows.group_for(label) for label in prepared["labels"]]
            logits = self.readout.label_logits(hidden, self.rows)
            return [self.readout._logsumexp([logits[i] for i in group]) for group in groups]
        with t.autocast("cuda", dtype=t.bfloat16):
            hidden = self.language(
                input_ids=ids, attention_mask=t.ones_like(ids), use_cache=False, return_dict=True
            ).last_hidden_state[:, -1, :]
            if self.provider == "sii":
                labels = [
                    self.counter.tokenizer.encode(s, add_special_tokens=False)[0]
                    for s in self.counter.labels[:n]
                ]
            else:
                labels = prepared["candidate_ids"]
            # Do not allocate [sequence_length,vocabulary] logits.
            logits = self.head(hidden).float()[0, labels]
        return logits.softmax(-1).cpu().tolist()

    def predict_payload(self, payload):
        prepared = self.counter.prepare(payload)
        if len(prepared["ids"]) > self.counter.limit:
            raise ValueError("Context overflow; truncation forbidden")
        if (
            self.provider == "h2o"
            and prepared["state_tokens"] > self.counter.config["prompt"]["max_state_tokens"]
        ):
            raise ValueError("H2O state limit exceeded; truncation forbidden")
        if self.provider == "kev" and prepared["encoded"]["state_tokens"] > 65536:
            raise ValueError("Kev state limit exceeded; truncation forbidden")
        values = self._scores(prepared)
        q, keys = prepared["question"], prepared["keys"]
        if self.provider == "h2o":
            cfg = self.counter.config
            answer = self.formatter.format_answer(
                q["type"],
                keys,
                list(q["criteria"]) if q["type"] == "score" else [],
                values,
                cfg["two_way"]["temperature"][q["type"]],
                cfg["two_way"]["noul_floor"],
            )
        else:
            answer = probability_answer(q, dict(zip(keys, values, strict=True)))
        return {
            "answers": {QUESTION_ID: answer},
            "usage": {"input_tokens": len(prepared["ids"]), "output_tokens": 0},
        }

    def _release_probe(self):
        """Check all six publisher anchors, before any dataset inference is allowed."""
        tree = ast.parse((self.counter.folder / "h2o_lightning_shim.py").read_text())
        probes = next(
            ast.literal_eval(n.value)
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "TWO_WAY_PROBES" for t in n.targets)
        )
        cfg, renderer = self.counter.config, self.counter.renderer
        worst = 0.0
        with self.torch.inference_mode():
            for body, reference in zip(probes, cfg["two_way"]["probe_reference"], strict=True):
                state = renderer.render_state(body["state"], cfg["prompt"]["state_compact"])
                for name, question in body["questions"].items():
                    keys, descriptions = renderer.options_of(question)
                    labels = cfg["labels"][: len(keys)]
                    user = renderer.user_turn(
                        state,
                        renderer.render_instr(question.get("instructions", "")),
                        keys,
                        descriptions,
                        labels,
                        cfg["prompt"]["layout"],
                    )
                    text = renderer.render_chat(
                        cfg["prompt"]["system"], user, cfg["prompt"]["prefill"], True
                    )
                    prepared = dict(
                        ids=self.counter.tokenizer.encode(text, add_special_tokens=False),
                        keys=keys,
                        labels=labels,
                    )
                    scores = self.formatter.probabilities(self._scores(prepared), 1.0)
                    worst = max(
                        worst,
                        *(
                            abs(p - reference[name][key])
                            for key, p in zip(keys, scores, strict=True)
                        ),
                    )
        if worst > cfg["two_way"]["probe_tolerance"]:
            raise ValueError(
                f"H2O publisher attention/readout parity failed: max probability difference {worst}"
            )
        self.release_check = {
            "status": "publisher_anchors_passed",
            "maximum_probability_difference": worst,
            "noul_floor": cfg["two_way"]["noul_floor"],
        }
