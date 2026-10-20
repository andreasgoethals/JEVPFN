"""Lazy pinned Clef/Intern runtimes. Never imported/loaded by ordinary notebooks.

Clef uses the documented low-level forward pass to retain full probability precision;
its convenience API rounds to four decimals. Both reject overflow before inference.
"""

import importlib.util
import sys

from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import file_sha256


def _module(provider, folder, filename):
    source = folder / filename
    if file_sha256(source) != file_sha256(paths.tokenizer_dir(provider) / filename):
        raise ValueError("Inference source differs from the audited publisher renderer")
    name = f"jevpfn_pinned_{provider}_runtime"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def probability_answer(question, probabilities):
    """Preserve native softmax floats; never round or rescale a provider's outputs."""
    kind = question["type"]
    if kind == "noul":
        return {"type": kind, "noul": probabilities["true"]}
    best = max(probabilities, key=probabilities.get)
    result = {"type": kind, "probabilities": probabilities, "confidence": probabilities[best]}
    if kind == "choice":
        result["choice"] = best
    else:
        result.update(
            score=sum(int(k) * p for k, p in probabilities.items()),
            legend={str(i): s for i, s in enumerate(question["criteria"])},
        )
    return result


class ClefRuntime:
    def __init__(self, folder, limit):
        self.publisher = _module("cloudflare", folder, "joint_schema_model.py")
        self.model, self.processor = self.publisher.load_release_model(
            folder, device="cuda", local_files_only=True
        )
        self.limit = limit

    def predict(self, request):
        return self.predict_many([request])[0]

    def predict_many(self, requests):
        # Upstream silently slices state at max_length. Count/encode without slicing,
        # then reject; never allow its implicit truncation to change our evidence.
        records = [
            self.publisher.encode_record(self.processor.tokenizer, request, max_length=sys.maxsize)
            for request in requests
        ]
        if any(len(record.input_ids) > self.limit for record in records):
            raise ValueError("Clef context overflow; truncation forbidden")
        device = next(self.model.parameters()).device
        batch = self.publisher.collate_records(
            records, self.processor.tokenizer.pad_token_id, device
        )
        result = []
        for request, record, logits in zip(requests, records, self.model(batch), strict=True):
            answers = {}
            for question, values in zip(record.questions, logits, strict=True):
                probabilities = dict(
                    zip(question.option_ids, values.float().softmax(-1).tolist(), strict=True)
                )
                answers[question.question_id] = probability_answer(
                    request["questions"][question.question_id], probabilities
                )
            result.append(
                {
                    "answers": answers,
                    "usage": {"input_tokens": len(record.input_ids), "output_tokens": 0},
                }
            )
        return result


def load_runtime(provider, folder, limit):
    if provider == "cloudflare":
        return ClefRuntime(folder, limit)
    if provider == "intern":
        module = _module(provider, folder, "inference.py")
        temperature = load_yaml("extractors")["candidates"][provider]["calibration_temperature"]
        if temperature != module.DEFAULT_TEMPERATURE:
            raise ValueError("Intern calibration must match the pinned publisher release")
        return module.DecisionEngine(
            checkpoint=str(folder),
            device="cuda",
            max_length=limit,
            temperature=temperature,
            backend="hf",
        )
    raise ValueError("Unsupported additional runtime")
