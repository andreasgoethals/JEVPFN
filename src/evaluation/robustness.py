"""RQ1B. Fresh, paired one-factor interventions on the frozen question bank."""

import argparse
import csv
import json
import sqlite3
from contextlib import closing
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.cluster.freeze import source_hash
from src.evaluation.research_questions import random_probe_requests
from src.jev.inputs import _read_set, iter_prepared_inputs
from src.jev.provider import response_features
from src.jev.requests import IntendedRequest
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import canonical_json, digest, write_json


def catalogue():
    with paths.config_file("question_versions.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))
    by = {r["dataset"]: r for r in rows}
    if len(by) != len(rows):
        raise ValueError("Duplicate reviewed question")
    return by


def sample(path, dataset, limit):
    cfg = load_yaml("rq1/behavior")
    wording = catalogue()[dataset]
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        identity, recipe, _, entry = _read_set(db, None)
        question = entry["variants"][recipe["question_variant"]]["question"]
        if (
            question["text"] != wording["original"]
            or len(
                {wording[k] for k in ("original", "paraphrase_1", "paraphrase_2", "paraphrase_3")}
            )
            != 4
        ):
            raise ValueError("Reviewed sentences differ from the frozen bank or each other")
        all_ids, field = {}, []
        for n, key, text in db.execute(
            "SELECT input_no,payload_sha256,text_json FROM unique_inputs WHERE input_set_id=?",
            (identity,),
        ):
            all_ids[key] = n
            if len(json.loads(text)) >= 2:
                field.append(key)

    def draw(keys, stream):
        return (
            set(
                random_probe_requests(
                    keys, dataset_id=dataset, limit=limit, seed=cfg["seed"], study=stream
                )
            )
            if keys
            else set()
        )

    words, fields = draw(all_ids, "wording"), draw(field, "field_order")
    replay = draw(words | fields, "replay")
    trials = {}
    for key in sorted(words | fields):
        variants = ["original"]
        if key in replay:
            variants += ["repeat_1", "repeat_2"]
        if key in words:
            variants += ["paraphrase_1", "paraphrase_2", "paraphrase_3"]
            if question["kind"] == "choice":
                variants += ["option_order"]
        if key in fields:
            variants += ["field_order"]
        trials[all_ids[key]] = {"key": key, "variants": variants}
    return identity, wording, trials


def intervene(original, variant, wording):
    payload = deepcopy(original)
    q = payload["question"]
    if q["text"] != wording["original"]:
        raise ValueError("Reference sentence differs from its reviewed original")
    if variant.startswith("paraphrase_"):
        q["text"] = wording[variant]
    elif variant == "field_order":
        text = payload["state"]["text"]
        if len(text) < 2:
            raise ValueError("Field order requires two observed fields")
        payload["state"]["text"] = dict(reversed(list(text.items())))
    elif variant == "option_order":
        if q["kind"] != "choice":
            raise ValueError("Option order requires multiclass")
        q["allowed_answers"].reverse()
        if "answer_descriptions" in q:
            q["answer_descriptions"].reverse()
    elif variant not in {"original", "repeat_1", "repeat_2"}:
        raise ValueError("Unknown intervention")
    return payload


def arena_inputs(counter, payload):
    from src.jev.provider import semantic_question
    from src.jev.requests import require_nonempty_text

    require_nonempty_text(payload)
    from src.jev.provider import QUESTION_ID

    state, q = payload["state"], semantic_question(payload["question"])
    questions = {QUESTION_ID: q}
    if counter.provider == "knowline":
        _, prompts, _ = counter.engine._render(state, questions, False)
        return [counter.tokenizer.encode(prompts[QUESTION_ID][0], add_special_tokens=False)]
    st, ins, opts, _ = counter.model_type.encode(counter.renderer, state, questions)
    inputs = []
    orders = [opts]
    if counter.policy["orders"] == 2 and q["type"] == "choice" and q.get("permute") is not False:
        n = len(opts[0]) // 2
        orders.append([opts[0][n:] + opts[0][:n]])
    for order in orders:
        units = counter.model_type.units(counter.renderer, st, ins, order,
                                      counter.policy["reads"], counter.policy["reread_max"])
        inputs.extend(u[0] for u in units)
    return inputs



def model_input(counter, payload):
    if getattr(counter, "provider", None) in {"knowline", "sieve"}:
        return digest(arena_inputs(counter, payload))
    if hasattr(counter, "prepare"):
        return digest(counter.prepare(payload)["ids"])
    if getattr(counter, "provider", None) in {"cloudflare", "intern"}:
        from src.jev.provider import QUESTION_ID, semantic_question

        body = {
            "state": payload["state"],
            "questions": {QUESTION_ID: semantic_question(payload["question"])},
        }
        if counter.provider == "cloudflare":
            import sys

            encoded = counter.renderer["encode_record"](
                counter.tokenizer, body, max_length=sys.maxsize
            )
            return digest(list(encoded.input_ids))
        compiled = counter.renderer["compile_row"](counter.renderer["validate_request"](body))
        ids = counter.tokenizer.apply_chat_template(
            compiled.messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
            add_vision_id=True,
            return_dict=False,
        )
        return digest(list(ids))
    parts = counter.render_parts(payload)
    return digest(
        [
            counter.tokenizer(s, add_special_tokens=special, truncation=False)["input_ids"]
            for s, special in parts
        ]
    )


def readout(request, response, variant):
    values = response_features(request, response)
    if "p_positive" in values:
        return [1 - values["p_positive"], values["p_positive"]]
    probabilities = [v for k, v in sorted(values.items()) if k.startswith("p_")]
    if variant == "option_order":
        probabilities.reverse()
    return probabilities


def effect(reference, result, regression):
    p, q = map(np.asarray, (reference, result))
    diff = q - p
    both_unique = (p == p.max()).sum() == 1 and (q == q.max()).sum() == 1
    out = dict(
        tv=float(np.abs(diff).sum() / 2),
        signed=diff.tolist(),
        flip=bool(np.argmax(p) != np.argmax(q)) if both_unique else None,
    )
    if regression:
        levels = np.arange(-4, 5)
        out.update(
            mean_change=float(diff @ levels),
            variance_change=float(diff @ (levels**2) - (q @ levels) ** 2 + (p @ levels) ** 2),
        )
    return out


def run(dataset, provider, *, pilot, acceptance, client=None):
    from src.evaluation.hosted_robustness import HostedReplay
    from src.jev.catalog import SELF_HOSTED
    from src.jev.self_hosted import SelfHostedClient

    hosted = provider == "jev" and isinstance(client, HostedReplay)
    if provider not in SELF_HOSTED and not hosted:
        raise ValueError("Use a separately budgeted hosted protocol for Jev")
    cfg = load_yaml("rq1/behavior")
    limit = min(3, cfg["requests_per_dataset"]) if pilot else cfg["requests_per_dataset"]
    path = paths.dataset_question_path(dataset)
    identity, wording, trials = sample(path, dataset, limit)
    if not pilot:
        from src.evaluation.qualification import benchmark_evidence

        benchmark_evidence(acceptance, {"phase": "experiment_1", "dataset_id": dataset,
                                        "model": "linear", "feature_set": "non_text_plus_joint",
                                        "provider": provider})
    client = client or SelfHostedClient(provider, allow_model_runs=True)
    native_signature = (
        client.input_signature if hosted else lambda payload: model_input(client.counter, payload)
    )
    design = dict(
        observability="outbound_request_body" if hosted else "tokenizer_input",
        dataset=dataset,
        provider=provider,
        settings=digest(client.settings),
        source=identity,
        experiment_code=source_hash(Path(__file__)),
        wording=digest(wording),
        sample=digest(trials),
        pilot=pilot,
        config=digest(cfg),
    )
    if not pilot:
        from src.evaluation.qualification import decision_selector

        selector = decision_selector(load_yaml("experiment_1/main"), provider)
        if selector["settings_sha256"] != design["settings"]:
            raise ValueError("Robustness settings differ from the frozen RQ2 extractor")
        passed = set()
        for dataset_id in cfg["qualifying_datasets"]:
            with closing(
                sqlite3.connect(
                    paths.dataset_question_path(dataset_id).resolve().as_uri() + "?mode=ro",
                    uri=True,
                )
            ) as bank:
                frozen_id, _, _, _ = _read_set(bank, None)
            for file in paths.results_dir("robustness", dataset_id, provider, phase="rq1").glob(
                "*.json"
            ):
                receipt = json.loads(file.read_text(encoding="utf-8"))
                tested = receipt.get("design", {})
                if (
                    tested.get("pilot") is True
                    and tested.get("dataset") == dataset_id
                    and receipt.get("completed") is True
                    and tested.get("source") == frozen_id
                    and tested.get("wording") == digest(catalogue()[dataset_id])
                    and all(
                        tested.get(k) == design[k]
                        for k in ("settings", "experiment_code", "config", "provider")
                    )
                ):
                    passed.add(dataset_id)
        if passed != set(cfg["qualifying_datasets"]):
            raise ValueError(
                "Require matching real robustness pilots across the registered datasets"
            )
    if hosted:
        client.start(design)
    folder = paths.results_dir("robustness", dataset, provider, phase="rq1")
    folder.mkdir(parents=True, exist_ok=True)
    store = folder / (digest(design) + ".sqlite3")
    with file_lock(store.with_suffix(".lock"), timeout=1), closing(sqlite3.connect(store)) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS attempts (input_no INTEGER, variant TEXT, status TEXT, detail TEXT)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS results (input_no INTEGER, variant TEXT, data TEXT, PRIMARY KEY(input_no,variant))"
        )
        db.commit()
        numbers = sorted(trials)
        counts = {}
        for start in range(0, len(numbers), 500):
            records = iter_prepared_inputs(
                path, input_set_id=identity, input_numbers=numbers[start : start + 500]
            )
            for record in records:
                n = record["input_no"]
                if record["payload_sha256"] != trials[n]["key"]:
                    raise ValueError("Sample fingerprint changed")
                original = json.loads(canonical_json(record["payload"]))
                reference_input = native_signature(original)
                reference = None
                for variant in trials[n]["variants"]:
                    found = db.execute(
                        "SELECT data FROM results WHERE input_no=? AND variant=?", (n, variant)
                    ).fetchone()
                    if found:
                        result = json.loads(found[0])
                    else:
                        payload = intervene(original, variant, wording)
                        native = native_signature(payload)
                        if variant == "field_order" and native == reference_input:
                            result = dict(status="canonicalized_by_runtime", native=native)
                        else:
                            if (
                                variant
                                in {"option_order", "paraphrase_1", "paraphrase_2", "paraphrase_3"}
                                and native == reference_input
                            ):
                                raise ValueError(
                                    "Intended intervention did not reach the tokenizer"
                                )
                            # Preserve insertion order for the field intervention. No production response cache.
                            request = IntendedRequest(
                                json.dumps(payload, ensure_ascii=False),
                                canonical_json(client.settings),
                                "{}",
                            )
                            try:
                                response = (
                                    client.evaluate_trial(request, variant)
                                    if hosted
                                    else client.evaluate(request)
                                )
                                probs = readout(request, response, variant)
                            except Exception as exc:
                                with db:
                                    db.execute(
                                        "INSERT INTO attempts VALUES(?,?,?,?)",
                                        (n, variant, "failed", type(exc).__name__),
                                    )
                                raise
                            result = dict(
                                status="measured",
                                native=native,
                                probabilities=probs,
                                response_sha256=digest(response),
                                reported_model=response.get("model"),
                                reported_version=response.get("version"),
                            )
                            if variant != "original":
                                result["effect"] = effect(
                                    reference,
                                    probs,
                                    payload["question"]["kind"] == "ordered_choice",
                                )
                        with db:
                            db.execute(
                                "INSERT INTO results VALUES(?,?,?)",
                                (n, variant, canonical_json(result)),
                            )
                            db.execute(
                                "INSERT INTO attempts VALUES(?,?,?,?)",
                                (n, variant, result["status"], None),
                            )
                    if variant == "original":
                        reference = result["probabilities"]
                    label = (variant, result["status"])
                    counts[label] = counts.get(label, 0) + 1
        if sum(counts.values()) != sum(len(v["variants"]) for v in trials.values()):
            raise ValueError("Incomplete frozen sample")
        # Keep raw text out of analysis outputs. Native-equivalent interventions
        # have separate coverage and never enter effect averages as observed zeroes.
        measures, paired = {}, {}
        for number, var, raw in db.execute("SELECT input_no,variant,data FROM results"):
            e = json.loads(raw).get("effect")
            if e:
                measures.setdefault(var, []).append(e)
                paired.setdefault(number, {})[var] = e["tv"]
        summary = {}
        for var, items in measures.items():
            flips = [x["flip"] for x in items if x["flip"] is not None]
            summary[var] = dict(
                measured=len(items),
                mean_tv=float(np.mean([x["tv"] for x in items])),
                median_tv=float(np.median([x["tv"] for x in items])),
                unique_argmax_pairs=len(flips),
                flip_rate=float(np.mean(flips)) if flips else None,
                mean_absolute_shift=np.mean(np.abs([x["signed"] for x in items]), axis=0).tolist(),
                mean_signed=np.mean([x["signed"] for x in items], axis=0).tolist(),
            )
            for key in ("mean_change", "variance_change"):
                if key in items[0]:
                    summary[var][key] = float(np.mean([x[key] for x in items]))
        contrasts = {}
        for variant in sorted(measures.keys() - {"repeat_1", "repeat_2"}):
            differences = [item[variant] - (item["repeat_1"] + item["repeat_2"]) / 2
                           for item in paired.values() if {variant, "repeat_1", "repeat_2"} <= item.keys()]
            contrasts[variant] = dict(matched_requests=len(differences),
                                      excess_tv=float(np.mean(differences)) if differences else None)
        return write_json(
            store.with_suffix(".json"),
            dict(
                design=design,
                decision_settings=client.settings,
                completed=True,
                sample=trials,
                counts=[dict(variant=v, status=s, count=n) for (v, s), n in counts.items()],
                effects=summary,
                replay_contrasts=contrasts,
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dataset")
    mode.add_argument("--pilot-suite", action="store_true")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--acceptance", type=Path)
    args = parser.parse_args()
    if args.pilot_suite:
        from src.jev.self_hosted import SelfHostedClient

        client = SelfHostedClient(args.provider, allow_model_runs=True)
        for dataset in load_yaml("rq1/behavior")["qualifying_datasets"]:
            print(
                run(dataset, args.provider, pilot=True, acceptance=None, client=client), flush=True
            )
    else:
        print(run(args.dataset, args.provider, pilot=args.pilot, acceptance=args.acceptance))
