"""Group C, C1: Strands hidden states at three prompt positions and two layers (label-free).

Strands 2B (Qwen3.5-2B-Base torso, 24 decoder layers) reads the prompt causally:

    <state>\\n TEXT \\n</state>\\n <question type=..>\\n HEADER \\n QUESTION \\n<options>\\n
    1. option ... \\n</options>\\n</question>\\n<answer>

For every frozen feature-creation request (the active prepared input set: per text column and
joint, with its exact deduplication) one forward saves, in float16, the hidden state at
``state_end`` (last token of the state text), ``question_end`` (last token of the question
text) and ``answer`` (the final ``<answer>`` token, the pointer head's query), each at layer 12
and layer 24. ``hidden_states[0]`` is the embedding output and ``hidden_states[l]`` decoder
layer l's output; ``hidden_states[24]`` must equal ``last_hidden_state``, the pointer head's
own input (checked on every request). For JOINT requests the last-layer option-end vectors
(the head's keys) are saved too; C2 trains the head on them.

Contract on every request: the released head applied to the saved float16 vectors reproduces
the probabilities of the same float32 forward through Strands' own head and fitted
temperatures (`probability_tolerance`); the first requests of each dataset are also compared
with the publisher's ``SystemOneEngine.evaluate``. Stores are per dataset on GPFS PROJECT
(SQLite, resumable by input number) with ``manifest.json`` (SHA-256 of every file).

CLI: ``--estimate`` (local, read-only storage estimate), ``--dataset D --execute`` (B200 job,
``scripts/slurm/C1_strands_hidden.slurm``), ``--dataset D --finalize``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import paths
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import digest, file_sha256, write_json

SCHEMA = "jevpfn.strands_hidden.v2"
CONFIG = "C1"
POSITIONS = ("state_end", "question_end", "answer")
LAYERS = (12, 24)
#: Order of the six vectors inside one stored block.
BLOCK = tuple((position, layer) for position in POSITIONS for layer in LAYERS)
STATE_OPEN = "<state>\n"
OPTIONS_MARK = "\n<options>\n"
ANSWER = "<answer>"
LAYER_NORM_EPS = 1e-5  # torch.nn.LayerNorm default, used by Strands' PointerHead
STORE = "hidden.sqlite3"
HEAD = "head.npz"


def config():
    cfg = load_yaml(CONFIG)
    extraction = cfg["extraction"]
    if (tuple(extraction["layers"].values()) != LAYERS or tuple(extraction["positions"]) != POSITIONS
            or extraction["num_hidden_layers"] != LAYERS[-1]):
        raise ValueError("C1 positions/layers differ from the approved 3 x (12, 24) design")
    return cfg


def block_index(position: str, layer: int) -> int:
    return BLOCK.index((position, int(layer)))


# ------------------------------------------------------------------ prompt positions (pure)


def _last_token(offsets, start: int, end: int) -> int:
    """Last token that holds a character of [start, end); special tokens have empty spans."""
    found = -1
    for j, (lo, hi) in enumerate(offsets):
        if hi > lo and lo < end and hi > start:
            found = j
    if found < 0:
        raise ValueError(f"No token covers characters [{start}, {end})")
    return found


def option_token_index(offsets, spans) -> list[int]:
    """Strands' rule (prompting.option_token_index): the last token inside each option line."""
    out = []
    for a, b in spans:
        last = -1
        for j, (lo, hi) in enumerate(offsets):
            if hi > lo and lo >= a and hi <= b:
                last = j
        if last < 0:
            raise ValueError(f"Option span ({a},{b}) has no token")
        out.append(last)
    return out


def prompt_positions(state_text, content, question_text, instructions, option_spans,
                     state_offsets, question_offsets) -> dict:
    """Token indices in the full (state + question) sequence; never guesses a position.

    State and question are tokenized separately, exactly as Strands' ``fit_window`` does
    (state with special tokens, question without), so indices are additive.
    """
    if not content or not instructions:
        raise ValueError("Empty state or question text has no request")
    start = len(STATE_OPEN)
    if not state_text.startswith(STATE_OPEN) or state_text[start:start + len(content)] != content:
        raise ValueError("State text is not <state>\\n + content")
    marker = question_text.find(OPTIONS_MARK)
    if marker < 0 or not question_text[:marker].endswith(instructions):
        raise ValueError("Question text does not end with the question before <options>")
    if not question_text.endswith(ANSWER) or question_offsets[-1][1] != len(question_text):
        raise ValueError("The prompt must end with the <answer> token")
    n_state, n_question = len(state_offsets), len(question_offsets)
    # state_end: the last token of the RENDERED state content (the request's JSON, so its
    # closing token, e.g. "}"), before the newline and </state>: an end-of-segment summary of the whole
    # state, analogous to <answer> for the full prompt (owner decision, 20 November 2026).
    # question_end: the last token of the question sentence itself (`instructions`, e.g. its
    # final "."), before the newline and <options>; neither the header nor the options are included.
    return {
        "state_end": _last_token(state_offsets, start, start + len(content)),
        "question_end": n_state + _last_token(question_offsets, marker - len(instructions), marker),
        "answer": n_state + n_question - 1,
        "options": [n_state + i for i in option_token_index(question_offsets, option_spans)],
        "tokens": n_state + n_question,
    }


# ------------------------------------------------------------------ released head (numpy)


def head_logits(answer, options, head) -> np.ndarray:
    """Strands' PointerHead: LayerNorm, q/k projections, scaled dot product (one row or many).

    ``answer`` [d] or [B, d]; ``options`` [K, d] or [B, K, d]; logits [K] or [B, K].
    """
    def norm(x):
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + LAYER_NORM_EPS) * head["norm.weight"] + head["norm.bias"]

    q = norm(answer) @ head["q.weight"].T.astype(np.float64) + head["q.bias"]
    k = norm(options) @ head["k.weight"].T.astype(np.float64) + head["k.bias"]
    scale = q.shape[-1] ** -0.5
    return np.einsum("...kd,...d->...k", k, q) * scale


def head_probabilities(answer, options, head, temperature=1.0) -> np.ndarray:
    logits = head_logits(answer, options, head) / float(temperature)
    logits = logits - logits.max(axis=-1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=-1, keepdims=True)


def temperature_for(kind: str, head_meta: dict) -> float:
    return float((head_meta.get("temperature_by_kind") or {}).get(kind, head_meta.get("temperature", 1.0)))


def check_reproduction(answer16, options16, head, temperature, reference, *, tolerance) -> float:
    """Max absolute difference between the head on stored float16 vectors and the reference."""
    observed = head_probabilities(np.asarray(answer16, np.float32), np.asarray(options16, np.float32),
                                  head, temperature)
    deviation = float(np.max(np.abs(observed - np.asarray(reference, float))))
    if not np.isfinite(deviation) or deviation > tolerance:
        raise ValueError(f"Stored vectors do not reproduce Strands' probabilities ({deviation:.5f})")
    return deviation


def published_vector(answer: dict, kind: str, n_options: int) -> np.ndarray:
    """Strands' own answer in canonical option order (noul: [P(false), P(true)])."""
    if kind == "noul":
        return np.array([1 - answer["noul"], answer["noul"]], float)
    values = list(answer["probabilities"].values())
    if len(values) != n_options:
        raise ValueError("Published answer has another option count")
    return np.asarray(values, float)


# ------------------------------------------------------------------ per-dataset store


def store_dir(dataset_id: str) -> Path:
    return paths.group_c_dir("C1", "strands_hidden", paths.output_component(dataset_id))


def _connect(path: Path, *, readonly=False):
    if readonly:
        return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    db = sqlite3.connect(path, timeout=120)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS vectors (
            input_no INTEGER PRIMARY KEY, kind TEXT NOT NULL, n_options INTEGER NOT NULL,
            tokens INTEGER NOT NULL, positions TEXT NOT NULL, block BLOB NOT NULL,
            options BLOB, probabilities BLOB NOT NULL, deviation REAL NOT NULL,
            sha256 TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS engine_checks (
            input_no INTEGER PRIMARY KEY REFERENCES vectors(input_no),
            max_abs_difference REAL NOT NULL
        ) WITHOUT ROWID;
    """)
    return db


def encode_record(input_no, kind, positions, block, options, probabilities, deviation) -> tuple:
    block = np.ascontiguousarray(block, dtype="<f2")
    if block.shape[0] != len(BLOCK):
        raise ValueError("A block holds exactly six vectors")
    # A residual-stream value beyond the float16 range would be stored as inf; refuse it here
    # instead of failing later as a "partially missing" vector in the predictor PCA.
    if not np.isfinite(block).all() or (options is not None and not np.isfinite(
            np.asarray(options, dtype="<f2")).all()):
        raise ValueError("Hidden state is not finite in float16; never stored")
    options_blob = None if options is None else np.ascontiguousarray(options, dtype="<f2").tobytes()
    probabilities = np.ascontiguousarray(probabilities, dtype="<f4")
    if (probabilities.ndim != 1 or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.isclose(probabilities.sum(), 1.0, atol=1e-5, rtol=0)
            or not np.isfinite(deviation) or deviation < 0):
        raise ValueError("Hidden-state record has invalid probabilities or reproduction error")
    payload = block.tobytes() + (options_blob or b"") + probabilities.tobytes()
    return (int(input_no), kind, int(len(probabilities)), int(positions["tokens"]),
            json.dumps(positions, sort_keys=True), block.tobytes(), options_blob,
            probabilities.tobytes(), float(deviation), hashlib.sha256(payload).hexdigest())


def write_records(path: Path, recipe: dict, records, engine_checks=()) -> None:
    with closing(_connect(path)) as db, db:
        stored = db.execute("SELECT value FROM meta WHERE key='recipe'").fetchone()
        if stored is None:
            db.execute("INSERT INTO meta VALUES ('recipe', ?)", (json.dumps(recipe, sort_keys=True),))
        elif json.loads(stored[0]) != recipe:
            raise ValueError("Hidden-state store belongs to another recipe; never mixed")
        db.executemany("INSERT INTO vectors VALUES (?,?,?,?,?,?,?,?,?,?)", records)
        db.executemany("INSERT INTO engine_checks VALUES (?,?)", engine_checks)


def done_inputs(path: Path) -> set[int]:
    if not Path(path).exists():
        return set()
    with closing(_connect(path, readonly=True)) as db:
        return {n for (n,) in db.execute("SELECT input_no FROM vectors")}


def read_vectors(path: Path, input_numbers, position: str, layer: int, hidden: int) -> dict:
    """{input_no: float32 vector} for one position/layer (only the requested slice is decoded)."""
    index = block_index(position, layer)
    wanted = sorted({int(n) for n in input_numbers})
    out = {}
    with closing(_connect(path, readonly=True)) as db:
        for start in range(0, len(wanted), 900):
            chunk = wanted[start:start + 900]
            marks = ",".join("?" for _ in chunk)
            for number, blob in db.execute(
                    f"SELECT input_no, block FROM vectors WHERE input_no IN ({marks})", chunk):
                block = np.frombuffer(blob, dtype="<f2").reshape(len(BLOCK), hidden)
                out[number] = block[index].astype(np.float32)
    if len(out) != len(wanted):
        raise ValueError("Hidden-state store lacks requested inputs")
    return out


def read_training_vectors(path: Path, input_numbers, hidden: int) -> dict:
    """{input_no: (answer L24 [d], options L24 [K, d])} for C2 head-only training."""
    index = block_index("answer", 24)
    wanted = sorted({int(n) for n in input_numbers})
    out = {}
    with closing(_connect(path, readonly=True)) as db:
        for start in range(0, len(wanted), 900):
            chunk = wanted[start:start + 900]
            marks = ",".join("?" for _ in chunk)
            for number, n_options, blob, options in db.execute(
                    f"SELECT input_no, n_options, block, options FROM vectors WHERE input_no IN ({marks})",
                    chunk):
                if options is None:
                    raise ValueError("A joint input lacks its option-end vectors")
                block = np.frombuffer(blob, dtype="<f2").reshape(len(BLOCK), hidden)
                keys = np.frombuffer(options, dtype="<f2").reshape(n_options, hidden)
                out[number] = (block[index].astype(np.float32), keys.astype(np.float32))
    if len(out) != len(wanted):
        raise ValueError("Hidden-state store lacks requested joint inputs")
    return out


def save_head(folder: Path, state: dict, temperature: float, temperature_by_kind: dict) -> Path:
    target = Path(folder) / HEAD
    partial = target.with_suffix(".partial.npz")
    np.savez(partial, **{k: np.asarray(v, dtype=np.float32) for k, v in state.items()},
             meta=np.array(json.dumps({"temperature": temperature,
                                       "temperature_by_kind": temperature_by_kind})))
    os.replace(partial, target)
    return target


def load_head(folder: Path) -> tuple[dict, dict]:
    with np.load(Path(folder) / HEAD, allow_pickle=False) as data:
        head = {k: data[k].astype(np.float64) for k in data.files if k != "meta"}
        meta = json.loads(str(data["meta"]))
    return head, meta


# ------------------------------------------------------------------ prepared inputs and rows


def input_plan(dataset_id: str):
    """(input_set_id, recipe, row x slot matrix of input numbers (-1 = empty), joint inputs)."""
    from src.jev.inputs import _read_set

    source = paths.dataset_question_path(dataset_id)
    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        identity, recipe, summary, _ = _read_set(db, None)
        slots = len(recipe["slots"])
        matrix = np.full((recipe["row_count"], slots), -1, dtype=np.int64)
        for row, slot, number in db.execute(
                "SELECT row_position, slot, input_no FROM row_map WHERE input_set_id=?", (identity,)):
            if number is not None:
                matrix[row, slot] = number
    joint = [i for i, s in enumerate(recipe["slots"]) if s["representation"] == "joint"]
    if len(joint) != 1:
        raise ValueError("Prepared inputs need exactly one joint slot")
    joint_inputs = {int(n) for n in matrix[:, joint[0]] if n >= 0}
    return identity, recipe, summary, matrix, joint_inputs


def option_count(dataset_id: str) -> int:
    task = load_yaml("dataset_semantics")["datasets"][dataset_id]
    return {"binary_classification": 2, "regression": 9}.get(task["task_type"], len(task.get("classes", [])))


def storage_estimate(hidden: int = 2048) -> pd.DataFrame:
    """Bytes per dataset from the local prepared input sets (read-only, no model)."""
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries

    rows = []
    for entry in selected_entries(load_catalog(), load_yaml("B1")):
        _, _, summary, _, joint = input_plan(entry["id"])
        k = option_count(entry["id"])
        vectors = summary["unique_inputs"] * len(BLOCK) * hidden * 2
        keys = len(joint) * k * hidden * 2
        rows.append({"dataset_id": entry["id"], "unique_inputs": summary["unique_inputs"],
                     "joint_inputs": len(joint), "options": k,
                     "gb": (vectors + keys) / 1e9})
    frame = pd.DataFrame(rows)
    return pd.concat([frame, pd.DataFrame([{"dataset_id": "TOTAL",
                                            "unique_inputs": frame.unique_inputs.sum(),
                                            "joint_inputs": frame.joint_inputs.sum(),
                                            "gb": frame.gb.sum()}])], ignore_index=True)


# ------------------------------------------------------------------ manifest


def manifest_path(dataset_id: str) -> Path:
    return store_dir(dataset_id) / "manifest.json"


def read_manifest(dataset_id: str, *, verify=True) -> dict:
    """Complete-store manifest; checksums are verified before any vector is read."""
    target = manifest_path(dataset_id)
    body = json.loads(target.read_text(encoding="utf-8"))
    if body.get("schema") != SCHEMA or body.get("status") != "complete":
        raise ValueError(f"{dataset_id}: hidden-state store lacks current per-input evidence")
    if body["artifact_id"] != _artifact_identity(body):
        raise ValueError(f"{dataset_id}: hidden-state manifest checksum differs")
    if body.get("dataset_id") != dataset_id or set(body["files"]) != {STORE, HEAD}:
        raise ValueError(f"{dataset_id}: hidden-state manifest has unexpected members")
    if verify:
        for name, checksum in body["files"].items():
            if file_sha256(target.parent / name) != checksum:
                raise ValueError(f"{dataset_id}: {name} changed after publication")
    return body


def _artifact_identity(body):
    return digest({k: v for k, v in body.items() if k not in {"created_utc", "artifact_id"}})


def active_artifact_id(dataset_id):
    """Check small sealed identities even when a worker already has this view in memory."""
    from src.jev.inputs import _read_set

    manifest = read_manifest(dataset_id, verify=False)
    questions = paths.dataset_question_path(dataset_id)
    with closing(sqlite3.connect(questions.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        active = _read_set(db, None)[0]
    if manifest["recipe"]["input_set_id"] != active:
        raise ValueError("Hidden-state store differs from the active prepared inputs")
    return manifest["artifact_id"]


def finalize(dataset_id: str) -> dict:
    """Publish the manifest only when every planned input is stored and every contract held."""
    folder = store_dir(dataset_id)
    with file_lock(folder / ".extraction.lock"):
        return _finalize(dataset_id)


def _finalize(dataset_id):
    folder = store_dir(dataset_id)
    identity, _, summary, matrix, joint = input_plan(dataset_id)
    planned = np.unique(matrix[matrix >= 0]).tolist()
    extraction = config()["extraction"]
    tolerance = extraction["probability_tolerance"]
    store = folder / STORE
    with closing(_connect(store, readonly=True)) as db:
        recipe = json.loads(db.execute("SELECT value FROM meta WHERE key='recipe'").fetchone()[0])
        stored = [n for (n,) in db.execute("SELECT input_no FROM vectors ORDER BY input_no")]
        keyed = {n for (n,) in db.execute("SELECT input_no FROM vectors WHERE options IS NOT NULL")}
        least, worst = db.execute("SELECT MIN(deviation), MAX(deviation) FROM vectors").fetchone()
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='engine_checks' AND type='table'").fetchone():
            raise ValueError(f"{dataset_id}: hidden-state store lacks per-input engine checks")
        checks = dict(db.execute("SELECT input_no, max_abs_difference FROM engine_checks ORDER BY input_no"))
    if (recipe.get("schema") != SCHEMA or recipe.get("dataset_id") != dataset_id
            or recipe["input_set_id"] != identity or stored != planned
            or len(planned) != summary["unique_inputs"] or keyed != joint):
        raise ValueError(f"{dataset_id}: stored inputs or joint option keys differ from the prepared input plan")
    expected = planned[:extraction["engine_check_requests"]]
    if (not expected or list(checks) != expected or least is None or least < 0
            or not np.isfinite(worst) or worst > tolerance
            or any(not np.isfinite(value) or not 0 <= value <= tolerance for value in checks.values())):
        raise ValueError(f"{dataset_id}: reproduction contract did not pass")
    head_hash = file_sha256(folder / HEAD)
    if recipe["head_sha256"] != head_hash:
        raise ValueError(f"{dataset_id}: released head changed after extraction")
    contracts = {"status": "passed", "requests": len(expected), "input_numbers": expected,
                 "max_abs_difference": max(checks.values()), "tolerance": tolerance}
    body = {
        "schema": SCHEMA, "status": "complete", "dataset_id": dataset_id, "recipe": recipe,
        "unique_inputs": len(planned), "joint_inputs": len(joint), "max_probability_deviation": worst,
        "engine_check": contracts, "created_utc": datetime.now(UTC).isoformat(),
        "files": {STORE: file_sha256(store), HEAD: head_hash},
    }
    body["artifact_id"] = _artifact_identity(body)
    if manifest_path(dataset_id).exists():
        previous = read_manifest(dataset_id, verify=False)
        if previous["artifact_id"] != body["artifact_id"]:
            raise ValueError(f"{dataset_id}: the published hidden-state store changed")
        return previous
    write_json(manifest_path(dataset_id), body)
    return body


# ------------------------------------------------------------------ predictor features


def column_names(view: str, position: str, layer: int, group: int, hidden: int) -> list[str]:
    scope = f"text_{group:03d}" if view == "per_column" else "joint"
    return [f"hidden__{position}__L{layer}__{scope}__{j:04d}" for j in range(hidden)]


def vector_frame(dataset, *, view: str, position: str, layer: int) -> pd.DataFrame:
    """Rows x groups of one position/layer (per column: one group per text column; joint: one
    group from the joint request). Empty text stays missing (NaN), never imputed here."""
    dataset_id = dataset.metadata.dataset_id
    manifest = read_manifest(dataset_id)
    identity, recipe, _, matrix, _ = input_plan(dataset_id)
    if (manifest["recipe"]["input_set_id"] != identity or len(matrix) != len(dataset.frame)
            or recipe["data_identity"] != dataset.source["data_identity"]
            or recipe["task_schema_sha256"] != dataset.metadata.fingerprint):
        raise ValueError("Hidden-state store differs from the active prepared inputs")
    hidden = manifest["recipe"]["hidden_size"]
    slots = [i for i, s in enumerate(recipe["slots"])
             if s["representation"] == ("per_column" if view == "per_column" else "joint")]
    numbers = {int(n) for i in slots for n in matrix[:, i] if n >= 0}
    vectors = read_vectors(store_dir(dataset_id) / STORE, numbers, position, layer, hidden)
    frames, groups = [], []
    for group, slot in enumerate(slots):
        values = np.full((len(matrix), hidden), np.nan, dtype=np.float32)
        present = matrix[:, slot] >= 0
        if present.any():
            values[present] = np.stack([vectors[int(n)] for n in matrix[present, slot]])
        names = column_names(view, position, layer, group, hidden)
        frames.append(pd.DataFrame(values, columns=names, index=[f"csv:{i}" for i in range(len(matrix))]))
        groups.append(names)
    result = pd.concat(frames, axis=1)
    result.attrs.update(feature_artifact_ids=[manifest["artifact_id"]], embedding_groups=groups,
                        decision_model=manifest["recipe"].get("decision_model"))
    return result


# ------------------------------------------------------------------ extraction (B200)


def extract(dataset_id: str, *, execute=False, limit: int | None = None) -> dict:
    """Resumable B200 extraction for one dataset. ``limit`` bounds a pilot (no manifest)."""
    cfg = config()
    extraction = cfg["extraction"]
    if not execute or extraction["enabled"] is not True:
        raise RuntimeError("C1 extraction is disabled; enable representations.extraction and pass --execute")
    if not paths.on_vsc() or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("C1 extraction runs only inside a VSC B200 allocation")
    if (os.environ.get("SLURM_CLUSTER_NAME"), os.environ.get("SLURM_JOB_PARTITION")) != ("mindwell", "gpu_b200"):
        raise RuntimeError("C1 extraction requires Mindwell gpu_b200")
    if limit is not None and limit < 1:
        raise ValueError("A pilot limit must be positive")
    with file_lock(store_dir(dataset_id) / ".extraction.lock"):
        return _extract(dataset_id, extraction, limit)


def _extract(dataset_id, extraction, limit):
    from src.evaluation.design import decision_versions
    from src.jev.self_hosted import SelfHostedClient, inference_environment

    inference_environment()
    identity, recipe_inputs, summary, matrix, joint = input_plan(dataset_id)
    client = SelfHostedClient("strands", allow_model_runs=True)  # A100/storage/freeze/checkpoint gates
    producer, = decision_versions(["strands"])
    if (client.settings["model"] != producer["model"] or client.settings["version"] != producer["version"]
            or client.settings["request_configuration"].get("runtime_revision") != producer.get("runtime_revision")):
        raise ValueError("The Strands runtime differs from its registered model revision")
    torch = client.torch
    engine, model = client.model, client.model.model
    from strands_decider import modeling, prompting

    from src.jev.inputs import iter_prepared_inputs, prepared_request
    from src.jev.provider import QUESTION_ID, semantic_question

    torso_cfg = model.torso.config
    layers = int(torso_cfg.num_hidden_layers)
    if layers != extraction["num_hidden_layers"]:
        raise ValueError(f"Strands torso has {layers} layers, not {extraction['num_hidden_layers']}")
    hidden = int(model.hidden_size(model.torso))
    folder = store_dir(dataset_id)
    folder.mkdir(parents=True, exist_ok=True)
    head_state = {k: v.detach().float().cpu().numpy() for k, v in model.head.state_dict().items()}
    head_meta = {"temperature": float(model.config.temperature),
                 "temperature_by_kind": dict(model.config.temperature_by_kind or {})}
    if not (folder / HEAD).exists():
        save_head(folder, head_state, **head_meta)
    head, stored_meta = load_head(folder)
    if stored_meta != head_meta or any(not np.array_equal(head[k], head_state[k]) for k in head_state):
        raise ValueError("Saved head differs from the loaded Strands head")
    recipe = {
        "schema": SCHEMA, "dataset_id": dataset_id, "input_set_id": identity,
        "settings_sha256": digest(client.settings), "checkpoint": client.settings["request_configuration"]["checkpoint_sha256"],
        "model": client.settings["model"], "decision_model": producer,
        "hidden_size": hidden, "num_hidden_layers": layers,
        "positions": list(POSITIONS), "layers": list(LAYERS), "block": [list(b) for b in BLOCK],
        "option_ends": "joint_inputs_layer_24", "dtype": "float16",
        "head_sha256": file_sha256(folder / HEAD), "code_sha256": file_sha256(Path(__file__)),
    }
    store = folder / STORE
    if store.exists():
        with closing(_connect(store, readonly=True)) as db:
            saved = db.execute("SELECT value FROM meta WHERE key='recipe'").fetchone()
            if not saved or json.loads(saved[0]) != recipe:
                raise ValueError("Existing hidden-state inputs belong to another recipe")
    done = done_inputs(store)
    check_ids = set(np.unique(matrix[matrix >= 0])[:extraction["engine_check_requests"]].tolist())
    tolerance = extraction["probability_tolerance"]
    backend = engine.tok.backend_tokenizer
    backend.no_padding()
    backend.no_truncation()
    pending, checks, started, new = [], [], time.perf_counter(), 0
    log = paths.logs_dir("C1") / "strands_hidden" / f"{dataset_id}_{os.environ['SLURM_JOB_ID']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    for record in iter_prepared_inputs(paths.dataset_question_path(dataset_id), input_set_id=identity):
        number = record["input_no"]
        if number in done:
            continue
        if limit is not None and new >= limit:
            break
        payload = record["payload"]
        question = semantic_question(payload["question"])
        request = client.request_class(state=payload["state"], questions={QUESTION_ID: question})
        parsed = request.questions[QUESTION_ID]
        rendered = prompting.render_question(parsed)
        state_text = prompting.render_state(request.state)
        state_enc = backend.encode(state_text, add_special_tokens=True)
        question_enc = backend.encode(rendered.text, add_special_tokens=False)
        positions = prompt_positions(
            state_text, prompting.render_content(request.state), rendered.text,
            prompting.render_content(parsed.instructions), rendered.option_spans,
            state_enc.offsets, question_enc.offsets)
        if positions["tokens"] != client.counter(payload) or positions["tokens"] > model.config.max_length:
            raise ValueError("Token count differs from the audited Strands count; stop")
        ids = torch.tensor([state_enc.ids + question_enc.ids], device="cuda")
        mask = torch.ones_like(ids)
        with torch.inference_mode():
            out = model.torso(input_ids=ids, attention_mask=mask, use_cache=False,
                              output_hidden_states=True, return_dict=True)
            states = out.hidden_states
            if len(states) != layers + 1 or not torch.equal(states[layers], out.last_hidden_state):
                raise ValueError("hidden_states indexing differs from the pointer head's input")
            last = states[layers].float()
            decide = modeling.pool_last_token(last, mask)  # the head's own query (Strands code)
            opt_idx = torch.tensor([positions["options"]], device="cuda")
            keys = modeling.gather_options(last, opt_idx)
            if not torch.equal(decide[0], last[0, positions["answer"]]):
                raise ValueError("The <answer> position differs from Strands' pooled query")
            logits = model.head(decide, keys)
            kind = rendered.kind
            temperature = temperature_for(kind, head_meta)
            probabilities = torch.softmax(logits[0] / temperature, dim=-1).double().cpu().numpy()
            block = torch.stack([states[layer][0, positions[pos]] for pos, layer in BLOCK])
            block = block.float().cpu().numpy().astype(np.float16)
            keys16 = keys[0].cpu().numpy().astype(np.float16)
        deviation = check_reproduction(block[block_index("answer", 24)], keys16, head, temperature,
                                       probabilities, tolerance=tolerance)
        if number in check_ids:
            response = client.evaluate(prepared_request(record, settings=client.settings,
                                                        provenance={"dataset_id": dataset_id}))
            published = published_vector(response["publisher_response"]["answers"][QUESTION_ID], kind,
                                         len(probabilities))
            engine_deviation = float(np.max(np.abs(published - probabilities)))
            if not np.isfinite(engine_deviation) or engine_deviation > tolerance:
                raise ValueError(f"Strands engine answer differs from the extraction forward ({engine_deviation:.5f})")
            checks.append((number, engine_deviation))
        pending.append(encode_record(number, kind, positions, block,
                                     keys16 if number in joint else None, probabilities, deviation))
        new += 1
        if len(pending) >= extraction["commit_every"]:
            write_records(store, recipe, pending, checks)
            pending, checks = [], []
            print(f"{dataset_id}: {len(done) + new:,}/{summary['unique_inputs']:,} inputs", flush=True)
    if pending:
        write_records(store, recipe, pending, checks)
    report = {"dataset_id": dataset_id, "new_inputs": new, "stored": len(done) + new,
              "planned": summary["unique_inputs"], "seconds": time.perf_counter() - started,
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
              "job_id": os.environ.get("SLURM_JOB_ID")}
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report) + "\n")
    if limit is None:
        report["manifest"] = _finalize(dataset_id)["artifact_id"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--limit", type=int, help="pilot: at most this many new inputs, no manifest")
    args = parser.parse_args()
    if args.estimate:
        print(storage_estimate().to_string(index=False))
    elif args.finalize:
        print(json.dumps(finalize(args.dataset), indent=2))
    elif args.execute:
        print(json.dumps(extract(args.dataset, execute=True, limit=args.limit), indent=2))
    else:
        print("C1 extraction prepared; pass --estimate, or --dataset D --execute in a B200 job.")


if __name__ == "__main__":
    main()
