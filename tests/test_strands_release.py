"""C1 publication and C2 training/pilot regressions. All stores and model doubles are local fixtures."""

import json
import math
import sqlite3
import sys
from contextlib import closing, nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.evaluation import strands_adaptation as sa
from src.jev import inputs
from src.jev import strands_hidden as sh
from src.utils.serialization import digest, file_sha256, write_json


@pytest.fixture
def hidden_store(tmp_path, monkeypatch):
    matrix = np.arange(3, 73, 2).reshape(7, 5)
    joint = set(matrix[:, -1].tolist())
    planned = sorted(matrix.ravel().tolist())
    monkeypatch.setattr(sh, "store_dir", lambda dataset_id: tmp_path)
    monkeypatch.setattr(sh, "input_plan", lambda dataset_id: (
        "inputs", {}, {"unique_inputs": len(planned)}, matrix, joint))
    (tmp_path / sh.HEAD).write_bytes(b"serialized head")
    recipe = {"schema": sh.SCHEMA, "dataset_id": "D", "input_set_id": "inputs",
              "head_sha256": file_sha256(tmp_path / sh.HEAD)}
    positions = {"tokens": 10, "state_end": 1, "question_end": 4, "answer": 9, "options": [5, 6]}
    records = [sh.encode_record(i, "noul", positions, np.ones((6, 4)),
                                np.ones((2, 4)) if i in joint else None, [0.3, 0.7], 0.0002)
               for i in planned]
    checks = [(i, 0.0001) for i in planned[:sh.config()["extraction"]["engine_check_requests"]]]
    return tmp_path / sh.STORE, recipe, records, checks


def test_c1_exact_reproduction_evidence_survives_resume_and_identity_is_stable(hidden_store, tmp_path):
    store, recipe, records, checks = hidden_store
    sh.write_records(store, recipe, records[:5], checks[:5])
    sh.write_records(store, recipe, records[5:], checks[5:])
    first = sh.finalize("D")
    original = (tmp_path / "manifest.json").read_bytes()
    assert first["engine_check"]["input_numbers"] == [n for n, _ in checks]
    assert first["engine_check"]["requests"] == 32
    assert sh.finalize("D") == sh.read_manifest("D") == first
    assert (tmp_path / "manifest.json").read_bytes() == original
    assert sh._artifact_identity(dict(first, created_utc="later")) == first["artifact_id"]


@pytest.mark.parametrize("changed", ["wrong_input", "wrong_joint", "missing_check", "wrong_check", "head"])
def test_c1_matching_counts_alone_do_not_qualify_a_store(hidden_store, changed, tmp_path):
    store, recipe, records, checks = hidden_store
    sh.write_records(store, recipe, records, checks)
    with closing(sqlite3.connect(store)) as db, db:
        if changed == "wrong_input":
            db.execute("UPDATE vectors SET input_no=10000 WHERE input_no=?", (records[-1][0],))
        elif changed == "wrong_joint":
            db.execute("UPDATE vectors SET options=CASE WHEN options IS NULL THEN X'00' ELSE NULL END "
                       "WHERE input_no IN (?,?)", (records[0][0], records[4][0]))
        elif changed == "missing_check":
            db.execute("DELETE FROM engine_checks WHERE input_no=?", (checks[0][0],))
        elif changed == "wrong_check":
            db.execute("UPDATE engine_checks SET input_no=? WHERE input_no=?", (records[-1][0], checks[0][0]))
        else:
            (tmp_path / sh.HEAD).write_bytes(b"changed")
    with pytest.raises(ValueError, match="inputs|reproduction|released head"):
        sh.finalize("D")
    assert not (tmp_path / "manifest.json").exists()


def test_c1_legacy_aggregate_flag_cannot_supply_missing_input_evidence(hidden_store, tmp_path):
    store, recipe, records, _ = hidden_store
    sh.write_records(store, recipe, records)
    with closing(sqlite3.connect(store)) as db, db:
        db.execute("DROP TABLE engine_checks")
        db.execute("INSERT INTO meta VALUES ('engine_check', ?)", (json.dumps({"status": "passed"}),))
    with pytest.raises(ValueError, match="per-input engine checks"):
        sh.finalize("D")
    write_json(tmp_path / "manifest.json", {"schema": "jevpfn.strands_hidden.v1", "status": "complete"})
    with pytest.raises(ValueError, match="per-input evidence"):
        sh.read_manifest("D")


def test_bundled_c1_views_reuse_exact_identity_and_evict_on_view_or_store_change(monkeypatch, tmp_path):
    from src.evaluation import campaign
    from src.evaluation.features import HIDDEN, PER_COLUMN_PLUS_HIDDEN

    questions = tmp_path / "questions.sqlite3"
    with closing(sqlite3.connect(questions)) as db, db:
        inputs._schema(db)
        db.execute("CREATE TABLE question_banks (bank_id TEXT PRIMARY KEY, entry_json TEXT, entry_sha256 TEXT)")
        db.execute("INSERT INTO question_banks VALUES (?,?,?)", ("bank", "{}", digest({})))
        ids = []
        for variant in ("main", "reworded"):
            recipe = {"bank_id": "bank", "entry_sha256": digest({}), "question_variant": variant}
            ids.append(digest(recipe))
            db.execute("INSERT INTO input_sets VALUES (?,?,?,?)",
                       (ids[-1], json.dumps(recipe), json.dumps({"input_set_id": ids[-1]}), "fixture"))
    inputs.activate_input_set(questions, ids[0])
    monkeypatch.setattr(sh.paths, "dataset_question_path", lambda _: questions)
    seal = {"artifact_id": "first", "recipe": {"input_set_id": ids[0]}}
    monkeypatch.setattr(sh, "read_manifest", lambda *a, **kw: seal)
    first = {"dataset_id": "D", "feature_set": HIDDEN, "hidden_view": "per_column",
             "hidden_position": "answer", "hidden_layer": 24}
    second = {**first, "hidden_view": "joint"}
    loads = []

    def view(arm):
        key = campaign._retain_for_arm(arm)
        def load():
            loads.append((arm["hidden_view"], seal["artifact_id"]))
            return {"score": [1.0]}
        return campaign._cached(key, load)

    with campaign.task_cache():
        view(first)["score"].append(2.0)
        # C1b reuses the same unmodified view as C1a.
        assert view({**first, "feature_set": PER_COLUMN_PLUS_HIDDEN, "provider": "strands"}) == {"score": [1.0]}
        assert loads == [("per_column", "first")]
        view(second)
        assert [k[2] for k in campaign._TASK_CACHE if k[0] == "hidden"] == ["joint"]
        view(first)
        inputs.activate_input_set(questions, ids[1])
        with pytest.raises(ValueError, match="active prepared inputs"):
            view(first)
        seal.update(artifact_id="second", recipe={"input_set_id": ids[1]})
        view(first)
        assert loads == [("per_column", "first"), ("joint", "first"), ("per_column", "first"),
                         ("per_column", "second")]
        campaign._retain_for_arm({"dataset_id": "D", "feature_set": "non_text"})
        assert not any(k[0] == "hidden" for k in campaign._TASK_CACHE)


def _dataset(n_train, n_test, *, kind="regression", dataset_id="D"):
    classes = [] if kind == "regression" else list(range(3 if kind == "multiclass_classification" else 2))
    y = np.arange(n_train + n_test, dtype=float) if not classes else np.arange(n_train + n_test) % len(classes)
    dataset = SimpleNamespace(metadata=SimpleNamespace(dataset_id=dataset_id, target="y", fingerprint="schema",
                                                      task_type=kind, classes=classes),
                              source={"data_identity": "raw"}, frame=pd.DataFrame({"y": y}))
    return dataset, {"split_id": "outer", "folds": [{"train": list(range(n_train)),
                                                     "test": list(range(n_train, n_train + n_test))}]}


class _FakeLoss:
    def __init__(self, weights):
        self.weights = weights

    def __mul__(self, value):
        self.weights.append(value)
        return self

    def backward(self):
        pass


class _Scored:
    def __init__(self, values):
        self.values = values

    def exp(self):
        return self

    detach = float = cpu = exp

    def numpy(self):
        return self.values


@pytest.fixture
def native_double(monkeypatch):
    calls = SimpleNamespace(train=[], weights=[], validation=1.0, peak_resets=[], optimizer=[], schedule=[],
                            score=[], casts=[])

    def autocast(device, *, dtype, enabled):
        calls.casts.append((device, dtype, enabled))
        return nullcontext()

    torch = SimpleNamespace(cuda=SimpleNamespace(reset_peak_memory_stats=lambda: calls.peak_resets.append(1),
                                                 synchronize=lambda: None, max_memory_allocated=lambda: 50),
                            no_grad=nullcontext, autocast=autocast, bfloat16="bf16",
                            nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda *a, **k: None)))
    monkeypatch.setitem(sys.modules, "torch", torch)

    class Fit:
        def __init__(self, cell, dataset, cfg, *, seed):
            self.model = self.head = SimpleNamespace(train=lambda: None, eval=lambda: None)
            self.checkpoint = "weights"

        def parameters(self):
            return []

        def optimizer(self, steps):
            calls.schedule.append(steps)
            return (SimpleNamespace(step=lambda: calls.optimizer.append(1), zero_grad=lambda **k: None),
                    SimpleNamespace(step=lambda: None))

        def prompt_batch(self, payloads, targets, kind, *, train, rng):
            if train:
                calls.train.extend(zip(payloads, targets))
            elif targets is None:
                calls.score.extend(payloads)
            values = _Scored(np.full((len(payloads), 9), 1 / 9))
            return values, [None] * len(payloads), _FakeLoss(calls.weights) if train else calls.validation

    monkeypatch.setattr(sa, "_Fit", Fit)
    return calls


@pytest.mark.parametrize("n_train", [24, 57])
def test_c2_epoch_uses_every_optimizer_row_and_isolates_regression_bins(monkeypatch, native_double, n_train):
    calls = native_double
    dataset, split = _dataset(n_train, 10)
    joint = np.arange(len(dataset.frame))
    monkeypatch.setattr(sa, "_joint_inputs", lambda _: ("questions", joint))
    monkeypatch.setattr(inputs, "iter_prepared_inputs", lambda path, **kw: (
        {"input_no": n, "payload": n} for n in sorted(kw["input_numbers"])))

    def run():
        return sa.train_fit(dataset, split, 0, 0, "lora_only", seed=5, max_epochs=1, input_set_id="questions")

    result = run()
    optimizer, val = result["optimizer_rows"], result["validation_rows"]
    assert sorted(row for row, _ in calls.train) == optimizer
    assert set(optimizer).isdisjoint(val)
    assert set(calls.score) == set(result["held_out_rows"] + split["folds"][0]["test"])
    steps = math.ceil(len(optimizer) / 32)
    assert result["history"][0]["examples"] == len(optimizer)
    assert result["history"][0]["optimizer_steps"] == len(calls.optimizer) == steps
    assert calls.schedule == [steps] and sum(calls.weights) == pytest.approx(steps)
    assert calls.peak_resets == [1] and result["peak_gpu_bytes"] == 50
    assert set(calls.casts) == {("cuda", "bf16", False)}
    original_train = list(calls.train)
    dataset.frame.loc[val, "y"] += 10000
    calls.train.clear()
    repeated = run()
    assert repeated["bin_edges"] == result["bin_edges"] and calls.train == original_train
    calls.validation = float("nan")
    with pytest.raises(ValueError, match="non-finite validation"):
        run()


def test_non_normalized_fits_cannot_cancel_in_test_average():
    train, test = np.arange(12), np.array([12])
    parts = train % 3
    predictions = []
    for k in range(3):
        values = np.full((13, 2), np.nan)
        values[train[parts == k]] = [0.5, 0.5]
        values[test] = [0.5, 0.5 + (k - 1) / 10]  # the mean sums to 1, two component fits do not
        predictions.append(values)
    with pytest.raises(ValueError, match="Each fit"):
        sa.assemble_fold(13, train, test, parts, predictions, [train[parts != k] for k in range(3)],
                         nonempty=np.ones(13, bool))


def test_native_adapter_dropout_and_gradient_checkpointing_follow_the_training_recipe():
    settings = sa.recipe_values()
    changes = []
    core = SimpleNamespace(is_gradient_checkpointing=True,
                           gradient_checkpointing_disable=lambda: changes.append("disabled"),
                           gradient_checkpointing_enable=lambda **opts: changes.append(opts))
    adapter = SimpleNamespace(lora_dropout=settings["lora_dropout"])
    torso = SimpleNamespace(peft_config={"default": adapter}, base_model=SimpleNamespace(model=core),
                            named_parameters=lambda: [("lora_A", SimpleNamespace(requires_grad=True))])
    cfg = SimpleNamespace(lora_targets=settings["lora_targets"], lora_alpha=96, lora_r=48)
    fit = sa._Fit.__new__(sa._Fit)
    fit.model, fit.recipe = SimpleNamespace(config=cfg, torso=torso), settings
    fit._check_adapter()
    fit.recipe["gradient_checkpointing"] = True
    fit._check_adapter()
    assert changes == ["disabled", {"gradient_checkpointing_kwargs": {"use_reentrant": False}}]
    adapter.lora_dropout = 0.0
    with pytest.raises(ValueError, match="LoRA dropout"):
        fit._check_adapter()


def _token_evidence(dataset_id, lengths, input_id):
    audit = {"provider": "strands", "fixture": True}
    audit_id = digest(audit)
    receipt = {"provider": "strands", "audit_id": audit_id, "status": "counted_all_inputs",
               "overflow_unique_inputs": 0, "datasets": [{"dataset_id": dataset_id, "input_set_id": input_id}]}
    write_json(sa.paths.manifests_dir("0") / "before_extraction" / "token_audit.json",
               {"self_hosted": [receipt]})
    counts = sa.paths.token_audit_dir() / f"strands_{audit_id}.sqlite3"
    counts.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(counts)) as db, db:
        db.execute("CREATE TABLE identity (id INTEGER PRIMARY KEY, recipe_json TEXT)")
        db.execute("INSERT INTO identity VALUES (1,?)", (json.dumps(audit),))
        db.execute("CREATE TABLE counts (dataset_id TEXT, input_no INTEGER, input_tokens INTEGER)")
        db.executemany("INSERT INTO counts VALUES (?,?,?)", [(dataset_id, n, int(val)) for n, val in enumerate(lengths)])


@pytest.mark.parametrize(("kind", "cell"), [(t, c) for t in sa.TASK_TYPES for c in sa.CELLS])
def test_c2_pilot_resumes_scores_seals_and_gates_only_matching_work(monkeypatch, kind, cell):
    dataset_id = sa.smallest_dataset(kind)
    dataset, outer = _dataset(280, 80, kind=kind, dataset_id=dataset_id)
    monkeypatch.setattr(sa, "_dataset_split", lambda _: (dataset, outer))
    joint = np.arange(len(dataset.frame))
    joint[[199, 299]] = -1  # missing in both outer partitions
    lengths = np.arange(1, len(joint) + 1)
    input_id = digest("questions")
    _token_evidence(dataset_id, lengths, input_id)
    monkeypatch.setattr(sa, "_joint_inputs", lambda _: (input_id, joint))
    runtime = {"container_id": digest("image"), "checkpoint": digest("weights")}
    monkeypatch.setattr(sa, "_require_native_runtime", lambda: runtime)
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    from src.evaluation.design import decision_versions

    registered, = decision_versions(["strands"])
    sources = {**runtime, "input_set_id": input_id, "decision_model": registered}
    if cell == "head_only":
        sources["hidden_artifact"] = digest("vector-store")
        monkeypatch.setattr(sh, "read_manifest", lambda _: {"artifact_id": digest("vector-store")})
    monkeypatch.setattr(sa, "_source_assets", lambda dataset_id, cell: dict(sources))
    calls, interrupted = [], [False]

    def train(dataset, split, fold, part, cell, *, seed, max_epochs, input_set_id):
        assert input_set_id == input_id
        calls.append(part)
        if part == 1 and not interrupted[0]:
            interrupted[0] = True
            raise RuntimeError("fixture interruption")
        cfg = sa.config()
        train = np.array(split["folds"][fold]["train"])
        assignment = sa.crossfit_parts(train, dataset_id=dataset_id, split_id=split["split_id"], fold=fold,
                                       seed=cfg["crossfit"]["seed"])
        opt, val, held = sa._fit_rows(train, assignment, joint, part, dataset_id=dataset_id, fold=fold, cfg=cfg)
        width = cfg["regression_bins"] if kind == "regression" else len(dataset.metadata.classes)
        probabilities = np.full((len(dataset.frame), width), np.nan)
        scored = np.concatenate([held, split["folds"][fold]["test"]])
        probabilities[scored[joint[scored] >= 0]] = 1 / width
        return {"part": part, "cell": cell, "checkpoint": runtime["checkpoint"], "probabilities": probabilities,
                "optimizer_rows": opt.tolist(), "validation_rows": val.tolist(), "held_out_rows": held.tolist(),
                "best_validation_loss": 0.5, "seconds": 2.0, "peak_gpu_bytes": 100,
                "history": [{"examples": len(opt), "optimizer_steps": math.ceil(len(opt) / 32),
                             "validation_loss": 0.5, "train_seconds": 1.0}]}

    monkeypatch.setattr(sa, "train_fit", train)
    cfg = {**sa.config(), "pilot_enabled": True}
    monkeypatch.setattr(sa, "config", lambda: cfg)
    assert not sa.pilot(task_type=kind, cell=cell)["execution"]
    with pytest.raises(FileNotFoundError):
        sa._qualified_pilot(kind, cell)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        sa.pilot(execute=True, task_type=kind, cell=cell)
    report = sa.pilot(execute=True, task_type=kind, cell=cell)
    assert calls == [0, 1, 1, 2] and report["parts"] == 3
    assert report["max_joint_tokens"] == {"train": 280, "test": 360}
    assert (report["train_rows"], report["test_rows"]) == (256, 64)
    dataset, split, recipe, folder, receipt = sa._pilot_inputs(dataset_id, cell)
    with pytest.raises(ValueError, match="reviewed"):
        sa._qualified_pilot(kind, cell)
    acceptance = receipt.parent / "approval.json"
    entry = {"path": receipt.name, "sha256": file_sha256(receipt)}
    write_json(acceptance, {"stage": "c2_adaptation_pilot", "accepted": True, "reviewed_by": "fixture",
                            "evidence": [entry]})
    cfg["pilot_acceptance"] = str(acceptance)
    assert sa._qualified_pilot(kind, cell) == report
    reloaded = sa._read_bundle(folder, recipe, dataset)
    selected = np.array(recipe["train"] + recipe["test"])
    assert np.isnan(reloaded["probabilities"][[199, 299]]).all()
    assert np.isfinite(reloaded["probabilities"][selected[joint[selected] >= 0]]).all()
    assert set(reloaded["files"]) == {"probabilities.npy", *(f"fit_{k}.{ext}" for k in range(3)
                                                               for ext in ("npy", "json"))}
    assert not (sa.bundle_dir(dataset_id, 0, cell) / "manifest.json").exists()
    assert sa.pilot(execute=True, task_type=kind, cell=cell) == report
    assert file_sha256(receipt) == entry["sha256"]
    assert calls == [0, 1, 1, 2]
    with pytest.raises(RuntimeError, match="training is disabled"):
        sa.run_fold(dataset_id, 0, cell, execute=True)
    cfg["training_enabled"] = True
    result = sa.run_fold(dataset_id, 0, cell, execute=True)
    assert calls[-3:] == [0, 1, 2]
    with monkeypatch.context() as cpu:
        cpu.setattr(sa, "_source_assets", lambda *args: pytest.fail("consumer tried to open producer assets"))
        cpu.setattr(sa, "_require_native_runtime", lambda: pytest.fail("consumer tried to select a GPU runtime"))
        assert sa.read_bundle(dataset, outer, 0, cell)["artifact_id"] == result["artifact_id"] != report["artifact_id"]
        with pytest.raises(ValueError, match="partition and recipe"):
            sa.read_bundle(dataset, {**outer, "split_id": "another split"}, 0, cell)
    sources["input_set_id"] = "changed"
    with pytest.raises(ValueError, match="pilot|partition and recipe"):
        sa.run_fold(dataset_id, 0, cell, execute=True)
    sources["input_set_id"] = input_id
    (folder / "fit_1.npy").write_bytes(b"changed after publication")
    with pytest.raises(ValueError, match="changed"):
        sa._qualified_pilot(kind, cell)
    production = sa.bundle_dir(dataset_id, 0, cell)
    (production / "fit_1.npy").unlink()
    with pytest.raises(ValueError, match="file missing or changed"):
        sa.read_bundle(dataset, outer, 0, cell)


@pytest.mark.parametrize("value", [False, "false", "true", 1, None])
def test_c2_switches_require_boolean_true_before_selecting_a_runtime(monkeypatch, value):
    cfg = {**sa.config(), "training_enabled": value, "pilot_enabled": value}
    monkeypatch.setattr(sa, "config", lambda: cfg)
    monkeypatch.setattr(sa, "_require_native_runtime", lambda: pytest.fail("an invalid switch started a native job"))
    assert all(not row["execution_enabled"] for row in sa.training_plan())
    assert not sa.pilot()["execution"]
    with pytest.raises(RuntimeError, match="pilot is disabled"):
        sa.pilot(execute=True)
    with pytest.raises(RuntimeError, match="training is disabled"):
        sa.run_fold("fixture", 0, "head_only", execute=True)


def test_c2_recipe_pins_checkpoint_image_prepared_inputs_and_head_only_vectors(tmp_path, monkeypatch):
    from src.cluster import containers, runtime_profiles
    from src.jev import checkpoints

    raw_questions = tmp_path / "questions.sqlite3"
    raw_questions.touch()
    monkeypatch.setattr(sa.paths, "dataset_question_path", lambda _: raw_questions)
    dataset, split = _dataset(48, 12)
    prepared = {"dataset_id": "D", "data_identity": "raw", "task_schema_sha256": "schema", "row_count": 60}
    monkeypatch.setattr(inputs, "_read_set", lambda *args: ("questions", prepared, None, None))
    monkeypatch.setattr(checkpoints, "checkpoint_path", lambda _: tmp_path)
    from src.evaluation.design import decision_versions

    registered, = decision_versions(["strands"])
    weights = {"provider": "strands", "model": registered["model"], "revision": checkpoints.PINS["strands"],
               "base": {"revision": registered["base_model_revision"]}, "files": {"weights": "hash"}}
    write_json(tmp_path / "_JEVPFN_CHECKPOINT.json", weights)
    monkeypatch.setattr(containers, "container_dir", lambda *args, **kwargs: tmp_path)
    seal = {"provider": "strands", "runtime_profile": runtime_profiles.BLACKWELL, "files": {"image": "hash"}}
    seal["container_id"] = digest(seal)
    write_json(tmp_path / containers.MARKER, seal)
    hidden = {"recipe": {"input_set_id": "questions", "checkpoint": digest(weights), "decision_model": registered},
              "artifact_id": "vectors"}
    monkeypatch.setattr(sh, "read_manifest", lambda *a, **kw: hidden)
    assert sa.fold_recipe(dataset, split, 0, "head_only")["sources"] == {
        "input_set_id": "questions", "checkpoint": digest(weights), "container_id": seal["container_id"],
        "hidden_artifact": "vectors",
        "decision_model": registered}
    assert "hidden_artifact" not in sa.fold_recipe(dataset, split, 0, "lora_only")["sources"]
    weights["files"]["weights"] = "another"
    write_json(tmp_path / "_JEVPFN_CHECKPOINT.json", weights)
    with pytest.raises(ValueError, match="C2 head-only vectors"):
        sa.fold_recipe(dataset, split, 0, "head_only")
    seal["container_id"] = "stale"
    write_json(tmp_path / containers.MARKER, seal)
    with pytest.raises(ValueError, match="sealed B200 runtime"):
        sa.fold_recipe(dataset, split, 0, "lora_only")
    prepared["data_identity"] = "outdated"
    with pytest.raises(ValueError, match="different rows"):
        sa.fold_recipe(dataset, split, 0, "lora_only")
