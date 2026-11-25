"""Group C (Strands only): C1 hidden-state positions, head reproduction and store; C2 cross-fitting,
targets and averaging; arm counts; recipe provenance; comparisons and exports. No GPU, no model."""

import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation import group_c
from src.evaluation import strands_adaptation as sa
from src.jev import strands_hidden as sh
from src.utils import paths


@pytest.mark.parametrize(("feature_set", "component"), [(group_c.HIDDEN, "hidden"), (group_c.ADAPTED, "adaptation")])
def test_group_c_fits_and_exports_require_the_observed_strands_revision(feature_set, component):
    from src.evaluation import benchmark_export
    from src.evaluation.design import decision_versions

    registered, = decision_versions(["strands"])
    wrong_sources = [None, {**registered, "version": "a" * 40}, {**registered, "base_model_revision": "b" * 40},
                     {**registered, "runtime_revision": "c" * 40}]
    assert group_c.decision_model_for_fit([registered] * 5) == registered
    for wrong in wrong_sources:
        with pytest.raises(ValueError, match="Strands model and revision"):
            group_c.decision_model_for_fit([registered, wrong])
    recipes = [{"feature_set": feature_set, "representation_recipe": {component: {"decision_model": source}}}
               for source in [*wrong_sources, registered]]
    baseline = {"feature_set": "non_text", "representation_recipe": {}}
    scope = {"decision_model_versions": [registered]}
    assert list(benchmark_export.cohort_recipes([baseline, *recipes], scope, {})) == [baseline, recipes[-1]]

# ------------------------------------------------------------------ synthetic Strands prompt


def _render(content, instructions, options):
    """Strands' prompting.render_state + render_question text and option spans (choice)."""
    state = f"<state>\n{content}\n</state>\n"
    lines, spans, cursor = [], [], 0
    for i, name in enumerate(options):
        line = f"{i + 1}. {name}"
        lines.append(line)
        spans.append((cursor, cursor + len(line)))
        cursor += len(line) + 1
    prefix = f'<question type="choice">\nSelect exactly one option.\n{instructions}\n<options>\n'
    question = prefix + "\n".join(lines) + "\n</options>\n</question>\n<answer>"
    return state, question, [(len(prefix) + a, len(prefix) + b) for a, b in spans]


def _tokens(text, *, bos=False, merge_newline=True):
    """Word/punctuation tokens with offsets; a trailing newline merges into the previous token
    (as byte-level BPE merges ".\\n"), so positions are tested across merged boundaries."""
    pattern = r" ?[A-Za-z0-9]+|[^A-Za-z0-9\s]\n?|\n| +" if merge_newline else r"[A-Za-z0-9]+|\S|\s"
    offsets = [(0, 0)] if bos else []
    offsets += [(m.start(), m.end()) for m in re.finditer(pattern, text)]
    return offsets


def test_positions_on_a_synthetic_prompt_name_the_right_tokens():
    content = '{\n  "review": "Great fit."\n}'
    instructions = "Which rating did the reviewer give?"
    state, question, spans = _render(content, instructions, ["1 star", "5 stars"])
    s_off, q_off = _tokens(state, bos=True), _tokens(question)
    pos = sh.prompt_positions(state, content, question, instructions, spans, s_off, q_off)
    full = [state[a:b] for a, b in s_off] + [question[a:b] for a, b in q_off]
    assert full[pos["state_end"]].startswith("}")  # last content token, merged with "\n"
    assert full[pos["state_end"] + 1].startswith("<")  # then </state>
    assert full[pos["question_end"]].startswith("?")  # last question token, before <options>
    assert question[q_off[pos["question_end"] - len(s_off)][0]:].lstrip("?\n").startswith("<options>")
    assert full[pos["answer"]] == ">" and pos["answer"] == len(full) - 1 == pos["tokens"] - 1
    assert [full[i].strip() for i in pos["options"]] == ["star", "stars"]
    with pytest.raises(ValueError, match="<answer>"):
        sh.prompt_positions(state, content, question[:-1], instructions, spans, s_off, q_off[:-1])
    with pytest.raises(ValueError, match="state"):
        sh.prompt_positions(state, content + "x", question, instructions, spans, s_off, q_off)


def test_positions_use_the_pinned_strands_renderer_when_available():
    folder = paths.tokenizer_dir("strands")
    if not (folder / "prompting.py").is_file():
        pytest.skip("Pinned Strands renderer is not staged locally")
    from src.jev.tokenization import _strands_renderer

    r = _strands_renderer(folder)
    question = r["ChoiceQuestion"]("Which class?", {"c000": '"a"', "c001": '"b"'})
    rendered = r["render_question"](question)
    state = r["render_state"]({"task": "t", "text": {"x": "hello world."}})
    content = r["render_content"]({"task": "t", "text": {"x": "hello world."}})
    s_off, q_off = _tokens(state), _tokens(rendered.text)
    pos = sh.prompt_positions(state, content, rendered.text, "Which class?", rendered.option_spans,
                              s_off, q_off)
    assert len(pos["options"]) == 2 and pos["answer"] == pos["tokens"] - 1
    full = [state[a:b] for a, b in s_off] + [rendered.text[a:b] for a, b in q_off]
    # Documented positions: the closing token of the rendered state; the question's final token.
    assert full[pos["state_end"]].strip() == content.rstrip()[-1]
    assert full[pos["question_end"]].startswith("?")


# ------------------------------------------------------------------ released head (numpy)


def _head(rng, hidden=16, dim=8):
    return {"norm.weight": rng.normal(1, 0.1, hidden), "norm.bias": rng.normal(0, 0.1, hidden),
            "q.weight": rng.normal(size=(dim, hidden)), "q.bias": rng.normal(size=dim),
            "k.weight": rng.normal(size=(dim, hidden)), "k.bias": rng.normal(size=dim)}


def _reference(answer, options, head, temperature):
    def norm(x):
        return (x - x.mean()) / np.sqrt(x.var() + 1e-5) * head["norm.weight"] + head["norm.bias"]
    q = head["q.weight"] @ norm(answer) + head["q.bias"]
    logits = np.array([(head["k.weight"] @ norm(o) + head["k.bias"]) @ q for o in options])
    logits = logits / np.sqrt(len(q)) / temperature
    return np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()


def test_saved_answer_vector_reproduces_strands_probabilities_through_the_head():
    rng = np.random.default_rng(3)
    head = _head(rng)
    answer, options = rng.normal(size=16), rng.normal(size=(5, 16))
    reference = _reference(answer, options, head, 1.7)
    assert np.allclose(sh.head_probabilities(answer, options, head, 1.7), reference)
    batched = sh.head_probabilities(np.stack([answer, answer]), np.stack([options, options]), head, 1.7)
    assert np.allclose(batched, [reference, reference])
    deviation = sh.check_reproduction(answer.astype(np.float16), options.astype(np.float16), head, 1.7,
                                      reference, tolerance=0.005)
    assert deviation < 0.005
    with pytest.raises(ValueError, match="reproduce"):
        sh.check_reproduction(answer[::-1].astype(np.float16), options.astype(np.float16), head, 1.7,
                              reference, tolerance=0.005)
    meta = {"temperature": 1.0, "temperature_by_kind": {"choice": 1.3}}
    assert sh.temperature_for("choice", meta) == 1.3 and sh.temperature_for("noul", meta) == 1.0
    assert np.allclose(sh.published_vector({"noul": 0.25}, "noul", 2), [0.75, 0.25])
    assert np.allclose(sh.published_vector({"probabilities": {"0": 0.2, "1": 0.8}}, "score", 2), [0.2, 0.8])


def test_numpy_head_equals_torch_pointer_head_when_torch_is_available():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    head = _head(rng)
    norm = torch.nn.LayerNorm(16).double()
    q, k = torch.nn.Linear(16, 8).double(), torch.nn.Linear(16, 8).double()
    with torch.no_grad():
        for module, prefix in ((norm, "norm"), (q, "q"), (k, "k")):
            module.weight.copy_(torch.tensor(head[f"{prefix}.weight"]))
            module.bias.copy_(torch.tensor(head[f"{prefix}.bias"]))
    answer, options = rng.normal(size=(1, 16)), rng.normal(size=(1, 4, 16))
    d = q(norm(torch.tensor(answer))).unsqueeze(-1)
    logits = (k(norm(torch.tensor(options))) @ d).squeeze(-1) * 8 ** -0.5
    assert np.allclose(sh.head_logits(answer, options, head), logits.detach().numpy())


def test_store_round_trip_is_float16_exact_and_resumable(tmp_path):
    rng = np.random.default_rng(1)
    hidden, store = 16, tmp_path / "hidden.sqlite3"
    block = rng.normal(size=(6, hidden)).astype(np.float16)
    keys = rng.normal(size=(3, hidden)).astype(np.float16)
    positions = {"state_end": 1, "question_end": 4, "answer": 9, "options": [5, 6, 7], "tokens": 10}
    recipe = {"schema": sh.SCHEMA, "hidden_size": hidden}
    sh.write_records(store, recipe, [sh.encode_record(7, "choice", positions, block, keys, [0.2, 0.3, 0.5], 0.0),
                                     sh.encode_record(8, "choice", positions, block, None, [0.2, 0.3, 0.5], 0.0)])
    assert sh.done_inputs(store) == {7, 8}
    assert np.array_equal(sh.read_vectors(store, [7], "question_end", 12, hidden)[7],
                          block[sh.block_index("question_end", 12)].astype(np.float32))
    answer, options = sh.read_training_vectors(store, [7], hidden)[7]
    assert np.array_equal(answer, block[sh.block_index("answer", 24)]) and np.array_equal(options, keys)
    with pytest.raises(ValueError, match="option-end"):
        sh.read_training_vectors(store, [8], hidden)
    with pytest.raises(ValueError, match="another recipe"):
        sh.write_records(store, {**recipe, "hidden_size": 32}, [])
    head = _head(rng)
    sh.save_head(tmp_path, head, 1.0, {"score": 2.0})
    loaded, meta = sh.load_head(tmp_path)
    assert meta["temperature_by_kind"] == {"score": 2.0}
    assert all(np.allclose(loaded[k], head[k], atol=1e-6) for k in head)
    assert sh.BLOCK == (("state_end", 12), ("state_end", 24), ("question_end", 12),
                        ("question_end", 24), ("answer", 12), ("answer", 24))


@pytest.mark.filterwarnings("ignore:overflow encountered in cast:RuntimeWarning")
def test_float16_overflow_is_refused_before_storage():
    """A middle-layer residual value beyond 65,504 would become inf in float16."""
    positions = {"state_end": 1, "question_end": 4, "answer": 9, "options": [5, 6], "tokens": 10}
    block = np.zeros((6, 4), dtype=np.float32)
    block[sh.block_index("state_end", 12), 0] = 7e4
    with pytest.raises(ValueError, match="not finite in float16"):
        sh.encode_record(1, "noul", positions, block, None, [0.5, 0.5], 0.0)
    keys = np.zeros((2, 4), dtype=np.float32)
    keys[1, 2] = -1e5
    with pytest.raises(ValueError, match="not finite in float16"):
        sh.encode_record(1, "noul", positions, np.zeros((6, 4)), keys, [0.5, 0.5], 0.0)
    sh.encode_record(1, "noul", positions, np.full((6, 4), 6e4), np.zeros((2, 4)), [0.5, 0.5], 0.0)


# ------------------------------------------------------------------ C2 cross-fitting


def test_crossfit_parts_are_balanced_label_free_and_fixed():
    rows = list(range(0, 3000, 3))
    args = dict(dataset_id="D", split_id="s", fold=2, seed=20261120)
    parts = sa.crossfit_parts(rows, **args)
    assert sorted(Counter(parts.tolist()).values()) == [333, 333, 334]
    assert np.array_equal(parts, sa.crossfit_parts(rows, **args))  # deterministic
    shuffled = rows[::-1]
    again = dict(zip(shuffled, sa.crossfit_parts(shuffled, **args)))
    assert all(again[r] == p for r, p in zip(rows, parts))  # depends on row identity only
    assert not np.array_equal(parts, sa.crossfit_parts(rows, **{**args, "fold": 3}))
    import inspect

    assert not {"y", "labels", "target"} & set(inspect.signature(sa.crossfit_parts).parameters)
    validation = sa.validation_rows(rows, seed=1, key="D|2|0")
    assert len(validation) == 100 and set(validation) <= set(rows)
    assert len(sa.validation_rows(range(50_000), seed=1, key="k")) == 1000  # capped


def test_no_row_is_scored_by_a_model_that_saw_it_and_test_rows_average_three_fits():
    n, train, test = 40, np.arange(30), np.arange(30, 40)
    parts = sa.crossfit_parts(train, dataset_id="D", split_id="s", fold=0, seed=1)
    probabilities, seen = [], []
    for k in range(3):
        fit_rows, held = sa.fit_partition(train, parts, k)
        assert not set(fit_rows) & set(held) and set(fit_rows) | set(held) == set(train)
        values = np.full((n, 2), np.nan)
        values[held] = [k / 2, 1 - k / 2]
        values[test] = [0.1 * (k + 1), 1 - 0.1 * (k + 1)]
        values[test[-1]] = np.nan  # empty joint text: no request, stays missing
        probabilities.append(values)
        seen.append(fit_rows)
    nonempty = np.ones(n, dtype=bool)
    nonempty[test[-1]] = False
    out = sa.assemble_fold(n, train, test, parts, probabilities, seen, nonempty=nonempty)
    for row in train:
        assert out[row, 0] == parts[row] / 2  # scored by the fit that held out its part
    assert np.allclose(out[test[:-1], 0], 0.2) and np.isnan(out[test[-1]]).all()
    leaked = [np.concatenate([s, train[parts == 0][:1]]) for s in seen]
    with pytest.raises(ValueError, match="saw a row"):
        sa.assemble_fold(n, train, test, parts, probabilities, leaked, nonempty=nonempty)
    with pytest.raises(ValueError, match="saw a row"):
        sa.assemble_fold(n, train, test, parts, probabilities, [np.append(s, test[0]) for s in seen],
                         nonempty=nonempty)


def test_nine_equal_frequency_bins_map_lowest_first_to_strands_levels():
    rng = np.random.default_rng(0)
    training = rng.normal(size=900)
    levels, edges = sa.regression_levels(training, training)
    assert len(edges) == 8 and np.all(np.diff(edges) >= 0)
    assert sorted(Counter(levels.tolist()).values()) == [100] * 9
    assert sa.regression_levels(training, [-99, 99])[0].tolist() == [0, 8]
    tied, edges = sa.regression_levels([1] * 9 + [2] * 9, [1, 2])
    assert len(edges) == 8 and tied.tolist() == [4, 8]  # ties kept, never renumbered
    order = list(reversed(range(9)))
    target = sa.ordinal_target(0, 9, 0.1, order)
    assert target[8] == pytest.approx(0.9) and target[7] == pytest.approx(0.1) and target.sum() == pytest.approx(1)
    middle = sa.ordinal_target(4, 9, 0.1)
    assert middle[[3, 4, 5]].tolist() == pytest.approx([0.05, 0.9, 0.05])
    assert sa.class_slots(["no", "yes", "yes"], ["no", "yes"], "binary_classification").tolist() == [0, 1, 1]
    assert sa.class_slots(["b", "c"], ["a", "b", "c"], "multiclass_classification").tolist() == [1, 2]


def test_option_augmentation_follows_the_strands_collator():
    rng = random.Random(0)
    orders = [sa.option_order("score", 9, rng, 0.5) for _ in range(200)]
    assert {None, tuple(reversed(range(9)))} == {None if o is None else tuple(o) for o in orders}
    shuffled = [sa.option_order("choice", 4, rng, 0.5) for _ in range(50)]
    assert all(sorted(o) == [0, 1, 2, 3] for o in shuffled) and len({tuple(o) for o in shuffled}) > 5
    rendered = np.array([[0.1, 0.2, 0.7]])
    assert sa.canonical_probabilities(rendered, [[2, 0, 1]], 3)[0].tolist() == [0.2, 0.7, 0.1]


def test_recipe_values_carry_their_source_lines_in_the_pinned_snapshot():
    lines = (paths.REPO_ROOT / sa.SNAPSHOT).read_text(encoding="utf-8").split("\n")
    expected = {"lora_dropout": "lora_dropout: float = 0.05", "adam_betas": "(0.9, 0.95)",
                "schedule": "math.cos", "bf16_autocast": "bf16_autocast: bool = False",
                "cudnn_sdp": "enable_cudnn_sdp(False)", "temperature": "c.temperature, c.temperature_by_kind = 1.0"}
    for name, line in sa.recipe_sources().items():
        text = lines[line - 1]
        assert expected.get(name, f"{name}:") in text, (name, line, text)
    values = sa.recipe_values()
    assert (values["lora_r"], values["lora_alpha"], values["lr"], values["head_lr"]) == (16, 32, 1e-4, 1e-3)
    assert {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"} <= set(values["lora_targets"])
    assert values["ordinal_smoothing"] == 0.1 and values["reverse_score_prob"] == 0.5


class _Recorder:
    def fit(self, x, y):
        self.columns = list(x.columns)
        return self

    def predict(self, x):
        return np.zeros(len(x))


def test_adapted_features_use_exact_outer_rows_and_b1_widths():
    split = {"folds": [{"train": [0, 1, 2], "test": [3]}, {"train": [1, 2, 3], "test": [0]}]}
    dataset = type("D", (), {"metadata": type("M", (), {"task_type": "binary_classification"})()})()
    bundles = [{"probabilities": np.full((4, 2), 0.25)}, {"probabilities": np.full((4, 2), 0.75)}]
    x = pd.DataFrame({"a": [1.0, 2, 3]}, index=["csv:1", "csv:2", "csv:3"])
    control = sa.AdaptedControl(_Recorder(), dataset, split, bundles, "lora_head").fit(x, [0, 1, 0])
    assert control.fold == 1 and control.estimator.columns == ["a", "adapted__lora_head__joint__p_true"]
    with pytest.raises(ValueError, match="inner refits"):
        control.fit(x.iloc[:2], [0, 1])
    assert len(sa.feature_columns("regression", np.zeros((2, 9)), "head_only")) == 9


def test_joint_hidden_group_keeps_text_presence_flags_and_missing_rows():
    rng = np.random.default_rng(2)
    vectors = pd.DataFrame(rng.normal(size=(12, 6)), columns=[f"h{i}" for i in range(6)],
                           index=[f"csv:{i}" for i in range(12)])
    vectors.iloc[0] = np.nan  # no joint request: every text empty
    vectors.attrs.update(feature_artifact_ids=["store"], embedding_groups=[list(vectors.columns)])
    base = pd.DataFrame({"x": np.arange(12.0), "text_presence__000__missing": [1] + [0] * 11,
                         "text_presence__001__missing": [1, 1] + [0] * 10}, index=vectors.index)
    base.attrs["feature_artifact_ids"] = []
    features = group_c.add_hidden(base, vectors)
    control = group_c.hidden_control(_Recorder(), features.attrs["embedding_groups"],
                                     {"pca_components": 3}, seed=1).fit(features, np.zeros(12))
    assert control.estimator.columns[:3] == ["x", "text_presence__000__missing", "text_presence__001__missing"]
    assert len(control.estimator.columns) == 6
    assert np.isnan(control.transform(features).iloc[0, -1])


# ------------------------------------------------------------------ arms, plan and comparisons


def test_group_c_arm_counts():
    from src.evaluation.campaign import benchmark_plan

    plan = [a for a in benchmark_plan() if a["phase"] in ("representations", "rq3")]
    count = Counter(a["feature_set"] for a in plan)
    assert count == {"non_text_plus_hidden": 1512, "non_text_plus_per_column_plus_hidden": 126,
                     "non_text_plus_joint_plus_hidden": 126, "non_text_plus_adapted_joint": 378}
    one = [a for a in plan if a["dataset_id"] == "REG_TEXT_BOOK_PRICE" and a["model"] == "linear"]
    hidden = {(a["hidden_view"], a["hidden_position"], a["hidden_layer"]) for a in one
              if a["feature_set"] == "non_text_plus_hidden"}
    assert len(hidden) == 12 and {a["pca_components"] for a in one if a.get("hidden_view") == "joint"} == {64}
    assert all(a["provider"] == "strands" and a["view"] == "probabilities" for a in one
               if a["feature_set"] != "non_text_plus_hidden" and a["phase"] == "representations")
    assert {a["adaptation_cell"] for a in one if a["phase"] == "rq3"} == set(sa.CELLS)
    assert not [a for a in plan if a["width_status"] == "unsupported_width"]
    assert [(a["model"], a["dataset_id"], a["feature_set"], a["kumo_max_columns_override"])
            for a in plan if a.get("kumo_max_columns_override")] == [
        ("kumo_tabular", "REG_TEXT_ROTTEN_TOMATOES", "non_text_plus_per_column_plus_hidden", 548)]
    fits = sa.training_plan()
    assert len(fits) == 810 and sum(f["cell"] != "head_only" for f in fits) == 540
    assert not any(f["execution_enabled"] for f in fits)
    assert sa.smallest_dataset() == "REG_TEXT_BOOK_READABILITY"


def _folds(rows):
    base = dict(task_type="regression", model_id="linear", checkpoint_id="c", split_id="s")
    return pd.DataFrame([dict(base, dataset_id=d, representation=r, outer_fold=f, value=v + 0.01 * f)
                         for d, r, v in rows for f in range(5)])


def test_contrasts_are_fold_paired_and_signed_toward_a():
    head = _folds([("D1", "hidden/joint/answer/L24", 0.6), ("D1", "decision/joint/strands", 0.5),
                   ("D2", "hidden/joint/answer/L24", 0.4), ("D2", "decision/joint/strands", 0.5)])
    spec = [c for c in group_c.c1_contrasts()
            if c["comparison"] == "answer_last_vs_strands_probabilities" and c["view"] == "joint"]
    out = group_c.contrast_gains(head, spec).set_index("dataset_id")
    assert out.loc["D1", "raw_gain"] == pytest.approx(0.1) and out.loc["D2", "raw_gain"] == pytest.approx(-0.1)
    assert out.loc["D1", "headroom_share"] == pytest.approx(0.1 / (1 - 0.52))  # B's five-fold mean
    assert group_c.contrast_gains(head.iloc[:0], spec).empty
    with pytest.raises(ValueError, match="Several fold records"):  # stale duplicate records
        group_c.contrast_gains(pd.concat([head, head.iloc[:1]]), spec)
    names = {c["comparison"] for c in group_c.c2_contrasts()}
    assert {"lora_vs_zero_shot", "head_only_vs_zero_shot", "lora_vs_head_only", "lora_head_vs_lora_only"} <= names
    assert not any("frozen" in c["a"] + c["b"] for c in group_c.c1_contrasts())


def test_exports_wait_when_absent(tmp_path):
    frames, receipt = group_c.read("C1", tmp_path / "missing.zip")
    assert frames == {} and receipt["status"] == "awaiting_export"
    assert group_c.export_path("C2").parts[-3:] == ("results", "group_c", "c2_adaptation.zip")
    assert group_c.export_path("C2").parts[-5:-3] == ("C", "exp11_adaptation")
    assert group_c.export_path("C1").parts[-5:-3] == ("C", "exp10_representations")


def test_group_c_slurm_scripts_use_b200_without_a100_and_lf():
    for name in ("c1_strands_hidden.slurm", "c2_adaptation.slurm"):
        text = (paths.REPO_ROOT / "scripts/slurm" / name).read_bytes()
        assert b"\r" not in text and b"--partition=gpu_b200" in text and b"a100" not in text.lower()
    export = Path(paths.REPO_ROOT / "scripts/slurm/group_c_export.slurm").read_bytes()
    assert b"batch_graniterapids" in export and b"\r" not in export
    # C2's log belongs to C/exp11_adaptation: the documented submission overrides the C1 default.
    assert b"--output=output_JEVPFN/C/exp11_adaptation/logs/" in export
