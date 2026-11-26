"""Exercise the routes used by RQ1 and RQ2 during deterministic B0 preflight."""

import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.evaluation import robustness as robust
from src.evaluation.geometry import encoder_distances
from src.evaluation.geometry_study import neighbours, relation
from src.evaluation.row_benefit import paired_cohorts, row_losses
from src.jev.inputs import prepare_dataset_inputs
from tests.test_prepared_inputs import preparation as preparation


def test_replay_is_fresh_and_resume_keeps_only_completed_trials(request, monkeypatch, tmp_path):
    dataset, bank = request.getfixturevalue("preparation")
    saved = prepare_dataset_inputs(dataset, bank, variant="accepted")
    ds = dataset.metadata.dataset_id
    original = bank["datasets"][ds]["variants"]["accepted"]["question"]["text"]
    wording = dict(
        original=original, paraphrase_1="One?", paraphrase_2="Two?", paraphrase_3="Three?"
    )
    monkeypatch.setattr(robust, "catalogue", lambda: {ds: wording})
    monkeypatch.setattr(robust.paths, "dataset_question_path", lambda _: saved["path"])
    monkeypatch.setattr(
        robust.paths, "results_dir", lambda *args, **kwargs: tmp_path.joinpath(*args)
    )

    class Counter:
        def prepare(self, p):
            return dict(ids=list(json.dumps(p, sort_keys=True).encode()))

    calls = []

    def evaluate(r):
        calls.append(r.payload_json)
        if len(calls) == 3:
            raise RuntimeError("interrupted")
        return {"score": len(calls) % 2, "model": "strands-test", "version": "test-revision"}

    client = SimpleNamespace(settings={"provider": "strands", "model": "strands-test", "version": "test-revision"},
                             counter=Counter(), evaluate=evaluate)
    monkeypatch.setattr(robust, "readout", lambda r, s, v: [s["score"], 1 - s["score"]])
    with pytest.raises(RuntimeError, match="interrupted"):
        robust.run(ds, "strands", pilot=True, acceptance=None, client=client)
    report = robust.run(ds, "strands", pilot=True, acceptance=None, client=client)
    result = json.loads(report.read_text())
    assert result["completed"]
    assert result["decision_settings"] == client.settings
    with sqlite3.connect(report.with_suffix(".sqlite3")) as db:
        assert db.execute("SELECT DISTINCT json_extract(data,'$.reported_model'), json_extract(data,'$.reported_version') "
                          "FROM results WHERE json_extract(data,'$.status')='measured'").fetchall() == [
                              ("strands-test", "test-revision")]
    counts = {(r["variant"], r["status"]): r["count"] for r in result["counts"]}
    assert counts["field_order", "canonicalized_by_runtime"] >= 1
    assert counts["repeat_1", "measured"] == counts["repeat_2", "measured"] == 3
    n = len(calls)
    assert n == 1 + sum(count for (v, s), count in counts.items() if s == "measured")
    assert robust.run(ds, "strands", pilot=True, acceptance=None, client=client) == report
    assert len(calls) == n


def test_option_alignment_and_replay_ties(monkeypatch):
    payload = {
        "state": {"text": {"a": "Hello", "b": "World"}},
        "question": {
            "text": "Which?",
            "kind": "choice",
            "allowed_answers": ["a", "b", "c"],
            "answer_descriptions": ["A", "B", "C"],
        },
    }
    changed = robust.intervene(payload, "option_order", {"original": "Which?"})
    assert changed["question"]["answer_descriptions"] == ["C", "B", "A"]
    assert payload["question"]["allowed_answers"] == ["a", "b", "c"]
    monkeypatch.setattr(
        robust, "response_features", lambda *a: dict(p_c001=0.3, p_c002=0.2, p_c000=0.5)
    )
    assert robust.readout(None, None, "option_order") == [0.2, 0.3, 0.5]
    assert robust.effect([0.5, 0.5], [0.0, 1.0], False)["flip"] is None
    a, b = np.eye(9)[[0, 8]]
    assert robust.effect(a, b, True)["mean_change"] == 8


def test_distance_null_resamples_identities_and_handles_ties():
    rng = np.random.default_rng(42)
    dist = encoder_distances(rng.normal(size=(25, 4)))
    out = relation(dist, dist, 1)
    assert out["spearman"] == pytest.approx(1)
    assert out["neighbour_overlap"] == pytest.approx(1)
    assert np.abs(out["null_mean"]) < 0.20
    assert out["text_block_spearman_range"] == pytest.approx([1, 1])
    assert relation(dist, dist * 0, 1)["spearman"] is None
    weights = neighbours(np.ones((5, 5)), 2)
    np.testing.assert_array_equal(weights, (1 - np.eye(5)) * 0.5)


def test_paired_losses_use_source_rows_and_training_only_scales():
    ref = pd.DataFrame(
        dict(
            row_id=["csv:1", "csv:0"],
            y_true=[1, 2],
            prediction=[0.0, 3.0],
            text_present=[True, False],
            joint_text_seen_in_train=[False, False],
        )
    )
    aug = ref.assign(prediction=[1.0, 2.0])
    rows, edges = paired_cohorts(ref, aug, [], 2, [2.0, 4.0, 8.0], [100.0, np.nan])
    all_rows = next(r for r in rows if r["axis"] == "all")
    assert all_rows["baseline_loss_sum"] == 1 and all_rows["augmented_loss_sum"] == 0
    assert edges == [3.0, 4.0, 6.0]
    assert row_losses(ref, [], 0)[1] is None and np.isnan(row_losses(ref, [], 0)[0]).all()
    with pytest.raises(ValueError, match="Unmatched"):
        paired_cohorts(ref, aug.iloc[::-1], [], 2, [2.0], [100.0, np.nan])
    c = ref.assign(y_true=["neg", "pos"], p_c000=[0.8, 0.25], p_c001=[0.2, 0.75])
    loss, entropy = row_losses(c, ["neg", "pos"], None)
    np.testing.assert_allclose(loss, -np.log([0.8, 0.75]))
    assert np.all((entropy >= 0) & (entropy <= 1))
    with pytest.raises(ValueError, match="sum to one"):
        row_losses(c.assign(p_c000=0.99), ["neg", "pos"], None)


def test_five_fold_prediction_store_to_rq2e_summary(isolated_output, monkeypatch):
    from sklearn.dummy import DummyClassifier

    from src.cluster.predictor_contracts import feature_cases, fixtures
    from src.evaluation import row_benefit as b
    from src.evaluation.protocol import evaluate_cv, shared_splits

    ds = next(d for d in fixtures() if d.metadata.task_type == "binary_classification")
    _, features, _ = next(feature_cases(ds, "linear"))
    name = ds.metadata.dataset_id
    monkeypatch.setattr(b, "load_catalog", lambda: {"datasets": [{"id": name}]})
    monkeypatch.setattr(b, "load_dataset", lambda *a, **kw: ds)
    seed = b.load_yaml("B1")["random_seed"]
    spec = shared_splits(ds, seed=seed, folds=5)
    for phase, feature_set in [("B1", "non_text"), ("B2", "embedding")]:
        evaluate_cv(
            ds,
            features,
            DummyClassifier,
            specification=spec,
            model_id="linear",
            checkpoint_id="fixture",
            feature_set=feature_set,
            phase=phase,
            allow_model_runs=True,
            representation_recipe={"provider": None, "feature_set": feature_set},
            estimator_settings={"seed": seed},
            feature_artifact_ids=[],
        )
    monkeypatch.setattr(b.paths, "durable_project_override", b.paths.staging_override)
    body = json.loads(b.run(name).read_text())
    assert len(body["folds"]) == 5 and not body["incomplete_or_unmatched"]
    by, balanced = b.summaries()
    all_rows = by[by.axis == "all"].iloc[0]
    assert all_rows.rows == len(ds.frame)
    assert all_rows.benefit == 0 and balanced.datasets.eq(1).all()
    assert all_rows.phase == "B2" and balanced.phase.eq("B2").all()


def test_geometry_joins_raw_encoder_vectors_to_decision_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("JEVPFN_DECISION_COHORT", "all")  # Two synthetic registered providers.
    import sqlite3
    from contextlib import closing

    from src.evaluation import design, geometry_export
    from src.evaluation import geometry_study as gs
    from src.evaluation.benchmark_export import cohort_key
    from src.utils.serialization import digest

    dataset = SimpleNamespace(
        metadata=SimpleNamespace(text_columns=["note"]),
        frame=pd.DataFrame({"note": ["a", "b", None, "c", "a", "d"]}),
        source={"data_identity": "frozen"},
    )
    monkeypatch.setattr(gs, "load_catalog", lambda: {})
    monkeypatch.setattr(gs, "selected_entries", lambda *a: [{"id": "d"}])
    monkeypatch.setattr(gs, "load_dataset", lambda *a, **k: dataset)
    select = dict(
        model="model",
        version="test",
        settings_sha256="settings",
        question_bank_id="bank",
        question_variant="original",
    )
    cfg = dict(seed=42, texts_per_group=500, encoders=["enc"])
    main = {"candidate_extractors": ["p", "q"], "feature_selectors": {"p": {"provider": "p", **select}}}
    registry = {"candidates": {p: {"model": "model", "version": "test", "delivery": "hosted"} for p in ("p", "q")}}
    configs = {"A3": cfg, "B1": main, "extractors": registry}
    for module in (gs, design, geometry_export):
        monkeypatch.setattr(module, "load_yaml", configs.__getitem__)
    store = tmp_path / "vectors.sqlite3"
    monkeypatch.setattr(gs.em, "encoder_config", lambda name: {})
    monkeypatch.setattr(gs.em, "cache_path", lambda *a: store)
    monkeypatch.setattr(gs.em, "validate_dataset", lambda *a, **k: None)
    with closing(sqlite3.connect(store)) as db, db:
        db.execute("CREATE TABLE metadata(key TEXT,value TEXT)")
        db.execute("INSERT INTO metadata VALUES('artifact_id','encoder-identity')")
        db.execute("CREATE TABLE embeddings(key TEXT,vector BLOB)")
        for i, t in enumerate("abcd"):
            db.execute(
                "INSERT INTO embeddings VALUES(?,?)",
                (digest(t), np.array([1, i], dtype="<f4").tobytes()),
            )
    monkeypatch.setattr(
        gs,
        "notebook_inventory",
        lambda d: pd.DataFrame(
            [
                dict(
                    provider="p",
                    **select,
                    representation="per_column",
                    feature_store=str(store),
                    portable=False,
                    artifact_id="features",
                )
            ]
        ),
    )
    manifest = {"identity": {"data_identity": "frozen"}, "text_columns": ["note"]}
    block = SimpleNamespace(manifest=manifest)
    monkeypatch.setattr(gs, "load_block", lambda *a, **k: block)
    monkeypatch.setattr(gs, "payload_hashes", lambda *a: {})
    rows = np.array(["csv:" + str(i) for i in reversed(range(6))])
    prob = np.array([0.9, 0.1, 0.6, np.nan, 0.3, 0.1])
    group = SimpleNamespace(
        row_ids=rows,
        input_hashes=np.array(["d", "a", "c", None, "b", "a"]),
        signature="shared",
        probabilities=np.stack([prob, 1 - prob], axis=1),
    )
    monkeypatch.setattr(gs, "normalize_group", lambda *a: group)
    from src.jev import inputs
    monkeypatch.setattr(inputs, "prepared_row", lambda _, pos: [
        {"representation": "per_column", "text_columns": ["note"], "payload_sha256": dataset.frame.note.iloc[pos]}
    ])
    with pytest.raises(ValueError, match="current pilot"):
        gs.run("d")
    with pytest.raises(ValueError, match="every selected provider"):
        gs.run("d", pilot=True)
    main["candidate_extractors"] = ["p"]
    pilot = gs.run("d", pilot=True)
    pilot_bytes = pilot.read_bytes()
    full = gs.run("d")
    assert pilot != full and pilot.read_bytes() == pilot_bytes
    assert tmp_path in full.parents and cohort_key("all") in pilot.parts
    assert full.parent == geometry_export.export_path("all").parent
    result = json.loads(full.read_text())
    assert result["protocol"] == json.loads(pilot_bytes)["protocol"]
    assert result["protocol"]["decision_model_versions"] == design.decision_versions(["p"])
    archived = geometry_export.collect(full.parent, members=["p"])
    assert archived["sources"].file.to_list() == [full.name]
    assert result["encoders"] == {"enc": "encoder-identity"}
    assert result["missing_providers"] == []
    assert result["comparisons"][0]["pairs"] == 6
    sample = result["samples"][0]
    assert len(set(sample["text_keys"])) == len(sample["rows"]) == 4
    for key, row in zip(sample["text_keys"], sample["rows"], strict=True):
        assert key == digest(dataset.frame.note.iloc[int(row.split(":")[1])])
    group.input_hashes[[0, 2]] = group.input_hashes[[2, 0]]
    with pytest.raises(ValueError, match="frozen row"):
        gs.run("d", pilot=True)
    group.input_hashes[[0, 2]] = group.input_hashes[[2, 0]]
    manifest["identity"]["data_identity"] = "other"
    with pytest.raises(ValueError, match="Source rows"):
        gs.run("d")


def test_owner_jev_replay_bypasses_other_variants_but_reuses_paid_attempts(
    tmp_path, isolated_output, monkeypatch
):
    from src.evaluation.hosted_robustness import VARIANTS, HostedReplay
    from src.jev.requests import IntendedRequest
    from src.jev.spending import JEV_CEILING_USD, campaign_committed
    from src.utils import paths
    from src.utils.config import load_yaml
    from src.utils.serialization import canonical_json, digest, file_sha256, write_json
    from tests.test_spending_policy import request

    monkeypatch.delenv("VSC_DATA")
    monkeypatch.delenv("JEVPFN_STAGING_ROOT", raising=False)
    cache_folder = tmp_path / "paid"
    monkeypatch.setattr(paths, "jev_cache_path", lambda: cache_folder)
    cfg = load_yaml("feature_creation")
    review = cfg["live_extraction"]
    cfg["jev"].update(model="jev-1.14.0", version="1.14.0", request_configuration={})
    cfg["cost"]["input_price_per_million"] = 0.042
    with pytest.raises(ValueError, match="separately funded"):
        HostedReplay(cfg, "fixture", pilot=True)
    review.update(
        purpose="rq1b_controlled_replays",
        enabled=True,
        workers=1,
        accepted_pilot_manifest="fixture",
        owner_budget_eur=None,
        dataset_budget_allocations_usd={"fixture": 1.0},
        campaign_spending_limit_usd=1.0,
        max_billed_input_tokens_per_attempt=64000,
        requests_per_second=1,
        input_tokens_per_second=100000,
        token_rate_policy="reserved_upper_bound",
        context_envelope=dict(
            verified=True,
            model="jev-1.14.0",
            evidence="test",
            max_request_utf8_bytes=10000,
            max_billed_input_tokens=64000,
        ),
    )
    calls = []

    def factory(cfg, allow_network):
        fake = SimpleNamespace()

        def evaluate(r):
            calls.append(r.payload_json)
            return dict(
                model=cfg["jev"]["model"],
                answers={"semantic": {"type": "noul", "noul": 0.4}},
                usage={"input_tokens": 100},
            )

        fake.evaluate = evaluate
        return fake

    signed = tmp_path / "review.json"
    evidence = write_json(tmp_path / "verified.json", {"fixture": True})
    review["accepted_pilot_manifest"] = str(signed)

    def client():
        c = HostedReplay(cfg, "fixture", pilot=True, transport_factory=factory)
        c.gate = SimpleNamespace(acquire=lambda: None, defer=lambda *_: None)
        write_json(
            signed,
            {
                "stage": "jev_pilot",
                "accepted": True,
                "reviewed_by": "fixture",
                "input_sets": {"fixture": "frozen"},
                "settings_sha256": digest(c.settings),
                "context_envelope_sha256": digest(review["context_envelope"]),
                "evidence": [{"path": str(evidence), "sha256": file_sha256(evidence)}],
            },
        )
        c.start({"source": "frozen", "settings": digest(c.settings)})
        return c

    a = client()
    orig = request()
    req = IntendedRequest(orig.payload_json, canonical_json(a.settings), orig.provenance_json)
    assert a.limit.dataset_budget_usd * len(VARIANTS) * 2 == 1.0
    assert a.evaluate_trial(req, "original") == a.evaluate_trial(req, "repeat_1")
    assert len(calls) == 2
    a = client()
    a.evaluate_trial(req, "repeat_1")
    assert len(calls) == 2
    a.evaluate_trial(req, "repeat_2")
    assert len(calls) == 3
    # The prior single-marker layout remains usable as evidence for a distinct release.
    old_marker = a.directory / "design.jev-1.14.0.json"
    legacy = a.directory / "design.json"
    old_marker.replace(legacy)
    legacy_bytes = legacy.read_bytes()
    cfg["jev"].update(model="jev-1.15.0", version="1.15.0")
    review["context_envelope"]["model"] = "jev-1.15.0"
    b = client()
    new_req = IntendedRequest(orig.payload_json, canonical_json(b.settings), orig.provenance_json)
    assert b.evaluate_trial(new_req, "original")["model"] == "jev-1.15.0"
    assert a.evaluate_trial(req, "original")["model"] == "jev-1.14.0"
    assert legacy.read_bytes() == legacy_bytes and len(calls) == 4
    resumed = client()
    assert resumed.evaluate_trial(new_req, "original")["model"] == "jev-1.15.0" and len(calls) == 4
    with pytest.raises(ValueError, match="design changed"):
        b.start({"source": "frozen", "settings": digest(b.settings), "different_sample": True})
    replays = a.directory.parent.parent
    charged = 4 * 100 * cfg["cost"]["input_price_per_million"] / 1e6
    assert campaign_committed(cache_folder, replay_folder=replays) == pytest.approx(charged)
    from src.evaluation.study_audit import jev_budget, jev_spend

    assert jev_budget(jev_spend(cache_folder, replay_folder=replays))["recorded_charges_usd"] == pytest.approx(charged)
    write_json(cache_folder / "_control/jev_smoke/state.json",
               {"attempts": [{"charged_usd": JEV_CEILING_USD - charged}]})
    # Reuse is free. A different, chargeable replay must respect the account ceiling.
    resumed.evaluate_trial(new_req, "original")
    with pytest.raises(RuntimeError, match="budget_stop"):
        resumed.evaluate_trial(new_req, "paraphrase_1")
    assert len(calls) == 4
