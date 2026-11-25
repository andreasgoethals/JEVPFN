"""Hosted-Jev provenance reports dated versions and replay evidence, read-only."""

import json
import sqlite3

from src.evaluation import jev_provenance as prov


def _store(path, rows):
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE responses (request_key TEXT, kind TEXT, payload_json TEXT, "
                   "settings_json TEXT, response_json TEXT, created_at TEXT)")
        for i, (version, model, created) in enumerate(rows):
            db.execute("INSERT INTO responses VALUES (?,?,?,?,?,?)", (
                str(i), "verified_response", "{}",
                json.dumps({"provider": "jev", "version": version}),
                json.dumps({"model": model, "answers": {}}), created))


def test_version_ledger_dates_versions_and_flags_mismatches(tmp_path):
    _store(tmp_path / "A.sqlite3", [("1.14.0", "jev-1.14.0", "2026-11-01"),
                                    ("1.14.0", "jev-1.14.0", "2026-11-03"),
                                    ("1.14.0", "jev-1.15.0", "2026-11-04")])
    ledger = prov.version_ledger(tmp_path)
    ok = ledger[~ledger.version_mismatch].iloc[0]
    assert ok.responses == 2 and ok.first_response_utc == "2026-11-01" and ok.last_response_utc == "2026-11-03"
    assert ledger.version_mismatch.sum() == 1


def test_report_counts_actual_replies_excludes_mocks_and_keeps_unknown_models(tmp_path, monkeypatch):
    root = tmp_path / "tier"
    cache = root / "jev_cache"
    cache.mkdir(parents=True)
    path = cache / "A.sqlite3"
    _store(path, [("1.14.0", "jev-1.15.0", "2026-11-01"), ("1.14.0", "jev-1.15.0", "2026-11-02"),
                  ("1.14.0", None, "2026-11-03"), ("1.14.0", "mock-output", "2026-11-04")])
    with sqlite3.connect(path) as db:
        db.execute("UPDATE responses SET kind='mock' WHERE request_key='3'")
    monkeypatch.setattr(prov.paths, "jev_cache_path", lambda relative: root / relative)
    results = prov.report()
    assert results["version_mismatches"] == 2 and results["version_unconfirmed"] == 1
    assert sum(r["responses"] for r in results["versions"]) == 3
    assert {r["reported_model"] for r in results["versions"]} == {"jev-1.15.0", None}
    assert all(r["requested_model"] is None for r in results["versions"])  # Unknown in this old fixture.
    json.dumps(results, allow_nan=False)


def _round(root, suite, name, model, base, repeat):
    folder = root / suite / "rounds" / name
    folder.mkdir(parents=True)
    (folder / "case__base.json").write_text(json.dumps({"model": model, "answers": base}))
    (folder / "case__repeat.json").write_text(json.dumps({"model": model, "answers": repeat}))


def test_replay_noise_and_version_drift(tmp_path):
    _round(tmp_path, "suite", "a" * 64, "jev-1.13.0", {"s": {"p": 0.97}}, {"s": {"p": 0.97}})
    _round(tmp_path, "suite", "b" * 64, "jev-1.14.0", {"s": {"p": 0.95}}, {"s": {"p": 0.96}})
    replays = prov.replay_evidence(tmp_path)
    assert sorted(round(x, 6) for x in replays.max_probability_change) == [0.0, 0.01]
    drift = prov.version_drift(replays, tmp_path)
    assert len(drift) == 1 and abs(drift.iloc[0].max_probability_change - 0.02) < 1e-9


def test_replay_with_unknown_or_changed_model_cannot_measure_same_version_noise(tmp_path):
    _round(tmp_path, "suite", "a" * 64, None, {"p": 0.8}, {"p": 0.5})
    _round(tmp_path, "suite", "a" * 63 + "b", "jev-1.14.0", {"p": 0.1}, {"p": 0.5})
    repeat = tmp_path / "suite/rounds" / ("a" * 63 + "b") / "case__repeat.json"
    repeat.write_text(json.dumps({"model": "jev-1.15.0", "answers": {"p": 0.7}}))
    rows = prov.replay_evidence(tmp_path)
    assert not rows.same_reported_model.any() and rows.max_probability_change.isna().all()
    assert prov.version_drift(rows, tmp_path).empty
