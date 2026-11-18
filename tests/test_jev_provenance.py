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
