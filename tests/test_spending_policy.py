"""Pilot-to-full budget changes retain paid usage; all requests are synthetic."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.jev.cache import ResponseCache
from src.jev.extraction import DatasetExtractor, ExtractionLimits
from src.jev.merge import merge_cache
from src.jev.requests import IntendedRequest
from src.jev.spending import policy_settings
from src.utils import paths
from src.utils.serialization import canonical_json, file_sha256, write_json


def request(text="first"):
    return IntendedRequest(
        canonical_json(
            {
                "state": {"text": {"review": text}},
                "question": {
                    "kind": "binary_probability",
                    "text": "Does the review favor approval?",
                },
            }
        ),
        canonical_json(
            {
                "provider": "jev",
                "model": "jev-1.14.0",
                "version": "1.14.0",
                "api_mapping_version": "typesafe-systemone-v1",
                "request_configuration": {},
            }
        ),
        canonical_json({"dataset_id": "fixture", "row_id": "csv:0"}),
    )


@pytest.fixture
def extraction(tmp_path, isolated_output):
    class Client:
        calls = 0

        def evaluate(self, _):
            self.calls += 1
            return {
                "model": "jev-1.14.0",
                "answers": {"semantic": {"type": "noul", "noul": 0.4}},
                "usage": {"input_tokens": 10},
            }

    cache = ResponseCache(tmp_path / "fixture.sqlite3", "fixture")
    client = Client()
    limits = ExtractionLimits(0.00011, 1, 100, 10, 1000)

    def start(selected=limits, approval=None):
        return DatasetExtractor(
            cache,
            client,
            selected,
            gate=SimpleNamespace(acquire=lambda: None),
            validate_context=lambda _: None,
            policy_revision_manifest=approval,
        )

    return SimpleNamespace(cache=cache, client=client, limits=limits, start=start)


def approval(tmp_path, extractor, limits):
    evidence = tmp_path / "pilot.json"
    write_json(evidence, {"fixture": "checked billing report"})
    path = tmp_path / "policy_review.json"
    write_json(
        path,
        {
            "accepted": True,
            "reviewed_by": "test reviewer",
            "stage": "jev_spending_policy",
            "evidence": [{"path": evidence.name, "sha256": file_sha256(evidence)}],
            "policies": {
                "fixture": {
                    "previous_policy_sha256": extractor.policy_sha256,
                    "policy": policy_settings(limits),
                }
            },
        },
    )
    return path


def test_reviewed_revision_preserves_usage_old_prices_and_cached_calls(extraction, tmp_path):
    old = extraction.start()
    assert old.evaluate(request()) == "completed"
    # A larger allocation and a new price/bound affect only subsequent requests.
    revised = replace(
        extraction.limits,
        dataset_budget_usd=0.00012,
        input_price_per_million=2,
        max_billed_input_tokens=50,
    )
    new = extraction.start(revised, approval(tmp_path, old, revised))
    assert new.evaluate(request()) == "cache_hit"
    assert new.evaluate(request("second")) == "completed"
    # Prior pilot spend remains counted: 0.00003 + 0.00010 reservation > 0.00012.
    assert new.evaluate(request("third")) == "budget_stop"
    assert extraction.client.calls == 2
    with extraction.cache._connect() as db:
        assert db.execute("SELECT SUM(spent_usd) FROM extraction_jobs").fetchone()[
            0
        ] == pytest.approx(0.00003)
        assert db.execute(
            "SELECT spent_usd,policy_sha256 FROM extraction_jobs WHERE request_key=?",
            (request().cache_key,),
        ).fetchone() == (0.00001, old.policy_sha256)
        history = db.execute(
            "SELECT policy_sha256,previous_policy_sha256,authorization_json FROM extraction_policy_history ORDER BY revision"
        ).fetchall()
        assert [(r[0], r[1]) for r in history] == [
            (old.policy_sha256, None),
            (new.policy_sha256, old.policy_sha256),
        ]
        assert json.loads(history[1][2])["reviewed_by"] == "test reviewer"
    # Unchanged restarts need no new approval and must not create another revision.
    assert extraction.start(revised).policy_sha256 == new.policy_sha256
    with extraction.cache._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM extraction_policy_history").fetchone()[0] == 2
    with pytest.raises(ValueError, match="policy changed"):
        old.evaluate(request("not dispatched by stale process"))
    assert extraction.client.calls == 2


@pytest.mark.parametrize(
    "damage", ["missing", "wrong_old", "wrong_new", "evidence", "not_accepted"]
)
def test_policy_revision_requires_matching_review(extraction, tmp_path, damage):
    old = extraction.start()
    revised = replace(extraction.limits, dataset_budget_usd=1)
    path = approval(tmp_path, old, revised)
    body = json.loads(path.read_text())
    if damage == "wrong_old":
        body["policies"]["fixture"]["previous_policy_sha256"] = "another-policy"
    elif damage == "wrong_new":
        body["policies"]["fixture"]["policy"]["price"] = 99
    elif damage == "evidence":
        (tmp_path / "pilot.json").write_text("changed")
    elif damage == "not_accepted":
        body["accepted"] = False
    write_json(path, body)
    with pytest.raises(ValueError):
        extraction.start(revised, None if damage == "missing" else path)
    with extraction.cache._connect() as db:
        assert json.loads(
            db.execute("SELECT settings_json FROM extraction_policy").fetchone()[0]
        ) == policy_settings(extraction.limits)
        assert db.execute("SELECT COUNT(*) FROM extraction_policy_history").fetchone()[0] == 1
    assert extraction.client.calls == 0


@pytest.mark.parametrize("status", ["sending", "ambiguous", "ambiguous_usage"])
def test_policy_cannot_clear_unresolved_reservations(extraction, tmp_path, status):
    old = extraction.start()
    assert old._reserve(request()) == "claimed"
    with extraction.cache._connect() as db:
        db.execute("UPDATE extraction_jobs SET status=?", (status,))
    revised = replace(extraction.limits, dataset_budget_usd=1)
    with pytest.raises(ValueError, match="reservations"):
        extraction.start(revised, approval(tmp_path, old, revised))
    with extraction.cache._connect() as db:
        assert db.execute("SELECT reserved_usd,status FROM extraction_jobs").fetchone() == (
            0.0001,
            status,
        )


def test_review_cannot_reduce_budget_below_paid_usage(extraction, tmp_path):
    old = extraction.start()
    old.evaluate(request())
    revised = replace(extraction.limits, dataset_budget_usd=0.000001)
    with pytest.raises(ValueError, match="below spending"):
        extraction.start(revised, approval(tmp_path, old, revised))
    assert extraction.client.calls == 1


def test_equivalent_number_spellings_reuse_the_active_identity(extraction):
    old = extraction.start()
    restarted = extraction.start(replace(extraction.limits, input_price_per_million=1.0))
    assert restarted.policy_sha256 == old.policy_sha256
    assert restarted.evaluate(request()) == "completed"


def test_legacy_policy_migrates_without_losing_paid_or_ambiguous_jobs(extraction):
    with extraction.cache._connect() as db:
        db.executescript("""
            CREATE TABLE extraction_policy(id INTEGER PRIMARY KEY, settings_json TEXT);
            CREATE TABLE extraction_jobs(
                request_key TEXT PRIMARY KEY, request_json TEXT NOT NULL, status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0, reserved_usd REAL NOT NULL DEFAULT 0,
                spent_usd REAL NOT NULL DEFAULT 0, input_tokens INTEGER, raw_response_json TEXT,
                error_type TEXT, owner_session TEXT);
        """)
        db.execute(
            "INSERT INTO extraction_policy VALUES(1,?)",
            (canonical_json(policy_settings(extraction.limits)),),
        )
        db.execute(
            "INSERT INTO extraction_jobs(request_key,request_json,status,attempts,reserved_usd,spent_usd) VALUES(?,?,'ambiguous',1,0.0001,0.00001)",
            (request().cache_key, request().to_json()),
        )
    migrated = extraction.start()
    assert migrated.evaluate(request()) == "ambiguous"
    assert extraction.client.calls == 0
    with extraction.cache._connect() as db:
        row = db.execute(
            "SELECT status,attempts,reserved_usd,spent_usd,policy_sha256 FROM extraction_jobs"
        ).fetchone()
        assert row == ("ambiguous", 1, 0.0001, 0.00001, migrated.policy_sha256)
        history = json.loads(
            db.execute("SELECT authorization_json FROM extraction_policy_history").fetchone()[0]
        )
        assert history == {"kind": "legacy_allocation"}


def test_cache_merge_preserves_policy_history_without_adopting_remote_budget(extraction, tmp_path):
    old = extraction.start()
    old.evaluate(request())
    revised = replace(extraction.limits, dataset_budget_usd=1)
    extraction.start(revised, approval(tmp_path, old, revised))
    local = ResponseCache(paths.dataset_cache_path("fixture"), "fixture")
    result = merge_cache(extraction.cache.path, local.path)
    assert result["responses_added"] == 1 and result["ledger_records_added"] == 4
    assert local.get(request(), kind="verified_response")["usage"]["input_tokens"] == 10
    with local._connect() as db:
        assert (
            db.execute(
                "SELECT COUNT(*) FROM imported_extraction_records WHERE table_name='extraction_policy_history'"
            ).fetchone()[0]
            == 2
        )
        assert not db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='extraction_policy'"
        ).fetchone()
    assert merge_cache(extraction.cache.path, local.path)["ledger_records_added"] == 0
