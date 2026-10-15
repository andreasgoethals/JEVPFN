"""No weights/network: resume, coverage, storage and opaque-hosted-tokenizer boundaries."""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.cluster import token_audit
from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.jev.decision_tokenization import ExactCounter
from src.jev.inputs import iter_prepared_inputs, prepare_dataset_inputs
from src.jev.question_bank import build_question_bank, save_question_bank
from src.jev.self_hosted import token_evidence
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import file_sha256, write_json


@pytest.fixture
def inputs(isolated_output):
    frame = pd.DataFrame(
        {"text": ["same", "same", "longer", "third", None], "target": [0, 1, 1, 0, 0]}
    )
    meta = extract_metadata(
        frame,
        {"task_type": "cls", "target": "target"},
        dataset_id="fixture",
        dataset_name="Fixture",
        text_columns=["text"],
    )
    dataset = LoadedDataset(frame, meta, {}, {"data_identity": "fixed"}, [])
    bank = build_question_bank([meta], load_yaml("question_design"))
    save_question_bank(bank)
    summary = prepare_dataset_inputs(dataset, bank, variant="accepted")
    return [
        {
            "dataset_id": "fixture",
            "path": str(summary["path"]),
            "input_set_id": summary["input_set_id"],
            "input_file_sha256": file_sha256(Path(summary["path"])),
            "source_rows": 5,
            "unique_inputs": 3,
        }
    ]


def test_exact_counts_resume_without_recounting_or_removing_rows(inputs):
    calls = []
    counter = SimpleNamespace(identity={"counter_id": "synthetic"}, limit=4)

    def count(payloads):
        calls.extend(payloads)
        return [len(p["state"]["text"]["text"]) for p in payloads]

    counter.count_many = count
    report = token_audit.audit_provider("liquid", inputs, counter=counter, batch_size=2)
    assert len(calls) == 3  # Repeated input and two representation slots count once.
    assert report["overflow_unique_inputs"] == 2 and report["overflow_source_rows"] == 2
    assert report["unique_inputs"] == 3 and report["total_input_tokens"] == 15
    assert not report["rows_removed"] and not report["inference_executed"]
    assert Path(report["counts_database"]).is_relative_to(paths.large_outputs_dir())
    assert list((paths.logs_dir("experiment_0") / "before_extraction").glob("*.log"))
    assert list((paths.manifests_dir("experiment_0") / "before_extraction").glob("*.json"))
    again = token_audit.audit_provider("liquid", inputs, counter=counter, batch_size=1)
    assert len(calls) == 3 and again["audit_id"] == report["audit_id"]
    assert inputs[0]["input_file_sha256"] == file_sha256(Path(inputs[0]["path"]))


def test_interrupted_token_audit_resumes_committed_batch(inputs):
    count = 0

    def interrupted(payloads):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("interrupted")
        return [2] * len(payloads)

    counter = SimpleNamespace(
        identity={"counter_id": "synthetic"}, limit=10, count_many=interrupted
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        token_audit.audit_provider("strands", inputs, counter=counter, batch_size=2)
    remaining = []
    counter.count_many = lambda rows: remaining.extend(rows) or [2] * len(rows)
    report = token_audit.audit_provider("strands", inputs, counter=counter)
    assert len(remaining) == 1 and report["unique_inputs"] == 3
    assert len(list(iter_prepared_inputs(inputs[0]["path"], after_input_no=1))) == 1


@pytest.mark.parametrize("provider", ["jev"])
def test_hosted_tokenizers_are_not_faked(provider):
    with pytest.raises(ValueError, match="unverified|unavailable"):
        ExactCounter(provider)


def test_bad_counter_result_cannot_be_saved_as_exact(inputs):
    counter = SimpleNamespace(
        identity={"counter_id": "broken"}, limit=10, count_many=lambda rows: [True] * len(rows)
    )
    with pytest.raises(ValueError, match="positive integer"):
        token_audit.audit_provider("liquid", inputs, counter=counter)


def test_gpu_gate_rejects_incomplete_detailed_counts(inputs):
    counter = SimpleNamespace(
        identity={"counter_id": "synthetic"}, limit=32768, count_many=lambda rows: [2] * len(rows)
    )
    report = token_audit.audit_provider("liquid", inputs, counter=counter)
    write_json(
        paths.manifests_dir("experiment_0") / "before_extraction" / "token_audit.json",
        {"self_hosted": [report]},
    )
    _, database = token_evidence("liquid", counter)
    with sqlite3.connect(database) as db:
        db.execute("DELETE FROM counts WHERE input_no=0")
    with pytest.raises(ValueError, match="incomplete"):
        token_evidence("liquid", counter)


def test_notebook_exact_counts_reject_changed_prepared_inputs(inputs, monkeypatch):
    counter = SimpleNamespace(
        identity={"counter_id": "synthetic"}, limit=10, count_many=lambda rows: [2] * len(rows)
    )
    report = token_audit.audit_provider("liquid", inputs, counter=counter)
    write_json(
        paths.manifests_dir("experiment_0") / "before_extraction" / "token_audit.json",
        {"self_hosted": [report], "created_utc": "fixture", "hosted_unverified": ["jev"]},
    )
    monkeypatch.setattr(token_audit, "selected_inputs", lambda: inputs)
    scope, table = token_audit.saved_exact_token_tables()
    assert scope["datasets"] == 1 and table.iloc[0].total_input_tokens == 6
    inputs[0]["input_file_sha256"] = "changed"
    scope, table = token_audit.saved_exact_token_tables()
    assert scope["status"] == "historical_inputs_changed" and table.empty


def test_missing_chat_template_dependency_fails_before_counting(monkeypatch):
    from unittest.mock import Mock

    counter = SimpleNamespace(count_many=Mock(side_effect=ImportError("jinja2 missing")))
    monkeypatch.setattr(token_audit, "ExactCounter", lambda _: counter)
    inputs, pool = Mock(), Mock()
    monkeypatch.setattr(token_audit, "selected_inputs", inputs)
    monkeypatch.setattr(token_audit, "ProcessPoolExecutor", pool)
    monkeypatch.setattr("sys.argv", ["token_audit", "--provider", "perplexity"])
    with pytest.raises(RuntimeError, match="Jinja2"):
        token_audit.main()
    inputs.assert_not_called()
    pool.assert_not_called()


def test_every_task_is_rendered_in_tokenizer_precheck(monkeypatch):
    seen = []
    counter = SimpleNamespace(
        identity={"counter_id": "test"},
        count_many=lambda rows: seen.extend(rows) or [5] * len(rows),
    )
    monkeypatch.setattr(token_audit, "ExactCounter", lambda _: counter)
    assert token_audit.check_tokenizers(["intern"])[0]["task_counts"] == [5, 5, 5]
    assert {r["question"]["kind"] for r in seen} == {
        "binary_probability",
        "choice",
        "ordered_choice",
    }


def test_completed_worker_report_survives_other_worker_failure(isolated_output, monkeypatch):
    from concurrent.futures import Future

    paths.staging_root().mkdir(parents=True)
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    monkeypatch.setattr(token_audit, "selected_inputs", lambda: [])
    monkeypatch.setattr(token_audit, "check_tokenizers", lambda _: None)
    monkeypatch.setattr(
        "sys.argv", ["token_audit", "--provider", "liquid", "--provider", "perplexity"]
    )

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, provider, inputs):
            future = Future()
            if provider == "liquid":
                future.set_result({"provider": provider, "status": "counted_all_inputs"})
            else:
                future.set_exception(ValueError("fixture worker failure"))
            return future

    monkeypatch.setattr(token_audit, "ProcessPoolExecutor", Pool)
    with pytest.raises(RuntimeError, match="preserved"):
        token_audit.main()
    saved = json.loads(
        (paths.manifests_dir("experiment_0") / "before_extraction/token_audit.json").read_text()
    )
    assert saved["self_hosted"] == [{"provider": "liquid", "status": "counted_all_inputs"}]
