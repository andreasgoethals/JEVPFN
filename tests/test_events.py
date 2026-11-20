"""`src/utils/events.py`: DATA logs stay compact; per-batch traces go to PROJECT."""

import json


def test_progress_events_go_to_the_project_trace_not_the_data_log(isolated_output):
    from src.utils import paths
    from src.utils.events import append_event, trace_path

    log = paths.logs_dir("feature_creation") / "text_embeddings.log"
    append_event(log, "embedding_started", dataset_id="D", unique=3, pending=3)
    for completed in (1, 2, 3):
        append_event(log, "embedding_progress", dataset_id="D", completed=completed, total=3)
    append_event(log, "embedding_completed", dataset_id="D", artifact_id="a")

    data_events = [json.loads(line)["event"] for line in log.read_text(encoding="utf-8").splitlines()]
    assert data_events == ["embedding_started", "embedding_completed"]
    trace = trace_path(log)
    assert trace == (paths.large_outputs_dir() / paths.phase_folder("feature_creation")
                     / "logs" / "text_embeddings.trace.log")
    assert trace.is_relative_to(isolated_output / "staging")
    assert not trace.is_relative_to(paths.data_root())
    progress = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert [p["completed"] for p in progress] == [1, 2, 3]
