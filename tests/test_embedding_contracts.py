import numpy as np
import pytest

from src.cluster.embedding_contracts import exercise, run
from src.utils.config import load_yaml


def test_encoder_report_binds_source_runtime_job_and_failure(isolated_output, monkeypatch):
    import json

    from src.cluster import embedding_contracts, evidence, freeze
    from src.utils import paths

    identity = {
        "source_sha256": "fixed",
        "runtime_pins_sha256": "pinned",
        "prepared_inputs_sha256": "inputs",
    }
    monkeypatch.setattr(evidence, "current_identity", lambda: identity)
    monkeypatch.setattr(freeze, "source_snapshot", lambda: {"git_head": "commit"})
    monkeypatch.setenv("SLURM_JOB_ID", "321")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "mindwell")

    def fail(_):
        raise ValueError("synthetic initialization failure")

    with pytest.raises(ValueError, match="synthetic"):
        embedding_contracts.run(execute=True, encoder_factory=fail)
    path = next(
        (paths.manifests_dir("experiment_0") / "before_extraction/embeddings").glob("*.json")
    )
    saved = json.loads(path.read_text())
    assert saved["status"] == "failed"
    assert saved["validation_identity"] == identity
    assert saved["git_head"] == "commit" and saved["job_id"] == "321"
    assert saved["cluster"] == "mindwell" and saved["finished_utc"] >= saved["created_utc"]


def test_bounded_encoder_probe_exercises_cache_resume_missing_and_chunks(isolated_output, tmp_path):
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(range(len(text.split())))

        def num_special_tokens_to_add(self, _):
            return 2

    class Encoder:
        tokenizer = Tokenizer()
        identity = {"runtime": "synthetic"}

        def __call__(self, texts):
            return np.tile([1.0, 0.0, 0.0], (len(texts), 1))

    cfg = load_yaml("feature_creation/text_embeddings") | {"dimensions": 3}
    report = exercise(cfg, Encoder(), tmp_path / "fixture.sqlite3")
    assert report["unique_texts_encoded"] == 3 and report["long_text_chunks"] > 1
    assert report["resume_without_inference"]
    with pytest.raises(RuntimeError, match="execute"):
        run()
