import numpy as np
import pytest

from src.cluster.embedding_contracts import exercise, run
from src.utils.config import load_yaml


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
