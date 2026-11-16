import os

import pytest

from src.cluster.native_qualification import verified_links
from src.utils.serialization import file_sha256


def test_qualification_links_verified_assets_without_changing_source(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    file = source / "model.safetensors"
    file.write_bytes(b"immutable weights")
    records = {file.name: dict(sha256=file_sha256(file), bytes=file.stat().st_size)}
    verified_links(source, target, records)
    assert os.path.samefile(file, target / file.name)
    verified_links(source, target, records)
    assert file.read_bytes() == b"immutable weights"


def test_bad_download_cannot_partially_populate_qualification(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    (source / "a").write_bytes(b"good")
    (source / "b").write_bytes(b"bad")
    records = {"a": dict(sha256=file_sha256(source / "a"), bytes=4),
               "b": dict(sha256="wrong", bytes=3)}
    with pytest.raises(ValueError, match="receipt"):
        verified_links(source, target, records)
    assert not target.exists()
