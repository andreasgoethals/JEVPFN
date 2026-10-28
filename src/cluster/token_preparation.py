"""Bounded CPU processes for publisher token preparation; one SQLite writer.

The reviewed renderer and token-cache schema are unchanged. Workers never open
the result database. Counts and prepared tokens commit together in the parent.
"""

import json
import math
import multiprocessing
import os
import zlib
from concurrent.futures import ProcessPoolExecutor

from src.jev.catalog import ADDED, ExactCounter
from src.utils.serialization import digest

_counter = None


def initialize(provider, expected_identity):
    global _counter
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _counter = ExactCounter(provider)
    if _counter.identity != expected_identity:
        raise ValueError("CPU worker tokenizer identity differs from coordinator")


def prepare_batch(payloads):
    result = []
    for payload in payloads:
        value = _counter.prepare(payload)
        count = len(value["ids"])
        if count < 1:
            raise ValueError("Publisher prepared an empty token input")
        result.append(
            (
                digest(payload),
                digest(value),
                zlib.compress(json.dumps(value, ensure_ascii=False).encode(), level=1),
                count,
            )
        )
    return result


def resources(provider, *, sharded=False):
    """Large publisher renderers need parallel CPU preparation, never a GPU."""
    if provider in ADDED:
        return ["--cpus-per-task=8", "--mem=27200M", "--time=12:00:00" if sharded else "--time=24:00:00"]
    return ["--cpus-per-task=2", "--mem=6800M", "--time=04:00:00"]


class PreparedTokenPool:
    def __init__(self, provider, counter, workers, *, executor_factory=ProcessPoolExecutor):
        self.counter = counter
        self.workers = workers
        self.pool = executor_factory(
            max_workers=workers,
            initializer=initialize,
            initargs=(provider, counter.identity),
            mp_context=multiprocessing.get_context("spawn"),
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def count_many(self, payloads):
        chunk = max(1, math.ceil(len(payloads) / self.workers))
        batches = [payloads[i : i + chunk] for i in range(0, len(payloads), chunk)]
        # At most one batch per worker is in flight; do not queue the dataset.
        records = [row for batch in self.pool.map(prepare_batch, batches) for row in batch]
        if len(records) != len(payloads):
            raise ValueError("CPU preparation returned incomplete results")
        for payload, (key, sha, content, count) in zip(payloads, records, strict=True):
            if key != digest(payload) or type(count) is not int or count < 1:
                raise ValueError("CPU preparation lost input order or returned invalid counts")
            value = json.loads(zlib.decompress(content))
            if digest(value) != sha or len(value["ids"]) != count:
                raise ValueError("CPU prepared-token checksum/count mismatch")
        db = self.counter.cache_db
        for key, sha, content, _ in records:
            prior = db.execute(
                "SELECT prepared_sha256,content FROM prepared_tokens WHERE payload_sha256=?", (key,)
            ).fetchone()
            if prior is not None and prior[0] != sha:
                raise ValueError("Prepared-token cache conflicts with publisher rendering")
            if prior is not None and digest(json.loads(zlib.decompress(prior[1]))) != sha:
                raise ValueError("Existing prepared-token cache checksum mismatch")
            db.execute("INSERT OR IGNORE INTO prepared_tokens VALUES (?,?,?)", (key, sha, content))
        return [row[3] for row in records]
