"""Audit item A6: the Chroma collection is reused when the documents are unchanged, rebuilt only on change, a failed
first ingest is retried later, and concurrent starts build it once.

A fake embedding model stands in for sentence-transformers: offline and fast. Real embedded Chroma in a temp folder.
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

from llm_approach.rag import embeddings, store


class FakeModel:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay, self.calls = delay, 0

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return np.array([[1.0, 0.0, 0.0] for _ in texts])


@pytest.fixture
def kb(monkeypatch, tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "a.md").write_text("---\ntitle: A\n---\nFirst paragraph about alpha.\n\nSecond paragraph about beta.\n", encoding="utf-8")
    (docs / "b.md").write_text("Only paragraph of b.\n", encoding="utf-8")
    model = FakeModel()
    monkeypatch.setattr(store, "DOCUMENTS_DIR", docs)
    monkeypatch.setenv("LLM_APPROACH_CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(embeddings, "get_shared_model", lambda: model)
    clock = {"now": 1000.0}
    monkeypatch.setattr(store, "_clock", lambda: clock["now"])
    store._reset_caches()
    ingests = []
    real_ingest = store.ingest
    monkeypatch.setattr(store, "ingest", lambda: (ingests.append(1), real_ingest())[1])
    yield type("KB", (), {"docs": docs, "model": model, "ingests": ingests, "clock": clock, "tmp": tmp_path})
    store._reset_caches()


def restart():
    """A new process: the in-process caches are empty, the folder on disk is not."""
    store._reset_caches()


def test_the_first_start_builds_the_index_and_records_a_fingerprint(kb):
    stats = store._ensure_ready()
    assert stats == {"documents_loaded": 2, "chunks_indexed": 3, "embedding_model_available": True}
    assert len(kb.ingests) == 1
    assert (kb.tmp / "chroma" / store.FINGERPRINT_FILE).read_text(encoding="utf-8").strip()
    assert not (kb.tmp / "chroma" / store.LOCK_FILE).exists()  # the build lock is released


def test_the_second_start_reuses_the_collection_without_rebuilding(kb):
    store._ensure_ready()
    encodes = kb.model.calls
    restart()
    stats = store._ensure_ready()
    assert stats["chunks_indexed"] == 3 and stats["embedding_model_available"] is True
    assert len(kb.ingests) == 1 and kb.model.calls == encodes  # no ingest, no re-embedding of the documents
    assert store.cached_stats() == stats


def test_a_changed_document_rebuilds_and_drops_the_old_chunks(kb):
    store._ensure_ready()
    old = store._stored_fingerprint()
    (kb.docs / "b.md").write_text("Rewritten b.\n\nWith a second paragraph.\n", encoding="utf-8")
    restart()
    stats = store._ensure_ready()
    assert len(kb.ingests) == 2 and stats["chunks_indexed"] == 4
    assert store._stored_fingerprint() != old
    assert store._get_collection().count() == 4  # rebuilt, not appended to


@pytest.mark.parametrize("change", ["model", "chunk_size"])
def test_a_new_embedding_model_or_chunking_rebuilds(kb, monkeypatch, change):
    store._ensure_ready()
    if change == "model":
        monkeypatch.setattr(embeddings, "MODEL_NAME", "another-model")
    else:
        monkeypatch.setattr(store, "CHUNK_MAX_CHARS", 300)
    restart()
    store._ensure_ready()
    assert len(kb.ingests) == 2


def test_an_incomplete_collection_is_rebuilt_even_when_the_fingerprint_matches(kb):
    store._ensure_ready()
    collection = store._get_collection()
    collection.delete(ids=collection.get()["ids"][:1])
    restart()
    assert store._ensure_ready()["chunks_indexed"] == 3 and len(kb.ingests) == 2
    assert store._get_collection().count() == 3


def test_a_build_that_dies_leaves_no_matching_fingerprint_and_no_lock(kb, monkeypatch):
    store._ensure_ready()
    (kb.docs / "a.md").write_text("Changed.\n", encoding="utf-8")
    restart()

    def dies():
        raise RuntimeError("killed")

    monkeypatch.setattr(store, "ingest", dies)
    with pytest.raises(RuntimeError):
        store._ensure_ready()
    assert store._stored_fingerprint() is None and not (kb.tmp / "chroma" / store.LOCK_FILE).exists()


def test_a_failed_first_ingest_is_not_cached_for_the_life_of_the_process(kb, monkeypatch):
    state = {"broken": True}
    calls = []

    def model():
        calls.append(1)
        if state["broken"]:
            raise OSError("model download failed")
        return FakeModel()

    monkeypatch.setattr(embeddings, "get_shared_model", model)
    first = store._ensure_ready()
    assert first["embedding_model_available"] is False and store.cached_stats() is None
    seen = len(calls)
    assert store._ensure_ready() is first and len(calls) == seen  # inside the pause: no new attempt on every question
    state["broken"] = False
    kb.clock["now"] += store.RETRY_FAILED_AFTER_S + 1
    later = store._ensure_ready()
    assert later["embedding_model_available"] is True and later["chunks_indexed"] == 3
    seen = len(calls)
    assert store._ensure_ready() is later and len(calls) == seen  # the good result is kept
    assert store.cached_stats() == later


def test_a_failed_first_ingest_with_no_client_is_retried_too(kb, monkeypatch):
    real = store._get_client.__wrapped__
    monkeypatch.setattr(store, "_get_client", __import__("functools").lru_cache(maxsize=1)(lambda: None))
    assert store._ensure_ready()["embedding_model_available"] is False
    monkeypatch.setattr(store, "_get_client", __import__("functools").lru_cache(maxsize=1)(real))
    store._ready_state["failed"] = store._ready_state["failed"]  # still failed, inside the pause
    kb.clock["now"] += store.RETRY_FAILED_AFTER_S + 1
    assert store._ensure_ready()["chunks_indexed"] == 3


def test_concurrent_starts_build_the_collection_once(kb, monkeypatch):
    slow = FakeModel(delay=0.3)
    monkeypatch.setattr(embeddings, "get_shared_model", lambda: slow)
    results, errors = [], []

    def start():
        try:
            results.append(store._sync_index())  # each caller as a separate process would: no shared in-process lock
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    store._get_client()  # a process has one client, built under the in-process lock; threads here stand in for processes
    threads = [threading.Thread(target=start) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(kb.ingests) == 1  # one builder, the others waited and reused
    assert all(r == {"documents_loaded": 2, "chunks_indexed": 3, "embedding_model_available": True} for r in results)
    assert store._get_collection().count() == 3 and not (kb.tmp / "chroma" / store.LOCK_FILE).exists()


def test_a_stale_lock_from_a_dead_process_is_taken_over(kb):
    store._persist_dir().mkdir(parents=True, exist_ok=True)
    lock = store._persist_dir() / store.LOCK_FILE
    lock.write_text("", encoding="utf-8")
    old = time.time() - store.LOCK_STALE_S - 10
    os.utime(lock, (old, old))
    assert store._ensure_ready()["chunks_indexed"] == 3 and not lock.exists()


def test_a_busy_lock_times_out_as_a_failure_that_is_retried_later(kb, monkeypatch):
    store._persist_dir().mkdir(parents=True, exist_ok=True)
    lock = store._persist_dir() / store.LOCK_FILE
    lock.write_text("", encoding="utf-8")  # fresh: another process is building
    monkeypatch.setattr(store, "LOCK_WAIT_S", 0.3)
    monkeypatch.setattr(store, "_sleep", lambda seconds: kb.clock.update(now=kb.clock["now"] + seconds))  # waiting moves the fake clock
    stats = store._ensure_ready()
    assert stats["embedding_model_available"] is False and lock.exists() and not kb.ingests  # not ours: left alone
    lock.unlink()
    kb.clock["now"] += store.RETRY_FAILED_AFTER_S + 1
    assert store._ensure_ready()["chunks_indexed"] == 3
