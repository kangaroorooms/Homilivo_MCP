"""v2's RAG storage is its own: default folder, own variable, own collection, never the app's data/chroma.

A fake embedding model stands in for sentence-transformers, so these tests are fast and offline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from llm_approach import reuse
from llm_approach.rag import embeddings, store


class FakeModel:
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        return np.array([[1.0, 0.0, 0.0] for _ in texts])


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # the relative default folder lands here, not in the repo
    monkeypatch.delenv("LLM_APPROACH_CHROMA_PERSIST_DIR", raising=False)
    monkeypatch.delenv("CHROMA_PERSIST_DIR", raising=False)
    monkeypatch.setattr(embeddings, "get_shared_model", lambda: FakeModel())
    store._reset_caches()
    yield
    store._reset_caches()


def test_default_folder_is_chroma_v2():
    assert store._persist_dir() == Path("data/chroma_v2")
    assert store.COLLECTION_NAME == "homilivo_knowledge_v2"


def test_the_apps_variable_is_never_read(monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "data/chroma")
    assert store._persist_dir() == Path("data/chroma_v2")


def test_own_variable_can_move_the_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_APPROACH_CHROMA_PERSIST_DIR", str(tmp_path / "elsewhere"))
    assert store._persist_dir() == tmp_path / "elsewhere"


def test_a_path_equal_to_the_apps_folder_is_refused(monkeypatch, caplog):
    monkeypatch.setenv("LLM_APPROACH_CHROMA_PERSIST_DIR", "data/chroma")
    with caplog.at_level("ERROR"):
        assert store._persist_dir() == Path("data/chroma_v2")
    assert "data/chroma" in caplog.text


def test_ingest_writes_only_under_chroma_v2_and_never_creates_the_apps_folder(tmp_path):
    stats = store.ingest()
    assert stats["embedding_model_available"] is True and stats["chunks_indexed"] > 0
    assert (tmp_path / "data" / "chroma_v2").is_dir()
    assert not (tmp_path / "data" / "chroma").exists()


def test_ingest_leaves_a_collection_of_the_apps_name_alone(tmp_path, monkeypatch):
    import chromadb

    app_dir = tmp_path / "data" / "chroma"
    app_dir.mkdir(parents=True)
    app_collection = chromadb.PersistentClient(path=str(app_dir)).get_or_create_collection("homilivo_knowledge")
    app_collection.add(ids=["a"], embeddings=[[0.0, 1.0, 0.0]], documents=["app chunk"])
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(app_dir))  # even a shared .env cannot redirect v2
    store.ingest()
    again = chromadb.PersistentClient(path=str(app_dir)).get_collection("homilivo_knowledge")
    assert again.count() == 1


def test_reuse_uses_the_local_copies():
    reuse.get_answer_platform_question.cache_clear()
    assert reuse.get_answer_platform_question().__module__ == "llm_approach.rag.answer"
    assert reuse.get_knowledge_preload().__module__ == "llm_approach.rag.store"
