"""
Copied from tests/test_knowledge.py at commit 7dbcba3 (copied, not imported), imports changed to llm_approach.rag.
The 11 tests that cover the knowledge layer itself: chunking, retrieval, ingestion, grounding.

Not copied (6 of the original 17), because they test app/ code that llm_approach does not contain:
  - test_platform_knowledge_keywords_are_detected, test_room_search_text_is_not_misrouted_as_platform_knowledge,
    test_route_message_classifies_platform_knowledge_question: the keyword router in app/experiments/agent_modes.py
    (v2 routes PLATFORM_INFO with the LLM decision, covered by the pipeline tests).
  - test_chat_platform_knowledge_question_is_grounded_and_does_not_call_the_api,
    test_chat_room_search_response_has_no_sources, test_knowledge_health_endpoint: app/server.py's /chat and
    /knowledge/health with the app's test client.

Differences from the original: the Chroma folder is a temporary directory (the original used the real
data/chroma), and a test that needs the real embedding model is skipped when the model cannot be loaded.
"""

from __future__ import annotations

import pytest

from llm_approach.rag import answer as answer_module
from llm_approach.rag import store as store_module
from llm_approach.rag.answer import NOT_AVAILABLE_RESPONSE, answer_platform_question
from llm_approach.rag.store import _chunk_text, _load_documents, index_stats, ingest, retrieve


@pytest.fixture(autouse=True, scope="module")
def _temp_chroma(tmp_path_factory):
    """v2's collection lives in a temporary folder for this module; nothing under data/ is touched."""
    mp = pytest.MonkeyPatch()
    mp.setenv("LLM_APPROACH_CHROMA_PERSIST_DIR", str(tmp_path_factory.mktemp("chroma_v2")))
    store_module._reset_caches()
    yield
    store_module._reset_caches()
    mp.undo()


@pytest.fixture(scope="module")
def embedding_model():
    """The real sentence-transformers model; the test is skipped when it cannot be loaded (no download offline)."""
    from llm_approach.rag.embeddings import get_shared_model

    try:
        return get_shared_model()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"embedding model not available: {type(exc).__name__}")


def _known_document_titles() -> set[str]:
    return {chunk.title for chunk in _load_documents()}


# ============================================================
# CHUNKING
# ============================================================

def test_chunk_text_splits_on_paragraph_boundaries():
    text = "First topic sentence.\n\nSecond, unrelated topic sentence."

    chunks = _chunk_text(text)

    assert chunks == [
        "First topic sentence.",
        "Second, unrelated topic sentence.",
    ]


def test_chunk_text_splits_an_oversized_paragraph():
    from llm_approach.rag.store import CHUNK_MAX_CHARS

    text = "x" * (CHUNK_MAX_CHARS * 2 + 10)

    chunks = _chunk_text(text)

    assert len(chunks) == 3
    assert all(len(chunk) <= CHUNK_MAX_CHARS for chunk in chunks)


def test_chunk_text_handles_empty_input():
    assert _chunk_text("") == []
    assert _chunk_text("   \n\n  ") == []


# ============================================================
# RETRIEVAL
# ============================================================

def test_retrieve_returns_nothing_for_an_unrelated_query(embedding_model):
    assert retrieve("find rooms in Jaipur under 15000") == []


def test_retrieve_returns_nothing_for_an_empty_query():
    assert retrieve("") == []
    assert retrieve("   ") == []


def test_retrieve_finds_a_platform_document_for_a_policy_question(embedding_model):
    results = retrieve("what are the platform fees?")

    assert results
    assert results[0].score >= 0.35
    assert results[0].chunk.title in _known_document_titles()


def test_index_stats_reports_at_least_one_document(embedding_model):
    stats = index_stats()

    assert stats["documents_loaded"] >= 1
    assert stats["chunks_indexed"] >= stats["documents_loaded"]
    assert stats["embedding_model_available"] is True


def test_multiple_documents_in_the_folder_are_all_retrievable(embedding_model):
    """
    Ingestion must scan the whole documents/ folder, not just one
    hardcoded file - so a query aimed at each shipped document must
    surface that document's own doc_id somewhere in its results.
    """
    stats = index_stats()
    assert stats["documents_loaded"] >= 2

    seen_doc_ids: set[str] = set()
    for query in (
        "what are the platform fees?",
        "how do I contact the owner of a listing?",
        "what can this assistant help me search for?",
        "does this assistant confirm or cancel bookings?",
    ):
        for result in retrieve(query):
            seen_doc_ids.add(result.chunk.doc_id)

    assert {"about_this_assistant", "homilivo_faq"} <= seen_doc_ids


def test_ingestion_is_idempotent(tmp_path, monkeypatch, embedding_model):
    """
    Re-running ingestion (e.g. after a redeploy, or a document being
    added/edited) must rebuild the collection to match what's on disk,
    never duplicating or accumulating stale chunks.
    """
    monkeypatch.setenv("LLM_APPROACH_CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    store_module._reset_caches()

    try:
        first = ingest()
        second = ingest()

        assert first == second
        assert first["chunks_indexed"] > 0

        collection = store_module._get_collection()
        assert collection.count() == first["chunks_indexed"]
    finally:
        store_module._reset_caches()


# ============================================================
# GROUNDING (llm_approach/rag/answer.py)
# ============================================================

def test_answer_platform_question_is_grounded_for_a_known_topic(embedding_model):
    result = answer_platform_question("how do I contact the owner of a listing?")

    assert result["grounded"] is True
    assert result["message"] != NOT_AVAILABLE_RESPONSE
    assert result["sources"]
    assert set(result["sources"]) <= _known_document_titles()


def test_answer_platform_question_never_fabricates_when_nothing_retrieves(monkeypatch):
    monkeypatch.setattr(answer_module, "retrieve", lambda query: [])

    result = answer_platform_question("what is the exact cancellation fee percentage?")

    assert result["grounded"] is False
    assert result["message"] == NOT_AVAILABLE_RESPONSE
    assert result["sources"] == []
