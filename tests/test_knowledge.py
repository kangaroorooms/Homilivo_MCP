"""
Tests for the RAG/knowledge layer: app/knowledge/store.py (chunking,
retrieval on the embedded Chroma collection, source tracking,
relevance threshold, idempotent ingestion) and app/knowledge/answer.py
(grounding), plus its wiring into app/experiments/agent_modes.py's
PLATFORM_KNOWLEDGE route and the /knowledge/health endpoint.

Two documents ship in app/knowledge/documents/ today
(about_this_assistant.md and the placeholder homilivo_faq.md), so
several assertions below check "the source is one of the known
indexed documents" rather than pinning a single hardcoded title - the
point being tested is that retrieval is grounded and cites a real
source, not which specific document happens to score highest as the
corpus grows.
"""

from __future__ import annotations

from app.experiments.agent_modes import (
    MESSAGE_PLATFORM_KNOWLEDGE,
    ConversationState,
    is_platform_knowledge_question,
    route_message,
)
from app.knowledge import answer as answer_module
from app.knowledge import store as store_module
from app.knowledge.answer import NOT_AVAILABLE_RESPONSE, answer_platform_question
from app.knowledge.store import _chunk_text, _load_documents, index_stats, ingest, retrieve


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
    from app.knowledge.store import CHUNK_MAX_CHARS

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

def test_retrieve_returns_nothing_for_an_unrelated_query():
    assert retrieve("find rooms in Jaipur under 15000") == []


def test_retrieve_returns_nothing_for_an_empty_query():
    assert retrieve("") == []
    assert retrieve("   ") == []


def test_retrieve_finds_a_platform_document_for_a_policy_question():
    results = retrieve("what are the platform fees?")

    assert results
    assert results[0].score >= 0.35
    assert results[0].chunk.title in _known_document_titles()


def test_index_stats_reports_at_least_one_document():
    stats = index_stats()

    assert stats["documents_loaded"] >= 1
    assert stats["chunks_indexed"] >= stats["documents_loaded"]
    assert stats["embedding_model_available"] is True


def test_multiple_documents_in_the_folder_are_all_retrievable():
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


def test_ingestion_is_idempotent(tmp_path, monkeypatch):
    """
    Re-running ingestion (e.g. after a redeploy, or a document being
    added/edited) must rebuild the collection to match what's on disk,
    never duplicating or accumulating stale chunks.
    """
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
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
# GROUNDING (app/knowledge/answer.py)
# ============================================================

def test_answer_platform_question_is_grounded_for_a_known_topic():
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


# ============================================================
# DETERMINISTIC ROUTING
# ============================================================

def test_platform_knowledge_keywords_are_detected():
    for text in (
        "how does verification work",
        "what are the platform fees",
        "how do i contact the owner",
        "how does booking work",
    ):
        assert is_platform_knowledge_question(text) is True


def test_room_search_text_is_not_misrouted_as_platform_knowledge():
    assert is_platform_knowledge_question("find a 2bhk in jaipur under 15000") is False


def test_route_message_classifies_platform_knowledge_question():
    route = route_message("what are the platform fees?", ConversationState())

    assert route.message_type == MESSAGE_PLATFORM_KNOWLEDGE


# ============================================================
# END-TO-END: /chat
# ============================================================

def test_chat_platform_knowledge_question_is_grounded_and_does_not_call_the_api(
    client, auth_headers, monkeypatch,
):
    """
    A platform-policy question must be answered from the knowledge
    base, not by calling the Homilivo API - so the outbound HTTP call
    must never fire for this message.
    """
    def fail_if_called(request, timeout=None):
        raise AssertionError("Homilivo API must not be called for a platform question")

    monkeypatch.setattr("app.server.urllib.request.urlopen", fail_if_called)

    response = client.post(
        "/chat",
        json={"message": "what are the platform fees?", "mode": "fast"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["room_count"] == 0
    assert body["sources"]
    assert set(body["sources"]) <= _known_document_titles()
    assert "fee" in body["message"].lower()


def test_chat_room_search_response_has_no_sources(client, auth_headers, mock_homilivo_success):
    response = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    )

    assert response.json()["sources"] == []


def test_knowledge_health_endpoint(client):
    response = client.get("/knowledge/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"healthy", "degraded"}
    assert body["documents_loaded"] >= 1
