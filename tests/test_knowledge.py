"""
Tests for the RAG/knowledge layer: app/knowledge/store.py (chunking,
retrieval, source tracking, relevance threshold) and
app/knowledge/answer.py (grounding), plus its wiring into
app/experiments/agent_modes.py's PLATFORM_KNOWLEDGE route and the
/knowledge/health endpoint.
"""

from __future__ import annotations

from app.experiments.agent_modes import (
    MESSAGE_PLATFORM_KNOWLEDGE,
    ConversationState,
    is_platform_knowledge_question,
    route_message,
)
from app.knowledge import answer as answer_module
from app.knowledge.answer import NOT_AVAILABLE_RESPONSE, answer_platform_question
from app.knowledge.store import _chunk_text, index_stats, retrieve


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


def test_retrieve_finds_the_platform_document_for_a_policy_question():
    results = retrieve("what are the platform fees?")

    assert results
    assert results[0].chunk.title == "About this assistant"
    assert results[0].score >= 0.35


def test_index_stats_reports_at_least_one_document():
    stats = index_stats()

    assert stats["documents_loaded"] >= 1
    assert stats["chunks_indexed"] >= stats["documents_loaded"]


# ============================================================
# GROUNDING (app/knowledge/answer.py)
# ============================================================

def test_answer_platform_question_is_grounded_for_a_known_topic():
    result = answer_platform_question("how do I contact the owner of a listing?")

    assert result["grounded"] is True
    assert result["message"] != NOT_AVAILABLE_RESPONSE
    assert result["sources"] == ["About this assistant"]


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
    assert body["sources"] == ["About this assistant"]
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
