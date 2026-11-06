"""Knowledge-base hardening (RAG audit, batch A): a blocked knowledge reply keeps the rooms on screen, a malformed document is
skipped, a query without letters is not retrieved, check_docs flags guardrail-blocked chunks, duplicates and unclosed front
matter, and the debug log reuses the answer's own retrieval.

Offline: scripted LLM, fake retriever where retrieval matters, no model, no network."""

from __future__ import annotations

import logging

import pytest

from llm_approach import knowledge_status, pipeline
from llm_approach.actions import TurnBudget, platform
from llm_approach.rag import answer as answer_module
from llm_approach.rag import check_docs, store
from llm_approach.rag.store import KnowledgeChunk
from llm_approach.schema import Decision, Intent, Slots
from llm_approach.state import LLMConversationState
from llm_approach.tests.conftest import dec
from llm_approach.tests.test_price_routing import REFUND, fake_retrieve, scored


# ---- a blocked knowledge reply keeps the rooms -------------------------------------------------------------------------


def test_a_blocked_knowledge_reply_keeps_the_rooms_on_screen(pipe):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    pipe.turn("the second one", dec("ROOM_FOLLOWUP", op="ROOM_DETAILS", ref=2))
    rooms_before = list(pipe.state.rooms_on_screen)
    selected_before = pipe.state.selected_room_index
    assert rooms_before and selected_before == 2
    pipe.output_ok = False
    r = pipe.turn("what personal information do you collect", dec("PLATFORM_INFO"))
    assert r.reply == pipeline.OUTPUT_BLOCKED_REPLY and r.metrics["action"] == "output_blocked"
    assert r.metrics["error_class"] == "output_blocked"
    assert r.rooms is None  # None: the cards on screen stay as they are
    assert pipe.state.rooms_on_screen == rooms_before and pipe.state.selected_room_index == selected_before


def test_a_blocked_search_reply_still_drops_its_rooms(pipe):
    pipe.output_ok = False
    r = pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    assert r.rooms == [] and pipe.state.rooms_on_screen == []


# ---- a malformed document is skipped ---------------------------------------------------------------------------------


GOOD = "---\ntitle: Good\nsource: test\nverified: false\n---\n\nGood paragraph one.\n\nGood paragraph two.\n"


@pytest.fixture
def documents(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DOCUMENTS_DIR", tmp_path)
    yield tmp_path
    store._skipped_documents.clear()


def test_a_malformed_document_is_skipped_and_the_rest_still_load(documents, caplog):
    (documents / "a_good.md").write_text(GOOD, encoding="utf-8")
    (documents / "b_broken.md").write_text("---\ntitle: [Broken\nverified: : :\n---\n\nNever indexed.\n", encoding="utf-8")
    (documents / "c_list.md").write_text("---\n- not\n- a mapping\n---\n\nNever indexed either.\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=store.__name__):
        chunks = store._load_documents()
    assert [c.text for c in chunks] == ["Good paragraph one.", "Good paragraph two."]
    skipped = store.skipped_documents()
    assert set(skipped) == {"b_broken.md", "c_list.md"} and skipped["c_list.md"] == "TypeError"
    assert knowledge_status.skipped_documents() == 2
    assert "file=b_broken.md" in caplog.text and f"error={skipped['b_broken.md']}" in caplog.text
    assert "Never indexed" not in caplog.text  # the file name and error class only


def test_the_skip_count_follows_the_last_load(documents):
    (documents / "a_good.md").write_text(GOOD, encoding="utf-8")
    (documents / "b_broken.md").write_text("---\ntitle: [Broken\n---\n\nx\n", encoding="utf-8")
    store._load_documents()
    assert knowledge_status.skipped_documents() == 1
    (documents / "b_broken.md").unlink()
    store._load_documents()
    assert knowledge_status.skipped_documents() == 0


# ---- a query without letters is not retrieved ---------------------------------------------------------------------------


@pytest.fixture
def readiness(monkeypatch):
    calls = []

    def ready():
        calls.append(1)
        return {"documents_loaded": 0, "chunks_indexed": 0, "embedding_model_available": True}

    monkeypatch.setattr(store, "_ensure_ready", ready)
    return calls


@pytest.mark.parametrize("query", ["🏠💰❓", "a", "?", "12345", " 7 🙂 ", "🙏 a"])
def test_a_query_with_fewer_than_two_letters_is_not_retrieved(readiness, query):
    result = answer_module.answer_platform_question(query)
    assert result == {"grounded": False, "message": answer_module.NOT_AVAILABLE_RESPONSE, "sources": []}
    assert readiness == []  # nothing loaded, nothing embedded


@pytest.mark.parametrize("query", ["ok", "refund", "रिफंड नीति", "退款", "plan 2"])
def test_two_letters_in_any_script_are_enough(readiness, query):
    store.retrieve(query)
    assert readiness == [1]


# ---- check_docs ----------------------------------------------------------------------------------------------------------


def _chunk(text, doc_id="d"):
    return KnowledgeChunk(doc_id, "T", "S", False, text)


def test_check_docs_flags_a_chunk_the_output_guardrail_blocks():
    assert "blocked by the output guardrail" in check_docs.chunk_problems(_chunk("Keep your password private."))
    assert "blocked by the output guardrail" not in check_docs.chunk_problems(_chunk("Keep your login details private."))


def test_check_docs_flags_a_duplicate_document_and_an_unclosed_front_matter(tmp_path):
    (tmp_path / "a.md").write_text(GOOD, encoding="utf-8")
    (tmp_path / "b.md").write_text(GOOD.replace("title: Good", "title: Copy"), encoding="utf-8")
    (tmp_path / "c.md").write_text("---\ntitle: Open\nverified: false\n\nBody that would be indexed with the front matter.\n",
                                   encoding="utf-8")
    problems = check_docs.document_problems(tmp_path)
    assert "b.md: duplicate of a.md" in problems
    assert "c.md: front matter is never closed" in problems
    assert not any(p.startswith("a.md") for p in problems)


def test_check_docs_flags_duplicate_chunk_text():
    problems = check_docs.duplicate_chunks([_chunk("Same text.", "a"), _chunk("Other text.", "a"), _chunk("Same text.", "b")])
    assert len(problems) == 1 and problems[0].startswith("b: duplicate chunk text (also in a)")


def test_the_shipped_documents_have_no_file_problems():
    assert check_docs.document_problems(store.DOCUMENTS_DIR) == []
    assert check_docs.duplicate_chunks(store._load_documents()) == []


@pytest.mark.parametrize("text, expected", [
    ("It can search rooms for you.", "dangling opener"),
    ("Use the Delete Account page.", "dangling opener"),
    ("Ask us (see section 11).", "section reference"),
    ("Please read our Privacy Policy.", "Please read our"),
    ("1. Search: What happens Pick your city.", "table leftover"),
    ("Refund: Time Within 5-7 business days.", "table leftover"),
])
def test_check_docs_wording_warnings(text, expected):
    assert any(expected in w for w in check_docs.chunk_warnings(_chunk(text)))
    assert check_docs.chunk_problems(_chunk(text)) == []  # a warning, not a flagged chunk


def test_wording_warnings_use_the_display_text():
    faq = _chunk("Do I need an account? You can browse listings freely.", "homilivo_faq")
    assert check_docs.chunk_warnings(faq) == []  # the question is removed before the opener is checked
    assert check_docs.chunk_warnings(_chunk("Itinerary planning is not offered.")) == []


def test_strict_mode_is_a_named_setting(monkeypatch):
    monkeypatch.delenv(check_docs.STRICT_ENV, raising=False)
    assert check_docs.strict() is False
    monkeypatch.setenv(check_docs.STRICT_ENV, "1")
    assert check_docs.strict() is True


# ---- the debug log reuses the answer's retrieval ------------------------------------------------------------------------


def test_the_debug_log_does_not_retrieve_again(monkeypatch, caplog):
    retrieve = fake_retrieve({"raw words": [scored("faq", REFUND, 0.6)], "rewrite words": [scored("faq", REFUND, 0.75)]})
    monkeypatch.setattr(answer_module, "retrieve", retrieve)

    def no_second_retrieval(*args, **kwargs):
        raise AssertionError("the debug log retrieved again")

    monkeypatch.setattr(store, "retrieve", no_second_retrieval)
    monkeypatch.setattr(store, "_load_documents", lambda: [KnowledgeChunk("faq", "faq", "t", True, REFUND)])
    platform._cached_chunk_positions.cache_clear()
    caplog.set_level(logging.DEBUG, logger=platform.logger.name)
    try:
        d = Decision(intent=Intent.PLATFORM_INFO, slots=Slots(), standalone_query="rewrite words", confidence=0.9)
        r = platform.handle_platform_info(d, {}, LLMConversationState(), TurnBudget(), user_message="raw words")
    finally:
        platform._cached_chunk_positions.cache_clear()
    assert r.reply == REFUND
    assert retrieve.calls == ["raw words", "rewrite words"]  # once each, by the answer path only
    assert "query='raw words'" in caplog.text and "rewritten='rewrite words'" in caplog.text
    assert "raw_top3=['faq#0=0.600']" in caplog.text and "rewrite_top3=['faq#0=0.750']" in caplog.text
    assert "used=rewrite" in caplog.text
