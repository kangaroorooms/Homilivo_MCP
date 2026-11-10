"""Knowledge replies read like answers: best chunk only, no label prefix, no FAQ question; chunk_mode in the front matter.

No model and no network: retrieval is replaced by fixed RetrievedChunk lists.
"""

from __future__ import annotations

import re

import pytest

from llm_approach.rag import answer as answer_module
from llm_approach.rag import reply, store
from llm_approach.rag.store import KnowledgeChunk, RetrievedChunk, _chunk_text, _load_documents


def rc(doc_id, text, score):
    return RetrievedChunk(KnowledgeChunk(doc_id, f"Title of {doc_id}", "src", False, text), score)


FAQ_A = "Which cities does Homilivo cover? Homilivo lists rental flats across India."
FAQ_B = "Do I need an account? You can browse listings freely."
TERMS = "Homilivo terms and conditions - What Homilivo is: Homilivo is an online marketplace."


# ---- 1. chunking -------------------------------------------------------------------------------------------------


def _docs(tmp_path, monkeypatch, faq_front="", other_front=""):
    (tmp_path / "faq.md").write_text(f"---\ntitle: F\nsource: s\nverified: false\n{faq_front}---\n\n{FAQ_A}\n\n{FAQ_B}\n", encoding="utf-8")
    (tmp_path / "other.md").write_text(f"---\ntitle: O\nsource: s\nverified: false\n{other_front}---\n\n{TERMS}\n\nSecond short paragraph.\n", encoding="utf-8")
    monkeypatch.setattr(store, "DOCUMENTS_DIR", tmp_path)


def test_one_per_paragraph_keeps_each_faq_entry_separate(tmp_path, monkeypatch):
    _docs(tmp_path, monkeypatch, faq_front="chunk_mode: one_per_paragraph\n")
    faq = [c.text for c in _load_documents() if c.doc_id == "faq"]
    assert faq == [FAQ_A, FAQ_B]


def test_other_documents_are_chunked_as_before(tmp_path, monkeypatch):
    _docs(tmp_path, monkeypatch, faq_front="chunk_mode: one_per_paragraph\n")
    other = [c.text for c in _load_documents() if c.doc_id == "other"]
    assert other == [TERMS, "Second short paragraph."]
    assert _chunk_text("a" * 1300) == [_chunk_text("a" * 1300)[0], "a" * 600, "a" * 100]  # the 600-character split is unchanged
    assert len(_chunk_text("a" * 1300)[0]) == 600


def test_one_per_paragraph_still_splits_an_oversized_paragraph():
    assert [len(c) for c in _chunk_text("x" * 1300, one_per_paragraph=True)] == [600, 600, 100]


def test_the_shipped_faq_is_one_per_paragraph_and_every_entry_is_its_own_chunk():
    text = (store.DOCUMENTS_DIR / "homilivo_faq.md").read_text(encoding="utf-8")
    assert "chunk_mode: one_per_paragraph" in text.split("---")[1]
    faq = [c.text for c in _load_documents() if c.doc_id == "homilivo_faq"]
    assert all(len(c) < 600 and c.count("? ") >= 1 for c in faq)  # one question per chunk, none over the cap
    assert any(c.startswith("Which cities do you cover, and in which cities is Homilivo available?") for c in faq)
    assert any(c.startswith("Do I need an account?") for c in faq)


# ---- 2 and 3. the displayed text ---------------------------------------------------------------------------------


def test_a_faq_reply_has_no_question_and_no_label():
    out = reply.select_for_reply([rc("homilivo_faq", FAQ_A, 0.8)])
    assert [r.chunk.text for r in out] == ["Homilivo lists rental flats across India."]


def test_a_non_faq_reply_has_the_label_stripped_and_keeps_its_text():
    out = reply.select_for_reply([rc("terms_and_conditions", TERMS, 0.8)])
    assert out[0].chunk.text == "Homilivo is an online marketplace."
    q = "Is a question? Yes it is - really: but this is no label."  # a "?" before the dash: not a label
    assert reply.clean_text(q, "terms_and_conditions") == q  # only the FAQ loses its question


def test_label_is_stripped_in_the_final_message_and_the_stored_chunk_is_not_changed(monkeypatch):
    results = [rc("terms_and_conditions", TERMS, 0.8)]
    monkeypatch.setattr(answer_module, "retrieve", lambda q: results)
    out = answer_module.answer_platform_question("what is homilivo")
    assert out["message"] == "Homilivo is an online marketplace." and out["grounded"] is True
    assert results[0].chunk.text == TERMS  # display only


# ---- 4. only the best chunk --------------------------------------------------------------------------------------


def test_only_the_best_chunk_is_returned_even_when_the_others_are_close():
    far = reply.select_for_reply([rc("a", "A one.", 0.80), rc("b", "B one.", 0.70), rc("c", "C one.", 0.69)])
    assert [r.chunk.text for r in far] == ["A one."]
    close_other_doc = reply.select_for_reply([rc("a", "A one.", 0.80), rc("b", "B one.", 0.80), rc("c", "C one.", 0.80)])
    assert [r.chunk.doc_id for r in close_other_doc] == ["a"]
    assert not hasattr(reply, "SECOND_CHUNK_SCORE_MARGIN")


def test_the_final_message_is_the_best_chunk_alone(monkeypatch):
    results = [rc("a", "A one.", 0.80), rc("b", "B one.", 0.80), rc("c", "C one.", 0.80)]
    monkeypatch.setattr(answer_module, "retrieve", lambda q: results)
    out = answer_module.answer_platform_question("q")
    assert out["message"] == "A one." and out["sources"] == ["Title of a"] and out["grounded"] is True


def test_empty_results_select_nothing():
    assert reply.select_for_reply([]) == []


# ---- 5. text without "? " is unchanged ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["No question here, just a statement.", "Ends with a question mark?", "Is it?Yes"])
def test_text_without_a_question_and_space_is_unchanged(text):
    assert reply.clean_text(text, "homilivo_faq") == text


def test_only_the_first_question_is_removed():
    assert reply.clean_text("First? Answer one? Still the answer.", "homilivo_faq") == "Answer one? Still the answer."


def test_no_faq_answer_starts_with_yes_or_no_after_the_display_transform():
    faq = [c.text for c in _load_documents() if c.doc_id == "homilivo_faq"]
    assert faq
    for text in faq:
        assert text == text.lstrip() and text.count("? ") == 1, text[:60]  # one entry per paragraph, one question
        shown = reply.clean_text(text, "homilivo_faq")
        assert shown != text and not re.match(r"(?i)(yes|no)\b", shown), shown[:60]
