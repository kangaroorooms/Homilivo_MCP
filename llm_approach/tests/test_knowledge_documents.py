"""The shipped knowledge documents (Homilivo website drafts, temporary) are safe to be quoted word for word.

Replies are the retrieved chunk text unchanged, so every chunk must already be plain text: short, no markdown, no links,
no domain but a bare homilivo.com, and no marker word. Also covers the unverified status flag. Reads the files only; no
model, no network.
"""

from __future__ import annotations

import re

from llm_approach import knowledge_status, reply_safety
from llm_approach.actions.platform import has_internal_marker
from llm_approach.rag import check_docs, store

_ALLOWED_EMAIL = "support@homilivo.com"
_MARKDOWN = re.compile(r"^\s*#|\*\*|\||\[\[|\]\]|^\s*```|\]\(", re.MULTILINE)
_LINK_OR_DOMAIN = re.compile(r"https?://\S+|www\.\S+|\b[\w-]+\.(?:com|in|org|net|io|co)\b", re.IGNORECASE)


def _chunks():
    chunks = store._load_documents()
    assert chunks, "no documents found"
    return chunks


def test_every_chunk_is_short_plain_text():
    for chunk in _chunks():
        t = chunk.text
        assert len(t) < store.CHUNK_MAX_CHARS, (chunk.doc_id, len(t))
        assert not _MARKDOWN.search(t), (chunk.doc_id, t[:60])
        assert not has_internal_marker(t), (chunk.doc_id, t[:60])
        assert reply_safety.find_unsafe(t) is None, (chunk.doc_id, t[:60])


def test_no_chunk_has_a_link_or_a_domain_other_than_homilivo():
    for chunk in _chunks():
        leftover = chunk.text.replace(_ALLOWED_EMAIL, "")
        for found in _LINK_OR_DOMAIN.findall(leftover):
            assert found.lower().rstrip(".,;") == "homilivo.com", (chunk.doc_id, found)


def test_every_chunk_has_a_title_and_a_source():
    for chunk in _chunks():
        assert chunk.title and chunk.source == "Homilivo official documents, draft v1 (pending team lead approval)", chunk.doc_id
        assert chunk.verified is False


def test_every_document_says_verified_false_in_its_front_matter():
    assert knowledge_status.unverified() is True
    for path in sorted(knowledge_status.DOCUMENTS.glob("*.md")):
        head = knowledge_status._FRONT_MATTER.match(path.read_text(encoding="utf-8"))
        assert head and knowledge_status._VERIFIED_FALSE.search(head.group(1)), path.name


def test_check_docs_flags_nothing_in_the_shipped_documents():
    assert [(c.doc_id, check_docs.chunk_problems(c)) for c in _chunks() if check_docs.chunk_problems(c)] == []


def test_check_docs_flags_bad_chunks():
    def chunk(text, title="T", source="S"):
        return store.KnowledgeChunk("d", title, source, False, text)

    assert check_docs.chunk_problems(chunk("## Heading"))
    assert check_docs.chunk_problems(chunk("---"))
    assert check_docs.chunk_problems(chunk("x" * 601))
    assert check_docs.chunk_problems(chunk("this is a placeholder"))
    assert check_docs.chunk_problems(chunk("see https://example.org"))
    assert check_docs.chunk_problems(chunk("<b>hi</b>"))
    assert check_docs.chunk_problems(chunk("visit evil.example"))
    assert check_docs.chunk_problems(chunk("fine text", title=""))
    assert check_docs.chunk_problems(chunk("fine text", source="unknown"))
    assert check_docs.chunk_problems(chunk("Email support@homilivo.com or see homilivo.com")) == []


def test_check_docs_refusals_use_the_real_guards():
    assert check_docs.refusals("Email support@homilivo.com.") == []
    assert check_docs.refusals("This is a placeholder.")
    assert check_docs.refusals("Visit https://evil.example/x")
