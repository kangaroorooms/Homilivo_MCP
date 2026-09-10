"""
Lightweight RAG-style knowledge base for stable PLATFORM questions
("how does verification work?", "what are the platform fees?", "how
do I contact an owner?", "how does booking work?"), as opposed to LIVE
LISTING questions (price/location/type/amenities of a specific search
result), which are always answered from the Homilivo API - see the
PLATFORM_KNOWLEDGE routing branch in app/experiments/agent_modes.py.

WHY ONLY ONE DOCUMENT SHIPS TODAY
----------------------------------
No Homilivo FAQ/policy documentation exists anywhere in this
repository (README.md and docs/info.md were both checked - neither
contains platform policy content). Fabricating verification steps,
fee amounts, or booking rules would violate this project's grounding
requirements, so the only document shipped today
(documents/about_this_assistant.md) states facts that are verifiable
from this codebase itself - what THIS ASSISTANT can and cannot do -
and explicitly says it does not have verified information about
Homilivo's actual policies.

When real Homilivo FAQ/policy documents become available, drop them
into app/knowledge/documents/ as markdown files with the same
frontmatter shape (title/source/verified) and they are picked up
automatically on the next process start - no code changes needed. A
question that does not match any indexed document simply retrieves
nothing (see MIN_RELEVANCE_SCORE below), and the caller must say so
rather than asking an LLM to improvise an answer.

PIPELINE
--------
load documents -> split into paragraphs -> chunk to CHUNK_MAX_CHARS ->
embed every chunk once with the shared SentenceTransformer (the same
model app.guardrails.input_guardrail and app.experiments.interpreter
already use) -> at query time, embed the query and rank chunks by
cosine similarity -> keep the top TOP_K chunks that clear
MIN_RELEVANCE_SCORE, each carrying its source document title for
citation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"

CHUNK_MAX_CHARS = 600
TOP_K = 3
MIN_RELEVANCE_SCORE = 0.35


@dataclass(frozen=True)
class KnowledgeChunk:
    doc_id: str
    title: str
    source: str
    verified: bool
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: KnowledgeChunk
    score: float


def _parse_document(path: Path) -> tuple[dict[str, Any], str]:
    """Split a markdown file's YAML frontmatter from its body."""
    raw = path.read_text(encoding="utf-8")

    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) == 3:
            _, frontmatter_raw, body = parts
            import yaml
            metadata = yaml.safe_load(frontmatter_raw) or {}
            return metadata, body.strip()

    return {}, raw.strip()


def _chunk_text(text: str) -> list[str]:
    """
    One markdown paragraph = one chunk, so each chunk stays focused on
    a single topic (better retrieval precision than merging adjacent
    paragraphs on a small, hand-authored corpus like this one). A
    paragraph longer than CHUNK_MAX_CHARS is split into fixed-size
    windows as a safety net against a single oversized block.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []

    for paragraph in paragraphs:
        if len(paragraph) <= CHUNK_MAX_CHARS:
            chunks.append(paragraph)
            continue

        for start in range(0, len(paragraph), CHUNK_MAX_CHARS):
            window = paragraph[start:start + CHUNK_MAX_CHARS].strip()
            if window:
                chunks.append(window)

    return chunks


def _load_documents() -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []

    if not DOCUMENTS_DIR.exists():
        return chunks

    for path in sorted(DOCUMENTS_DIR.glob("*.md")):
        metadata, body = _parse_document(path)
        title = metadata.get("title", path.stem)
        source = metadata.get("source", "unknown")
        verified = bool(metadata.get("verified", False))

        for chunk_text in _chunk_text(body):
            chunks.append(
                KnowledgeChunk(
                    doc_id=path.stem,
                    title=title,
                    source=source,
                    verified=verified,
                    text=chunk_text,
                )
            )

    return chunks


@lru_cache(maxsize=1)
def _get_index():
    """
    (chunks, embedding_matrix), computed once per process.

    Returns (chunks, None) when the embedding model is unavailable, so
    retrieve() can degrade to "nothing retrieved" instead of raising -
    matching how app.experiments.interpreter degrades when the model
    fails to load.
    """
    chunks = _load_documents()

    if not chunks:
        return [], None

    try:
        from app.embeddings import get_shared_model
        model = get_shared_model()
    except Exception:  # noqa: BLE001 - degrade gracefully
        return chunks, None

    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return chunks, embeddings


def warmup() -> None:
    """Force the index to build ahead of the first request."""
    _get_index()


def index_stats() -> dict[str, Any]:
    """Summary used by GET /knowledge/health."""
    chunks, embeddings = _get_index()

    return {
        "documents_loaded": len({chunk.doc_id for chunk in chunks}),
        "chunks_indexed": len(chunks),
        "embedding_model_available": embeddings is not None,
    }


def retrieve(
    query: str,
    top_k: int = TOP_K,
    min_score: float = MIN_RELEVANCE_SCORE,
) -> list[RetrievedChunk]:
    """
    Return the top_k indexed chunks whose cosine similarity to `query`
    is at least min_score, ranked highest first. Empty list means
    "nothing in the knowledge base is confidently relevant" - callers
    must treat that as ungrounded, not fall back to guessing.
    """
    if not query or not query.strip():
        return []

    chunks, embeddings = _get_index()

    if not chunks or embeddings is None:
        return []

    try:
        from app.embeddings import get_shared_model
        model = get_shared_model()
    except Exception:  # noqa: BLE001
        return []

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    scores = embeddings @ query_embedding

    ranked_indices = scores.argsort()[::-1][:top_k]

    return [
        RetrievedChunk(chunk=chunks[i], score=float(scores[i]))
        for i in ranked_indices
        if scores[i] >= min_score
    ]
