# Copied from app/knowledge/store.py at commit 7dbcba3 (copied, not imported).
# Changes: own storage (own env variable, default data/chroma_v2, own collection name, never the app's folder); embeddings import (marked '# v2:');
# the index is reused when the document fingerprint matches (rebuilt only on change, under a cross-process lock), and a failed first
# ingest is not cached for the life of the process (tried again after a pause)
# Drift is checked by llm_approach/tests/test_copy_drift.py.
# ==== end of v2 provenance header ====
"""
Lightweight RAG-style knowledge base for stable PLATFORM questions
("how does verification work?", "what are the platform fees?", "how
do I contact an owner?", "how does booking work?"), as opposed to LIVE
LISTING questions (price/location/type/amenities of a specific search
result), which are always answered from the Homilivo API - see the
PLATFORM_KNOWLEDGE routing branch in app/experiments/agent_modes.py.

BACKING STORE: EMBEDDED CHROMA
-------------------------------
Retrieval is served from an embedded (in-process, no separate
server) Chroma collection persisted under CHROMA_PERSIST_DIR (default
data/chroma - see .env.example). This replaced an earlier in-process
numpy array cached with lru_cache; the retrieval behavior it produces
is unchanged: paragraph chunking, the same MiniLM embeddings, cosine
similarity, top-K, and the relevance floor below.

Documents (app/knowledge/documents/*.md) are the source of truth, not
the Chroma collection - ingest() rebuilds the collection from whatever
markdown files are on disk, so editing/adding/removing a document and
re-ingesting is always safe and never accumulates stale or duplicate
chunks (see ingest() below and app/knowledge/ingest.py for the
standalone re-runnable script).

A question that does not match any indexed document with enough
confidence simply retrieves nothing (see MIN_RELEVANCE_SCORE below),
and the caller must say so rather than asking an LLM to improvise an
answer.

PIPELINE
--------
load documents -> split into paragraphs -> chunk to CHUNK_MAX_CHARS ->
embed every chunk once with the shared SentenceTransformer (the same
model app.guardrails.input_guardrail and app.experiments.interpreter
already use) -> store the embeddings in the Chroma collection -> at
query time, embed the query and let Chroma rank chunks by cosine
similarity -> keep the top TOP_K chunks that clear
MIN_RELEVANCE_SCORE, each carrying its source document title for
citation.
"""

from __future__ import annotations

import hashlib  # v2: fingerprint
import json  # v2: fingerprint
import logging  # v2: new
import os
import threading  # v2: ready state lock
import time  # v2: build lock and retry pause
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"

CHUNK_MAX_CHARS = 600
TOP_K = 3
MIN_RELEVANCE_SCORE = 0.35

COLLECTION_NAME = "homilivo_knowledge_v2"  # v2: was: COLLECTION_NAME = "homilivo_knowledge"
# v2: own storage, never the app's: its own variable, its own default folder, its own collection name
DEFAULT_PERSIST_DIR = "data/chroma_v2"  # v2: new
PERSIST_DIR_ENV = "LLM_APPROACH_CHROMA_PERSIST_DIR"  # v2: new
APP_PERSIST_DIR = Path("data/chroma")  # v2: new, the app/ folder that must never be used


def _persist_dir() -> Path:
    # v2: CHROMA_PERSIST_DIR (the app's variable) is never read: a shared .env must not point v2 at data/chroma
    path = Path(os.getenv(PERSIST_DIR_ENV, "").strip() or DEFAULT_PERSIST_DIR)  # v2: new
    if path.resolve() == APP_PERSIST_DIR.resolve():  # v2: new
        logging.getLogger(__name__).error("%s points at the app's data/chroma: using %s", PERSIST_DIR_ENV, DEFAULT_PERSIST_DIR)  # v2: new
        path = Path(DEFAULT_PERSIST_DIR)  # v2: new
    return path  # v2: new


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
    """
    Scan DOCUMENTS_DIR for every markdown file and chunk each one.
    This is the whole folder, not a single hardcoded document - drop
    any number of *.md files in here (with or without frontmatter) and
    they are all picked up on the next ingest()/process start.
    """
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
def _get_client():
    """
    The embedded (no separate server/process) Chroma client, persisted
    to disk under CHROMA_PERSIST_DIR. Returns None if chromadb isn't
    importable or the persist directory can't be created, so callers
    degrade to "nothing retrieved" instead of raising - matching how
    app.experiments.interpreter degrades when its model fails to load.
    """
    try:
        import chromadb
    except Exception:  # noqa: BLE001 - degrade gracefully
        return None

    persist_dir = _persist_dir()

    try:
        persist_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(persist_dir))
    except Exception:  # noqa: BLE001
        return None


def _get_collection():
    client = _get_client()
    if client is None:
        return None

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def ingest() -> dict[str, Any]:
    """
    (Re-)build the Chroma collection from whatever markdown documents
    are currently in DOCUMENTS_DIR. Idempotent and safe to re-run any
    time: the collection is dropped and recreated from the documents
    on disk every call, so running this twice in a row - or after a
    document was added, edited, or removed - never duplicates or
    accumulates stale chunks; the indexed state always matches exactly
    what's on disk right now.

    Returns {"documents_loaded", "chunks_indexed",
    "embedding_model_available"} - the same shape GET /knowledge/health
    and index_stats() surface.
    """
    chunks = _load_documents()
    documents_loaded = len({chunk.doc_id for chunk in chunks})

    client = _get_client()
    if client is None:
        return {
            "documents_loaded": documents_loaded,
            "chunks_indexed": 0,
            "embedding_model_available": False,
        }

    try:
        from llm_approach.rag.embeddings import get_shared_model  # v2: was: from app.embeddings import get_shared_model
        model = get_shared_model()
    except Exception:  # noqa: BLE001
        return {
            "documents_loaded": documents_loaded,
            "chunks_indexed": 0,
            "embedding_model_available": False,
        }

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - fine, collection didn't exist yet
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if not chunks:
        return {
            "documents_loaded": 0,
            "chunks_indexed": 0,
            "embedding_model_available": True,
        }

    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    collection.add(
        ids=[f"{chunk.doc_id}::{i}" for i, chunk in enumerate(chunks)],
        embeddings=embeddings.tolist(),
        documents=[chunk.text for chunk in chunks],
        metadatas=[
            {
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "source": chunk.source,
                "verified": chunk.verified,
            }
            for chunk in chunks
        ],
    )

    return {
        "documents_loaded": documents_loaded,
        "chunks_indexed": len(chunks),
        "embedding_model_available": True,
    }


def _ensure_ready() -> dict[str, Any]:
    """Build the collection once per process, exactly like the old lru_cache'd index."""
    return _ready_stats()  # v2: was: return ingest()


def _reset_caches() -> None:
    """
    Test-only: drop the process-level client/readiness caches so a
    changed CHROMA_PERSIST_DIR or document set takes effect without a
    process restart. Production code never calls this - a fresh
    process is expected to pick up documents once at warmup(), exactly
    like the store this replaced.
    """
    _get_client.cache_clear()
    _ensure_ready.cache_clear()


def warmup() -> None:
    """Force the index to build ahead of the first request."""
    _ensure_ready()


def index_stats() -> dict[str, Any]:
    """Summary used by GET /knowledge/health."""
    return dict(_ensure_ready())


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

    stats = _ensure_ready()
    if not stats["embedding_model_available"] or stats["chunks_indexed"] == 0:
        return []

    try:
        from llm_approach.rag.embeddings import get_shared_model  # v2: was: from app.embeddings import get_shared_model
        model = get_shared_model()
    except Exception:  # noqa: BLE001
        return []

    collection = _get_collection()
    if collection is None:
        return []

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    if not result["ids"] or not result["ids"][0]:
        return []

    retrieved: list[RetrievedChunk] = []

    for doc_text, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        # Chroma's cosine space reports distance = 1 - cosine_similarity
        # for normalized vectors, so undo that to recover the same
        # similarity score the old numpy dot-product path produced.
        score = 1.0 - float(distance)
        if score < min_score:
            continue

        chunk = KnowledgeChunk(
            doc_id=metadata["doc_id"],
            title=metadata["title"],
            source=metadata["source"],
            verified=bool(metadata["verified"]),
            text=doc_text,
        )
        retrieved.append(RetrievedChunk(chunk=chunk, score=score))

    return retrieved


FINGERPRINT_FILE = "doc_fingerprint.txt"  # v2: written next to the collection after a successful build
LOCK_FILE = ".ingest.lock"  # v2: exists while one process builds the collection
LOCK_WAIT_S = 120.0  # v2: the longest a start waits for another process's build
LOCK_STALE_S = 300.0  # v2: a lock older than this was left by a process that died
RETRY_FAILED_AFTER_S = 60.0  # v2: a failed first ingest is tried again after this pause, not on every question

_ready_lock = threading.Lock()  # v2: new
_ready_state: dict[str, Any] = {"ok": None, "failed": None, "failed_at": 0.0}  # v2: ok = the cached good result
_clock = time.monotonic  # v2: replaced in tests
_sleep = time.sleep  # v2: replaced in tests


def _fingerprint(chunks: list[KnowledgeChunk]) -> str:  # v2: new
    # v2: Same documents, same chunking, same embedding model and collection: the stored index is current.
    from llm_approach.rag.embeddings import MODEL_NAME  # v2: new
    digest = hashlib.sha256(f"{MODEL_NAME}|{CHUNK_MAX_CHARS}|{COLLECTION_NAME}".encode("utf-8"))  # v2: new
    for chunk in chunks:  # v2: new
        row = [chunk.doc_id, chunk.title, chunk.source, chunk.verified, chunk.text]  # v2: new
        digest.update(json.dumps(row, default=str, ensure_ascii=False).encode("utf-8"))  # v2: new
    return digest.hexdigest()  # v2: new


def _stored_fingerprint() -> str | None:  # v2: new
    try:  # v2: new
        return (_persist_dir() / FINGERPRINT_FILE).read_text(encoding="utf-8").strip() or None  # v2: new
    except OSError:  # v2: new
        return None  # v2: new


def _forget_fingerprint() -> None:  # v2: new
    try:  # v2: new
        (_persist_dir() / FINGERPRINT_FILE).unlink()  # v2: new
    except OSError:  # v2: new
        pass  # v2: new


def _store_fingerprint(value: str) -> None:  # v2: new
    path = _persist_dir() / FINGERPRINT_FILE  # v2: new
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # v2: new
    try:  # v2: new
        tmp.write_text(value, encoding="utf-8")  # v2: new
        os.replace(tmp, path)  # v2: new
    except OSError:  # v2: new
        logging.getLogger(__name__).warning("knowledge index fingerprint could not be saved")  # v2: new


def _acquire_build_lock() -> bool:  # v2: new
    # v2: An exclusive lock file in the persist folder: only one process rebuilds, the others wait and then reuse.
    path = _persist_dir() / LOCK_FILE  # v2: new
    deadline = _clock() + LOCK_WAIT_S  # v2: new
    while True:  # v2: new
        try:  # v2: new
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))  # v2: new
            return True  # v2: new
        except FileExistsError:  # v2: new
            try:  # v2: new
                if time.time() - path.stat().st_mtime > LOCK_STALE_S:  # v2: new
                    path.unlink()  # v2: new
                    continue  # v2: new
            except OSError:  # v2: new
                continue  # v2: new
            if _clock() >= deadline:  # v2: new
                return False  # v2: new
            _sleep(0.2)  # v2: new
        except OSError:  # v2: new
            return False  # v2: new


def _release_build_lock() -> None:  # v2: new
    try:  # v2: new
        (_persist_dir() / LOCK_FILE).unlink()  # v2: new
    except OSError:  # v2: new
        pass  # v2: new


def _reusable(chunks: list[KnowledgeChunk], fingerprint: str) -> dict[str, Any] | None:  # v2: new
    # v2: The stored collection is current when the fingerprint matches and every chunk is in it. None: rebuild.
    if not chunks or _stored_fingerprint() != fingerprint:  # v2: new
        return None  # v2: new
    try:  # v2: new
        collection = _get_collection()  # v2: new
        if collection is None or collection.count() != len(chunks):  # v2: new
            return None  # v2: new
    except Exception:  # noqa: BLE001  # v2: new
        return None  # v2: new
    stats = {"documents_loaded": len({c.doc_id for c in chunks}), "chunks_indexed": len(chunks)}  # v2: new
    try:  # v2: new
        from llm_approach.rag.embeddings import get_shared_model  # v2: new
        get_shared_model()  # v2: new, queries need the model: load it now, as a rebuild would
    except Exception:  # noqa: BLE001  # v2: new
        return {**stats, "embedding_model_available": False}  # v2: new
    return {**stats, "embedding_model_available": True}  # v2: new


def _sync_index() -> dict[str, Any]:  # v2: new
    chunks = _load_documents()  # v2: new
    fingerprint = _fingerprint(chunks)  # v2: new
    if _get_client() is None:  # v2: new
        return {"documents_loaded": len({c.doc_id for c in chunks}), "chunks_indexed": 0, "embedding_model_available": False}  # v2: new
    reused = _reusable(chunks, fingerprint)  # v2: new
    if reused is not None:  # v2: new
        return reused  # v2: new
    if not _acquire_build_lock():  # v2: new
        return {"documents_loaded": len({c.doc_id for c in chunks}), "chunks_indexed": 0, "embedding_model_available": False}  # v2: new
    try:  # v2: new
        reused = _reusable(chunks, fingerprint)  # v2: another process may have built it while this one waited
        if reused is not None:  # v2: new
            return reused  # v2: new
        _forget_fingerprint()  # v2: a build that dies half way must not leave a matching fingerprint
        stats = ingest()  # v2: new
        if stats["embedding_model_available"] and stats["chunks_indexed"] == len(chunks):  # v2: new
            _store_fingerprint(fingerprint)  # v2: new
        return stats  # v2: new
    finally:  # v2: new
        _release_build_lock()  # v2: new


def _ready_stats() -> dict[str, Any]:  # v2: new
    # v2: A good result is kept for the life of the process; a failed one is kept only for RETRY_FAILED_AFTER_S.
    with _ready_lock:  # v2: new
        if _ready_state["ok"] is not None:  # v2: new
            return _ready_state["ok"]  # v2: new
        failed = _ready_state["failed"]  # v2: new
        if failed is not None and _clock() - _ready_state["failed_at"] < RETRY_FAILED_AFTER_S:  # v2: new
            return failed  # v2: new
        if failed is not None:  # v2: new
            _get_client.cache_clear()  # v2: a remembered "no client" must not outlive the pause
        stats = _sync_index()  # v2: new
        if stats["embedding_model_available"]:  # v2: new
            _ready_state.update(ok=stats, failed=None)  # v2: new
        else:  # v2: new
            _ready_state.update(failed=stats, failed_at=_clock())  # v2: new
        return stats  # v2: new


def cached_stats() -> dict[str, Any] | None:  # v2: new
    # v2: The good result if one exists, without doing any work (for /ready after a failed start).
    ok = _ready_state["ok"]  # v2: new
    return dict(ok) if ok is not None else None  # v2: new


def _forget_ready() -> None:  # v2: new
    _ready_state.update(ok=None, failed=None, failed_at=0.0)  # v2: new


_ensure_ready.cache_clear = _forget_ready  # v2: _reset_caches clears it like the lru_cache it replaced
