# Copied from app/knowledge/ingest.py at commit 7dbcba3 (copied, not imported).
# Changes: import path and usage line (marked '# v2:')
# Drift is checked by llm_approach/tests/test_copy_drift.py.
# ==== end of v2 provenance header ====
"""
Idempotent ingestion for the knowledge base's embedded Chroma
collection. Scans every markdown file in app/knowledge/documents/
(not just one) and rebuilds the collection from what's currently on
disk - safe to re-run any time a document is added, edited, or
removed; running it twice in a row never duplicates or accumulates
stale chunks (see ingest() in app/knowledge/store.py).

Usage:
    python -m llm_approach.rag.ingest  # v2: was: python -m app.knowledge.ingest
"""

from __future__ import annotations

from llm_approach.rag.store import ingest  # v2: was: from app.knowledge.store import ingest


def main() -> None:
    stats = ingest()
    print(
        f"Ingested {stats['documents_loaded']} document(s), "
        f"{stats['chunks_indexed']} chunk(s) into Chroma "
        f"(embedding model available: {stats['embedding_model_available']})."
    )


if __name__ == "__main__":
    main()
