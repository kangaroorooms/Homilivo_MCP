# Provenance of llm_approach/rag/

Copied from app/ at commit 7dbcba3 (copied, not imported). Python files carry a provenance header and are
drift-checked by llm_approach/tests/test_copy_drift.py. The two documents are byte-for-byte copies, also drift-checked.

| Copy | Source |
| --- | --- |
| rag/embeddings.py | app/embeddings.py |
| rag/store.py | app/knowledge/store.py |
| rag/answer.py | app/knowledge/answer.py |
| rag/ingest.py | app/knowledge/ingest.py |
| rag/documents/about_this_assistant.md | app/knowledge/documents/about_this_assistant.md |
| rag/documents/homilivo_faq.md | app/knowledge/documents/homilivo_faq.md |

Own storage: `LLM_APPROACH_CHROMA_PERSIST_DIR`, default `data/chroma_v2`, collection `homilivo_knowledge_v2`.
The app's `CHROMA_PERSIST_DIR` (data/chroma) is never read, and a path equal to data/chroma is refused.
This file is outside documents/ on purpose: every *.md in documents/ is indexed.
