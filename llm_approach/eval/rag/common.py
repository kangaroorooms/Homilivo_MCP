"""Shared setup for the knowledge-base evaluation: a temporary index (never data/chroma_v2), the cached embedding model only.

Importing this module sets HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE (unless already set), so the model comes from the local cache
and nothing is downloaded. Run the scripts from the repository root: ``python -m llm_approach.eval.rag.<script>``.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("LLM_APPROACH_METRICS", "")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # the rupee sign and Devanagari on a Windows console

TMP_INDEX = Path(tempfile.gettempdir()) / "llm_approach_rag_eval_chroma"
RESULTS = Path(tempfile.gettempdir()) / "llm_approach_rag_eval_results.json"  # written by eval_rag, read by the other scripts


def use_temp_index(path: Path | None = None):
    """Point the knowledge store at a temporary folder and drop its caches; returns the store module."""
    from llm_approach.rag import store

    folder = path or TMP_INDEX
    os.environ[store.PERSIST_DIR_ENV] = str(folder)
    store._reset_caches()
    store._forget_ready()
    assert "chroma_v2" not in str(store._persist_dir())
    return store


def chunk_ids():
    """``([(doc_id, index within doc, chunk), ...], {(doc_id, text): index})`` in index order."""
    from llm_approach.rag import store

    pos: dict[str, int] = {}
    out, by_text = [], {}
    for c in store._load_documents():
        i = pos.get(c.doc_id, 0)
        pos[c.doc_id] = i + 1
        out.append((c.doc_id, i, c))
        by_text[(c.doc_id, c.text)] = i
    return out, by_text
