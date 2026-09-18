"""
Shared SentenceTransformer loading.

app/guardrails/input_guardrail.py (intent classification) and
app/experiments/interpreter.py (operation fallback) both need semantic
embeddings from the SAME model. Before this module existed, each loaded
its own independent SentenceTransformer("all-MiniLM-L6-v2") instance -
doubling memory, duplicating the one-time load cost, and leaving only
one of the two warmed at FastAPI startup (so the first real user
request silently paid the other's multi-second load latency).

This is now the single place the model is instantiated. Each caller
keeps its own existing failure-handling around calling
get_shared_model() - this module only removes the duplicate load, it
does not change what happens on failure.
"""

from __future__ import annotations

from functools import lru_cache

# Multilingual model supporting 50+ languages
# including Hindi and Hinglish for Indian users.
# Produces 384-dim embeddings — same as previous
# all-MiniLM-L6-v2, Chroma collection compatible.
# Switched in P2-09b after fixing anchor/example
# regressions in input_guardrail.py and interpreter.py
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def get_shared_model():
    """
    Load the SentenceTransformer model once per process and hand back
    the same instance to every caller.

    functools.lru_cache serializes concurrent first calls internally,
    so this is safe without an extra lock. It does NOT cache a failed
    call (lru_cache never caches exceptions), so a load failure simply
    propagates to the caller every time, exactly as a direct
    SentenceTransformer(...) call would have before this module
    existed - callers are responsible for their own degradation.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)
