"""
app/experiments/interpreter.py

Semantic operation classifier for room follow-up questions.

WHY THIS EXISTS
---------------
detect_room_operation() in agent_modes.py maps arbitrary user phrasing onto a
FIXED set of operation slots. Doing that with keyword tuples forces one new
keyword per phrasing, so the tuples grow without bound. This module maps open
phrasing onto the same closed slot set using sentence-embedding similarity
against a small anchor table, so new phrasings cost zero (or at most one or two
anchor lines), not a new branch.

CONTRACT
--------
    classify_operation(text: str) -> str | None

Returns exactly one of the operation constants below (identical strings to the
ones agent_modes.py already switches on), or None when it is not confident. On
None the caller falls back exactly as it does today, so behaviour degrades
safely and never crashes.

RULES
-----
- The SentenceTransformer model is loaded ONCE (module-level singleton).
- Anchor embeddings are computed ONCE and cached.
- No LLM call. This is a local encode + cosine, single-digit milliseconds on CPU.
- Ordinals and the handful of unambiguous superlative/keyword tokens are handled
  deterministically by the CALLER, before this runs, so they never reach the
  model. This module only sees the productive long tail.
- If sentence-transformers or the model is unavailable, every call returns None
  and the system keeps working on the deterministic core alone.
"""

from __future__ import annotations

import re
import threading
from typing import Optional

import numpy as np


# ============================================================
# OPERATION CONSTANTS  (must stay identical to agent_modes.py)
# ============================================================

LOWEST_PRICE = "LOWEST_PRICE"
HIGHEST_PRICE = "HIGHEST_PRICE"
ROOM_PRICE = "ROOM_PRICE"
ROOM_TYPE = "ROOM_TYPE"
LOCATION = "LOCATION"
STUDENT_SUITABILITY = "STUDENT_SUITABILITY"
FAMILY_SUITABILITY = "FAMILY_SUITABILITY"
PROFESSIONAL_SUITABILITY = "PROFESSIONAL_SUITABILITY"
AMENITY = "AMENITY"
ROOM_DETAILS = "ROOM_DETAILS"


# ============================================================
# ANCHOR TABLE
# ============================================================
# A few prototype phrasings per slot. To support a new way of asking something,
# add ONE line here (or nothing, if it already clusters). Never add branches.

ANCHORS: dict[str, list[str]] = {
    LOWEST_PRICE: [
        "which one is the cheapest",
        "least expensive room among these",
        "which is the most affordable option",
        "which one is easiest on my budget",
        "which will cost me the least",
        "lightest on my pocket",
        # Idiom-aware phrasings mirroring HIGHEST_PRICE's idiom anchors below,
        # so financial idioms don't skew toward the wrong price direction.
        "which one would set me back the least",
        "which costs the least amount",
        "which one is the least expensive",
        "what is the most affordable option",
        "which would cost me the least",
        "show me the most budget friendly option",
        # Hindi/Hinglish phrasings (additional anchors only - none of
        # the English anchors above are removed).
        #
        # "sabse sasta kaun sa hai", "sabse kam kimat wala",
        # "sasta wala dikhao" and "sabse affordable kaun sa" were
        # removed: their "kaun sa hai"/"wala" determiner cadence turned
        # out to generically match ANY Hindi reference query (e.g.
        # "pehle wale ke baare mein batao" - an ordinal/detail
        # question with no price meaning at all), not just price
        # questions, so they were false-pulling unrelated follow-ups
        # into LOWEST_PRICE. "kam se kam kitna hai" carries the same
        # meaning without that generic cadence and is kept as the
        # slot's Hindi anchor.
        "kam se kam kitna hai",
    ],
    HIGHEST_PRICE: [
        "which one is the most expensive",
        "the priciest option",
        "which has the highest rent",
        "which will cost me the most",
        "which is the biggest expense",
        "which will burn the biggest hole in my pocket",
        "which one hits my wallet the hardest",
        # Hindi/Hinglish phrasings (additional anchors only - none of
        # the English anchors above are removed).
        #
        # "sabse zyada kimat wala", "mehanga wala dikhao",
        # "sabse upar wala" and "sabse expensive kaun sa" were removed
        # for the same reason as LOWEST_PRICE above - same generic
        # "kaun sa hai"/"wala" cadence false-pulling unrelated Hindi
        # reference queries. "jo sabse mehanga hai" is kept as the
        # slot's Hindi anchor.
        "jo sabse mehanga hai",
    ],
    ROOM_PRICE: [
        "what is its price",
        "how much is the rent for this one",
        "what is the monthly cost of this room",
        "how expensive is this one",
        "what would I pay for it",
    ],
    ROOM_TYPE: [
        "what type of room is it",
        "how many bhk is this",
        "what kind of room is this",
        "is it a 2bhk or a pg",
    ],
    LOCATION: [
        "where is it located",
        "which area is this room in",
        "what part of the city is it in",
        "where can I find this place",
    ],
    STUDENT_SUITABILITY: [
        "is it suitable for students",
        "can a college student stay here",
        "would this work for a student",
    ],
    FAMILY_SUITABILITY: [
        "is it suitable for a family",
        "can a family live here",
        "would this be good for families",
    ],
    PROFESSIONAL_SUITABILITY: [
        "is it suitable for working professionals",
        "would this work for a professional",
        "is it okay for office workers",
    ],
    AMENITY: [
        "does it have parking",
        "is the room furnished",
        "does it come with wifi",
        "what facilities are included",
        "is there an attached bathroom",
        "does it have air conditioning",
        "does it come with a washing machine"
    ],
    ROOM_DETAILS: [
        "tell me more about this room",
        "give me the details of this property",
        "describe this one for me",
    ],
}


# ============================================================
# TUNABLES  (tune against the golden conversation, then freeze)
# ============================================================

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Minimum cosine similarity to the best-matching anchor to accept a slot.
ACCEPT_THRESHOLD = 0.45

# The winning slot must beat the runner-up slot by at least this much,
# otherwise the query is treated as ambiguous and we return None.
# Raised from 0.04 to 0.06 as defense-in-depth alongside the LOWEST_PRICE/
# HIGHEST_PRICE anchor cleanup above: no single anchor was solely
# responsible for the false-pull regression, so a stronger required
# margin adds a second layer of protection against similarly ambiguous
# short Hindi/Hinglish anchors in the future.
SLOT_MARGIN = 0.06

# Stricter floor for the two price-direction slots. Off-topic or meta
# phrasing ("kuch aur dikhao", "i dont understand", "ye wala nahi chahiye")
# scores 0.46-0.59 against the price anchors and used to win HIGHEST_PRICE
# with a wide margin (the margin only measures distance from the runner-up,
# not absolute confidence). Genuine Tier-B price phrasings score >= ~0.67;
# Hinglish "sabse sasta/kam" are resolved by Tier A and never rely on this.
PRICE_SLOT_THRESHOLD = 0.65
_PRICE_SLOTS = frozenset({LOWEST_PRICE, HIGHEST_PRICE})


# ============================================================
# LAZY SINGLETON STATE
# ============================================================

_model = None
_model_failed = False
_lock = threading.Lock()

_anchor_matrix: Optional[np.ndarray] = None   # (num_anchors, dim), L2-normalized
_anchor_labels: Optional[list[str]] = None    # parallel slot label per anchor row


def _load_model():
    """
    Load the SentenceTransformer once. Returns the model or None on failure.

    Delegates the actual instantiation to app.embeddings.get_shared_model()
    so this module and app.guardrails.input_guardrail never load two
    independent copies of the same model - but keeps its own _model /
    _model_failed memoization so a prior failure here is remembered
    instead of retried on every follow-up question.
    """
    global _model, _model_failed

    if _model is not None:
        return _model

    if _model_failed:
        return None

    with _lock:
        if _model is not None:
            return _model
        if _model_failed:
            return None
        try:
            from app.embeddings import get_shared_model
            _model = get_shared_model()
        except Exception as error:  # noqa: BLE001 - degrade gracefully
            print(f"⚠ interpreter: embedding model unavailable, "
                  f"falling back to deterministic core only ({error})")
            _model_failed = True
            return None

    return _model


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _build_anchor_index() -> bool:
    """Encode the anchor table once and cache it. Returns True on success."""
    global _anchor_matrix, _anchor_labels

    if _anchor_matrix is not None:
        return True

    model = _load_model()
    if model is None:
        return False

    with _lock:
        if _anchor_matrix is not None:
            return True

        phrases: list[str] = []
        labels: list[str] = []
        for slot, examples in ANCHORS.items():
            for phrase in examples:
                phrases.append(phrase)
                labels.append(slot)

        embeddings = np.asarray(
            model.encode(phrases, convert_to_numpy=True),
            dtype=np.float32,
        )
        _anchor_matrix = _normalize_rows(embeddings)
        _anchor_labels = labels

    return True


# ============================================================
# PUBLIC API
# ============================================================

def warmup() -> None:
    """
    Optional: force model load + anchor encoding ahead of the first request.
    Call this from FastAPI startup so the first user message is not slow.
    """
    _build_anchor_index()


def classify_operation(text: str) -> Optional[str]:
    """
    Map a follow-up question onto one operation slot, or None if not confident.

    This is intentionally called ONLY for the long tail: the caller resolves
    ordinals and the obvious keyword cases deterministically first.
    """
    if not text or not text.strip():
        return None

    if not _build_anchor_index():
        return None  # model unavailable -> caller keeps its deterministic result

    model = _load_model()
    if model is None:
        return None

    query = re.sub(r"\s+", " ", text.strip().lower())

    q = np.asarray(
        model.encode([query], convert_to_numpy=True),
        dtype=np.float32,
    )
    q = _normalize_rows(q)[0]

    sims = _anchor_matrix @ q  # cosine, since everything is L2-normalized

    # Best similarity per slot.
    best_by_slot: dict[str, float] = {}
    for label, score in zip(_anchor_labels, sims):
        score = float(score)
        if score > best_by_slot.get(label, -1.0):
            best_by_slot[label] = score

    ranked = sorted(best_by_slot.items(), key=lambda kv: kv[1], reverse=True)
    top_slot, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -1.0

    threshold = (
        PRICE_SLOT_THRESHOLD if top_slot in _PRICE_SLOTS else ACCEPT_THRESHOLD
    )
    if top_score < threshold:
        return None
    if (top_score - second_score) < SLOT_MARGIN:
        return None

    return top_slot
