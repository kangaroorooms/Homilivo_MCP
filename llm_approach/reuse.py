"""Adapter for the pieces of ``app/`` that llm_approach does not own yet.

Phase 1 (part A) moved the safe components into local copies (``prompt_safety``,
``fast_parser``, ``guardrails.output_guardrail``, ``security.demo_auth``,
``security.rate_limit``); the getters for them below now return the local copies, so llm_approach
no longer reads those from ``app/``. What still comes from ``app/`` through lazy getters:
``is_security_blocked`` (input guardrail) and the knowledge base / RAG.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

from llm_approach import fast_parser, prompt_safety


# ---- pure symbols (local copy: prompt_safety) -------------------------------------


def get_groq_model() -> str:
    return prompt_safety.GROQ_MODEL


def get_fact_grounding_rule() -> str:
    return prompt_safety.FACT_GROUNDING_RULE


def get_prompt_injection_rule() -> str:
    return prompt_safety.PROMPT_INJECTION_DEFENSE_RULE


def get_neutralize() -> Callable[[Any], Any]:
    """``_neutralize_prompt_delimiters(value) -> value`` (private upstream)."""
    return prompt_safety._neutralize_prompt_delimiters


def get_sanitize_room() -> Callable[[dict], dict]:
    """``sanitize_room_for_llm_context(room) -> room``."""
    return prompt_safety.sanitize_room_for_llm_context


def get_unsupported_action_response() -> str:
    return prompt_safety.UNSUPPORTED_ACTION_RESPONSE


def get_out_of_scope_response() -> str:
    return prompt_safety.OUT_OF_SCOPE_RESPONSE


def get_show_alternatives_response() -> str:
    return prompt_safety.SHOW_ALTERNATIVES_RESPONSE


def get_not_verified_amenity_response() -> str:
    return prompt_safety.NOT_VERIFIED_AMENITY_RESPONSE


# ---- light: fast_parser (local copy) --------------------------------------------


def get_known_cities() -> set[str]:
    """Lowercase canonical city names."""
    return set(fast_parser.KNOWN_CANONICAL_CITIES)


def get_city_aliases() -> dict[str, str]:
    """Lowercase alias -> canonical display name."""
    return dict(fast_parser.CITY_ALIASES)


def get_extract_search_parameters() -> Callable[[str], dict]:
    """``extract_search_parameters(query) -> {city, min_price, max_price,
    room_types, ideal_for, ...}``: the old rule-based slot extractor (fallback)."""
    return fast_parser.extract_search_parameters


# ---- guardrails: output guardrail is local; the input guardrail still comes from app/ (torch) -------------------


@lru_cache(maxsize=1)
def get_is_security_blocked() -> Callable[[str], tuple[bool, str]]:
    """``is_security_blocked(message) -> (blocked, refusal_text)``."""
    from app.guardrails import is_security_blocked

    return is_security_blocked


@lru_cache(maxsize=1)
def get_output_guardrail() -> Callable[[str], tuple[bool, str]]:
    """``validate_ai_response(response) -> (is_valid, text_or_reason)``."""
    from llm_approach.guardrails.output_guardrail import validate_ai_response

    return validate_ai_response


# ---- light but only needed by the Phase 5 server -------------------------


@lru_cache(maxsize=1)
def get_rate_limit_module() -> Any:
    from llm_approach.security import rate_limit

    return rate_limit


@lru_cache(maxsize=1)
def get_demo_auth_module() -> Any:
    from llm_approach.security import demo_auth

    return demo_auth


def get_knowledge_preload() -> Callable[[], None]:
    """``llm_approach.rag.store.warmup()``: loads the knowledge-base index (v2's own copy and storage). Lazy."""
    from llm_approach.rag.store import warmup

    return warmup


@lru_cache(maxsize=1)
def get_answer_platform_question() -> Callable[[str], dict]:
    """``answer_platform_question(query) -> {grounded, message, sources}``.
    Lazy: retrieval loads the embedding model."""
    from llm_approach.rag.answer import answer_platform_question

    return answer_platform_question
