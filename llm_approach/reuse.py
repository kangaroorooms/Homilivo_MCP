"""Single adapter for every import from ``app/``.

Nothing else in ``llm_approach`` imports from ``app`` directly. Heavy symbols
(anything that pulls in torch, sentence_transformers, ollama or mcp) are
exposed through lazy getters and cached after the first call.

Pure symbols that live in ``app/experiments/agent_modes.py`` (constants, the
room sanitiser and the delimiter helper) are loaded by executing only those
definitions from the original source file, so importing this module or
calling those getters never loads torch. ``LLM_APPROACH_FULL_IMPORT=1``
forces a normal import of ``agent_modes`` instead, and a normal import is
also the fallback if the extraction ever fails.
"""

from __future__ import annotations

import ast
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from app import fast_parser  # light: stdlib re only

_REPO_ROOT = Path(__file__).resolve().parents[1]
_AGENT_MODES_PATH = _REPO_ROOT / "app" / "experiments" / "agent_modes.py"

_PURE_NAMES = (
    "GROQ_MODEL",
    "FACT_GROUNDING_RULE",
    "PROMPT_INJECTION_DEFENSE_RULE",
    "_PROMPT_FENCE_DELIMITERS",
    "_neutralize_prompt_delimiters",
    "_SANITIZE_FIELD_MAX_LENGTHS",
    "_SANITIZE_DEFAULT_MAX_LENGTH",
    "_CONTROL_CHAR_RE",
    "_MULTI_SPACE_RE",
    "_MULTI_NEWLINE_RE",
    "_sanitize_string_value",
    "sanitize_room_for_llm_context",
    "UNSUPPORTED_ACTION_RESPONSE",
    "OUT_OF_SCOPE_RESPONSE",
    "SHOW_ALTERNATIVES_RESPONSE",
    "NOT_VERIFIED_AMENITY_RESPONSE",
)


def _top_level_name(node: ast.stmt) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


@lru_cache(maxsize=1)
def _agent_modes_pure() -> dict[str, Any]:
    """Return the pure agent_modes symbols without importing the module."""

    if os.getenv("LLM_APPROACH_FULL_IMPORT") != "1":
        try:
            tree = ast.parse(_AGENT_MODES_PATH.read_text(encoding="utf-8"))
            wanted = set(_PURE_NAMES)
            namespace: dict[str, Any] = {
                "__name__": "llm_approach._agent_modes_pure",
                "re": re,
                "Any": Any,
            }
            for node in tree.body:
                if _top_level_name(node) in wanted:
                    exec(
                        compile(
                            ast.Module(body=[node], type_ignores=[]),
                            str(_AGENT_MODES_PATH),
                            "exec",
                        ),
                        namespace,
                    )
            if all(name in namespace for name in _PURE_NAMES):
                return {name: namespace[name] for name in _PURE_NAMES}
        except Exception:
            pass  # fall through to the real import

    from app.experiments import agent_modes  # heavy

    return {name: getattr(agent_modes, name) for name in _PURE_NAMES}


# ---- pure symbols from agent_modes -------------------------------------


def get_groq_model() -> str:
    return _agent_modes_pure()["GROQ_MODEL"]


def get_fact_grounding_rule() -> str:
    return _agent_modes_pure()["FACT_GROUNDING_RULE"]


def get_prompt_injection_rule() -> str:
    return _agent_modes_pure()["PROMPT_INJECTION_DEFENSE_RULE"]


def get_neutralize() -> Callable[[Any], Any]:
    """``_neutralize_prompt_delimiters(value) -> value`` (private upstream)."""
    return _agent_modes_pure()["_neutralize_prompt_delimiters"]


def get_sanitize_room() -> Callable[[dict], dict]:
    """``sanitize_room_for_llm_context(room) -> room``."""
    return _agent_modes_pure()["sanitize_room_for_llm_context"]


def get_unsupported_action_response() -> str:
    return _agent_modes_pure()["UNSUPPORTED_ACTION_RESPONSE"]


def get_out_of_scope_response() -> str:
    return _agent_modes_pure()["OUT_OF_SCOPE_RESPONSE"]


def get_show_alternatives_response() -> str:
    return _agent_modes_pure()["SHOW_ALTERNATIVES_RESPONSE"]


def get_not_verified_amenity_response() -> str:
    return _agent_modes_pure()["NOT_VERIFIED_AMENITY_RESPONSE"]


# ---- light: app.fast_parser --------------------------------------------


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


# ---- heavy: guardrails (torch / sentence_transformers) -------------------


@lru_cache(maxsize=1)
def get_is_security_blocked() -> Callable[[str], tuple[bool, str]]:
    """``is_security_blocked(message) -> (blocked, refusal_text)``."""
    from app.guardrails import is_security_blocked

    return is_security_blocked


@lru_cache(maxsize=1)
def get_output_guardrail() -> Callable[[str], tuple[bool, str]]:
    """``validate_ai_response(response) -> (is_valid, text_or_reason)``."""
    from app.guardrails import validate_ai_response

    return validate_ai_response


# ---- light but only needed by the Phase 5 server -------------------------


@lru_cache(maxsize=1)
def get_rate_limit_module() -> Any:
    from app import rate_limit

    return rate_limit


@lru_cache(maxsize=1)
def get_demo_auth_module() -> Any:
    from app import demo_auth

    return demo_auth


def get_knowledge_preload() -> Callable[[], None]:
    """``app.knowledge.store.warmup()``: loads the knowledge-base index. Lazy (embedding model)."""
    from app.knowledge.store import warmup

    return warmup


@lru_cache(maxsize=1)
def get_answer_platform_question() -> Callable[[str], dict]:
    """``answer_platform_question(query) -> {grounded, message, sources}``.
    Lazy: retrieval loads the embedding model."""
    from app.knowledge.answer import answer_platform_question

    return answer_platform_question
