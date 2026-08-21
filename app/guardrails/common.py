"""
Shared utilities for guardrails.

This module intentionally contains only common helpers.
Individual guardrail logic belongs in its respective module.
"""

from __future__ import annotations

from typing import Any


MAX_USER_INPUT_LENGTH = 500
MAX_OUTPUT_LENGTH = 5000
MAX_SEARCH_RESULTS = 50


def is_non_empty_string(value: Any) -> bool:
    """Return True when value is a non-empty string."""
    return isinstance(value, str) and bool(value.strip())


def normalize_text(value: str) -> str:
    """Normalize text for deterministic comparisons."""
    return " ".join(value.lower().strip().split())


def safe_error_message(message: str) -> str:
    """
    Normalize guardrail error messages so callers receive
    consistent single-line messages.
    """
    return " ".join(str(message).strip().split())