"""Additive input security layer, run before the reused ``is_security_blocked``.

The app's hard block (``BLOCKED_PATTERNS`` in ``app/guardrails/input_guardrail.py``)
is not edited. It misses instruction-override phrasing, prompt-tag spoofing and
role-play jailbreaks (README Phase 4, security finding), so this layer adds
regexes for those gaps. Regex only: no torch, no network, safe to import.

A blocked message gets one fixed refusal. The refusal never names a pattern.
Patterns are deliberately narrow so normal room requests pass ("I can't ignore
my budget", "system of 2bhk rooms", "show me rooms", Hinglish requests).
"""

from __future__ import annotations

import re
import unicodedata

from llm_approach import textclean

# Same wording the app's hard block returns (app/guardrails/input_guardrail.py).
REFUSAL = "I can only help with accommodation and room searches."

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"))

_VERBS = r"(?:ignore|disregard|forget|override|bypass|skip|drop)"
_FILL = r"(?:(?:all|any|every|your|the|my|these|those|of|ur)\s+)*"
_AGE = r"(?:previous|prior|above|earlier|preceding|former|original|initial|existing)"
_THINGS = (
    r"(?:instructions?|prompts?|rules?|directions?|guidelines?|programming|"
    r"restrictions?|constraints?|safeguards?|guardrails?|system\s+messages?)"
)

_PATTERNS = [
    # --- instruction override: "ignore (all) previous instructions", "forget your rules"
    rf"\b{_VERBS}\s+{_FILL}{_AGE}\s+{_FILL}{_THINGS}\b",
    rf"\b{_VERBS}\s+(?:all|any|every|your|ur)\s+{_FILL}{_THINGS}\b",
    rf"\b{_VERBS}\s+(?:everything|anything)\s+(?:you\s+(?:were|have\s+been)\s+)?(?:told|said|instructed)\b",
    r"\bdo\s+not\s+follow\s+(?:your|the|any)\s+(?:previous\s+|prior\s+)?(?:instructions?|rules?|guidelines?)\b",
    r"\bnew\s+instructions?\s*:",
    # --- reveal system prompt / hidden instructions
    r"\bsystem\s*prompt\b",
    r"\b(?:hidden|secret|internal|original|initial)\s+(?:instructions?|prompt|rules|messages?)\b",
    r"\b(?:reveal|show|print|display|repeat|leak|dump|output|tell|give|share|expose)\b[^.?!\n]{0,25}"
    r"\b(?:your\s+(?:system\s+|initial\s+|original\s+|full\s+)?(?:instructions?|prompt|rules|programming)"
    r"|the\s+(?:system|initial|original|full)\s+(?:instructions?|prompt|rules))\b",
    r"\bwhat\s+(?:are|is|were)\s+your\s+(?:system\s+|initial\s+|original\s+|hidden\s+)?"
    r"(?:instructions?|prompt|rules|programming)\b",
    # --- persona / role override
    r"\byou\s+are\s+now\s+(?:a|an|the|in|my|dan|free|unrestricted|no\s+longer|going\s+to|acting|called|named)\b",
    r"\bfrom\s+now\s+on\s*,?\s+you\s+(?:are|will|must|can|should)\b",
    r"\b(?:pretend|imagine)\s+(?:to\s+be|you\s+are|you're|that\s+you\s+are)\b",
    r"\bact\s+as\s+(?:if\s+you\s+(?:have|had|were|are)\b|an?\s+(?:unrestricted|unfiltered|jailbroken|evil)\b)",
    r"\broleplay\s+as\b",
    # --- modes / jailbreak
    r"\b(?:developer|dev|dan|god|debug|admin|sudo|root|unrestricted|unfiltered|jailbreak|jailbroken)\s+mode\b",
    r"\bdo\s+anything\s+now\b",
    r"\bjail\s*break(?:ing|s|ed)?\b",
    r"\b(?:without|no)\s+(?:any\s+)?(?:restrictions|filters|guardrails|safety\s+rules)\b",
    r"\bprompt\s+inject\w*",
    # --- role-tag / prompt-structure injection
    r"</?\s*(?:verified_data|user_message|system|assistant|user|developer|tool|instructions?)\s*/?>",
    r"<\|[a-z_]+\|>",
    r"\[\s*/?\s*(?:system|inst|assistant)\s*\]",
    r"(?:^|[\n\r.;:!?]|\s)#{2,}\s*(?:system|instructions?|assistant)\b",
    r"(?:^|[\n\r.;!?]\s*|\s)(?:system|assistant|developer)\s*:\s*\S",
    # --- secrets
    r"\b(?:api|auth|access|bearer|secret|private|jwt|session|refresh|login)[\s_-]*(?:key|token)s?\b",
    r"\b(?:kangaroo|groq|openai|anthropic)[\s_-]*(?:token|key|api|secret)\w*\b",
    r"\b(?:passwords?|passcodes?|passwd|credentials?|secrets?)\b",
    r"\b(?:env|environment)\s+(?:variables?|vars?|file)\b",
    r"(?<!\w)\.env\b",
    r"\bos\.environ\b|\bprintenv\b|\bgetenv\b",
    r"\b(?:give|show|tell|reveal|print|share|send|leak|dump|display|what(?:'s|\s+is)|your)\b[^.?!\n]{0,25}"
    r"\btokens?\b(?!\s*(?:amount|money|payment|fees?|deposit|advance))",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]


def normalize(message: str) -> str:
    """NFKC, strip zero-width characters, collapse whitespace."""
    text = textclean.strip_invisible(unicodedata.normalize("NFKC", message)).translate(_ZERO_WIDTH)  # v2: every invisible character, not seven
    return re.sub(r"[ \t\f\v]+", " ", text).strip()


def is_extra_blocked(message: object) -> tuple[bool, str]:
    """``(True, REFUSAL)`` when the message matches an injection or secret-request pattern."""
    if not isinstance(message, str):
        return False, ""
    text = normalize(message)
    if any(p.search(text) for p in _COMPILED):
        return True, REFUSAL
    return False, ""
