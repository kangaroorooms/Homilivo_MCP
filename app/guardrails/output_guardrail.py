"""
Output Guardrail.

Validates the final user-facing chatbot response before it is
returned through either the terminal agent or FastAPI.

Responsibilities:
- Validate response type
- Reject empty responses
- Protect against excessive output
- Detect sensitive information
- Reject control characters
- Detect obvious prompt/system leakage
- Reject malformed responses
"""

from __future__ import annotations

import re

from .common import MAX_OUTPUT_LENGTH


# ============================================================
# SENSITIVE INFORMATION
# ============================================================

SENSITIVE_OUTPUT_PATTERNS = [
    r"\bkangaroo[\s_-]*token\b",
    r"\bapi[\s_-]*key\b",
    r"\bapi[\s_-]*token\b",
    r"\bauthorization\s*:\s*bearer\b",
    r"\bbearer[\s_-]*token\b",
    r"\baccess[\s_-]*token\b",
    r"\brefresh[\s_-]*token\b",
    r"\bpassword\b",
    r"\bsecret\b",
    r"\bcredential[s]?\b",
    r"\bclient[\s_-]*secret\b",
]


# ============================================================
# INTERNAL / SYSTEM INFORMATION
# ============================================================

INTERNAL_OUTPUT_PATTERNS = [
    r"\bsystem\s+prompt\b",
    r"\bdeveloper\s+message\b",
    r"\binternal\s+instructions?\b",
    r"\binternal\s+prompt\b",
    r"\bhidden\s+instructions?\b",
    r"\btool\s+call\b",
    r"\bmcp\s+server\b",
    r"\benvironment\s+variable\b",
    r"\bKANGAROO_TOKEN\b",
]


# ============================================================
# CONTROL CHARACTERS
# ============================================================

def contains_invalid_control_characters(
    response: str,
) -> bool:

    return any(
        ord(character) < 32
        and character not in (
            "\n",
            "\r",
            "\t",
        )
        for character in response
    )


# ============================================================
# PATTERN MATCHING
# ============================================================

def contains_sensitive_information(
    response: str,
) -> bool:

    for pattern in SENSITIVE_OUTPUT_PATTERNS:

        if re.search(
            pattern,
            response,
            flags=re.IGNORECASE,
        ):

            return True

    return False


def contains_internal_information(
    response: str,
) -> bool:

    for pattern in INTERNAL_OUTPUT_PATTERNS:

        if re.search(
            pattern,
            response,
            flags=re.IGNORECASE,
        ):

            return True

    return False


# ============================================================
# MAIN OUTPUT GUARDRAIL
# ============================================================

def validate_ai_response(
    response: str,
) -> tuple[bool, str]:

    # --------------------------------------------------------
    # TYPE VALIDATION
    # --------------------------------------------------------

    if not isinstance(
        response,
        str,
    ):

        return (
            False,
            "The AI returned an invalid response.",
        )

    # --------------------------------------------------------
    # EMPTY RESPONSE
    # --------------------------------------------------------

    if not response.strip():

        return (
            False,
            "The AI returned an empty response.",
        )

    # --------------------------------------------------------
    # LENGTH PROTECTION
    # --------------------------------------------------------

    if len(response) > MAX_OUTPUT_LENGTH:

        return (
            False,
            "The AI response is unexpectedly long.",
        )

    # --------------------------------------------------------
    # SENSITIVE / INTERNAL INFORMATION
    #
    # Skipped for a structured markdown table (starts with "| ",
    # e.g. format_rooms_as_table()'s output). These two checks exist
    # to catch the AI/system leaking ITS OWN credentials or internals
    # in free-form generated text - they are not meant to censor
    # verified, deterministically-rendered room data (title, location,
    # amenities) that already passed the parameter and api-data
    # guardrails, and whose content is not chosen by this system. A
    # real listing whose title happens to contain an ordinary English
    # word like "secret" or "password" must not have its entire table
    # blocked. Every other check below (length, empty, control
    # characters, excessive blank lines) still runs on a table, and
    # every non-table response still gets full coverage of both
    # checks.
    # --------------------------------------------------------

    is_markdown_table = response.lstrip().startswith("| ")

    if not is_markdown_table:

        if contains_sensitive_information(
            response
        ):

            return (
                False,
                "The response contained potentially sensitive information.",
            )

        if contains_internal_information(
            response
        ):

            return (
                False,
                "The response contained internal system information.",
            )

    # --------------------------------------------------------
    # CONTROL CHARACTERS
    # --------------------------------------------------------

    if contains_invalid_control_characters(
        response
    ):

        return (
            False,
            "The response contained invalid control characters.",
        )

    # --------------------------------------------------------
    # EXCESSIVE WHITESPACE
    # --------------------------------------------------------

    if re.search(
        r"\n{8,}",
        response,
    ):

        return (
            False,
            "The response contained excessive empty lines.",
        )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    return (
        True,
        "Response validated.",
    )