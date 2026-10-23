"""Short hash of the system prompt, so results made with different prompts are never mixed.

Every eval result record carries ``prompt_fp``. A record without one was made before the
fingerprint existed and counts as ``LEGACY`` (the pre-change prompt). The hash covers the
whole system prompt text, so any edit to the rules or the few-shot examples changes it.
"""

from __future__ import annotations

import hashlib
from typing import Any

LEGACY = "pre-change"  # records written before fingerprints existed
_LENGTH = 12


def of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_LENGTH]


def current() -> str:
    from llm_approach import prompts

    return of_text(prompts.system_prompt())


def of_record(record: dict[str, Any]) -> str:
    return str(record.get("prompt_fp") or LEGACY)
