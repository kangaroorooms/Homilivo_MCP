"""Cleaning of untrusted text before it enters the prompt.

The reused ``_neutralize_prompt_delimiters`` strips only exact ``<verified_data>`` style tags, so
``"< /verified_data >"`` or ``"</VERIFIED_DATA>"`` in a room title could close the data fence.
``neutralize`` runs for the user message, history, last bot message and every room field in the
state summary. Steps, in order:

1. NFKC (full-width ``＜`` becomes ``<``) and removal of zero-width/format and control characters,
   keeping tab, newline and carriage return;
2. the reused neutraliser (earlier behaviour is only extended);
3. removal of tag-like text that names a fence (any case, any spacing inside the tag, also a
   half tag such as ``</verified_data`` or ``verified_data>``) and of ``<system>``-style role tags;
4. defusing of role prefixes (``system:``, ``assistant:``, ``user:``, ``developer:``, ``tool:``) at
   the start of a line or right after a removed tag (including the reused neutraliser's ``[blocked-tag]``): the colon becomes ``-``.

The result is never longer than the input, and ordinary text (room names, prices, Hinglish)
is returned unchanged. Pure functions, no I/O, safe to import.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Any

from llm_approach import reuse

_ROLE_WORDS = ("system", "assistant", "user", "developer", "tool")
_MARK = "\x00"  # stands where a tag was removed; never survives

_KEEP_CONTROL = {"\t", "\n", "\r"}


@lru_cache(maxsize=1)
def fence_names() -> tuple[str, ...]:
    """Fence tag names the prompts use (``prompts.FENCE_NAMES``)."""
    from llm_approach import prompts

    return tuple(prompts.FENCE_NAMES)


@lru_cache(maxsize=1)
def _patterns() -> tuple[re.Pattern, re.Pattern, re.Pattern, re.Pattern]:
    names = "|".join(re.escape(n) for n in fence_names())
    tag_names = names + "|" + "|".join(_ROLE_WORDS) + "|im_start|im_end"
    # <  /name  >  with any spacing and case (inner spaces between letters are not allowed)
    full_tag = re.compile(rf"<\s*/?\s*(?:{tag_names})\s*/?\s*>", re.IGNORECASE)
    # a half tag: "</verified_data" without the closing ">" or "verified_data>" without the opening "<"
    half_open = re.compile(rf"<\s*/?\s*(?:{names})\b", re.IGNORECASE)
    half_close = re.compile(rf"\b(?:{names})\s*/?\s*>", re.IGNORECASE)
    role = re.compile(rf"(^|\n|{_MARK}|\[blocked-tag\])([ \t]*)({'|'.join(_ROLE_WORDS)})([ \t]*):", re.IGNORECASE)
    return full_tag, half_open, half_close, role


def _strip_invisible(text: str) -> str:
    return "".join(
        c for c in text
        if c in _KEEP_CONTROL or unicodedata.category(c) not in ("Cf", "Cc", "Cs", "Co", "Cn")
    )


def _nfkc(text: str) -> str:
    whole = unicodedata.normalize("NFKC", text)
    if len(whole) <= len(text):
        return whole
    # NFKC can expand (ligatures, "…"): keep only one-to-one replacements so the text never grows
    out = []
    for c in text:
        n = unicodedata.normalize("NFKC", c)
        out.append(n if len(n) == 1 else c)
    return "".join(out)


def clean_text(text: str) -> str:
    """Steps 1 to 4 for one string."""
    original_len = len(text)
    text = _strip_invisible(_nfkc(text))
    text = reuse.get_neutralize()(text)
    full_tag, half_open, half_close, role = _patterns()
    text = full_tag.sub(_MARK, text)
    text = half_open.sub(_MARK, text)
    text = half_close.sub(_MARK, text)
    text = role.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}-", text)
    text = text.replace(_MARK, "")
    return text if len(text) <= original_len else text[:original_len]


def neutralize(value: Any) -> Any:
    """``clean_text`` on every string inside ``value`` (str, list, tuple, dict); other types pass through."""
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, dict):
        return {k: neutralize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [neutralize(v) for v in value]
    if isinstance(value, tuple):
        return tuple(neutralize(v) for v in value)
    return value
