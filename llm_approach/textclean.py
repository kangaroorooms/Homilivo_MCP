"""Plain-text cleaning of untrusted strings. No dependencies on the rest of the package.

``clean_text`` is the one cleaner for third-party text (rooms API fields, subscription plans): tags, links, bare domains,
``javascript:``/``data:`` schemes, markdown characters, control and invisible characters are removed and the length is
capped. ``normalize_input`` is applied once to the user's message at entry, so the filters and the model see the same text.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Top-level domains a bare "name.tld" is recognised by (a letters-only list keeps "2.5" and "e.g." out).
TLDS = (
    "com net org in io co app dev xyz info biz me ly link click top site online shop store tk ml ga cf gq ru cn uk us de fr "
    "example test local invalid edu gov ai gg to cc ws vip pw tv cloud page live website space"
).split()
DOMAIN_RE = re.compile(
    r"(?<![\w@.-])(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+(?:" + "|".join(TLDS) + r")(?![\w-])", re.IGNORECASE
)
URL_RE = re.compile(r"(?:https?://|ftp://|www\.)\S+", re.IGNORECASE)
SCHEME_RE = re.compile(r"\b(?:javascript|vbscript|data|file|blob|mailto)\s*:\S*", re.IGNORECASE)
PROTO_REL_RE = re.compile(r"(?<![\w:])//\S+")
TAG_RE = re.compile(r"<[^>]*>")
STRIP_RE = re.compile(r"[\x00-\x1f\x7f<>\[\]()*_`#|~\\{}]")  # markdown and HTML characters, control characters

_KEEP_CONTROL = {"\t", "\n", "\r"}
_INVISIBLE = ("Cf", "Cc", "Cs", "Co", "Cn")  # format, control, surrogate, private use, unassigned (zero-width, bidi, tags)


def strip_invisible(text: str) -> str:
    """Remove every format/control/private/unassigned character except tab, newline and carriage return."""
    return "".join(c for c in text if c in _KEEP_CONTROL or unicodedata.category(c) not in _INVISIBLE)


def normalize_input(text: str) -> str:
    """NFKC, then strip invisible characters. The same text goes to the security filters and to the model."""
    return strip_invisible(unicodedata.normalize("NFKC", text))


def clean_text(value: Any, limit: int) -> str | None:
    """Plain text from an untrusted string: markup, links, domains, schemes, markdown and invisible characters removed,
    whitespace collapsed, cut at ``limit``. None when nothing is left or the value is not a string."""
    if not isinstance(value, str):
        return None
    text = strip_invisible(unicodedata.normalize("NFKC", value))
    for pattern in (TAG_RE, URL_RE, SCHEME_RE, PROTO_REL_RE, DOMAIN_RE):
        text = pattern.sub(" ", text)
    text = STRIP_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()[:limit].strip()
    return text or None
