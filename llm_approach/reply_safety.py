"""Links, markdown images/links and HTML in a reply.

``find_unsafe(text)`` returns a fixed reason word, or None when the text is clean. The only link allowed is ``homilivo.com``
(the subscription footer ``[homilivo.com](https://homilivo.com)``, ``https://homilivo.com`` and the bare name); a look-alike
such as ``homilivo.com.evil.example`` is not. Applied to every reply, whoever wrote it.
"""

from __future__ import annotations

import re

from llm_approach.textclean import DOMAIN_RE

ALLOWED_HOST = "homilivo.com"

_ALLOWED = (
    re.compile(r"\[homilivo\.com\]\(https://homilivo\.com/?\)", re.IGNORECASE),
    re.compile(r"(?<![\w.@/-])https://homilivo\.com/?(?![\w.@/:?#%=&-])(?!\.\w)", re.IGNORECASE),
    re.compile(r"(?<![\w.@/-])homilivo\.com(?![\w@/:?#%=&-])(?!\.\w)", re.IGNORECASE),
)
_CHECKS = (
    ("markdown_link", re.compile(r"!?\[[^\]]*\]\([^)]*\)|!?\[[^\]]*\]\[[^\]]*\]")),
    ("html", re.compile(r"<\s*/?\s*[a-z!?][^>]*>", re.IGNORECASE)),
    ("url", re.compile(r"\b[a-z][a-z0-9+.-]{1,15}://\S+|\bwww\.\S+|(?<![\w:])//[\w.-]+", re.IGNORECASE)),
    ("scheme", re.compile(r"\b(?:javascript|vbscript)\s*:|\bdata:\S|\bmailto:|\bfile:|\bblob:", re.IGNORECASE)),
    ("domain", DOMAIN_RE),
)


def find_unsafe(text: object) -> str | None:
    """Fixed reason word (markdown_link, html, url, scheme, domain) for the first problem found, else None."""
    if not isinstance(text, str):
        return None
    for pattern in _ALLOWED:
        text = pattern.sub(" ", text)
    for reason, pattern in _CHECKS:
        if pattern.search(text):
            return reason
    return None
