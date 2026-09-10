"""
Opaque demo-credential exchange for the /chat-ui demo page.

/chat's Authorization header is, by design, a "bring your own token"
(BYOT) contract: whatever the caller supplies is used, verbatim, as
the Homilivo API credential for that request (see app/server.py's
ContextVar-based per-request token isolation). That contract, and
every test that locks it down, is intentionally left untouched here.

This module adds a SEPARATE, opt-in mechanism so the bundled demo page
does not have to ask a presenter to paste a real Homilivo token into
the browser: if the server operator has configured KANGAROO_TOKEN as
a server-side environment variable, the demo page can request a
short-lived, random, meaningless-outside-this-process "demo token"
from GET /chat-ui/demo-token and send THAT as its Authorization
bearer. app/api.py recognizes a valid demo token and substitutes the
real server-side credential when calling Homilivo - the real secret
itself is never sent to, stored in, or derivable by the browser.

Any Authorization value that is NOT a currently-valid demo token is
passed through unchanged as a real BYOT credential, exactly as before.
"""

from __future__ import annotations

import os
import secrets
import threading
import time

DEMO_TOKEN_TTL_SECONDS = 3600.0
_MAX_TRACKED_TOKENS = 1000

_tokens: dict[str, float] = {}  # opaque token -> issued_at
_lock = threading.Lock()


def _purge_expired(now: float) -> None:
    expired = [
        token
        for token, issued_at in _tokens.items()
        if now - issued_at > DEMO_TOKEN_TTL_SECONDS
    ]
    for token in expired:
        del _tokens[token]


def server_credential_configured() -> bool:
    """True when the operator has set a server-side Homilivo credential."""
    return bool(os.getenv("KANGAROO_TOKEN"))


def issue_demo_token() -> str | None:
    """
    Mint a new opaque demo token, or None if no server-side credential
    is configured (nothing meaningful to redeem it for).
    """
    if not server_credential_configured():
        return None

    now = time.time()
    token = secrets.token_urlsafe(32)

    with _lock:
        if len(_tokens) >= _MAX_TRACKED_TOKENS:
            _purge_expired(now)
        _tokens[token] = now

    return token


def redeem_demo_token(candidate: str) -> str | None:
    """
    Return the real server-side Homilivo credential if `candidate` is a
    currently-valid demo token, else None (meaning: treat it as an
    ordinary BYOT credential instead).
    """
    now = time.time()

    with _lock:
        issued_at = _tokens.get(candidate)
        if issued_at is None:
            return None
        if now - issued_at > DEMO_TOKEN_TTL_SECONDS:
            del _tokens[candidate]
            return None

    return os.getenv("KANGAROO_TOKEN")
