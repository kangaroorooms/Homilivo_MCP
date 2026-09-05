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
from typing import Protocol

try:
    import redis as redis_lib
except ImportError:  # pragma: no cover - redis is always in requirements.txt
    redis_lib = None  # STATE_BACKEND=redis will raise a clear error if reached

DEMO_TOKEN_TTL_SECONDS = 3600.0
_MAX_TRACKED_TOKENS = 1000

_tokens: dict[str, float] = {}  # opaque token -> issued_at (memory backend only)
_lock = threading.Lock()


def _purge_expired(now: float) -> None:
    expired = [
        token
        for token, issued_at in _tokens.items()
        if now - issued_at > DEMO_TOKEN_TTL_SECONDS
    ]
    for token in expired:
        del _tokens[token]


class DemoAuthBackend(Protocol):
    def issue(self, token: str, now: float) -> None: ...
    def get_issued_at(self, token: str) -> float | None: ...
    def delete(self, token: str) -> None: ...


class InMemoryDemoAuthBackend:
    """
    Thin wrapper around the module-level `_tokens` dict (unchanged from
    before Redis support) - kept as a real module attribute, not hidden
    inside this class, so existing tests that reach into
    `demo_auth._tokens` directly for setup/assertions keep working.
    """

    def issue(self, token: str, now: float) -> None:
        with _lock:
            if len(_tokens) >= _MAX_TRACKED_TOKENS:
                _purge_expired(now)
            _tokens[token] = now

    def get_issued_at(self, token: str) -> float | None:
        return _tokens.get(token)

    def delete(self, token: str) -> None:
        with _lock:
            _tokens.pop(token, None)


class RedisDemoAuthBackend:
    """
    Redis-backed demo-token store: one instance issuing a token can be
    redeemed by ANY instance sharing the same Redis, instead of only
    the process that minted it. Redis's own EXPIRE enforces the TTL
    directly (no separate purge step needed).
    """

    _KEY_PREFIX = "homilivo:demo_token:"

    def __init__(self, redis_client=None, url: str | None = None) -> None:
        if redis_client is None and redis_lib is None:
            raise RuntimeError(
                "STATE_BACKEND=redis requires the 'redis' package "
                "(pip install -r requirements.txt)."
            )
        self._redis = redis_client or redis_lib.Redis.from_url(
            url or os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
        )

    def issue(self, token: str, now: float) -> None:
        self._redis.set(
            f"{self._KEY_PREFIX}{token}", now, ex=int(DEMO_TOKEN_TTL_SECONDS)
        )

    def get_issued_at(self, token: str) -> float | None:
        raw = self._redis.get(f"{self._KEY_PREFIX}{token}")
        return float(raw) if raw is not None else None

    def delete(self, token: str) -> None:
        self._redis.delete(f"{self._KEY_PREFIX}{token}")


def _create_default_backend() -> DemoAuthBackend:
    if os.getenv("STATE_BACKEND", "memory").strip().lower() == "redis":
        return RedisDemoAuthBackend()
    return InMemoryDemoAuthBackend()


_backend: DemoAuthBackend = _create_default_backend()


def get_demo_auth_backend() -> DemoAuthBackend:
    """Overridden in tests to inject a fresh/fake backend, mirroring
    app.rate_limit.get_rate_limiter()."""
    return _backend


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

    get_demo_auth_backend().issue(token, now)

    return token


def redeem_demo_token(candidate: str) -> str | None:
    """
    Return the real server-side Homilivo credential if `candidate` is a
    currently-valid demo token, else None (meaning: treat it as an
    ordinary BYOT credential instead).
    """
    now = time.time()
    backend = get_demo_auth_backend()

    issued_at = backend.get_issued_at(candidate)
    if issued_at is None:
        return None
    if now - issued_at > DEMO_TOKEN_TTL_SECONDS:
        backend.delete(candidate)
        return None

    return os.getenv("KANGAROO_TOKEN")
