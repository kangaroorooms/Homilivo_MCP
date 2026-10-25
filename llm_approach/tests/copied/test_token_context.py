# Copied from tests/test_token_context.py at commit 7ee27fb (copied, not imported).
# Changes: the import points at the copied tool. Every changed line is marked '# v2:'.
# Source under test is the copied tool llm_approach/tools/mcp_search.py (from app/server.py); app/ is not imported.

"""
Regression tests for the per-request Homilivo token context.

Previously api.py wrote the caller's token into a process-global
(os.environ["KANGAROO_TOKEN"]) and server.py read it back later in the
same request. Under real concurrency (FastAPI serving requests as
concurrent asyncio tasks on one worker), one request's token could be
overwritten by another's before the first request's outbound HTTP call
fired - leaking one user's Homilivo credential into another user's
search. These tests prove the ContextVar-based replacement keeps each
request's token isolated regardless of interleaving.
"""

import asyncio

from llm_approach.tools.mcp_search import set_kangaroo_token, _get_kangaroo_token  # v2: was: from app.server import set_kangaroo_token, _get_kangaroo_token


async def _simulate_request(token: str, delay_before_read: float) -> str | None:
    """
    Mimics one FastAPI request: bind a token to this task, await some
    work (standing in for guardrails/parsing/the Homilivo HTTP call),
    then read the token back. A leaking global would let a
    concurrently-running sibling with a different token clobber it
    during that await.
    """
    set_kangaroo_token(token)
    await asyncio.sleep(delay_before_read)
    return _get_kangaroo_token()


def test_concurrent_requests_do_not_leak_tokens():
    async def run():
        return await asyncio.gather(
            _simulate_request("token-A", 0.05),
            _simulate_request("token-B", 0.01),
            _simulate_request("token-C", 0.03),
        )

    token_a, token_b, token_c = asyncio.run(run())

    assert token_a == "token-A"
    assert token_b == "token-B"
    assert token_c == "token-C"


def test_many_concurrent_requests_each_keep_their_own_token():
    async def run():
        tasks = [
            _simulate_request(f"token-{i}", 0.001 * (i % 5))
            for i in range(50)
        ]
        return await asyncio.gather(*tasks)

    results = asyncio.run(run())

    assert results == [f"token-{i}" for i in range(50)]


def test_token_context_falls_back_to_env_when_unset(monkeypatch):
    """
    CLI / manual usage outside FastAPI (e.g. `python -m app.agent`):
    no per-request token is ever bound, so the lookup must fall back
    to the process environment variable - there is only one caller and
    no concurrency to protect against in that path.
    """
    monkeypatch.setenv("KANGAROO_TOKEN", "env-token")

    assert _get_kangaroo_token() == "env-token"


def test_token_context_prefers_bound_token_over_env(monkeypatch):
    monkeypatch.setenv("KANGAROO_TOKEN", "env-token")

    async def run():
        set_kangaroo_token("request-token")
        return _get_kangaroo_token()

    assert asyncio.run(run()) == "request-token"
