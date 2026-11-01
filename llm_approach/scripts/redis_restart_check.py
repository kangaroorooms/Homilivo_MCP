"""One-off check against a REAL Redis: a session survives a restart, and sessions stay isolated.

    python llm_approach/scripts/redis_restart_check.py

Uses the fake LLM decisions and the offline rooms fixture (no Gemini, Groq or rooms API call). Redis comes
from REDIS_URL (default redis://localhost:6379/0); keys are the store's own (prefix homilivo:v2:).

Steps: search + follow-up turn -> new store and client (a restart) -> same session id asks about wifi and
gets a rooms reply -> a new session id gets the "no rooms yet" reply -> the session key has a TTL -> only
the keys of the two session ids made here are deleted. Prints PASS/FAIL per step, with booleans and counts
only (never a key, a URL or a message). Exit 0 = all pass, 1 = a step failed, 2 = Redis not reachable.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_approach import config, llm_client, pipeline, reuse  # noqa: E402
from llm_approach.actions import followup  # noqa: E402
from llm_approach.schema import parse_decision  # noqa: E402
from llm_approach.session_store import RedisSessionStore, make_store  # noqa: E402

SEARCH_MSG = "2bhk in bangalore under 20k"
FOLLOWUP_MSG = "which one is the cheapest"
WIFI_MSG = "does the first one have wifi"

_DECISIONS = {
    SEARCH_MSG: {"intent": "NEW_SEARCH", "confidence": 0.95, "slots": {"city": "Bangalore", "bhk": 2, "max_price": 20000}},
    FOLLOWUP_MSG: {"intent": "ROOM_FOLLOWUP", "confidence": 0.95, "operation": "LOWEST_PRICE"},
    WIFI_MSG: {"intent": "ROOM_FOLLOWUP", "confidence": 0.95, "operation": "AMENITY", "room_reference": 1,
               "slots": {"amenity": "wifi"}},
}


async def _fake_decide(user_message, state_summary, budget):
    return parse_decision(_DECISIONS[user_message])


def _install_fakes() -> Callable[[], None]:
    """Fake LLM, fake guardrails, offline rooms fixture. Returns a function that puts everything back."""
    saved_env = {k: os.environ.get(k) for k in (config.OFFLINE_ROOMS_ENV, "KANGAROO_TOKEN", config.STATE_BACKEND_ENV)}
    saved = (llm_client.decide, reuse.get_is_security_blocked, reuse.get_output_guardrail, pipeline._store)
    llm_client.decide = _fake_decide
    reuse.get_is_security_blocked = lambda: (lambda m: (False, ""))
    reuse.get_output_guardrail = lambda: (lambda t: (True, t))
    os.environ[config.OFFLINE_ROOMS_ENV] = "1"
    os.environ[config.STATE_BACKEND_ENV] = "redis"
    os.environ.pop("KANGAROO_TOKEN", None)
    pipeline.configure(full_guardrails=True)

    def restore() -> None:
        llm_client.decide, reuse.get_is_security_blocked, reuse.get_output_guardrail, pipeline._store = saved
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    return restore


def _turn(store, session_id: str, message: str):
    return asyncio.run(pipeline.handle_turn(session_id, message, store=store))


def _delete_session_keys(client, session_ids: list[str]) -> int:
    """Delete only the keys belonging to these session ids. Returns how many keys existed."""
    cls = RedisSessionStore
    removed = 0
    for sid in session_ids:
        removed += client.delete(cls._KEY_PREFIX + sid, cls._LOCK_PREFIX + sid, cls._RESET_PREFIX + sid)
        client.zrem(cls._LRU_KEY, sid)
    return removed


def run_checks(new_client: Callable[[], object], emit: Callable[[str], None] = print) -> bool:
    """Run every step. ``new_client()`` returns a fresh Redis client (decode_responses=True) on each call, which
    is what a restart does. Returns True when all steps pass."""
    first_sid, other_sid = str(uuid.uuid4()), str(uuid.uuid4())
    results: list[bool] = []

    def report(step: str, ok: bool, detail: str = "") -> bool:
        results.append(ok)
        emit(f"{'PASS' if ok else 'FAIL'}  {step}{(' ' + detail) if detail else ''}")
        return ok

    restore = _install_fakes()
    client = None
    try:
        # 1. before the restart
        client = new_client()
        store = make_store(redis_client=client)
        searched = _turn(store, first_sid, SEARCH_MSG)
        followed = _turn(store, first_sid, FOLLOWUP_MSG)
        saved = store.peek(first_sid)
        rooms_before = len(saved.rooms_on_screen) if saved else 0
        report("1 search and follow-up saved", bool(searched.success and followed.success and rooms_before > 0),
               f"rooms={rooms_before} turns={saved.turn_count if saved else 0}")

        # 2. restart: throw everything away and build again against the same Redis
        del store, client, saved
        pipeline._store = None
        client = new_client()
        store = make_store(redis_client=client)
        report("2 new store and client built", isinstance(store, RedisSessionStore) and store.count() >= 1,
               f"is_shared={store.is_shared}")

        # 3. same session id after the restart
        reply = _turn(store, first_sid, WIFI_MSG)
        state = store.peek(first_sid)
        about_rooms = bool(reply.success and reply.reply != followup.SEARCH_FIRST and state and state.rooms_on_screen)
        report("3 same session remembered after restart", about_rooms,
               f"no_rooms_reply={reply.reply == followup.SEARCH_FIRST} rooms={len(state.rooms_on_screen) if state else 0}")

        # 4. a new session id must not see it
        isolated = _turn(store, other_sid, WIFI_MSG)
        report("4 new session is isolated", isolated.reply == followup.SEARCH_FIRST,
               f"no_rooms_reply={isolated.reply == followup.SEARCH_FIRST}")

        # 5. TTL on the key
        ttl = client.ttl(RedisSessionStore._KEY_PREFIX + first_sid)
        report("5 ttl set on session key", 0 < ttl <= int(config.SESSION_TTL_SECONDS), f"ttl_set={ttl > 0}")
    except Exception as exc:  # noqa: BLE001 - report the class only: the message may hold a URL
        report("unexpected error", False, f"error_class={type(exc).__name__}")
    finally:
        try:
            if client is None:
                client = new_client()
            removed = _delete_session_keys(client, [first_sid, other_sid])
            left = sum(client.exists(RedisSessionStore._KEY_PREFIX + s) for s in (first_sid, other_sid))
            report("6 cleanup of this run's keys", left == 0, f"deleted={removed} left={left}")
        except Exception as exc:  # noqa: BLE001
            report("6 cleanup of this run's keys", False, f"error_class={type(exc).__name__}")
        restore()
    return all(results)


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except Exception:  # noqa: BLE001 - python-dotenv missing or .env unreadable: use the environment
        pass
    import redis

    url = config.redis_url()

    def new_client():
        return redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)

    try:
        new_client().ping()
    except Exception as exc:  # noqa: BLE001
        print(f"Redis is not reachable (error_class={type(exc).__name__}). Start Redis or set REDIS_URL. Nothing was run.")
        return 2
    ok = run_checks(new_client)
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
