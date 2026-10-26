"""Read-only GET to the Homilivo rooms API, with one safe retry.

Same URL, auth header and query parameters as ``search_rooms`` in
``app/server.py``. The token comes from ``KANGAROO_TOKEN`` only and is never
logged. Retry once, only on timeout or connection error. Never on 4xx/5xx.
Amenity and area are not API parameters: they are filtered here, on the result.
An area search fetches ``config.ROOMS_AREA_FETCH_LIMIT`` rooms and shows at most
``config.ROOMS_LIMIT`` of the matches.

Transport (``LLM_APPROACH_SEARCH_TRANSPORT``, default ``mcp``): the live call goes through the copied in-process MCP
tool (``tools/mcp_search.py``), which is then the only retry layer (3 attempts, 5xx included). ``direct`` keeps this
module's own httpx client and its single retry. An injected ``http_get`` always means direct. The offline fixture is
answered here before either transport.

Two behaviours match the original: the whole search is bounded by ``config.search_timeout_s()``
(``MCP_TOOL_TIMEOUT_SECONDS``, 15 s), and the token can be bound per request (``set_rooms_token``,
the original's ``set_kangaroo_token``), falling back to ``KANGAROO_TOKEN``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import threading
import time
from contextvars import ContextVar, Token, copy_context
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from llm_approach import config, observability, sorting, switches, turn_context
from llm_approach.actions import TurnBudget
from llm_approach.guardrails.api_data_guardrail import validate_room_data
from llm_approach.state import clean_room

logger = logging.getLogger(__name__)

UNAVAILABLE_MESSAGE = "Room search is unavailable right now, please try again shortly."
_DEFAULT_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "rooms_bangalore.json"


@dataclass
class RoomsResult:
    ok: bool
    rooms: list[dict[str, Any]] = field(default_factory=list)  # allowlisted
    error_class: str | None = None
    status: int | None = None
    message: str | None = None
    api_count: int | None = None  # rooms returned by the API, before the area and amenity filters
    fetch_limit: int | None = None  # the limit the fetch asked for
    truncated: bool = False  # more matching rooms than the display limit
    dropped: int = 0  # rooms the API returned that contradicted the search (v2)
    area_matches: int | None = None  # rooms that matched the area (None: no area asked)
    areas: list[str] = field(default_factory=list)  # real locations of the fetched rooms (area searches)
    total: int | None = None  # meta.total: matching rooms before paging (None: meta missing or malformed)
    has_next: bool | None = None  # meta.hasNextPage (None: meta missing or malformed)
    page: int = 1  # the page this result is for


@dataclass
class RawResponse:
    status_code: int
    body: Any  # parsed JSON or None


HttpGet = Callable[..., Any]

# The Homilivo token of the current request. A ContextVar, not os.environ: concurrent requests share one
# process, so a global write from one request could reach another's search (see app/server.py).
_rooms_token_var: ContextVar[str | None] = ContextVar("llm_approach_rooms_token", default=None)


def set_rooms_token(token: str | None) -> Token:
    """Bind the Homilivo token to the current request's task only. Never logged."""
    return _rooms_token_var.set(token or None)


def reset_rooms_token(handle: Token) -> None:
    _rooms_token_var.reset(handle)


def get_rooms_token() -> str | None:
    """The per-request token if one was bound, otherwise the process ``KANGAROO_TOKEN``."""
    return _rooms_token_var.get() or os.getenv("KANGAROO_TOKEN") or None


def _httpx_get(url: str, params: dict, headers: dict, timeout: float) -> Any:
    return httpx.get(url, params=params, headers=headers, timeout=timeout)


def normalize_token(value: Any) -> str:
    """Lowercase, spaces/underscores to hyphens: 'Hot Water' == 'hot-water'."""
    return re.sub(r"[\s_]+", "-", str(value).strip().lower())


# ---- params ---------------------------------------------------------------


def _room_types_param(search: dict[str, Any]) -> str | None:
    types = search.get("room_types")
    if isinstance(types, str):
        types = [types]
    if types:
        return ",".join(types)
    bhk = search.get("bhk")
    return f"{bhk}BHK" if bhk else None


def build_params(
    search: dict[str, Any], *, limit: int | None = None, page: int = 1
) -> dict[str, Any]:
    """Same names as server.py: city, roomTypes, minPrice, maxPrice, idealFor, sort, limit.
    ``page`` is sent only for page > 1. No amenity: the API has no such parameter."""
    params: dict[str, Any] = {
        "city": search.get("city"),
        "minPrice": search.get("min_price"),
        "maxPrice": search.get("max_price"),
        "roomTypes": _room_types_param(search),
        "idealFor": search.get("ideal_for"),
        "sort": sorting.api_sort(search),
        "limit": limit or config.ROOMS_LIMIT,
        "page": page if page > 1 else None,
    }
    return {k: v for k, v in params.items() if v is not None}


# ---- filtering (offline fixture and live amenity) -------------------------


def _room_price(room: dict[str, Any]) -> float | None:
    for key in ("pricePerMonth", "price"):
        value = room.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def room_amenities(room: dict[str, Any]) -> list[str] | None:
    """Normalised amenities, or None when the room lists none at all."""
    for key in ("amenities", "amenity", "facilities", "facility", "features"):
        value = room.get(key)
        if value is None:
            continue
        items = [value] if isinstance(value, str) else value
        if isinstance(items, list):
            return [normalize_token(i) for i in items if isinstance(i, str)]
    return None


def has_amenity(room: dict[str, Any], amenity: str) -> bool:
    return normalize_token(amenity) in (room_amenities(room) or [])


def filter_by_amenity(rooms: list[dict[str, Any]], amenity: str | None) -> list[dict[str, Any]]:
    return rooms if not amenity else [r for r in rooms if has_amenity(r, amenity)]


def normalize_area(value: Any) -> str:
    """Lowercase, punctuation removed, spaces collapsed."""
    return " ".join(re.sub(r"[\W_]+", " ", str(value).lower()).split())


def room_matches_area(room: dict[str, Any], area: str) -> bool:
    """Whole-word match of the area inside the room's location or landmark."""
    wanted = normalize_area(area)
    if not wanted:
        return False
    for key in ("location", "landmark"):
        value = room.get(key)
        if isinstance(value, str) and f" {wanted} " in f" {normalize_area(value)} ":
            return True
    return False


def filter_by_area(rooms: list[dict[str, Any]], area: str | None) -> list[dict[str, Any]]:
    return rooms if not area else [r for r in rooms if room_matches_area(r, area)]


def distinct_locations(rooms: list[dict[str, Any]]) -> list[str]:
    """Real ``location`` values of the rooms, first seen first, case-insensitive."""
    seen: dict[str, str] = {}
    for room in rooms:
        location = clean_room(room).get("location")
        if isinstance(location, str) and location.strip():
            seen.setdefault(location.strip().lower(), location.strip())
    return list(seen.values())


def _canon_city(value: Any) -> str:
    """Lowercase, spaces collapsed, aliases mapped to the canonical name ("Bengaluru" -> "bangalore")."""
    from llm_approach import validators  # lazy: validators does not import this module, but keep the import graph flat

    text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    return validators._city_candidates().get(text, text)


def _type_key(value: Any) -> str:
    return re.sub(r"[\s_-]+", "", str(value or "")).upper()


def contradicts(room: dict[str, Any], search: dict[str, Any]) -> bool:
    """A room that clearly contradicts the search: another city, a price above the maximum or below the minimum, a room type
    that is not one asked for. A field the room or the search does not have never counts against the room."""
    want_city, room_city = search.get("city"), room.get("city")
    if want_city and room_city and _canon_city(want_city) != _canon_city(room_city):
        return True
    price = _room_price(room)
    if price is not None:
        if search.get("max_price") is not None and price > search["max_price"]:
            return True
        if search.get("min_price") is not None and price < search["min_price"]:
            return True
    wanted_types = _room_types_param(search)
    room_type = room.get("roomType") or room.get("type")
    if wanted_types and room_type:
        if _type_key(room_type) not in {_type_key(t) for t in wanted_types.split(",")}:
            return True
    return False


def drop_off_criteria(rooms: list[dict[str, Any]], search: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """``(rooms that fit, how many were dropped)``; every dropped room is counted (fixed word) in the metrics."""
    kept = [r for r in rooms if not contradicts(r, search)]
    for _ in range(len(rooms) - len(kept)):
        try:
            observability.record_turn_event("rooms_off_criteria")
        except Exception:  # noqa: BLE001 - metrics never break a search
            pass
    return kept, len(rooms) - len(kept)


def _finish(
    fetched: list[dict[str, Any]],
    search: dict[str, Any],
    *,
    fetch_limit: int,
    display_limit: int,
    status: int | None = None,
    total: int | None = None,
    has_next: bool | None = None,
    page: int = 1,
) -> RoomsResult:
    """The one place where area and amenity filters and the display cut happen,
    for the live API and the offline fixture alike."""
    area = search.get("area")
    fetched_all = fetched
    fetched, dropped = drop_off_criteria(fetched, search)  # v2: rooms that contradict the search are not shown
    matched = filter_by_area(fetched, area)
    found = filter_by_amenity(matched, search.get("amenity"))
    shown = found[:display_limit]
    return RoomsResult(
        ok=True,
        rooms=[clean_room(r) for r in shown],
        status=status,
        api_count=len(fetched_all),
        dropped=dropped,
        fetch_limit=fetch_limit,
        truncated=len(found) > len(shown),
        area_matches=len(matched) if area else None,
        areas=distinct_locations(fetched) if area else [],
        total=total,
        has_next=has_next,
        page=page,
    )


def _filter_fixture(rooms: list[dict[str, Any]], search: dict[str, Any]) -> list[dict[str, Any]]:
    city = (search.get("city") or "").strip().lower()
    wanted_types = _room_types_param(search)
    wanted = {t.strip().lower() for t in wanted_types.split(",")} if wanted_types else set()
    out = []
    for room in rooms:
        if city and str(room.get("city", "")).strip().lower() != city:
            continue
        if wanted and str(room.get("roomType", "")).strip().lower() not in wanted:
            continue
        price = _room_price(room)
        if search.get("min_price") is not None and (price is None or price < search["min_price"]):
            continue
        if search.get("max_price") is not None and (price is None or price > search["max_price"]):
            continue
        ideal = search.get("ideal_for")
        if ideal and ideal.lower() not in [str(i).lower() for i in room.get("idealFor") or []]:
            continue
        out.append(room)
    return out


def offline_path() -> Path | None:
    value = os.getenv(config.OFFLINE_ROOMS_ENV, "").strip()
    if not value:
        return None
    return _DEFAULT_FIXTURE if value == "1" else Path(value)


def _offline_search(
    path: Path, search: dict[str, Any], limit: int, display_limit: int, page: int = 1
) -> RoomsResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    rooms = data["rooms"] if isinstance(data, dict) else data
    # Like the live API: API filters, then paging (with the same meta); area and
    # amenity afterwards on the page.
    matching = _filter_fixture([r for r in rooms if isinstance(r, dict)], search)
    start = (page - 1) * limit
    return _finish(
        matching[start : start + limit],
        search,
        fetch_limit=limit,
        display_limit=display_limit,
        total=len(matching),
        has_next=page * limit < len(matching),
        page=page,
    )


# ---- HTTP -----------------------------------------------------------------


_SECRET_LIKE = re.compile(r"(?:Bearer\s+\S+|eyJ[\w-]{6,}\S*|gsk_\w+|(?:token|key|password|secret)\s*[=:]\s*\S+)", re.IGNORECASE)


def scrub_message(text: str, search: dict[str, Any] | None = None, token: str | None = None) -> str:
    """Remove what an API error message may echo back: the user's search values, the token and
    anything token-shaped. The message is logged and kept in ``RoomsResult.message``."""
    for value in (search or {}).values():
        for item in value if isinstance(value, (list, tuple)) else (value,):
            if isinstance(item, (str, int)) and not isinstance(item, bool) and len(str(item)) >= 2:
                text = re.sub(re.escape(str(item)), "[value]", text, flags=re.IGNORECASE)
    if token:
        text = text.replace(token, "[token]")
    return _SECRET_LIKE.sub("[redacted]", text)


def _error_message(body: Any, search: dict[str, Any] | None = None, token: str | None = None) -> str | None:
    if isinstance(body, dict) and isinstance(body.get("message"), str):
        return scrub_message(body["message"][:2000], search, token)[:200]  # scrub first: a cut must not split a value
    return None


def request_once(
    params: dict[str, Any],
    *,
    token: str,
    http_get: HttpGet | None = None,
    timeout: float | None = None,
) -> RawResponse:
    """One GET, no retry. Raises httpx errors on timeout / connection failure."""
    get = http_get or _httpx_get
    response = get(
        config.ROOMS_API_URL,
        params=params,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=timeout or config.ROOMS_API_TIMEOUT_S,
    )
    try:
        body = response.json()
    except Exception:
        body = None
    return RawResponse(status_code=response.status_code, body=body)


def extract_rooms(body: Any) -> list[dict[str, Any]] | None:
    """Rooms live under ``data`` (as server.py reads it); ``rooms`` accepted too."""
    if isinstance(body, dict):
        for key in ("data", "rooms"):
            if isinstance(body.get(key), list):
                return [r for r in body[key] if isinstance(r, dict)]
    return None


def parse_meta(body: Any) -> tuple[int | None, bool | None]:
    """``(meta.total, meta.hasNextPage)``; each None when missing or the wrong type."""
    meta = body.get("meta") if isinstance(body, dict) else None
    if not isinstance(meta, dict):
        return None, None
    total, has_next = meta.get("total"), meta.get("hasNextPage")
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        total = None
    if not isinstance(has_next, bool):
        has_next = None
    return total, has_next


# The guardrail's reason can hold a room id or a value, so only the field name is logged. Prefix -> field.
_ROOM_REASON_FIELDS = (
    ("Invalid room data format", "rooms"),
    ("Invalid room record", "rooms"),
    ("Invalid room ID", "id"),
    ("Invalid title", "title"),
    ("Invalid city", "city"),
    ("Invalid price", "pricePerMonth"),
    ("Invalid negative price", "pricePerMonth"),
    ("Invalid room type", "roomType"),
    ("Invalid idealFor", "idealFor"),
    ("Invalid location", "location"),
    ("Invalid availability status", "availabilityStatus"),
    ("Invalid images", "images"),
    ("Invalid image URL", "images"),
    ("Invalid rating", "rating"),
    ("Rating must be", "rating"),
    ("Invalid review count", "reviewsCount"),
    ("Review count cannot", "reviewsCount"),
)
_MISSING_PREFIX = "Room data missing required field: "


def failing_field(reason: str) -> tuple[str, str]:
    """``(field, kind)`` for a ``validate_room_data`` reason. Never returns the reason text itself."""
    if reason.startswith(_MISSING_PREFIX):
        name = reason[len(_MISSING_PREFIX):].strip()
        return (name if re.fullmatch(r"\w{1,40}", name) else "unknown"), "missing"
    for prefix, name in _ROOM_REASON_FIELDS:
        if reason.startswith(prefix):
            return name, "invalid"
    return "unknown", "invalid"


def _fail(error_class: str, status: int | None = None, message: str | None = None) -> RoomsResult:
    # Status, class and the API's own `message` only. Never headers or token.
    logger.error("rooms api failure status=%s class=%s message=%s", status, error_class, message)
    return RoomsResult(ok=False, error_class=error_class, status=status, message=message)


def search_rooms(
    search: dict[str, Any],
    budget: TurnBudget,
    *,
    limit: int | None = None,
    page: int = 1,
    http_get: HttpGet | None = None,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[float, float], float] = random.uniform,
) -> RoomsResult:
    """Run one search and record its latency (live API only, not the offline fixture).

    The live search is bounded by ``config.search_timeout_s()`` for the whole call, retry included. On
    timeout it returns ``error_class="search_timeout"`` (shown as ``UNAVAILABLE_MESSAGE``). The worker
    is left to finish on its own: it only talks to the rooms API and never touches session state, so a
    late result is dropped and cannot half-save anything.
    """
    started = time.monotonic()
    if offline_path() is not None:
        return _search_rooms(search, budget, limit=limit, page=page, http_get=http_get, sleep=sleep, jitter=jitter)
    cancelled = threading.Event()
    holder: dict[str, Any] = {}
    done = threading.Event()

    def work() -> None:
        try:
            holder["result"] = _search_rooms(
                search, budget, limit=limit, page=page, http_get=http_get, sleep=sleep, jitter=jitter,
                cancelled=cancelled,
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            holder["error"] = exc
        finally:
            done.set()

    timeout = config.search_timeout_s()
    threading.Thread(target=copy_context().run, args=(work,), name="rooms-search", daemon=True).start()
    if not done.wait(timeout):
        cancelled.set()
        logger.warning("rooms search timed out after %ss request_id=%s session=%s",
                       timeout, turn_context.request_id(), turn_context.session_hash())
        result = _fail("search_timeout")
        try:
            observability.record_search_timeout()
        except Exception:  # noqa: BLE001 - metrics never break a search
            pass
    elif "error" in holder:
        raise holder["error"]
    else:
        result = holder["result"]
    if result.error_class == "search_timeout" or not _use_mcp(http_get):  # the MCP tool records its own latency
        try:
            observability.record_api_latency((time.monotonic() - started) * 1000, result.ok)
        except Exception:  # noqa: BLE001 - metrics never break a search
            pass
    return result


def _use_mcp(http_get: HttpGet | None) -> bool:
    return http_get is None and config.search_transport() == "mcp"


def _direct_request(
    search: dict[str, Any],
    token: str,
    limit: int,
    page: int,
    http_get: HttpGet | None,
    sleep: Callable[[float], None],
    jitter: Callable[[float, float], float],
    cancelled: threading.Event | None,
) -> tuple[RoomsResult | None, RawResponse | None]:
    """The direct httpx client: one GET and one retry on timeout or connection error. ``(failure, None)`` or ``(None, raw)``."""
    params = build_params(search, limit=limit, page=page)
    for attempt in (1, 2):
        try:
            return None, request_once(params, token=token, http_get=http_get)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt == 1 and not (cancelled and cancelled.is_set()):
                sleep(jitter(*config.ROOMS_RETRY_JITTER_S))
                if cancelled and cancelled.is_set():  # timed out while waiting: no second request
                    return _fail("search_timeout"), None
                continue
            return _fail("timeout" if isinstance(exc, httpx.TimeoutException) else "connection_error"), None
        except Exception as exc:  # anything else: no retry
            return _fail(type(exc).__name__), None
    raise AssertionError("unreachable")


_MCP_FAILURES = (  # (text in the tool's error string, error_class); the tool returns only a string
    ("KANGAROO_TOKEN", "no_token"),
    ("timed out", "timeout"),
    ("empty response", "bad_payload"),
    ("malformed JSON", "bad_payload"),
    ("unexpected response", "bad_payload"),
    ("urlopen error", "connection_error"),
)


def mcp_failure(error: Any, search: dict[str, Any], token: str) -> RoomsResult:
    """Map the copied tool's ``error`` string to the same error classes the direct client uses. The text is scrubbed
    before it is logged or kept; only the class and status reach the pipeline."""
    text = error if isinstance(error, str) else ""
    status = None
    match = re.search(r"HTTP Error (\d{3})", text)
    if match:
        status, error_class = int(match.group(1)), f"http_{match.group(1)}"
    else:
        error_class = next((name for needle, name in _MCP_FAILURES if needle in text), "api_error")
    return _fail(error_class, status, scrub_message(text, search, token)[:200] or None)


def _mcp_request(
    search: dict[str, Any],
    token: str,
    limit: int,
    page: int,
    cancelled: threading.Event | None,
) -> tuple[RoomsResult | None, RawResponse | None]:
    """The search through the copied in-process MCP tool, bounded by the copied 15 s wrapper. The tool owns the
    retries, so none happen here. Runs in the search worker thread; the token is bound to this thread's context only."""
    from mcp import Client  # lazy: importing the pipeline or the server must not load the mcp package (tests check this)

    from llm_approach.tools import mcp_search, mcp_timeout

    types = _room_types_param(search)
    ideal = search.get("ideal_for")
    arguments = {
        "city": search.get("city"),
        "min_price": search.get("min_price"),
        "max_price": search.get("max_price"),
        "room_types": types.split(",") if types else None,
        "ideal_for": [ideal] if ideal else None,
        "sort": sorting.api_sort(search),
        "limit": limit,
        "page": page,
    }
    arguments = {k: v for k, v in arguments.items() if v is not None}

    async def call() -> Any:
        async with Client(mcp_search.mcp) as client:
            try:
                return await mcp_timeout._call_search_rooms_with_timeout(client, arguments)
            except asyncio.TimeoutError:  # caught inside the client: leaving it would wrap the error in an ExceptionGroup
                return None

    handle = mcp_search.set_kangaroo_token(token)
    try:
        result = asyncio.run(call())
        if result is None:  # the copied 15 s wrapper timed out
            if not (cancelled and cancelled.is_set()):  # past the outer limit the caller has already counted it
                try:
                    observability.record_search_timeout()
                except Exception:  # noqa: BLE001 - metrics never break a search
                    pass
            return _fail("search_timeout"), None
        api_data = json.loads(result.content[0].text)
    except Exception as exc:  # noqa: BLE001 - a broken MCP reply is a failed search, never a crash
        return _fail(type(exc).__name__), None
    finally:
        mcp_search._kangaroo_token_var.reset(handle)
    if not isinstance(api_data, dict) or api_data.get("success") is not True:
        return mcp_failure(api_data.get("error") if isinstance(api_data, dict) else None, search, token), None
    return None, RawResponse(status_code=200, body={"data": api_data.get("rooms"), "meta": api_data.get("meta")})


def _search_rooms(
    search: dict[str, Any],
    budget: TurnBudget,
    *,
    limit: int | None = None,
    page: int = 1,
    http_get: HttpGet | None = None,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[float, float], float] = random.uniform,
    cancelled: threading.Event | None = None,
) -> RoomsResult:
    """Run one search. Counts as the turn's single tool call.

    Blocking (up to two 15 s requests plus the jitter sleep): the Phase 4
    pipeline should call it through ``asyncio.to_thread``.
    """
    if not search.get("city"):
        raise ValueError("rooms_client.search_rooms needs a validated city")
    budget.use_tool_call()
    page = page if isinstance(page, int) and not isinstance(page, bool) and page > 1 else 1
    # Area is filtered after the fetch, so fetch more and still show the normal page.
    area_mode = bool(search.get("area"))
    limit = limit or (config.ROOMS_AREA_FETCH_LIMIT if area_mode else config.ROOMS_LIMIT)
    display_limit = config.ROOMS_LIMIT if area_mode else limit

    path = offline_path()
    if path is not None:
        return _offline_search(path, search, limit, display_limit, page)

    token = get_rooms_token()  # v2: per request first, like app/server.py _get_kangaroo_token
    if not token:
        return _fail("no_token")

    if _use_mcp(http_get):
        failure, raw = _mcp_request(search, token, limit, page, cancelled)
        if failure is not None:
            return failure
    else:
        failure, raw = _direct_request(search, token, limit, page, http_get, sleep, jitter, cancelled)
        if failure is not None:
            return failure
    assert raw is not None

    if raw.status_code >= 400:
        return _fail(f"http_{raw.status_code}", raw.status_code, _error_message(raw.body, search, token))
    rooms = extract_rooms(raw.body)
    if rooms is None:
        return _fail("bad_payload", raw.status_code, _error_message(raw.body, search, token))
    valid, reason = validate_room_data(rooms) if switches.room_data_check() else (True, "")  # one bad room rejects the whole result
    if not valid:
        field_name, kind = failing_field(reason)
        observability.record_room_data_rejection()
        logger.warning(
            "rooms api data rejected field=%s reason=%s request_id=%s session=%s",
            field_name, kind, turn_context.request_id(), turn_context.session_hash(),
        )
        return _fail("invalid_room_data", raw.status_code)
    total, has_next = parse_meta(raw.body)
    return _finish(
        rooms,
        search,
        fetch_limit=limit,
        display_limit=display_limit,
        status=raw.status_code,
        total=total,
        has_next=has_next,
        page=page,
    )
