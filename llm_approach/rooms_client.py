"""Read-only GET to the Homilivo rooms API, with one safe retry.

Same URL, auth header and query parameters as ``search_rooms`` in
``app/server.py``. The token comes from ``KANGAROO_TOKEN`` only and is never
logged. Retry once, only on timeout or connection error. Never on 4xx/5xx.
Amenity and area are not API parameters: they are filtered here, on the result.
An area search fetches ``config.ROOMS_AREA_FETCH_LIMIT`` rooms and shows at most
``config.ROOMS_LIMIT`` of the matches.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from llm_approach import config
from llm_approach.actions import TurnBudget
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
    area_matches: int | None = None  # rooms that matched the area (None: no area asked)
    areas: list[str] = field(default_factory=list)  # real locations of the fetched rooms (area searches)


@dataclass
class RawResponse:
    status_code: int
    body: Any  # parsed JSON or None


HttpGet = Callable[..., Any]


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


def build_params(search: dict[str, Any], *, limit: int | None = None) -> dict[str, Any]:
    """Same names as server.py: city, roomTypes, minPrice, maxPrice, idealFor, sort, limit.
    No amenity: the API has no such parameter."""
    params: dict[str, Any] = {
        "city": search.get("city"),
        "minPrice": search.get("min_price"),
        "maxPrice": search.get("max_price"),
        "roomTypes": _room_types_param(search),
        "idealFor": search.get("ideal_for"),
        "sort": config.ROOMS_SORT,
        "limit": limit or config.ROOMS_LIMIT,
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


def _finish(
    fetched: list[dict[str, Any]],
    search: dict[str, Any],
    *,
    fetch_limit: int,
    display_limit: int,
    status: int | None = None,
) -> RoomsResult:
    """The one place where area and amenity filters and the display cut happen,
    for the live API and the offline fixture alike."""
    area = search.get("area")
    matched = filter_by_area(fetched, area)
    found = filter_by_amenity(matched, search.get("amenity"))
    shown = found[:display_limit]
    return RoomsResult(
        ok=True,
        rooms=[clean_room(r) for r in shown],
        status=status,
        api_count=len(fetched),
        fetch_limit=fetch_limit,
        truncated=len(found) > len(shown),
        area_matches=len(matched) if area else None,
        areas=distinct_locations(fetched) if area else [],
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
    path: Path, search: dict[str, Any], limit: int, display_limit: int
) -> RoomsResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    rooms = data["rooms"] if isinstance(data, dict) else data
    # Like the live API: API filters and limit first; area and amenity afterwards on the page.
    page = _filter_fixture([r for r in rooms if isinstance(r, dict)], search)[:limit]
    return _finish(page, search, fetch_limit=limit, display_limit=display_limit)


# ---- HTTP -----------------------------------------------------------------


def _error_message(body: Any) -> str | None:
    if isinstance(body, dict) and isinstance(body.get("message"), str):
        return body["message"][:200]
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


def _fail(error_class: str, status: int | None = None, message: str | None = None) -> RoomsResult:
    # Status, class and the API's own `message` only. Never headers or token.
    logger.error("rooms api failure status=%s class=%s message=%s", status, error_class, message)
    return RoomsResult(ok=False, error_class=error_class, status=status, message=message)


def search_rooms(
    search: dict[str, Any],
    budget: TurnBudget,
    *,
    limit: int | None = None,
    http_get: HttpGet | None = None,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[float, float], float] = random.uniform,
) -> RoomsResult:
    """Run one search. Counts as the turn's single tool call.

    Blocking (up to two 15 s requests plus the jitter sleep): the Phase 4
    pipeline should call it through ``asyncio.to_thread``.
    """
    if not search.get("city"):
        raise ValueError("rooms_client.search_rooms needs a validated city")
    budget.use_tool_call()
    # Area is filtered after the fetch, so fetch more and still show the normal page.
    area_mode = bool(search.get("area"))
    limit = limit or (config.ROOMS_AREA_FETCH_LIMIT if area_mode else config.ROOMS_LIMIT)
    display_limit = config.ROOMS_LIMIT if area_mode else limit

    path = offline_path()
    if path is not None:
        return _offline_search(path, search, limit, display_limit)

    token = os.getenv("KANGAROO_TOKEN")
    if not token:
        return _fail("no_token")

    params = build_params(search, limit=limit)
    raw: RawResponse | None = None
    for attempt in (1, 2):
        try:
            raw = request_once(params, token=token, http_get=http_get)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt == 1:
                sleep(jitter(*config.ROOMS_RETRY_JITTER_S))
                continue
            return _fail("timeout" if isinstance(exc, httpx.TimeoutException) else "connection_error")
        except Exception as exc:  # anything else: no retry
            return _fail(type(exc).__name__)
        break
    assert raw is not None

    if raw.status_code >= 400:
        return _fail(f"http_{raw.status_code}", raw.status_code, _error_message(raw.body))
    rooms = extract_rooms(raw.body)
    if rooms is None:
        return _fail("bad_payload", raw.status_code, _error_message(raw.body))
    return _finish(
        rooms, search, fetch_limit=limit, display_limit=display_limit, status=raw.status_code
    )
