# Copied from app/experiments/agent_modes.py at commit fca91ac (the last commit that changed it; the other copies cite 7dbcba3) (copied, not imported).
# Changes: only PRICING_API_URL, PRICING_API_TIMEOUT_SECONDS and format_subscription_plans are copied (drift-checked, unchanged).
# fetch_subscription_plans is NOT copied line for line: its single httpx call is replaced by the v2 client below (2 attempts
# with backoff on timeout / 5xx, 15 s whole-call limit, response validation, no exception text or URL in the logs). The same
# endpoint, the same `city` query parameter, no auth header and the same two response shapes (bare list or {"data": [...]}).
# Drift is checked by llm_approach/tests/test_copy_drift.py.
# ==== end of v2 provenance header ====
"""The Homilivo subscription pricing API: fetch, validate and format the plans of a city.

Read-only GET, no token. Plans differ by city, so nothing here caches across cities or sessions. The log lines carry a fixed
error class, a status and (for bad data) a field name; never the city, the URL with its query, an exception message or a body.
"""

from __future__ import annotations  # v2: header for the copied file

import json  # v2: offline fixture
import logging  # v2: the original imports it at the top of agent_modes.py
import math  # v2: price check
import os  # v2: the original imports it at the top of agent_modes.py
import re  # v2: text cleaning
import threading  # v2: whole-call limit
import time  # v2: backoff
from dataclasses import dataclass, field  # v2: result type
from pathlib import Path  # v2: offline fixture
from typing import Any, Callable  # v2: the original imports Any at the top of agent_modes.py

import httpx  # v2: the original imports it inside fetch_subscription_plans

from llm_approach import config  # v2: offline switch name

logger = logging.getLogger(__name__)  # v2: module logger, as in agent_modes.py

# v2: httpx logs "HTTP Request: GET <url>?city=..." at INFO. The query must not reach the logs.
logging.getLogger("httpx").setLevel(logging.WARNING)  # v2: added
logging.getLogger("httpcore").setLevel(logging.WARNING)  # v2: added

PRICING_API_URL = os.getenv(
    "PRICING_API_URL",
    "https://kangaroo-rooms-backend.onrender.com"
    "/api/tenant-subscriptions/pricing",
)

# Matches server.py's HOMILIVO_API_TIMEOUT_SECONDS default.
PRICING_API_TIMEOUT_SECONDS = 10.0

PRICING_TOTAL_TIMEOUT_S = 15.0  # v2: the whole call, retries included (the same bound as a rooms search)
PRICING_MAX_ATTEMPTS = 2  # v2: the host is on a cold-start platform
PRICING_BACKOFF_S = 1.0  # v2: pause before the second attempt
MAX_PLANS = 10  # v2: validation limits
MAX_FEATURES = 10  # v2: added
MAX_NAME_CHARS = 40  # v2: added
MAX_FEATURE_CHARS = 120  # v2: added
MAX_PRICE = 1_000_000  # v2: added

_DEFAULT_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "pricing_plans.json"  # v2: offline plans by city


def format_subscription_plans(
    plans: list[dict[str, Any]],
    city: str,
) -> str:
    """
    Format subscription plans as Markdown. Uses .get() everywhere -
    never assumes an API field exists.
    """

    if not plans:

        return (
            "I wasn't able to find subscription plans for "
            f"{city.title()} right now. Please visit "
            "[homilivo.com](https://homilivo.com) to view plans."
        )

    lines = [
        "To connect with the owner, you'll need a Homilivo "
        f"subscription for **{city.title()}**:",
        "",
    ]

    for plan in plans:

        name = str(plan.get("plan", "Plan")).upper()

        price = plan.get("price", 0)

        try:

            price_text = f"₹{float(price):,.0f}"

        except (TypeError, ValueError):

            price_text = str(price)

        lines.append(f"**{name}** — {price_text}")

        for feature in plan.get("features", []) or []:

            lines.append(f"• {feature}")

        lines.append("")

    lines.append(
        "Visit [homilivo.com](https://homilivo.com) to subscribe."
    )

    return "\n".join(lines).strip()


# ---- v2: the city -----------------------------------------------------------

_CITY_RE = re.compile(r"[a-z][a-z .'\-]{1,48}")  # v2: letters, space, dot, apostrophe, hyphen; no digits or URL characters


def clean_city(value: Any) -> str | None:  # v2: new
    """A city from validated room or search data, lowercased like app/'s extract_verified_city; None when it is not a plain
    place name. The city is never taken from the user's text, and an odd-looking value never reaches a request."""
    if not isinstance(value, str):
        return None
    city = re.sub(r"\s+", " ", value).strip().lower()
    return city if _CITY_RE.fullmatch(city) else None


# ---- v2: validating the response ---------------------------------------------

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)  # v2: added
_TAG_RE = re.compile(r"<[^>]*>")  # v2: added
_STRIP_RE = re.compile(r"[\x00-\x1f\x7f<>\[\]()*_`#|~\\{}]")  # v2: markdown and HTML characters, control characters


def clean_text(value: Any, limit: int) -> str | None:  # v2: new
    """Plain text from an API string: tags, links, markdown characters and control characters removed, whitespace
    collapsed, cut at ``limit``. None when nothing is left or the value is not a string."""
    if not isinstance(value, str):
        return None
    text = _STRIP_RE.sub(" ", _URL_RE.sub(" ", _TAG_RE.sub(" ", value)))
    text = re.sub(r"\s+", " ", text).strip()[:limit].strip()
    return text or None


class BadPlanData(ValueError):  # v2: new
    """The response failed validation. ``field`` is a fixed field name, never a value."""

    def __init__(self, field_name: str) -> None:
        super().__init__(field_name)
        self.field = field_name


def validate_plans(body: Any) -> list[dict[str, Any]]:  # v2: new
    """The plans of a response: a bare list or ``{"data": [...]}``. One bad field rejects the whole response
    (``BadPlanData``). Only ``plan``, ``price`` and ``features`` are kept, as clean text."""
    plans = body.get("data") if isinstance(body, dict) else body
    if not isinstance(plans, list):
        raise BadPlanData("data")
    if len(plans) > MAX_PLANS:
        raise BadPlanData("data")
    clean: list[dict[str, Any]] = []
    for item in plans:
        if not isinstance(item, dict):
            raise BadPlanData("data")
        name = clean_text(item.get("plan"), MAX_NAME_CHARS)
        if name is None:
            raise BadPlanData("plan")
        price = item.get("price")
        if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or not 0 <= price <= MAX_PRICE:
            raise BadPlanData("price")
        raw_features = item.get("features")
        if raw_features is None:
            raw_features = []
        if not isinstance(raw_features, list) or len(raw_features) > MAX_FEATURES:
            raise BadPlanData("features")
        features = []
        for raw in raw_features:
            text = clean_text(raw, MAX_FEATURE_CHARS)
            if text is None:
                raise BadPlanData("features")
            features.append(text)
        clean.append({"plan": name, "price": price, "features": features})
    return clean


# ---- v2: fetching ---------------------------------------------------------------


@dataclass
class PricingResult:  # v2: new
    ok: bool
    plans: list[dict[str, Any]] = field(default_factory=list)
    error_class: str | None = None  # fixed words: timeout, connection_error, http_4xx, http_5xx, bad_json, bad_payload, error
    status: int | None = None


HttpGet = Callable[..., Any]  # v2: added


def _httpx_get(url: str, params: dict, headers: dict, timeout: float) -> Any:  # v2: new
    return httpx.get(url, params=params, headers=headers, timeout=timeout)


def offline_path() -> Path | None:  # v2: new
    """The fixture used instead of the API when the rooms offline switch is on (``LLM_APPROACH_OFFLINE_ROOMS``)."""
    return _DEFAULT_FIXTURE if os.getenv(config.OFFLINE_ROOMS_ENV, "").strip() else None


def _fail(error_class: str, status: int | None = None) -> PricingResult:  # v2: new
    logger.warning("pricing api failure class=%s status=%s", error_class, status)  # no city, URL, message or body
    return PricingResult(ok=False, error_class=error_class, status=status)


def _offline(city: str) -> PricingResult:  # v2: new
    try:
        data = json.loads(_DEFAULT_FIXTURE.read_text(encoding="utf-8"))
        return PricingResult(ok=True, plans=validate_plans(data.get(city, [])))
    except BadPlanData as exc:
        logger.warning("pricing data rejected field=%s", exc.field)
        return PricingResult(ok=False, error_class="bad_payload")
    except Exception as exc:  # noqa: BLE001
        return _fail(type(exc).__name__)


def _attempts(
    city: str, http_get: HttpGet | None, sleep: Callable[[float], None], cancelled: threading.Event
) -> PricingResult:  # v2: new
    """Up to ``PRICING_MAX_ATTEMPTS`` GETs. Only a timeout, a connection error or a 5xx is tried again."""
    get = http_get or _httpx_get
    result = PricingResult(ok=False, error_class="error")
    for attempt in range(1, PRICING_MAX_ATTEMPTS + 1):
        retryable = False
        try:
            response = get(
                PRICING_API_URL,
                params={"city": city},  # encoded by the client
                headers={"Accept": "application/json"},  # no Authorization: the API needs no token
                timeout=PRICING_API_TIMEOUT_SECONDS,
            )
            status = response.status_code
            if status >= 500:
                result, retryable = _fail("http_5xx", status), True
            elif status != 200:  # 4xx, and anything else that is not a plain success
                return _fail("http_4xx", status)
            else:
                try:
                    body = response.json()
                except Exception:  # noqa: BLE001 - the body is never echoed
                    return _fail("bad_json", status)
                try:
                    return PricingResult(ok=True, plans=validate_plans(body), status=status)
                except BadPlanData as exc:
                    logger.warning("pricing data rejected field=%s", exc.field)
                    return PricingResult(ok=False, error_class="bad_payload", status=status)
        except httpx.TimeoutException:
            result, retryable = _fail("timeout"), True
        except httpx.TransportError:
            result, retryable = _fail("connection_error"), True
        except Exception as exc:  # noqa: BLE001 - class name only
            return _fail(type(exc).__name__)
        if not retryable or attempt == PRICING_MAX_ATTEMPTS or cancelled.is_set():
            return result
        sleep(PRICING_BACKOFF_S)
        if cancelled.is_set():  # past the whole-call limit while waiting: no second request
            return result
    return result


def fetch_plans(
    cities: list[str],
    *,
    http_get: HttpGet | None = None,
    sleep: Callable[[float], None] = time.sleep,
    total_timeout: float = PRICING_TOTAL_TIMEOUT_S,
) -> dict[str, PricingResult]:  # v2: new
    """Plans for each city (already cleaned by ``clean_city``), fetched in parallel and bounded together by
    ``total_timeout``. Never raises; a city that did not finish in time is a ``timeout`` failure. Blocking: call it
    through ``asyncio.to_thread``. Every request carries its own city, so nothing is shared between cities."""
    results: dict[str, PricingResult] = {}
    if offline_path() is not None:
        for city in cities:
            results[city] = _offline(city)
        return results
    cancelled = threading.Event()
    done: dict[str, PricingResult] = {}
    workers = []

    def work(city: str) -> None:
        try:
            done[city] = _attempts(city, http_get, sleep, cancelled)
        except BaseException as exc:  # noqa: BLE001 - never lost
            done[city] = _fail(type(exc).__name__)

    for city in cities:
        worker = threading.Thread(target=work, args=(city,), name="pricing-fetch", daemon=True)
        worker.start()
        workers.append(worker)
    deadline = time.monotonic() + total_timeout
    for worker in workers:
        worker.join(max(0.0, deadline - time.monotonic()))
    cancelled.set()
    for city in cities:
        results[city] = done.get(city) or _fail("timeout")
    return results
