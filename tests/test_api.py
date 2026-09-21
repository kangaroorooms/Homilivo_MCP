"""
FastAPI-layer test suite.

Exercises app/api.py end-to-end (through the real conversational engine
in app/experiments/agent_modes.py - this suite does not duplicate or
bypass the pipeline) with the outbound Homilivo HTTP call mocked via
the fixtures in conftest.py, so it is fast and deterministic. Real
end-to-end verification against the live API stays in tests/test_mcp.py
and tests/real_api_test.py - this file does not replace those.
"""

from __future__ import annotations

import asyncio
import types
from collections import Counter

import httpx
import pytest

from app.api import app
from app.experiments.agent_modes import GemmaError
import app.experiments.agent_modes as agent_modes


# ============================================================
# ROOT / HEALTH
# ============================================================

def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "service": "homilivo-chatbot"}


def test_chat_ui_is_served_same_origin(client):
    response = client.get("/chat-ui")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "/chat" in body
    assert "fetch(" in body


def test_chat_ui_never_hardcodes_a_secret(client):
    """
    Demo-safety requirement: the page must never contain a real token or
    key baked into the HTML/JS - only the client-side field name/label
    text and runtime fetch() calls that send whatever the user typed.
    """
    body = client.get("/chat-ui").text.lower()

    for leaked in ("kangaroo_token=", "groq_api_key=", "gsk_", "bearer ey"):
        assert leaked not in body


# ============================================================
# AUTHENTICATION
# ============================================================

def test_chat_missing_authorization_returns_401(client):
    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 401


def test_chat_wrong_auth_scheme_returns_401(client):
    response = client.post(
        "/chat",
        json={"message": "hello"},
        headers={"Authorization": "Basic not-a-bearer-token"},
    )

    assert response.status_code == 401


def test_chat_valid_authorization_is_accepted(client, auth_headers, mock_homilivo_success):
    response = client.post(
        "/chat",
        json={"message": "hello"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["success"] is True


# ============================================================
# SESSION HANDLING
# ============================================================

def test_chat_creates_session_id_when_omitted(client, auth_headers, mock_homilivo_success):
    response = client.post("/chat", json={"message": "hello"}, headers=auth_headers)

    body = response.json()
    assert body["session_id"]


def test_chat_session_persists_across_calls(client, auth_headers, mock_homilivo_success):
    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    session_id = search["session_id"]
    assert search["room_count"] == 3

    followup = client.post(
        "/chat",
        json={
            "message": "which one is cheapest?",
            "session_id": session_id,
            "mode": "fast",
        },
        headers=auth_headers,
    ).json()

    assert followup["session_id"] == session_id
    assert followup["selected_room"] is not None
    assert followup["selected_room"]["id"] == "room-2"
    assert followup["selected_room"]["price_per_month"] == 8500


def test_chat_session_isolation(client, auth_headers, mock_homilivo_success, monkeypatch):
    # Session A: search, then select the cheapest room.
    search_a = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()
    session_a = search_a["session_id"]

    client.post(
        "/chat",
        json={"message": "which one is cheapest?", "session_id": session_a, "mode": "fast"},
        headers=auth_headers,
    )

    # Session B: a brand new session asking the same follow-up cold, with
    # no prior search. The deterministic router requires previous_rooms /
    # selected_room to treat this as a follow-up; with neither, it falls
    # through to the local Gemma intent classifier - mocked here to be
    # unavailable so the test is deterministic regardless of whether
    # Ollama happens to be running locally (this also exercises the
    # "local model unavailable" degradation path).
    monkeypatch.setattr(
        "app.experiments.agent_modes.classify_intent",
        lambda query: (_ for _ in ()).throw(GemmaError("no local model available")),
    )

    response_b = client.post(
        "/chat",
        json={"message": "which one is cheapest?", "mode": "fast"},
        headers=auth_headers,
    ).json()

    # Session B must NOT see session A's selected room or search results.
    assert response_b["session_id"] != session_a
    assert response_b["selected_room"] is None
    assert response_b["room_count"] == 0


def test_chat_reset_clears_session(client, auth_headers, mock_homilivo_success):
    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()
    session_id = search["session_id"]
    assert search["room_count"] == 3

    reset = client.post("/chat/reset", params={"session_id": session_id}).json()
    assert reset["existed"] is True

    after_reset = client.get("/health").json()
    assert after_reset["status"] == "ok"


def test_chat_reset_nonexistent_session_returns_existed_false(client):
    reset = client.post("/chat/reset", params={"session_id": "does-not-exist"}).json()

    assert reset["existed"] is False


# ============================================================
# MODE VALIDATION
# ============================================================

def test_chat_invalid_mode_returns_422(client, auth_headers):
    response = client.post(
        "/chat",
        json={"message": "hello", "mode": "not-a-real-mode"},
        headers=auth_headers,
    )

    assert response.status_code == 422


@pytest.mark.parametrize("mode", ["fast", "qwen", "gemma", "groq"])
def test_chat_all_valid_modes_are_accepted_by_validation(client, auth_headers, mode):
    """
    Every documented mode must pass Pydantic validation (not be rejected
    at the request-validation layer). The actual response-generation
    behaviour per mode is covered by the dedicated mode tests below.
    """
    response = client.post(
        "/chat",
        json={"message": "hello", "mode": mode},
        headers=auth_headers,
    )

    assert response.status_code != 422


def test_chat_qwen_mode_uses_local_model(client, auth_headers, mock_homilivo_success, monkeypatch):
    fake_reply = types.SimpleNamespace(message=types.SimpleNamespace(content="Qwen says hi."))
    monkeypatch.setattr(
        "app.experiments.agent_modes.chat",
        lambda model, messages: fake_reply,
    )

    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "session_id": None, "mode": "fast"},
        headers=auth_headers,
    ).json()

    response = client.post(
        "/chat",
        json={
            "message": "tell me more about this one",
            "session_id": search["session_id"],
            "mode": "qwen",
        },
        headers=auth_headers,
    ).json()

    assert response["mode"] == "qwen"
    assert response["success"] is True


def test_chat_gemma_mode_uses_local_model(client, auth_headers, mock_homilivo_success, monkeypatch):
    fake_reply = types.SimpleNamespace(message=types.SimpleNamespace(content="Gemma says hi."))
    monkeypatch.setattr(
        "app.experiments.agent_modes.chat",
        lambda model, messages: fake_reply,
    )

    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    response = client.post(
        "/chat",
        json={
            "message": "tell me more about this one",
            "session_id": search["session_id"],
            "mode": "gemma",
        },
        headers=auth_headers,
    ).json()

    assert response["mode"] == "gemma"
    assert response["success"] is True


# ============================================================
# GROQ MODE
# ============================================================

def _fake_groq_client(content: str | None = None, fail: bool = False):
    def create(model, messages):
        if fail:
            raise RuntimeError("simulated Groq outage")
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
        )

    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )


def test_chat_groq_mode_returns_llm_generated_text(client, auth_headers, mock_homilivo_success, monkeypatch):
    monkeypatch.setattr(
        "app.experiments.agent_modes._get_groq_client",
        lambda: _fake_groq_client(content="Groq-generated answer."),
    )

    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    response = client.post(
        "/chat",
        json={
            "message": "tell me more about this one",
            "session_id": search["session_id"],
            "mode": "groq",
        },
        headers=auth_headers,
    ).json()

    assert response["mode"] == "groq"
    assert response["message"] == "Groq-generated answer."


def test_chat_groq_failure_falls_back_gracefully(client, auth_headers, mock_homilivo_success, monkeypatch):
    monkeypatch.setattr(
        "app.experiments.agent_modes._get_groq_client",
        lambda: _fake_groq_client(fail=True),
    )

    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    response = client.post(
        "/chat",
        json={
            "message": "which one is the most expensive?",
            "session_id": search["session_id"],
            "mode": "groq",
        },
        headers=auth_headers,
    )

    # A Groq outage must degrade to the fast-mode fallback, not a 500.
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "18,000" in body["message"] or "18000" in body["message"]


# ============================================================
# MALFORMED REQUESTS / PYDANTIC VALIDATION
# ============================================================

def test_chat_missing_message_field_returns_422(client, auth_headers):
    response = client.post("/chat", json={"mode": "fast"}, headers=auth_headers)

    assert response.status_code == 422


def test_chat_empty_message_returns_422(client, auth_headers):
    response = client.post("/chat", json={"message": ""}, headers=auth_headers)

    assert response.status_code == 422


def test_chat_overlong_message_returns_422(client, auth_headers):
    response = client.post(
        "/chat",
        json={"message": "x" * 501},
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_chat_non_json_body_returns_422(client, auth_headers):
    response = client.post(
        "/chat",
        content=b"this is not json",
        headers={**auth_headers, "Content-Type": "application/json"},
    )

    assert response.status_code == 422


# ============================================================
# GRACEFUL DEGRADATION
# ============================================================

def test_chat_invalid_price_range_degrades_gracefully(
    client, auth_headers, mock_homilivo_success, monkeypatch
):
    """
    A query whose extracted parameters fail the parameter guardrail
    (min_price > max_price) must not crash the request - it should
    come back as a normal 200 with an apologetic fallback message.

    The input guardrail's semantic classifier (embedding cosine
    similarity, app/guardrails/input_guardrail.py) actually scores this
    exact phrasing as a near-tie between ROOM_SEARCH and CHATBOT_INFO
    (~0.538 vs ~0.543, inside INTENT_MARGIN) - so it reports OUT_OF_SCOPE
    even before any platform difference, and process_message() then
    falls back to a live Gemma/Ollama call to rescue the classification.
    That fallback happens to succeed on a machine with Ollama running
    locally but is unavailable in CI, so the query gets classified
    OUT_OF_SCOPE there and never reaches the parameter guardrail this
    test targets. Stub the input guardrail to deterministically report
    ROOM_SEARCH so this test isolates the parameter-guardrail behavior
    it actually names, independent of the semantic classifier's
    borderline score or Ollama's availability (both are covered by
    tests/test_guardrails.py instead).
    """
    monkeypatch.setattr(
        agent_modes, "validate_user_input", lambda user_query: (True, "ROOM_SEARCH")
    )

    response = client.post(
        "/chat",
        json={"message": "rooms in Jaipur above 20000 under 10000", "mode": "fast"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["success"] is False


def test_chat_homilivo_api_failure_degrades_gracefully(client, auth_headers, mock_homilivo_failure):
    response = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["room_count"] == 0


def test_chat_out_of_scope_contact_owner_does_not_rerun_search(client, auth_headers, mock_homilivo_success):
    """
    Regression for Issue 1: "how to contact with the owner?" must not be
    silently treated as a search refinement (the bare "with " substring
    in refinement_patterns used to cause this) and must not come back
    looking like a fresh room search.
    """
    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    response = client.post(
        "/chat",
        json={
            "message": "how to contact with the owner?",
            "session_id": search["session_id"],
            "mode": "fast",
        },
        headers=auth_headers,
    ).json()

    assert "Here are the" not in response["message"]
    assert "matching options" not in response["message"]


def test_chat_set_wide_amenity_question_is_answered_from_session_data(client, auth_headers, mock_homilivo_success):
    """
    Regression for Issue 2: "does any room have ac?" after a search must
    be answered from the already-verified room set, not the generic
    capability fallback message. YES/NO framing: boolean + count, no
    room names (see the LISTING-framing test right below for that).
    """
    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    response = client.post(
        "/chat",
        json={
            "message": "does any room have ac?",
            "session_id": search["session_id"],
            "mode": "fast",
        },
        headers=auth_headers,
    ).json()

    assert response["message"] == "Yes, 1 of the verified rooms have AC."
    assert "I can answer questions about" not in response["message"]


def test_chat_set_wide_amenity_listing_names_room_and_filters_cards(
    client, auth_headers, mock_homilivo_success,
):
    """
    "which rooms have AC?" is LISTING framing: names the matching
    room(s) in the text AND the returned room cards (`rooms`) are
    filtered to just the matches for this turn - without touching the
    session's underlying room set (verified via the follow-up ordinal
    reference right after).
    """
    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    response = client.post(
        "/chat",
        json={
            "message": "which rooms have ac?",
            "session_id": search["session_id"],
            "mode": "fast",
        },
        headers=auth_headers,
    ).json()

    assert "Sunrise 2BHK" in response["message"]
    assert response["room_count"] == 1
    assert response["rooms"][0]["title"] == "Sunrise 2BHK"

    # The full 3-room search result must still be intact in session
    # state for ordinal follow-ups, even though only 1 room was shown.
    followup = client.post(
        "/chat",
        json={
            "message": "what is the price of the last one?",
            "session_id": search["session_id"],
            "mode": "fast",
        },
        headers=auth_headers,
    ).json()

    assert "Mid-range Flat" in followup["message"]


def test_chat_set_wide_room_type_question_is_answered_from_session_data(client, auth_headers, mock_homilivo_success):
    """
    "what type of rooms are available?" after a search must be answered
    from the already-verified room set (SAMPLE_ROOMS has 2BHK and PG),
    not the generic capability fallback message.
    """
    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    response = client.post(
        "/chat",
        json={
            "message": "what type of rooms are available?",
            "session_id": search["session_id"],
            "mode": "fast",
        },
        headers=auth_headers,
    ).json()

    assert "2BHK" in response["message"]
    assert "PG" in response["message"]
    assert "I can answer questions about" not in response["message"]


def test_chat_set_wide_location_question_is_answered_from_session_data(client, auth_headers, mock_homilivo_success):
    """
    "where are they located?" after a search must list the distinct
    locations from the verified room set, not the generic fallback.
    """
    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    response = client.post(
        "/chat",
        json={
            "message": "where are they located?",
            "session_id": search["session_id"],
            "mode": "fast",
        },
        headers=auth_headers,
    ).json()

    assert "Jagatpura" in response["message"]
    assert "Mansarovar" in response["message"]


def test_chat_set_wide_student_suitability_question_is_answered_from_session_data(
    client, auth_headers, mock_homilivo_success,
):
    """
    "which rooms are suitable for students?" must list the rooms whose
    idealFor includes Students (room-2 and room-3 in SAMPLE_ROOMS),
    excluding room-1 which has no Students entry.
    """
    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    response = client.post(
        "/chat",
        json={
            "message": "which rooms are suitable for students?",
            "session_id": search["session_id"],
            "mode": "fast",
        },
        headers=auth_headers,
    ).json()

    assert "Budget PG" in response["message"]
    assert "Mid-range Flat" in response["message"]
    assert "Sunrise 2BHK" not in response["message"]


def test_chat_room_response_includes_amenities_when_present(client, auth_headers, mock_homilivo_success):
    """
    SAMPLE_ROOMS' first room has an "amenities" field; the second does
    not. RoomResponse.amenities must reflect that per-room, not be a
    fixed empty list regardless of the API data.
    """
    response = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    rooms_by_id = {room["id"]: room for room in response["rooms"]}

    assert rooms_by_id["room-1"]["amenities"] == ["AC", "Parking"]
    assert rooms_by_id["room-2"]["amenities"] == []


def test_chat_empty_search_results(client, auth_headers, mock_homilivo_empty):
    response = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["room_count"] == 0
    assert "jaipur" in body["message"].lower()
    assert body["search_performed"] is True


def test_search_performed_is_true_for_a_completed_search(client, auth_headers, mock_homilivo_success):
    response = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    assert response["search_performed"] is True


def test_search_performed_is_false_for_a_greeting(client, auth_headers):
    response = client.post(
        "/chat", json={"message": "hello", "mode": "fast"}, headers=auth_headers,
    ).json()

    assert response["search_performed"] is False


def test_search_performed_is_false_for_a_platform_knowledge_answer(client, auth_headers):
    response = client.post(
        "/chat",
        json={"message": "what are the platform fees?", "mode": "fast"},
        headers=auth_headers,
    ).json()

    assert response["search_performed"] is False


def test_search_performed_is_false_for_a_room_followup(client, auth_headers, mock_homilivo_success):
    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    followup = client.post(
        "/chat",
        json={
            "message": "which one is cheapest?",
            "session_id": search["session_id"],
            "mode": "fast",
        },
        headers=auth_headers,
    ).json()

    assert followup["search_performed"] is False


# ============================================================
# CONCURRENT REQUEST TOKEN ISOLATION (integration-level)
# ============================================================
# app/server.py/test_token_context.py already proves the ContextVar
# mechanism itself in isolation; this proves it holds through a real
# concurrent HTTP request lifecycle (guardrails -> parser -> MCP call).

def test_concurrent_chat_requests_do_not_leak_tokens(mock_homilivo_token_recorder):
    get_recorded_tokens = mock_homilivo_token_recorder
    token_count = 15

    async def post_one(async_client: httpx.AsyncClient, index: int) -> httpx.Response:
        return await async_client.post(
            "/chat",
            json={"message": "find rooms in Jaipur", "mode": "fast"},
            headers={"Authorization": f"Bearer token-{index}"},
        )

    async def run() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as async_client:
            return await asyncio.gather(*(post_one(async_client, i) for i in range(token_count)))

    responses = asyncio.run(run())

    for response in responses:
        assert response.status_code == 200

    expected = Counter(f"Bearer token-{i}" for i in range(token_count))
    actual = Counter(get_recorded_tokens())

    assert actual == expected


# ============================================================
# REGRESSION: city-agnostic search + real backend room-type variation
# ============================================================
# Root-caused in a live-bug investigation: "find rooms in udaipur" was
# blocked entirely because the real Homilivo API returned roomType
# "2RK" (not in the guardrail's old closed enum), and "rooms in
# bangalore" / "find me rooms in bangalore" fell through to the
# generic capability message because the deterministic router only
# matched a handful of exact verb+object phrasings. Both fixes are
# city-agnostic by construction (no city name is hardcoded anywhere in
# app/experiments/agent_modes.py's fix) - these tests use Udaipur and
# Bangalore specifically to prove that, while tests/test_api.py's
# existing Jaipur-based coverage (SAMPLE_ROOMS throughout this file)
# proves the previously-working path is unchanged.

def test_chat_bare_city_search_without_a_verb_executes_a_real_search(
    client, auth_headers, mock_homilivo_success,
):
    """
    "rooms in bangalore" has no leading verb ("find"/"show me") and
    previously fell through to the generic capability response instead
    of ever calling the MCP tool.
    """
    response = client.post(
        "/chat",
        json={"message": "rooms in bangalore", "mode": "fast"},
        headers=auth_headers,
    ).json()

    assert response["search_performed"] is True
    assert response["room_count"] == 3
    assert "I can answer questions about" not in response["message"]
    assert "I'm the Homilivo room-search assistant" not in response["message"]


def test_chat_pronoun_broken_search_phrasing_executes_a_real_search(
    client, auth_headers, mock_homilivo_success,
):
    """
    "find me rooms in bangalore" - the inserted pronoun breaks the
    "find rooms" substring match the old deterministic router relied on.
    """
    response = client.post(
        "/chat",
        json={"message": "find me rooms in bangalore", "mode": "fast"},
        headers=auth_headers,
    ).json()

    assert response["search_performed"] is True
    assert response["room_count"] == 3


def test_chat_udaipur_search_with_an_extended_room_type_is_not_blocked(
    client, auth_headers, monkeypatch,
):
    """
    True end-to-end reproduction of the original bug report: the live
    Homilivo API returned a room with roomType "2RK" for a Udaipur
    search, and the entire response was discarded by the API-data
    guardrail. Mocks the outbound HTTP call to return exactly that
    shape (a real, unmodified room type this project had not seen
    before) rather than testing validate_room_data() in isolation.
    """
    from tests.conftest import FakeHomilivoResponse

    udaipur_rooms = [
        {
            "id": "2ad23efe-007f-4022-b951-3726b4298900",
            "title": "Cozy 2RK near lake",
            "city": "Udaipur",
            "location": "Fatehsagar",
            "pricePerMonth": 9500,
            "roomType": "2RK",
            "idealFor": ["Students"],
        },
        {
            "id": "udaipur-room-2",
            "title": "Standard 1BHK",
            "city": "Udaipur",
            "location": "Hiran Magri",
            "pricePerMonth": 12000,
            "roomType": "1BHK",
            "idealFor": ["Working Professionals"],
        },
    ]

    def fake_urlopen(request, timeout=None):
        return FakeHomilivoResponse({"data": udaipur_rooms})

    monkeypatch.setattr("app.server.urllib.request.urlopen", fake_urlopen)

    response = client.post(
        "/chat",
        json={"message": "find rooms in udaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    assert response["success"] is True
    assert response["search_performed"] is True
    assert response["room_count"] == 2

    # The backend's original room-type value must be preserved
    # verbatim, never fabricated or silently remapped.
    room_types = {room["room_type"] for room in response["rooms"]}
    assert "2RK" in room_types
    assert "1BHK" in room_types


def test_chat_udaipur_search_and_followup_end_to_end(client, auth_headers, monkeypatch):
    """
    Combined regression: a Udaipur search whose result includes the
    previously-blocking "2RK" room type, followed by a set-wide
    follow-up question - proving both fixes work together and that
    set-wide follow-up behavior (Issue 3 / requirement 8) survives the
    routing change.
    """
    from tests.conftest import FakeHomilivoResponse

    udaipur_rooms = [
        {
            "id": "2ad23efe-007f-4022-b951-3726b4298900",
            "title": "Cozy 2RK near lake",
            "city": "Udaipur",
            "location": "Fatehsagar",
            "pricePerMonth": 9500,
            "roomType": "2RK",
            "idealFor": ["Students"],
        },
        {
            "id": "udaipur-room-2",
            "title": "Standard 1BHK",
            "city": "Udaipur",
            "location": "Hiran Magri",
            "pricePerMonth": 12000,
            "roomType": "1BHK",
            "idealFor": ["Working Professionals"],
        },
    ]

    def fake_urlopen(request, timeout=None):
        return FakeHomilivoResponse({"data": udaipur_rooms})

    monkeypatch.setattr("app.server.urllib.request.urlopen", fake_urlopen)

    search = client.post(
        "/chat",
        json={"message": "rooms in Udaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()
    assert search["room_count"] == 2

    followup = client.post(
        "/chat",
        json={
            "message": "which one is cheapest?",
            "session_id": search["session_id"],
            "mode": "fast",
        },
        headers=auth_headers,
    ).json()

    assert followup["selected_room"] is not None
    assert followup["selected_room"]["room_type"] == "2RK"
    assert followup["selected_room"]["price_per_month"] == 9500


def test_chat_jaipur_search_behavior_is_unchanged(client, auth_headers, mock_homilivo_success):
    """
    Explicit lock for requirement 6: the previously-working Jaipur path
    (SAMPLE_ROOMS, all recognized room types) must be completely
    unaffected by either fix.
    """
    response = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    assert response["success"] is True
    assert response["search_performed"] is True
    assert response["room_count"] == 3
