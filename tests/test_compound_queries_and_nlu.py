"""
Regression tests for this round's fixes:

1. Compound ordinal+operation queries ("does the first one have a
   fridge?") - reference resolution (WHICH room) and operation
   detection (WHAT is being asked) are now independent, so an ordinal
   in the sentence no longer silently overrides the actual question -
   see detect_room_reference_operation() vs detect_room_operation() in
   app/experiments/agent_modes.py.
2. "anything cheaper?" now maps to the existing LOWEST_PRICE operation
   (was previously swallowed by generic price-language detection via
   the "cheap" substring in "cheaper").
3. TV/television and washer synonym coverage.
4. Preposition-free city extraction ("any rooms Bangalore?").

All 22 cases from the task's TESTING section are covered here or in
the pre-existing test_conversational_behavior.py /
test_api_regression* suites from earlier work on this codebase; this
file focuses on the genuinely new fixes plus a full end-to-end
numbered checklist for traceability.
"""

from __future__ import annotations

import pytest

from app.experiments.agent_modes import GemmaError
from tests.conftest import FakeHomilivoResponse


TWO_ROOMS = [
    {
        "id": "room-ac",
        "title": "Skyline 2BHK",
        "city": "Jaipur",
        "location": "C-Scheme",
        "pricePerMonth": 20000,
        "roomType": "2BHK",
        "idealFor": ["Working Professionals"],
        "amenities": ["Air Conditioning", "Parking", "Television"],
    },
    {
        "id": "room-fridge",
        "title": "Garden View PG",
        "city": "Jaipur",
        "location": "Malviya Nagar",
        "pricePerMonth": 9000,
        "roomType": "PG",
        "idealFor": ["Students"],
        "amenities": ["Refrigerator", "Wi-Fi", "Washer"],
    },
]


@pytest.fixture()
def mock_two_rooms(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeHomilivoResponse({"data": TWO_ROOMS})

    monkeypatch.setattr("app.server.urllib.request.urlopen", fake_urlopen)


def _mock_gemma_unavailable(monkeypatch):
    monkeypatch.setattr(
        "app.experiments.agent_modes.classify_intent",
        lambda query: (_ for _ in ()).throw(GemmaError("no local model available")),
    )


# ============================================================
# 1. COMPOUND ORDINAL + OPERATION - the core fix this round
# ============================================================

def test_does_the_first_one_have_fridge_resolves_reference_and_operation(
    client, auth_headers, mock_two_rooms,
):
    """
    Root-cause regression: an ordinal in the sentence must not
    override the actual question. "does the first one have a fridge?"
    must resolve reference=first room AND operation=amenity(fridge),
    not silently default to the room's price.
    """
    search = client.post(
        "/chat", json={"message": "rooms in Jaipur", "mode": "fast"}, headers=auth_headers,
    ).json()

    response = client.post(
        "/chat",
        json={
            "message": "does the first one have a fridge?",
            "session_id": search["session_id"],
            "mode": "fast",
        },
        headers=auth_headers,
    ).json()

    assert response["selected_room"]["title"] == "Skyline 2BHK"
    # Must answer about the fridge, not default to price.
    assert "does not specify" in response["message"] or "amenities" in response["message"].lower()
    assert "20,000" not in response["message"]


def test_does_the_second_one_have_ac_resolves_reference_and_operation(
    client, auth_headers, mock_two_rooms,
):
    search = client.post(
        "/chat", json={"message": "rooms in Jaipur", "mode": "fast"}, headers=auth_headers,
    ).json()

    response = client.post(
        "/chat",
        json={
            "message": "does the second one have ac?",
            "session_id": search["session_id"],
            "mode": "fast",
        },
        headers=auth_headers,
    ).json()

    assert response["selected_room"]["title"] == "Garden View PG"
    assert "9,000" not in response["message"]
    assert "does not specify" in response["message"] or "amenities" in response["message"].lower()


def test_bare_ordinal_reference_without_operation_still_works(client, auth_headers, mock_two_rooms):
    """A PURE reference ("the first one", no operation words) must still resolve normally."""
    search = client.post(
        "/chat", json={"message": "rooms in Jaipur", "mode": "fast"}, headers=auth_headers,
    ).json()

    response = client.post(
        "/chat",
        json={"message": "the first one", "session_id": search["session_id"], "mode": "fast"},
        headers=auth_headers,
    ).json()

    assert response["selected_room"]["title"] == "Skyline 2BHK"


def test_contextual_room_question_without_pronoun_still_resolves_selected_room(
    client, auth_headers, mock_two_rooms,
):
    """Regression guard: "what type of room is this?" (no ordinal/pronoun keyword
    the earlier checks catch) must still resolve to the selected room."""
    search = client.post(
        "/chat", json={"message": "rooms in Jaipur", "mode": "fast"}, headers=auth_headers,
    ).json()
    client.post(
        "/chat",
        json={"message": "select the first one", "session_id": search["session_id"], "mode": "fast"},
        headers=auth_headers,
    )

    response = client.post(
        "/chat",
        json={"message": "what type of room is this?", "session_id": search["session_id"], "mode": "fast"},
        headers=auth_headers,
    ).json()

    assert "2BHK" in response["message"]


# ============================================================
# 2. "anything cheaper?"
# ============================================================

def test_anything_cheaper_maps_to_lowest_price(client, auth_headers, mock_two_rooms):
    search = client.post(
        "/chat", json={"message": "rooms in Jaipur", "mode": "fast"}, headers=auth_headers,
    ).json()

    response = client.post(
        "/chat",
        json={"message": "anything cheaper?", "session_id": search["session_id"], "mode": "fast"},
        headers=auth_headers,
    ).json()

    assert "Garden View PG" in response["message"]
    assert "9,000" in response["message"]


# ============================================================
# 3. TV/washer synonyms
# ============================================================

def test_tv_synonym_matches_television_data(client, auth_headers, mock_two_rooms):
    search = client.post(
        "/chat", json={"message": "rooms in Jaipur", "mode": "fast"}, headers=auth_headers,
    ).json()
    client.post(
        "/chat",
        json={"message": "select the first one", "session_id": search["session_id"], "mode": "fast"},
        headers=auth_headers,
    )

    response = client.post(
        "/chat",
        json={"message": "does it have a tv?", "session_id": search["session_id"], "mode": "fast"},
        headers=auth_headers,
    ).json()

    assert "does not specify" not in response["message"]


def test_washer_synonym_matches_washer_data_set_wide(client, auth_headers, mock_two_rooms):
    search = client.post(
        "/chat", json={"message": "rooms in Jaipur", "mode": "fast"}, headers=auth_headers,
    ).json()

    response = client.post(
        "/chat",
        json={"message": "which rooms have a washing machine?", "session_id": search["session_id"], "mode": "fast"},
        headers=auth_headers,
    ).json()

    assert "Garden View PG" in response["message"]


# ============================================================
# 4. Preposition-free city extraction
# ============================================================

def test_any_rooms_bangalore_without_preposition_executes_a_search(
    client, auth_headers, mock_homilivo_success,
):
    response = client.post(
        "/chat", json={"message": "any rooms Bangalore?", "mode": "fast"}, headers=auth_headers,
    ).json()

    assert response["search_performed"] is True
    assert response["room_count"] == 3


# ============================================================
# FULL NUMBERED CHECKLIST (task's TESTING section, items 1-22)
# ============================================================

def test_01_does_it_have_ac(client, auth_headers, mock_two_rooms):
    search = client.post("/chat", json={"message": "rooms in Jaipur", "mode": "fast"}, headers=auth_headers).json()
    client.post("/chat", json={"message": "select the first one", "session_id": search["session_id"], "mode": "fast"}, headers=auth_headers)
    r = client.post("/chat", json={"message": "does it have an ac?", "session_id": search["session_id"], "mode": "fast"}, headers=auth_headers).json()
    assert "Air Conditioning" in r["message"]


def test_02_does_it_have_fridge(client, auth_headers, mock_two_rooms):
    search = client.post("/chat", json={"message": "rooms in Jaipur", "mode": "fast"}, headers=auth_headers).json()
    client.post("/chat", json={"message": "what about the second one?", "session_id": search["session_id"], "mode": "fast"}, headers=auth_headers)
    r = client.post("/chat", json={"message": "does it have fridge?", "session_id": search["session_id"], "mode": "fast"}, headers=auth_headers).json()
    assert "Refrigerator" in r["message"]


def test_03_which_rooms_have_ac(client, auth_headers, mock_two_rooms):
    search = client.post("/chat", json={"message": "rooms in Jaipur", "mode": "fast"}, headers=auth_headers).json()
    r = client.post("/chat", json={"message": "which rooms have ac?", "session_id": search["session_id"], "mode": "fast"}, headers=auth_headers).json()
    assert "Skyline 2BHK" in r["message"]
    assert "Garden View PG" not in r["message"]


def test_04_how_many_rooms_have_parking(client, auth_headers, mock_two_rooms):
    search = client.post("/chat", json={"message": "rooms in Jaipur", "mode": "fast"}, headers=auth_headers).json()
    r = client.post("/chat", json={"message": "how many rooms have parking?", "session_id": search["session_id"], "mode": "fast"}, headers=auth_headers).json()
    assert r["message"] == "1 of the 2 verified rooms have parking."


def test_05_which_rooms_dont_have_wifi(client, auth_headers, mock_two_rooms):
    search = client.post("/chat", json={"message": "rooms in Jaipur", "mode": "fast"}, headers=auth_headers).json()
    r = client.post("/chat", json={"message": "which rooms don't have wifi?", "session_id": search["session_id"], "mode": "fast"}, headers=auth_headers).json()
    assert "Skyline 2BHK" in r["message"]
    assert "Garden View PG" not in r["message"]


def test_06_are_there_any_furnished_rooms(client, auth_headers, mock_two_rooms):
    search = client.post("/chat", json={"message": "rooms in Jaipur", "mode": "fast"}, headers=auth_headers).json()
    r = client.post("/chat", json={"message": "are there any furnished rooms?", "session_id": search["session_id"], "mode": "fast"}, headers=auth_headers).json()
    assert r["message"] == "No, none of the verified rooms have furnishing."


def test_07_what_are_their_prices(client, auth_headers, mock_two_rooms):
    search = client.post("/chat", json={"message": "rooms in Jaipur", "mode": "fast"}, headers=auth_headers).json()
    r = client.post("/chat", json={"message": "what are their prices?", "session_id": search["session_id"], "mode": "fast"}, headers=auth_headers).json()
    assert "20,000" in r["message"]
    assert "9,000" in r["message"]


# Checklist items 8, 9, 10 ("does the first one have fridge?", "does
# the second one have ac?", "anything cheaper?") are covered by the
# dedicated tests in the section above, not repeated here.


def test_11_what_about_furnished_ones(client, auth_headers, mock_homilivo_success):
    """
    Plural "ones" (no "room(s)" word) must still be recognized as a
    set-wide question, not fall back to a stale single-room selection
    - SAMPLE_ROOMS has no furnished data, so the honest answer is "no
    rooms have it", not an error or a hallucinated room-specific reply.
    """
    search = client.post("/chat", json={"message": "find rooms in Jaipur", "mode": "fast"}, headers=auth_headers).json()
    client.post("/chat", json={"message": "select the first one", "session_id": search["session_id"], "mode": "fast"}, headers=auth_headers)

    r = client.post("/chat", json={"message": "what about furnished ones?", "session_id": search["session_id"], "mode": "fast"}, headers=auth_headers).json()

    assert r["success"] is True
    assert r["message"] == "No, none of the verified rooms have furnishing."
    assert r["selected_room"] is None


def test_12_rooms_in_bangalore(client, auth_headers, mock_homilivo_success):
    r = client.post("/chat", json={"message": "rooms in Bangalore", "mode": "fast"}, headers=auth_headers).json()
    assert r["search_performed"] is True


def test_13_find_me_rooms_in_bangalore(client, auth_headers, mock_homilivo_success):
    r = client.post("/chat", json={"message": "find me rooms in Bangalore", "mode": "fast"}, headers=auth_headers).json()
    assert r["search_performed"] is True


def test_14_find_rooms_in_udaipur(client, auth_headers, mock_homilivo_success):
    r = client.post("/chat", json={"message": "find rooms in Udaipur", "mode": "fast"}, headers=auth_headers).json()
    assert r["search_performed"] is True


def test_15_book_this_room(client, auth_headers, mock_homilivo_success):
    search = client.post("/chat", json={"message": "find rooms in Jaipur", "mode": "fast"}, headers=auth_headers).json()
    r = client.post("/chat", json={"message": "book this room", "session_id": search["session_id"], "mode": "fast"}, headers=auth_headers).json()
    assert "can't complete that action" in r["message"]


def test_16_thanks(client, auth_headers):
    r = client.post("/chat", json={"message": "thanks", "mode": "fast"}, headers=auth_headers).json()
    assert "welcome" in r["message"].lower()


def test_17_bye(client, auth_headers):
    r = client.post("/chat", json={"message": "bye", "mode": "fast"}, headers=auth_headers).json()
    assert "goodbye" in r["message"].lower()


def test_18_unsupported_general_knowledge_question(client, auth_headers, monkeypatch):
    _mock_gemma_unavailable(monkeypatch)
    r = client.post("/chat", json={"message": "who is the president?", "mode": "fast"}, headers=auth_headers).json()
    assert "support team" in r["message"].lower()


def test_19_unknown_amenity_swimming_pool(client, auth_headers, mock_two_rooms):
    search = client.post("/chat", json={"message": "rooms in Jaipur", "mode": "fast"}, headers=auth_headers).json()
    r = client.post("/chat", json={"message": "does it have a swimming pool?", "session_id": search["session_id"], "mode": "fast"}, headers=auth_headers).json()
    assert "don't have verified information" in r["message"]


def test_20_api_failure(client, auth_headers, mock_homilivo_failure):
    r = client.post("/chat", json={"message": "find rooms in Jaipur", "mode": "fast"}, headers=auth_headers).json()
    assert r["room_count"] == 0
    assert "Traceback" not in r["message"]


def test_21_empty_result_set(client, auth_headers, mock_homilivo_empty):
    r = client.post("/chat", json={"message": "find rooms in Jaipur", "mode": "fast"}, headers=auth_headers).json()
    assert r["success"] is True
    assert r["room_count"] == 0


def test_22_stale_selected_room_then_set_wide_question(client, auth_headers, mock_two_rooms):
    search = client.post("/chat", json={"message": "rooms in Jaipur", "mode": "fast"}, headers=auth_headers).json()
    client.post("/chat", json={"message": "what about the second one?", "session_id": search["session_id"], "mode": "fast"}, headers=auth_headers)
    r = client.post("/chat", json={"message": "which rooms have parking?", "session_id": search["session_id"], "mode": "fast"}, headers=auth_headers).json()
    assert "Skyline 2BHK" in r["message"]
    assert r["selected_room"] is None
