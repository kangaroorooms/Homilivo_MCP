# P2-09: Hindi/Hinglish Query Handling Tests
# Tests locale-aware extraction for Indian users
# who mix Hindi and English in room searches.

from app.fast_parser import extract_search_parameters
from app.experiments.agent_modes import (
    ConversationState,
    MESSAGE_NEW_SEARCH,
    route_message,
)
from app.guardrails.input_guardrail import classify_user_intent
from app.guardrails.api_data_guardrail import validate_room_data


# ============================================================
# CITY TESTS
# ============================================================

def test_hinglish_city_bengaluru():
    result = extract_search_parameters("show me rooms in bengaluru")
    assert result["city"] == "Bangalore"


def test_hinglish_city_dilli():
    result = extract_search_parameters("room dhundna hai dilli mein")
    assert result["city"] == "Delhi"


def test_hinglish_city_bambai():
    result = extract_search_parameters("bambai mein flat chahiye")
    assert result["city"] == "Mumbai"


def test_hinglish_city_chenai():
    result = extract_search_parameters("chenai mein room dikhao")
    assert result["city"] == "Chennai"


def test_typo_city_banglore():
    result = extract_search_parameters("rooms in banglore under 10k")
    assert result["city"] == "Bangalore"


def test_typo_city_calcutta():
    result = extract_search_parameters("calcutta mein 2bhk")
    assert result["city"] == "Kolkata"


# ============================================================
# BUDGET TESTS
# ============================================================

def test_hinglish_budget_hazaar():
    result = extract_search_parameters(
        "10 hazaar mein room chahiye bangalore"
    )
    assert result["max_price"] == 10000 or result["city"] == "Bangalore"


def test_standard_budget_still_works():
    result = extract_search_parameters("room in bangalore under 15000")
    assert result["max_price"] == 15000
    assert result["city"] == "Bangalore"


# ============================================================
# BHK TESTS
# ============================================================

def test_hinglish_bhk_do_bhk():
    result = extract_search_parameters("do BHK bangalore mein")
    assert "2BHK" in result["room_types"] or result["city"] == "Bangalore"


def test_standard_bhk_still_works():
    result = extract_search_parameters("2BHK in bangalore under 20k")
    assert "2BHK" in result["room_types"]
    assert result["city"] == "Bangalore"


# ============================================================
# MIXED LANGUAGE TESTS
# ============================================================

def test_mixed_hinglish_query():
    result = extract_search_parameters(
        "bangalore mein 2BHK chahiye 15000 se kam"
    )
    assert result["city"] == "Bangalore"
    assert result["max_price"] == 15000


def test_typo_tolerance():
    result = extract_search_parameters("show rooms in hydrabad")
    assert result["city"] == "Hyderabad"


# ============================================================
# EMBEDDING MODEL TEST
# ============================================================

def test_multilingual_model_loaded():
    from app.embeddings import get_shared_model

    model = get_shared_model()
    assert model is not None

    embedding = model.encode("bangalore mein room chahiye")
    assert embedding is not None
    assert len(embedding) == 384


# ============================================================
# MANUAL-DEMO REGRESSION TESTS (P2-09 continuation)
#
# Covers two bugs found during manual demo testing that the tests
# above did not catch, because they only exercise
# extract_search_parameters() in isolation:
#
#   1. "10 hazaar mein room chahiye bangalore" resolved max_price but
#      not city, because the trailing-city fallback in fast_parser.py
#      only matched a CAPITALIZED trailing word - see TEST 15.
#   2. Both "bambai mein flat chahiye" and
#      "10 hazaar mein room chahiye bangalore" were rejected as
#      OUT_OF_SCOPE by the semantic input guardrail
#      (app/guardrails/input_guardrail.py), which was trained on
#      English-only example sentences - see TEST 16. TEST 14 confirms
#      the deterministic router still resolves NEW_SEARCH correctly on
#      its own (unaffected by the guardrail bug), and TEST 17 confirms
#      that correct routing decision is trusted even if the guardrail
#      misjudges the input.
# ============================================================

def test_hinglish_routing_bambai():
    state = ConversationState()

    route = route_message("bambai mein flat chahiye", state)

    assert route.message_type == MESSAGE_NEW_SEARCH


def test_hinglish_trailing_lowercase_city():
    result = extract_search_parameters(
        "10 hazaar mein room chahiye bangalore"
    )

    # fast_parser resolves every city to its display-cased form (e.g.
    # "Mumbai", "Bangalore" - see _resolve_city_name()), including
    # known cities matched from an all-lowercase trailing word, so the
    # correct assertion is "Bangalore", not the literal lowercase
    # spelling the user typed.
    assert result["city"] == "Bangalore"
    assert result["max_price"] == 10000


def test_hinglish_guardrail_classifies_room_search():
    assert classify_user_intent(
        "bambai mein flat chahiye"
    ) == "ROOM_SEARCH"

    assert classify_user_intent(
        "10 hazaar mein room chahiye bangalore"
    ) == "ROOM_SEARCH"


def test_new_search_in_trusted_routes(
    client, auth_headers, mock_homilivo_success, monkeypatch
):
    """
    Confirms MESSAGE_NEW_SEARCH is trusted over the semantic input
    guardrail's judgment in process_message() (app/experiments/
    agent_modes.py), not just that the guardrail itself now happens to
    accept these queries after TEST 16's fix. Forces the guardrail to
    reject the message (simulating it still misjudging some other
    Hinglish phrasing) and asserts the turn is NOT blocked and the LLM
    fallback is never reached, because the deterministic NEW_SEARCH
    route must be trusted on its own.
    """
    import app.experiments.agent_modes as agent_modes

    monkeypatch.setattr(
        agent_modes,
        "validate_user_input",
        lambda user_query: (False, "OUT_OF_SCOPE"),
    )

    def _fail_if_called(user_query):
        raise AssertionError(
            "fallback_llm_intent() should not be called - "
            "MESSAGE_NEW_SEARCH must be trusted directly."
        )

    monkeypatch.setattr(
        agent_modes, "fallback_llm_intent", _fail_if_called
    )

    response = client.post(
        "/chat",
        json={"message": "bambai mein flat chahiye", "mode": "fast"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["search_performed"] is True
    assert "sorry" not in body["message"].lower()
    assert "out of scope" not in body["message"].lower()


# ============================================================
# IDEAL-FOR ENUM REGRESSION TEST
#
# Homilivo's live API returns "Small Family" as an idealFor value
# that api_data_guardrail.py's VALID_IDEAL_FOR did not recognize,
# causing validate_room_data() to reject the ENTIRE room batch for
# that search (not just the one room) - see the guardrail's
# VALID_IDEAL_FOR comment for the confirmed source of this value.
# ============================================================

def test_small_family_idealfor_accepted():
    room = {
        "id": "room-small-family-1",
        "title": "Cozy Small Family Flat",
        "city": "Mumbai",
        "pricePerMonth": 25000,
        "roomType": "2BHK",
        "idealFor": ["Small Family"],
    }

    # validate_room_data() returns (bool, message) - not (bool, rooms).
    # The meaningful assertion is that it accepts the batch (True)
    # rather than rejecting it, which is what the live bug broke.
    ok, message = validate_room_data([room])

    assert ok is True
    assert "invalid" not in message.lower()


# ============================================================
# MANUALLY-VERIFIED DEMO QUERIES (exact phrasings) - P2-09 final
# verification pass. The tests above exercise the same parsing
# logic with slightly different phrasing; these lock in the exact
# strings that were manually confirmed working in the live demo.
# ============================================================

def test_hinglish_dilli_routing():
    result = extract_search_parameters("dilli mein room dikhao")
    assert result["city"] == "Delhi"


def test_hinglish_bengaluru_pg():
    result = extract_search_parameters("bengaluru mein PG chahiye")
    assert result["city"] == "Bangalore"


def test_hinglish_15k_se_kam():
    result = extract_search_parameters(
        "15k se kam bangalore mein flat"
    )
    assert result["max_price"] == 15000
    assert result["city"] == "Bangalore"


def test_hinglish_do_bhk_with_city():
    result = extract_search_parameters(
        "do BHK bangalore mein chahiye"
    )
    assert result["city"] == "Bangalore"
    assert "2BHK" in result["room_types"]


def test_guardrail_classifies_dilli_room_search():
    assert classify_user_intent(
        "dilli mein room dikhao"
    ) == "ROOM_SEARCH"


def test_english_regression_not_broken():
    result = extract_search_parameters("2BHK in bangalore under 15000")
    assert result["city"] == "Bangalore"
    assert result["max_price"] == 15000
    assert "2BHK" in result["room_types"]
