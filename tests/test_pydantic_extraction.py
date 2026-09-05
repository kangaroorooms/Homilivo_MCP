"""
Phase 6 - Pydantic-validated LLM extraction fallback.

Covers:
  1. SearchParameters schema validation (city/price/room_types/
     ideal_for/sort enums, price-range, unknown-field rejection).
  2. The fallback trigger (_extraction_is_unhelpful): only fires when
     the deterministic regex fast_parser found too little to act on.
  3. Clean/easy queries never call the LLM extraction fallback.
  4. The 3 golden-eval "messy" cases named in the Phase 6 brief, run
     through the real process_message() pipeline (via /chat) with the
     LLM extraction mocked - CI-safe, no live Ollama/Groq required,
     following the same pattern already used by
     tests/test_golden_eval.py's test_messy_search_detection_end_to_end_*.
  5. A failed/invalid LLM extraction never sends unvalidated data
     onward - the existing clarification-question behavior runs
     instead.

Note on scope: "any cheap rooms???" has no city, no number, and no
room-type word anywhere in the text. No extraction tier - regex or
LLM - can honestly produce a city for it without inventing one, which
this project's rules forbid. That query is (and was, before this
phase, via the pre-existing embedding-based input-guardrail fallback -
see app/guardrails/input_guardrail.py) already correctly recognized as
a room-search attempt; what Phase 6 proves for it is that the LLM
fallback is exercised and honestly reports nothing new, so the
existing "which city are you looking for?" clarification runs instead
of silently failing or fabricating a search.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import ValidationError

import app.experiments.agent_modes as agent_modes
from app.experiments.agent_modes import GemmaError
from app.schemas.chat import SearchParameters


def _no_gemma(monkeypatch) -> None:
    """Force the existing Gemma intent classifier unavailable, matching
    the pattern already used by tests/test_golden_eval.py so these
    tests are deterministic regardless of whether Ollama is running."""
    monkeypatch.setattr(
        agent_modes,
        "classify_intent",
        lambda query: (_ for _ in ()).throw(GemmaError("no local model available")),
    )


def _query_params(request) -> dict[str, list[str]]:
    return parse_qs(urlparse(request.full_url).query)


# ============================================================
# 1. SCHEMA VALIDATION
# ============================================================

def test_schema_accepts_and_normalizes_valid_output():
    params = SearchParameters(
        city="Jaipur",
        room_types=["pg"],
        ideal_for=["family"],
        max_price=15000,
    )
    assert params.to_search_arguments() == {
        "city": "Jaipur",
        "min_price": None,
        "max_price": 15000,
        "room_types": ["PG"],
        "ideal_for": ["Family"],
        "sort": "latest",
        "limit": 10,
    }


def test_schema_rejects_invalid_room_type():
    with pytest.raises(ValidationError):
        SearchParameters(room_types=["Mansion"])


def test_schema_rejects_invalid_ideal_for():
    with pytest.raises(ValidationError):
        SearchParameters(ideal_for=["Bachelors"])


def test_schema_rejects_invalid_sort():
    with pytest.raises(ValidationError):
        SearchParameters(sort="cheapest_first")


def test_schema_rejects_min_greater_than_max():
    with pytest.raises(ValidationError):
        SearchParameters(min_price=20000, max_price=10000)


def test_schema_rejects_negative_price():
    with pytest.raises(ValidationError):
        SearchParameters(min_price=-100)


def test_schema_rejects_unknown_field():
    """
    guardrails/parameter_guardrail.py's ALLOWED_PARAMETERS has no
    gender_preference field at all - a hallucinated one must be
    rejected outright here, never silently dropped or passed on to
    the parameter guardrail (which would itself reject it, but only
    after this schema was supposed to have already caught it).
    """
    with pytest.raises(ValidationError):
        SearchParameters(city="Jaipur", gender_preference="female")


def test_schema_rejects_wrong_types():
    with pytest.raises(ValidationError):
        SearchParameters(room_types="PG")  # must be a list, not a bare string


# ============================================================
# 2. FALLBACK TRIGGER LOGIC (unit-level, no LLM involved)
# ============================================================

def test_unhelpful_new_search_missing_city():
    assert agent_modes._extraction_is_unhelpful(
        {"city": None, "min_price": None, "max_price": None, "room_types": [], "ideal_for": []},
        is_refinement=False,
    ) is True


def test_unhelpful_new_search_price_with_no_type_or_ideal_for():
    assert agent_modes._extraction_is_unhelpful(
        {"city": "Bangalore", "min_price": None, "max_price": 20000, "room_types": [], "ideal_for": []},
        is_refinement=False,
    ) is True


def test_not_unhelpful_new_search_clean_city_only():
    """'find rooms in Jaipur' - a bare city with nothing else claimed
    is a complete, valid search, not a miss."""
    assert agent_modes._extraction_is_unhelpful(
        {"city": "Jaipur", "min_price": None, "max_price": None, "room_types": [], "ideal_for": []},
        is_refinement=False,
    ) is False


def test_not_unhelpful_new_search_when_room_type_and_price_found():
    assert agent_modes._extraction_is_unhelpful(
        {"city": "Mumbai", "min_price": None, "max_price": 10000, "room_types": ["PG"], "ideal_for": []},
        is_refinement=False,
    ) is False


def test_unhelpful_refinement_when_nothing_new_found():
    assert agent_modes._extraction_is_unhelpful(
        {"city": None, "min_price": None, "max_price": None, "room_types": [], "ideal_for": []},
        is_refinement=True,
    ) is True


def test_not_unhelpful_refinement_when_something_found():
    """Refinements normally omit the city on purpose - that alone must
    never count as a miss."""
    assert agent_modes._extraction_is_unhelpful(
        {"city": None, "min_price": None, "max_price": 8000, "room_types": [], "ideal_for": []},
        is_refinement=True,
    ) is False


# ============================================================
# 3. extract_search_parameters_via_llm(): tiering + validation
# ============================================================

def test_llm_extraction_falls_back_to_groq_when_gemma_unavailable(monkeypatch):
    def gemma_fails(_query):
        raise agent_modes.ParameterExtractionError("ollama not running")

    def groq_ok(_query):
        return {
            "city": "Pune", "min_price": None, "max_price": None,
            "room_types": [], "ideal_for": [], "sort": "latest", "limit": 10,
        }

    monkeypatch.setattr(agent_modes, "_extract_search_parameters_via_gemma", gemma_fails)
    monkeypatch.setattr(agent_modes, "_extract_search_parameters_via_groq", groq_ok)

    result = agent_modes.extract_search_parameters_via_llm("some messy query")

    assert result["city"] == "Pune"


def test_llm_extraction_returns_none_when_every_tier_fails(monkeypatch):
    def always_fails(_query):
        raise agent_modes.ParameterExtractionError("unavailable")

    monkeypatch.setattr(agent_modes, "_extract_search_parameters_via_gemma", always_fails)
    monkeypatch.setattr(agent_modes, "_extract_search_parameters_via_groq", always_fails)

    assert agent_modes.extract_search_parameters_via_llm("anything") is None


def test_llm_extraction_returns_none_on_schema_validation_failure(monkeypatch):
    """
    Malformed LLM output (an unrecognized room type here) must never
    be forwarded - extract_search_parameters_via_llm must swallow the
    ValidationError and return None, exactly like "no extraction tier
    available", so the caller falls back to the existing
    clarification-question behavior instead of sending bad data on.
    """
    monkeypatch.setattr(
        agent_modes,
        "_extract_search_parameters_via_gemma",
        lambda _query: {"city": "Jaipur", "room_types": ["Mansion"]},
    )
    monkeypatch.setattr(
        agent_modes,
        "_extract_search_parameters_via_groq",
        lambda _query: (_ for _ in ()).throw(agent_modes.ParameterExtractionError("no key")),
    )

    assert agent_modes.extract_search_parameters_via_llm("bad query") is None


# ============================================================
# 4. CLEAN QUERIES STAY ON THE FAST PATH (no LLM call at all)
# ============================================================

def test_clean_easy_query_never_calls_llm_extraction(client, auth_headers, mock_homilivo_success, monkeypatch):
    def fail_if_called(_query):
        raise AssertionError("LLM extraction fallback must not run for a clean query")

    monkeypatch.setattr(agent_modes, "extract_search_parameters_via_llm", fail_if_called)

    response = client.post(
        "/chat", json={"message": "find rooms in Jaipur", "mode": "fast"}, headers=auth_headers,
    ).json()

    assert response["search_performed"] is True
    assert _query_params(mock_homilivo_success[-1])["city"] == ["Jaipur"]


def test_clean_query_with_room_type_and_price_never_calls_llm_extraction(
    client, auth_headers, mock_homilivo_success, monkeypatch
):
    def fail_if_called(_query):
        raise AssertionError("LLM extraction fallback must not run for a clean query")

    monkeypatch.setattr(agent_modes, "extract_search_parameters_via_llm", fail_if_called)

    response = client.post(
        "/chat",
        json={"message": "PG in Mumbai under 10000", "mode": "fast"},
        headers=auth_headers,
    ).json()

    assert response["search_performed"] is True


# ============================================================
# 5. THE 3 GOLDEN-EVAL "MESSY" CASES, END TO END (mocked LLM)
# ============================================================

def test_messy_case_pg_near_jaipur_for_girls(client, auth_headers, mock_homilivo_success, monkeypatch):
    """'wat about pg options near jaipur for gurls' - regex finds
    room_types=[PG] but no city ('near' isn't a recognized city
    preposition). The LLM fallback must be invoked (spied) and its
    resolved city must reach the actual Homilivo search call.

    The slang/typo-heavy phrasing here also fails the embedding-based
    input guardrail on its own confidence (pre-existing, unrelated to
    Phase 6 - see app/guardrails/input_guardrail.py), so - exactly like
    tests/test_golden_eval.py's test_messy_search_detection_end_to_end_
    llm_available - the existing Gemma intent-classifier fallback is
    mocked to a realistic ROOM_SEARCH result rather than "unavailable",
    to reach the NEW_SEARCH branch this test is actually about.
    """
    monkeypatch.setattr(
        agent_modes,
        "classify_intent",
        lambda query: {
            "intent": "ROOM_SEARCH",
            "confidence": 0.9,
            "reason": "Mentions PG and Jaipur.",
            "search_query": query,
        },
    )

    calls = []

    def fake_llm_extract(query):
        calls.append(query)
        return {
            "city": "Jaipur", "min_price": None, "max_price": None,
            "room_types": ["PG"], "ideal_for": [], "sort": "latest", "limit": 10,
        }

    monkeypatch.setattr(agent_modes, "extract_search_parameters_via_llm", fake_llm_extract)

    response = client.post(
        "/chat",
        json={"message": "wat about pg options near jaipur for gurls", "mode": "fast"},
        headers=auth_headers,
    ).json()

    assert calls == ["wat about pg options near jaipur for gurls"]
    assert response["search_performed"] is True

    sent = _query_params(mock_homilivo_success[-1])
    assert sent["city"] == ["Jaipur"]
    assert sent["roomTypes"] == ["PG"]


def test_messy_case_blr_under_20k_for_wife_infers_family(client, auth_headers, mock_homilivo_success, monkeypatch):
    """'i need somewhere to stay in blr under 20k for me and my wife' -
    regex already resolves city=Bangalore and max_price=20000
    correctly; the only gap is the implicit-occupancy ideal_for=Family
    inference, which the LLM fallback fills in. Regex-found fields
    (city, max_price) must stay authoritative even though the fake
    LLM here returns none of them - only the ideal_for gap should
    come from the LLM."""
    _no_gemma(monkeypatch)

    calls = []

    def fake_llm_extract(query):
        calls.append(query)
        return {
            "city": None, "min_price": None, "max_price": None,
            "room_types": [], "ideal_for": ["Family"], "sort": "latest", "limit": 10,
        }

    monkeypatch.setattr(agent_modes, "extract_search_parameters_via_llm", fake_llm_extract)

    response = client.post(
        "/chat",
        json={
            "message": "i need somewhere to stay in blr under 20k for me and my wife",
            "mode": "fast",
        },
        headers=auth_headers,
    ).json()

    assert calls == ["i need somewhere to stay in blr under 20k for me and my wife"]
    assert response["search_performed"] is True

    sent = _query_params(mock_homilivo_success[-1])
    assert sent["city"] == ["Bangalore"]
    assert sent["maxPrice"] == ["20000"]
    assert sent["idealFor"] == ["Family"]


def test_messy_case_any_cheap_rooms_asks_for_city_honestly(client, auth_headers, mock_homilivo_success, monkeypatch):
    """'any cheap rooms???' has no city, no number, no room-type word
    anywhere in the text. The LLM fallback is exercised (regex found
    nothing at all) but must not invent a city - it correctly reports
    nothing new, and the EXISTING clarification behavior asks for the
    city instead of silently failing or fabricating a search."""
    _no_gemma(monkeypatch)

    calls = []

    def fake_llm_extract(query):
        calls.append(query)
        return {
            "city": None, "min_price": None, "max_price": None,
            "room_types": [], "ideal_for": [], "sort": "latest", "limit": 10,
        }

    monkeypatch.setattr(agent_modes, "extract_search_parameters_via_llm", fake_llm_extract)

    response = client.post(
        "/chat", json={"message": "any cheap rooms???", "mode": "fast"}, headers=auth_headers,
    ).json()

    assert calls == ["any cheap rooms???"]
    assert response["search_performed"] is False
    assert response["pending_clarification"] is True
    assert "city" in response["message"].lower()


def test_messy_case_llm_unavailable_falls_back_to_clarification(client, auth_headers, mock_homilivo_success, monkeypatch):
    """When every LLM tier is unavailable (extract_search_parameters_via_llm
    returns None, e.g. Ollama down and no GROQ_API_KEY set), the
    pipeline must behave exactly as it did before Phase 6 - ask for
    the missing city - never crash and never send a half-built search."""
    _no_gemma(monkeypatch)
    monkeypatch.setattr(agent_modes, "extract_search_parameters_via_llm", lambda _query: None)

    response = client.post(
        "/chat", json={"message": "any cheap rooms???", "mode": "fast"}, headers=auth_headers,
    ).json()

    assert response["search_performed"] is False
    assert response["pending_clarification"] is True
