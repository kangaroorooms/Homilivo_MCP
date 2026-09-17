"""
P1-05: Property Grounding Regression Tests
FIX 1: Skipped — existing missing-field handling
  already qualifies correctly per field. Tests
  below lock in this correct behavior.
FIX 2: Unified LLM grounding instruction wording
  across QWEN, GEMMA, GROQ system prompts.
FIX 3: Skipped — build_llm_room_context() already
  passes full room dict to LLM, nothing excluded.
REMAINING RISK: Output guardrail has zero
  fact-grounding check. Hallucinated facts that
  pass the existing guardrail checks are not caught.
  This requires a separate design decision.
"""

from __future__ import annotations

import re
import types

from app.experiments.agent_modes import (
    FACT_GROUNDING_RULE,
    answer_room_fact,
    generate_fast_response,
    generate_gemma_response,
    generate_groq_response,
    generate_qwen_response,
)

# Qualification phrases the current (unchanged) missing-field handling
# uses. Any one of these appearing in a response is accepted as "told
# the user the fact isn't available" rather than requiring one exact
# string, since answer_room_fact() intentionally uses field-specific
# wording (see P1-05 Step 1 findings).
QUALIFICATION_MARKERS = (
    "not specify",
    "not available",
    "don't have verified information",
    "does not specify",
)


def _contains_qualification(response: str) -> bool:
    lowered = response.lower()
    return any(marker in lowered for marker in QUALIFICATION_MARKERS)


# A plausible-looking but NOT-in-the-data value that a hallucinating
# implementation might invent. Used as a negative check.
INVENTED_PRICE_PATTERN = re.compile(r"₹\s*[\d,]+")


# ============================================================
# TEST 1 — missing price is never invented
# ============================================================

def test_answer_room_fact_missing_price_not_invented():
    room = {
        "id": "r1",
        "title": "Cozy Studio",
        "city": "Jaipur",
        "roomType": "1RK",
        # no pricePerMonth / price field at all
    }

    response = answer_room_fact("what is its price?", room)

    assert response
    assert response.strip() != ""
    assert not INVENTED_PRICE_PATTERN.search(response)
    assert _contains_qualification(response)


# ============================================================
# TEST 2 — missing amenities are never invented
# ============================================================

def test_answer_room_fact_missing_amenities_not_invented():
    room = {
        "id": "r2",
        "title": "Sunny Room",
        "city": "Pune",
        "roomType": "1BHK",
        "pricePerMonth": 12000,
        # no amenities/amenity/facilities/facility/features field
    }

    response = answer_room_fact("does it have parking?", room)

    assert response
    # Must decline rather than affirmatively claim the amenity exists,
    # and must not fabricate any OTHER amenity name either.
    assert _contains_qualification(response)
    assert "yes" not in response.lower().split(".")[0]
    for other_amenity in ("wifi", "fridge", "washing machine", "tv"):
        assert other_amenity not in response.lower()


# ============================================================
# TEST 3 — present price is used correctly, unchanged behavior
# ============================================================

def test_answer_room_fact_present_price_used_correctly():
    room = {
        "id": "r3",
        "title": "Deluxe Room",
        "city": "Mumbai",
        "roomType": "2BHK",
        "pricePerMonth": 15000,
    }

    response = answer_room_fact("what is its price?", room)

    assert response is not None
    assert ("15000" in response) or ("15,000" in response)


# ============================================================
# TEST 4 — missing location is never invented
# ============================================================

def test_answer_room_fact_missing_location_not_invented():
    room = {
        "id": "r4",
        "title": "Basic Room",
        "roomType": "1RK",
        "pricePerMonth": 9000,
        # no location/area/locality/city field
    }

    response = answer_room_fact("where is it located?", room)

    assert response
    assert _contains_qualification(response)
    # Must not have fabricated any known city name.
    for city in ("Mumbai", "Pune", "Jaipur", "Delhi", "Bangalore", "Chennai"):
        assert city.lower() not in response.lower()


# ============================================================
# TEST 5 — unified grounding instruction, identical across modes
# ============================================================

def test_llm_prompts_contain_unified_grounding_instruction(monkeypatch):
    captured_qwen_gemma: list = []
    captured_groq: list = []

    def fake_chat(model, messages):
        captured_qwen_gemma.append(messages)
        return types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))

    def fake_create(model, messages):
        captured_groq.append(messages)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
        )

    fake_groq_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=fake_create))
    )

    monkeypatch.setattr("app.experiments.agent_modes.chat", fake_chat)
    monkeypatch.setattr(
        "app.experiments.agent_modes._get_groq_client", lambda: fake_groq_client
    )

    rooms = [{"id": "r1", "title": "Room", "city": "Jaipur", "pricePerMonth": 10000, "roomType": "1RK"}]
    common_args = dict(
        user_query="what is its price?",
        rooms=rooms,
        conversation_history=[],
        selected_room=rooms[0],
        selected_room_id="r1",
        search_request={"city": "Jaipur"},
    )

    generate_qwen_response(**common_args)
    generate_gemma_response(**common_args)
    generate_groq_response(**common_args)

    qwen_system_prompt = captured_qwen_gemma[0][0]["content"]
    gemma_system_prompt = captured_qwen_gemma[1][0]["content"]
    groq_system_prompt = captured_groq[0][0]["content"]

    for system_prompt in (qwen_system_prompt, gemma_system_prompt, groq_system_prompt):
        assert FACT_GROUNDING_RULE in system_prompt

    # Not just "contains the same substring" - assert the substring is
    # exactly the shared constant text in every prompt (byte-identical
    # wording across modes).
    assert (
        qwen_system_prompt.count(FACT_GROUNDING_RULE)
        == gemma_system_prompt.count(FACT_GROUNDING_RULE)
        == groq_system_prompt.count(FACT_GROUNDING_RULE)
        == 1
    )


# ============================================================
# TEST 6 — fast mode omits missing fields instead of inventing them
# ============================================================

def test_fast_mode_missing_field_omits_not_invents():
    rooms = [
        {
            "id": "r5",
            "title": "No-Price Room",
            "city": "Jaipur",
            "roomType": "1RK",
            # no pricePerMonth / price field at all
        }
    ]
    search_request = {"city": "Jaipur"}

    response = generate_fast_response("find rooms in Jaipur", rooms, search_request)

    assert "No-Price Room" in response
    assert not INVENTED_PRICE_PATTERN.search(response)
    assert "/month" not in response
