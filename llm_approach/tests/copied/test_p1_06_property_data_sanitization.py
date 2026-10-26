# Copied from tests/test_p1_06_property_data_sanitization.py at commit 7dbcba3 (copied, not imported).
# Changes: import now points to llm_approach.prompt_safety; build_llm_room_context cases omitted. Every changed line is marked '# v2:'.
# Not copied from this file: cases that need app.api, TestClient, agent_modes or the MCP server.

from llm_approach.prompt_safety import sanitize_room_for_llm_context  # v2: was: from app.experiments.agent_modes import (build_llm_room_context, sanitize_room_for_llm_context)


def test_sanitize_truncates_long_title():
    room = {"title": "A" * 500}
    result = sanitize_room_for_llm_context(room)
    assert len(result["title"]) <= 200


def test_sanitize_truncates_long_description():
    room = {"description": "B" * 1000}
    result = sanitize_room_for_llm_context(room)
    assert len(result["description"]) <= 500


def test_sanitize_removes_control_characters():
    room = {"title": "Nice\x00\x01Room\x07Here"}
    result = sanitize_room_for_llm_context(room)
    assert "\x00" not in result["title"]
    assert "\x01" not in result["title"]
    assert "\x07" not in result["title"]
    assert result["title"] != ""


def test_sanitize_removes_null_bytes():
    room = {"title": "Nice Room\x00 in Bangalore"}
    result = sanitize_room_for_llm_context(room)
    assert "\x00" not in result["title"]
    assert "Nice Room" in result["title"]


def test_sanitize_preserves_normal_content():
    room = {
        "title": "Cozy 1BHK",
        "pricePerMonth": 15000,
        "location": "Koramangala, Bangalore",
    }
    result = sanitize_room_for_llm_context(room)
    assert result["title"] == "Cozy 1BHK"
    assert result["location"] == "Koramangala, Bangalore"
    assert result["pricePerMonth"] == 15000


def test_sanitize_handles_missing_fields_gracefully():
    room = {"title": "Only Title Present"}
    result = sanitize_room_for_llm_context(room)
    assert result["title"] == "Only Title Present"
    assert "description" not in result
    assert "location" not in result


def test_sanitize_handles_empty_dict():
    result = sanitize_room_for_llm_context({})
    assert isinstance(result, dict)
    assert result == {}


def test_sanitize_handles_exception_gracefully():
    result_none = sanitize_room_for_llm_context(None)
    assert result_none is None

    result_str = sanitize_room_for_llm_context("not a dict")
    assert result_str == "not a dict"
