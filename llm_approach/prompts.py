"""System prompt, few-shot examples and the state summary sent to the model."""

from __future__ import annotations

import json
from typing import Any

from llm_approach import config, reuse

_INTENT_LINES = """\
NEW_SEARCH - start a new room search ("find me a room in pune"). City missing: set needs_clarification and ask for the city.
REFINEMENT - change the current search: another city, budget, BHK or room type ("in udupi", "under 15k", "2bhk only", "something in a different area").
ROOM_FOLLOWUP - a question about the rooms already on screen ("which is cheapest", "does the second one have wifi"). Set operation.
SHOW_MORE - wants more or different results ("show more", "kuch aur dikhao").
REJECT_CURRENT - dislikes the rooms shown ("none of these are good").
PLATFORM_INFO - a question about Homilivo itself (fees, how it works, owners, subscription).
GENERAL_CHAT - greeting, thanks, small talk, goodbye.
UNSUPPORTED_ACTION - asks for something the bot cannot do ("book this room", "call the owner").
OUT_OF_SCOPE - unrelated to accommodation or Homilivo ("what is the speed of light").
UNCLEAR - you cannot tell what the user wants ("i dont know", "umm")."""

_OPERATION_LINES = """\
LOWEST_PRICE, HIGHEST_PRICE (cheapest / most expensive of the rooms on screen)
ROOM_PRICE (price of one room), ROOM_TYPE, LOCATION, AMENITY (set slots.amenity)
STUDENT_SUITABILITY, FAMILY_SUITABILITY, PROFESSIONAL_SUITABILITY
ROOM_DETAILS (general details of one room), COMPARE (set compare_references)"""

_SCHEMA = """\
{
  "intent": one of the 10 intents,
  "operation": one of the operations above for ROOM_FOLLOWUP, otherwise null,
  "slots": {"city": string|null, "bhk": integer|null, "min_price": integer|null,
            "max_price": integer|null, "amenity": string|null, "ideal_for": string|null,
            "area": string|null},
  "room_reference": 1-based integer position of a room on screen, or "selected", or null,
  "compare_references": [1-based integers] (COMPARE only, else []),
  "standalone_query": the request rewritten so it needs no context, or null,
  "confidence": number from 0 to 1,
  "needs_clarification": true when a needed detail is missing,
  "clarifying_question": short question to the user when needs_clarification, else null,
  "reply": short friendly reply (max 2 sentences) ONLY for GENERAL_CHAT, OUT_OF_SCOPE, UNSUPPORTED_ACTION, else null
}"""

_RULES = """\
- You classify and extract. You never state room facts, prices, availability or platform facts yourself; code does that.
- Use "reply" only for GENERAL_CHAT, OUT_OF_SCOPE and UNSUPPORTED_ACTION. Keep it to two sentences and steer back to finding a room. Never put prices, room details or Homilivo policies in it. Set reply to null for every other intent.
- Prices are integers in rupees ("15k" -> 15000, "1.5 lakh" -> 150000). "under X" sets max_price, "above X" sets min_price.
- Only fill a slot the user actually stated or that clearly continues the current search. Never invent a city, price or BHK. If the user did not name a city, city is null.
- Words like "pls", "urgent", "a different area", "somewhere" are not cities or areas.
- area is a locality or neighbourhood inside a city (Koramangala, Indiranagar), never a city. A city goes in city. Without a named locality, area is null.
- Room references are positions of rooms on screen (first = 1). Use "selected" for "it" / "this one" when selected_room_index is set. Use null when no room is meant.
- When rooms are on screen, "in <place>" or "under <price>" refines the current search (REFINEMENT). When no rooms are on screen, it starts a new search.
- When pending_question is set, read the message as an answer to that question.
- The user may write English, Hindi or Hinglish, with typos. Handle all of them.
- Confidence is how sure you are of the intent. Use below 0.6 when the message is ambiguous.
- Output one JSON object and nothing else."""


def _rooms() -> list[dict[str, Any]]:
    return [
        {"index": 1, "title": "Budget Single Room", "price": 7000, "type": "Single", "location": "Koramangala"},
        {"index": 2, "title": "Cozy 2BHK", "price": 15000, "type": "2BHK", "location": "Indiranagar"},
        {"index": 3, "title": "Studio Flat", "price": 12000, "type": "Studio", "location": "HSR Layout"},
    ]


def _state(
    rooms: bool = False,
    city: str | None = None,
    pending: str | None = None,
    last_bot: str | None = None,
    selected: int | None = None,
) -> dict[str, Any]:
    return {
        "current_search": {"city": city} if city else None,
        "pending_question": pending,
        "last_bot_message": last_bot,
        "rooms_on_screen": _rooms() if rooms else [],
        "selected_room_index": selected,
        "recent_turns": [],
    }


def _decision(intent: str, **fields: Any) -> dict[str, Any]:
    slot_keys = ("city", "bhk", "min_price", "max_price", "amenity", "ideal_for", "area")
    slots = {key: fields.pop(key, None) for key in slot_keys}
    out: dict[str, Any] = {
        "intent": intent,
        "operation": fields.pop("operation", None),
        "slots": slots,
        "room_reference": fields.pop("room_reference", None),
        "compare_references": fields.pop("compare_references", []),
        "standalone_query": fields.pop("standalone_query", None),
        "confidence": fields.pop("confidence", 0.95),
        "needs_clarification": fields.pop("needs_clarification", False),
        "clarifying_question": fields.pop("clarifying_question", None),
        "reply": fields.pop("reply", None),
    }
    assert not fields, f"unknown example fields: {fields}"
    return out


_SHOWN = "Here are the rooms I found."

# (state, user message, expected decision)
FEW_SHOT_EXAMPLES: list[tuple[dict[str, Any], str, dict[str, Any]]] = [
    (_state(), "find me a room in bangalore",
     _decision("NEW_SEARCH", city="Bangalore", standalone_query="find rooms in Bangalore")),
    (_state(), "2bhk bangalor under 15k",
     _decision("NEW_SEARCH", city="Bangalore", bhk=2, max_price=15000,
               standalone_query="find 2 BHK rooms in Bangalore under 15000")),
    (_state(), "mujhe pune mein room chahiye",
     _decision("NEW_SEARCH", city="Pune", standalone_query="find rooms in Pune")),
    (_state(), "find me a room",
     _decision("NEW_SEARCH", confidence=0.85, needs_clarification=True,
               clarifying_question="Which city are you looking for?",
               standalone_query="find me a room")),
    (_state(rooms=True, city="Bangalore", last_bot=_SHOWN), "find me a room",
     _decision("NEW_SEARCH", confidence=0.8, needs_clarification=True,
               clarifying_question="Which city are you looking for?",
               standalone_query="find me a room")),
    (_state(pending="city", last_bot="Which city are you looking for?"), "pune",
     _decision("NEW_SEARCH", city="Pune", standalone_query="find rooms in Pune")),
    (_state(rooms=True, city="Bangalore", last_bot=_SHOWN), "in udupi",
     _decision("REFINEMENT", city="Udupi", standalone_query="find rooms in Udupi")),
    (_state(rooms=True, city="Bangalore", last_bot=_SHOWN), "something in a different area",
     _decision("REFINEMENT", confidence=0.8, needs_clarification=True,
               clarifying_question="Which area would you like?",
               standalone_query="find rooms in a different area")),
    (_state(rooms=True, city="Bangalore", pending="clarify", last_bot="Which area would you like?"), "koramangala",
     _decision("REFINEMENT", area="Koramangala", standalone_query="find rooms in Koramangala, Bangalore")),
    (_state(city="Bangalore"), "rooms in koramangala",
     _decision("NEW_SEARCH", city="Bangalore", area="Koramangala",
               standalone_query="find rooms in Koramangala, Bangalore")),
    (_state(rooms=True, city="Bangalore", last_bot=_SHOWN), "under 15k in indiranagar",
     _decision("REFINEMENT", area="Indiranagar", max_price=15000,
               standalone_query="find rooms in Indiranagar under 15000")),
    (_state(), "i dont know",
     _decision("UNCLEAR", confidence=0.4, needs_clarification=True,
               clarifying_question="Could you tell me which city and budget you have in mind?")),
    (_state(rooms=True, city="Bangalore", last_bot=_SHOWN), "none of these are good",
     _decision("REJECT_CURRENT", standalone_query="the shown rooms are not suitable")),
    (_state(rooms=True, city="Bangalore", last_bot=_SHOWN), "kuch aur dikhao",
     _decision("SHOW_MORE", standalone_query="show more rooms")),
    (_state(rooms=True, city="Bangalore", last_bot=_SHOWN), "which one is cheapest",
     _decision("ROOM_FOLLOWUP", operation="LOWEST_PRICE")),
    (_state(rooms=True, city="Bangalore", last_bot=_SHOWN), "does the second one have wifi",
     _decision("ROOM_FOLLOWUP", operation="AMENITY", amenity="WiFi", room_reference=2)),
    (_state(rooms=True, city="Bangalore", last_bot=_SHOWN, selected=2), "how much is it",
     _decision("ROOM_FOLLOWUP", operation="ROOM_PRICE", room_reference="selected")),
    (_state(rooms=True, city="Bangalore", last_bot="The cheapest is Room 1 (Budget Single Room) at ₹7,000/month.", selected=1),
     "does it have wifi",
     _decision("ROOM_FOLLOWUP", operation="AMENITY", amenity="WiFi", room_reference="selected")),
    (_state(rooms=True, city="Bangalore", last_bot=_SHOWN, selected=3), "is it good for students",
     _decision("ROOM_FOLLOWUP", operation="STUDENT_SUITABILITY", room_reference="selected")),
    (_state(rooms=True, city="Bangalore", last_bot=_SHOWN), "compare the first two",
     _decision("ROOM_FOLLOWUP", operation="COMPARE", compare_references=[1, 2])),
    (_state(), "how does homilivo make money",
     _decision("PLATFORM_INFO", standalone_query="How does Homilivo make money?")),
    (_state(), "hi",
     _decision("GENERAL_CHAT", reply="Hi! I can help you find a room. Which city are you looking in?")),
    (_state(rooms=True, city="Bangalore", last_bot=_SHOWN), "thanks",
     _decision("GENERAL_CHAT", reply="You're welcome! Tell me if you want to refine the search.")),
    (_state(rooms=True, city="Bangalore", last_bot=_SHOWN), "book this room",
     _decision("UNSUPPORTED_ACTION",
               reply="I can't book rooms, but I can help you compare the ones shown or search again.")),
    (_state(), "what is the speed of light",
     _decision("OUT_OF_SCOPE",
               reply="That's outside what I can help with. I can help you find a room, though.")),
]


def _dump(value: Any, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent)


def _compact(value: Any) -> Any:
    """Drop null / empty / false entries so examples stay short."""
    if isinstance(value, dict):
        out = {k: _compact(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if v not in (None, [], {}, False, "")}
    return value


def _render_examples() -> str:
    blocks = []
    for state, message, decision in FEW_SHOT_EXAMPLES:
        shown = _compact(state)
        if "rooms_on_screen" in shown:
            shown["rooms_on_screen"] = "<the 3 example rooms>"
        blocks.append(
            f"<verified_data>{_dump(shown)}</verified_data> "
            f"<user_message>{message}</user_message>\n"
            f"Output: {_dump(_compact(decision))}"
        )
    return "\n\n".join(blocks)


_PROMPT_CACHE: list[str] = []


def system_prompt() -> str:
    """The full system prompt (built once; the reused rules load lazily)."""
    if not _PROMPT_CACHE:
        _PROMPT_CACHE.append(
            "You are the decision layer of the Homilivo accommodation chatbot. "
            "For each user message you return ONE JSON object describing what the user wants. "
            "You do not write the final answer; code validates your JSON and acts on it.\n\n"
            f"INTENTS\n{_INTENT_LINES}\n\n"
            f"OPERATIONS (only for ROOM_FOLLOWUP)\n{_OPERATION_LINES}\n\n"
            f"JSON SCHEMA\n{_SCHEMA}\n\n"
            f"RULES\n{_RULES}\n\n"
            f"{reuse.get_fact_grounding_rule()}\n"
            f"{reuse.get_prompt_injection_rule()}\n"
            "Each input has the conversation state inside <verified_data> and the user's "
            "message inside <user_message>. Examples:\n\n"
            f"{_render_examples()}"
        )
    return _PROMPT_CACHE[0]


def __getattr__(name: str) -> str:
    """``prompts.SYSTEM_PROMPT`` resolves lazily, so import needs no app code."""
    if name == "SYSTEM_PROMPT":
        return system_prompt()
    raise AttributeError(name)


# ---- state summary ------------------------------------------------------


def _room_view(index: int, room: dict[str, Any]) -> dict[str, Any]:
    price = room.get("pricePerMonth", room.get("price"))
    return {
        "index": index,
        "title": room.get("title"),
        "price": price,
        "type": room.get("roomType", room.get("type")),
        "location": room.get("location"),
    }


def _trim(value: Any, limit: int) -> Any:
    return value[:limit] if isinstance(value, str) else value


def _size(summary: dict[str, Any]) -> int:
    return len(_dump(summary))


def build_state_summary(state: dict[str, Any] | None) -> dict[str, Any]:
    """Compact, sanitised view of the conversation for the model.

    ``state`` is a plain dict with the keys current_search, pending_question,
    last_bot_message, rooms_on_screen (raw room dicts), selected_room_index and
    recent_turns (list of {role, content}). Every string is passed through the
    reused delimiter neutraliser; rooms go through the reused sanitiser and are
    cut down to index, title, price, type and location. The result stays within
    config.MAX_LLM_CONTEXT_CHARS, dropping the oldest history first.
    """
    state = state or {}
    sanitize = reuse.get_sanitize_room()
    neutralize = reuse.get_neutralize()

    raw_rooms = state.get("rooms_on_screen") or []
    rooms = [
        _room_view(i, sanitize(room))
        for i, room in enumerate(raw_rooms[: config.MAX_ROOMS_IN_PROMPT], start=1)
        if isinstance(room, dict)
    ]

    turns = [t for t in (state.get("recent_turns") or []) if isinstance(t, dict)]
    turns = turns[-2 * config.HISTORY_TURNS_IN_PROMPT :] if config.HISTORY_TURNS_IN_PROMPT else []
    turns = [
        {"role": t.get("role"), "content": _trim(t.get("content"), config.HISTORY_MESSAGE_CHARS)}
        for t in turns
    ]

    search = state.get("current_search")
    summary: dict[str, Any] = {
        "current_search": search if isinstance(search, dict) else None,
        "pending_question": state.get("pending_question"),
        "last_bot_message": _trim(state.get("last_bot_message"), config.LAST_BOT_MESSAGE_CHARS),
        "rooms_on_screen": rooms,
        "selected_room_index": state.get("selected_room_index"),
        "recent_turns": turns,
    }
    summary = neutralize(summary)

    limit = config.MAX_LLM_CONTEXT_CHARS
    while _size(summary) > limit:
        if summary["recent_turns"]:
            summary["recent_turns"].pop(0)
        elif len(summary["rooms_on_screen"]) > 1:
            summary["rooms_on_screen"].pop()
        elif summary["last_bot_message"]:
            summary["last_bot_message"] = summary["last_bot_message"][: len(summary["last_bot_message"]) // 2]
        elif summary["rooms_on_screen"]:
            summary["rooms_on_screen"].pop()
        else:
            summary["current_search"] = None
            summary["pending_question"] = _trim(summary["pending_question"], 100)
            break
    return summary


def build_messages(user_message: str, state_summary: dict[str, Any]) -> list[dict[str, str]]:
    """Chat messages for one decision call."""
    neutralize = reuse.get_neutralize()
    text = neutralize(str(user_message or ""))[: config.MAX_USER_MESSAGE_CHARS]
    # Summaries from build_state_summary are already neutralised; do it again
    # for dicts built elsewhere (cheap and idempotent).
    summary = neutralize(state_summary or {})
    user_content = (
        f"<verified_data>\n{_dump(summary)}\n</verified_data>\n"
        f"<user_message>\n{text}\n</user_message>"
    )
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_content},
    ]
