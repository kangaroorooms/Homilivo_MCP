"""Label space and pipeline-output -> label mapping for the intent-classifier PoC.

Gold labels describe what the user SHOULD get, independent of what the current
pipeline does. Labels reuse existing operation names from
app/experiments/agent_modes.py::detect_room_operation where they exist.
"""

LABELS = [
    "NEW_SEARCH", "REFINEMENT", "SHOW_MORE_OPTIONS", "REJECT_CURRENT",
    "LOWEST_PRICE", "HIGHEST_PRICE", "ROOM_PRICE", "ROOM_TYPE", "LOCATION",
    "AMENITY", "STUDENT_SUITABILITY", "FAMILY_SUITABILITY",
    "PROFESSIONAL_SUITABILITY", "ROOM_DETAILS", "ROOM_TABLE", "COMPARE_ROOMS",
    "ACKNOWLEDGMENT", "GREETING", "CLOSING", "PLATFORM_INFO",
    "UNSUPPORTED_ACTION", "OUT_OF_SCOPE", "UNCLEAR",
]

DEFINITIONS = {
    "NEW_SEARCH": "wants rooms; starts (or restarts) a search, even mid-conversation",
    "REFINEMENT": "changes constraints of the CURRENT search (price cap, area, type, 'cheaper ones')",
    "SHOW_MORE_OPTIONS": "wants other/more rooms than the ones shown, no new constraint",
    "REJECT_CURRENT": "says shown rooms are unsuitable, no new constraint given",
    "LOWEST_PRICE": "asks for the cheapest of the shown rooms",
    "HIGHEST_PRICE": "asks for the most expensive of the shown rooms",
    "ROOM_PRICE": "asks the rent/price of a shown room",
    "ROOM_TYPE": "asks the type (1bhk/pg/single...) of a shown room",
    "LOCATION": "asks where a shown room is / which is nearest somewhere",
    "AMENITY": "asks about facilities of a shown room (wifi, ac, parking...)",
    "STUDENT_SUITABILITY": "asks which shown room suits students",
    "FAMILY_SUITABILITY": "asks which shown room suits a family",
    "PROFESSIONAL_SUITABILITY": "asks which shown room suits working professionals",
    "ROOM_DETAILS": "asks for details/description/availability of one specific or all shown rooms, or picks one",
    "ROOM_TABLE": "asks for a table/list overview of shown rooms",
    "COMPARE_ROOMS": "asks to compare two or more shown rooms",
    "ACKNOWLEDGMENT": "ok/thanks-style reply with no request and no goodbye",
    "GREETING": "hello-style opener",
    "CLOSING": "ends the conversation / says goodbye",
    "PLATFORM_INFO": "asks about the chatbot or the Homilivo platform/policy",
    "UNSUPPORTED_ACTION": "asks for a write action the bot cannot do (book, pay, call, schedule, cancel)",
    "OUT_OF_SCOPE": "clear request unrelated to accommodation",
    "UNCLEAR": "no recoverable intent (empty, noise, fragment, pure punctuation)",
}
assert set(DEFINITIONS) == set(LABELS) and len(LABELS) == 23

# ---------------------------------------------------------------------------
# Existing pipeline output -> label.
# Pipeline output = (guardrail_allowed, guardrail_intent, route.message_type,
#                    route.operation). Defined here so (a)/(b)/(c) compare fairly.
# ---------------------------------------------------------------------------
ROUTE_TO_LABEL = {
    "GREETING": "GREETING",
    "CHATBOT_INFO": "PLATFORM_INFO",
    "PLATFORM_KNOWLEDGE": "PLATFORM_INFO",
    "EXIT": "CLOSING",
    "ACKNOWLEDGMENT": "ACKNOWLEDGMENT",
    "NEW_SEARCH": "NEW_SEARCH",
    "SEARCH_REFINEMENT": "REFINEMENT",
    "UNSUPPORTED_ACTION": "UNSUPPORTED_ACTION",
    # "UNKNOWN" -> handled in pipeline_label(): with LLM stubbed it ends as
    # OUT_OF_SCOPE (process_message's fallback_intent None branch)
}

# route ROOM_FOLLOWUP + operation -> label
OPERATION_TO_LABEL = {
    "LOWEST_PRICE": "LOWEST_PRICE",
    "RANKED_LOWEST_PRICE": "LOWEST_PRICE",       # partial: "2nd cheapest"
    "HIGHEST_PRICE": "HIGHEST_PRICE",
    "RANKED_HIGHEST_PRICE": "HIGHEST_PRICE",     # partial
    "ROOM_PRICE": "ROOM_PRICE",
    "ROOM_TYPE": "ROOM_TYPE",
    "LOCATION": "LOCATION",
    "AMENITY": "AMENITY",
    "STUDENT_SUITABILITY": "STUDENT_SUITABILITY",
    "FAMILY_SUITABILITY": "FAMILY_SUITABILITY",
    "PROFESSIONAL_SUITABILITY": "PROFESSIONAL_SUITABILITY",
    "ROOM_DETAILS": "ROOM_DETAILS",
    "ROOM_TABLE": "ROOM_TABLE",
    "ORDINAL_1": "ROOM_DETAILS",                 # partial: picks a room
    "ORDINAL_2": "ROOM_DETAILS",
    "ORDINAL_3": "ROOM_DETAILS",
    "LAST_ROOM": "ROOM_DETAILS",
}

# Pipeline outputs with NO clean mapping (reported in the README / step report):
NO_CLEAN_MAPPING = {
    "ROOM_FOLLOWUP/None":
        "operation=None is used both for 'show alternatives' and any unclassified follow-up; "
        "cannot distinguish SHOW_MORE_OPTIONS / REJECT_CURRENT / COMPARE_ROOMS. Mapped to "
        "SHOW_MORE_OPTIONS (best guess); REJECT_CURRENT and COMPARE_ROOMS may be unreachable "
        "as distinct operations (comparison is handled inside the follow-up answer, not routed).",
    "ROOM_FOLLOWUP/ORDINAL_n,LAST_ROOM":
        "pipeline says WHICH room, not WHAT is asked -> mapped to ROOM_DETAILS (lossy).",
    "ROOM_FOLLOWUP/RANKED_*":
        "rank detail is lost by mapping to LOWEST_PRICE / HIGHEST_PRICE.",
    "CLARIFICATION_RESPONSE":
        "answer to a pending-search question; no matching label. Mapped to REFINEMENT.",
    "CONVERSATION_META":
        "'what did I ask before' queries; no label in the requested space. Mapped to "
        "OUT_OF_SCOPE-adjacent: left UNMAPPED (counted wrong unless gold is absent).",
    "UNKNOWN":
        "router could not decide; in production an LLM decides. With LLM stubbed -> OUT_OF_SCOPE.",
    "Guardrail rejection reasons":
        "the guardrail returns one generic reject; it cannot say UNCLEAR vs OUT_OF_SCOPE vs "
        "UNSUPPORTED. Rejection -> OUT_OF_SCOPE (UNCLEAR for blank/whitespace input).",
    "UNCLEAR":
        "the pipeline has no UNCLEAR output at all.",
}


def pipeline_label(guard_allowed, route_type, operation, blank=False):
    """Map one pipeline output to a label. guard_allowed=None means 'guard bypassed'."""
    if guard_allowed is False:
        return "UNCLEAR" if blank else "OUT_OF_SCOPE"
    if route_type == "ROOM_FOLLOWUP":
        return OPERATION_TO_LABEL.get(operation, "SHOW_MORE_OPTIONS") if operation else "SHOW_MORE_OPTIONS"
    if route_type == "CLARIFICATION_RESPONSE":
        return "REFINEMENT"
    if route_type == "CONVERSATION_META":
        return None  # unmapped
    if route_type == "UNKNOWN":
        return "UNCLEAR" if blank else "OUT_OF_SCOPE"
    return ROUTE_TO_LABEL.get(route_type)
