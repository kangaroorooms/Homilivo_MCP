"""The full turn: length check, security, state, LLM decision, validation, action,
output guardrail, state update, metrics.

``handle_turn(session_id, message)`` never raises for a turn-level problem: any
exception becomes ``ERROR_REPLY`` and the history stays consistent. Heavy
guardrails (torch) load lazily; call ``preload()`` once at startup to pay that
cost up front.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from llm_approach import (
    config,
    extra_security,
    fallback,
    llm_client,
    metrics as metrics_mod,
    prompts,
    responder,
    reuse,
    validators,
)
from llm_approach.actions import ActionResult, TurnBudget, clarify, conversation, followup, platform, recall, search
from llm_approach.schema import Decision, Intent
from llm_approach.state import ERROR_REPLY, LLMConversationState, SessionStore

logger = logging.getLogger(__name__)

TOO_LONG_REPLY = (
    f"That message is a bit long. Please shorten it to {config.MAX_USER_MESSAGE_CHARS} characters or fewer."
)
EMPTY_REPLY = "Please type a message and I'll help you find a room."
OUTPUT_BLOCKED_REPLY = "Sorry, I can't share that. I can help you find a room, though. Which city are you looking in?"
RATE_LIMIT_REPLY = "I'm getting a lot of requests right now, please try again in a minute."
BLOCKED_PLACEHOLDER = "[message blocked]"  # stored in history instead of the blocked text

_SEARCH_SLOTS = ("city", "bhk", "min_price", "max_price")
_SLOT_NAMES = ("city", "area", "bhk", "min_price", "max_price", "amenity", "ideal_for")

_full_guardrails = True
_store: SessionStore | None = None


@dataclass
class TurnResult:
    reply: str
    rooms: list[dict[str, Any]] | None = None  # frontend cards; None = unchanged
    metrics: dict[str, Any] = field(default_factory=dict)
    success: bool = True  # False for blocked, too-long and error replies
    debug: dict[str, Any] | None = None  # decision details for the CLI; never persisted


# ---- configuration -------------------------------------------------------


def configure(*, full_guardrails: bool = True) -> None:
    """``full_guardrails=False`` skips the heavy app guardrails: only
    ``extra_security`` runs. For quick local tests, never for evaluation."""
    global _full_guardrails
    _full_guardrails = full_guardrails


def get_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store


def reset_session(session_id: str, store: SessionStore | None = None) -> bool:
    return (store or get_store()).reset(session_id)


def preload() -> None:
    """Load the heavy guardrails (torch, embedding model) once, at startup."""
    if not _full_guardrails:
        return
    reuse.get_is_security_blocked()("hello")
    reuse.get_output_guardrail()("Hello")


# ---- guardrails ----------------------------------------------------------


def _input_blocked(message: str) -> bool:
    if extra_security.is_extra_blocked(message)[0]:
        return True
    if _full_guardrails:
        return bool(reuse.get_is_security_blocked()(message)[0])
    return False


def _output_ok(reply: str) -> bool:
    if not _full_guardrails:
        return True
    return bool(reuse.get_output_guardrail()(reply)[0])


# ---- helpers -------------------------------------------------------------


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _log_withheld(store: SessionStore, session_id: str) -> None:
    """A too-long message still counts as a message the user sent: log a placeholder."""
    try:
        store.get_or_create(session_id).record_query(None)
    except ValueError:  # not a session id: nothing to log against
        pass


def _result(
    reply: str, rooms, m: metrics_mod.TurnMetrics, *, success: bool = True, debug=None, started: float
) -> TurnResult:
    m.total_ms = _ms(started)
    return TurnResult(reply=reply, rooms=rooms, metrics=m.to_dict(), success=success, debug=debug)


def _clarify_rejected(
    decision: Decision, rejected: list[str], state: LLMConversationState
) -> ActionResult | None:
    """A slot the search depends on failed validation: ask instead of searching."""
    if decision.intent not in (Intent.NEW_SEARCH, Intent.REFINEMENT):
        return None
    if "city" in rejected:
        if decision.intent is Intent.REFINEMENT and state.current_search.get("city"):
            state.set_pending("clarify", "refinement")
            return ActionResult(search.area_question(state), None, "ask_area")
        state.set_pending("city")
        return ActionResult(clarify.ASK_CITY, None, "ask_city")
    bad = [name for name in ("bhk", "min_price", "max_price") if name in rejected]
    if bad:
        state.set_pending("clarify")
        what = "budget" if any("price" in b for b in bad) else "number of BHK"
        return ActionResult(f"I didn't catch the {what}. Could you say it again?", None, "ask_slot")
    return None


def _reference_problem(decision: Decision, rejected: list[str], state: LLMConversationState) -> ActionResult | None:
    """The model pointed at a room that is not on screen: ask, never guess."""
    if decision.intent is not Intent.ROOM_FOLLOWUP or not state.rooms_on_screen:
        return None
    count = len(state.rooms_on_screen)
    if "room_reference" in rejected:
        return ActionResult(
            f"I only see {count} rooms on screen. Which one do you mean (1 to {count})?", None, "ask_which_room"
        )
    if "compare_references" in rejected:
        return ActionResult(
            f"Which two rooms would you like to compare? Pick numbers from 1 to {count}.", None, "ask_compare"
        )
    return None


_HANDLERS = {
    Intent.NEW_SEARCH: search.handle_new_search,
    Intent.REFINEMENT: search.handle_refinement,
    Intent.SHOW_MORE: search.handle_show_more,
    Intent.ROOM_FOLLOWUP: followup.handle_followup,
    Intent.REJECT_CURRENT: clarify.handle_reject,
    Intent.PLATFORM_INFO: platform.handle_platform_info,
    Intent.GENERAL_CHAT: conversation.handle_general_chat,
    Intent.UNSUPPORTED_ACTION: conversation.handle_unsupported,
    Intent.OUT_OF_SCOPE: conversation.handle_out_of_scope,
    Intent.UNCLEAR: clarify.handle_unclear,
    Intent.CONVERSATION_RECALL: recall.handle_recall,
}
_THREADED = {Intent.NEW_SEARCH, Intent.REFINEMENT, Intent.PLATFORM_INFO}  # blocking I/O


async def _decide_action(
    message: str,
    decision: Decision,
    state: LLMConversationState,
    budget: TurnBudget,
    m: metrics_mod.TurnMetrics,
    stored_facts: dict[str, Any] | None = None,
) -> tuple[ActionResult, dict[str, Any], list[str]]:
    validated, rejected = validators.validate_slots(
        decision.slots,
        rooms_count=len(state.rooms_on_screen),
        has_selected=state.selected_room_index is not None,
        room_reference=decision.room_reference,
        compare_references=decision.compare_references,
    )
    m.slots_valid = [n for n in _SLOT_NAMES if validated.get(n) is not None]
    m.slots_rejected = list(rejected)

    forced = _clarify_rejected(decision, rejected, state) or _reference_problem(decision, rejected, state)
    if forced is not None:
        return forced, validated, rejected

    low_confidence = decision.confidence < config.CONFIDENCE_THRESHOLD
    # Recall questions are answered from state and never go to the generic clarifier: the recall
    # action asks its own question when it has no target. Safety net: a message with a personal
    # recall cue that the model called UNCLEAR or was unsure about is answered the same way.
    net = None
    if decision.intent is not Intent.CONVERSATION_RECALL and (low_confidence or decision.intent is Intent.UNCLEAR):
        net = recall.detect_recall_cue(message)
    if decision.intent is Intent.CONVERSATION_RECALL or net is not None:
        if net is not None:
            state.set_last_query_intent(Intent.CONVERSATION_RECALL.value)
            decision = decision.model_copy(
                update={"intent": Intent.CONVERSATION_RECALL, "recall_target": None, "recall_targets": net}
            )
        return recall.handle_recall(decision, validated, state, budget, user_message=message), validated, rejected
    # needs_clarification on a search is answered by the search handler itself
    # (it asks for the city or area); on any other intent, ask a question.
    asks = decision.needs_clarification and decision.intent not in (Intent.NEW_SEARCH, Intent.REFINEMENT)
    if low_confidence or asks or decision.intent is Intent.UNCLEAR:
        return (
            clarify.handle_unclear(decision, validated, state, budget, user_message=message),
            validated,
            rejected,
        )

    if (decision.intent is Intent.GENERAL_CHAT and not stored_facts
            and clarify.is_bare_reply(message) and clarify.was_generic_prompt(state)):
        # "yes" to our generic question is not small talk: say what we can do instead of repeating it
        return ActionResult(clarify.help_message(state), None, "help"), validated, rejected

    if stored_facts and decision.intent is Intent.GENERAL_CHAT:
        return recall.acknowledge(stored_facts, state), validated, rejected  # fixed wording, not the model's

    handler = _HANDLERS[decision.intent]
    kwargs = {"user_message": message} if decision.intent is Intent.PLATFORM_INFO else {}
    if decision.intent in _THREADED:
        started = time.perf_counter()
        action = await asyncio.to_thread(handler, decision, validated, state, budget, **kwargs)
        m.api_ms = _ms(started)
    else:
        action = handler(decision, validated, state, budget, **kwargs)
    return action, validated, rejected


async def _run_turn(
    message: str, state: LLMConversationState, m: metrics_mod.TurnMetrics
) -> tuple[ActionResult, dict[str, Any] | None]:
    budget = TurnBudget()
    llm_budget = llm_client.LLMBudget()
    summary = prompts.build_state_summary(state.to_prompt_state())
    decision = await llm_client.decide(message, summary, llm_budget)
    m.llm_ms, m.tokens_in, m.tokens_out = llm_budget.latency_ms, llm_budget.tokens_in, llm_budget.tokens_out
    m.llm_calls, m.llm_call_ms = llm_budget.calls, list(llm_budget.call_ms)

    if decision is None and llm_budget.error_class == "RateLimitError":
        # Groq is throttling us: say so, rather than asking the user a misleading question.
        return ActionResult(RATE_LIMIT_REPLY, None, "rate_limited", error_class="RateLimitError"), None
    if decision is None:
        m.fallback_used = True
        started = time.perf_counter()
        action = await asyncio.to_thread(fallback.run_fallback, message, state, budget)
        m.api_ms = _ms(started) if budget.tool_calls else 0.0
        state.set_last_query_intent(Intent.NEW_SEARCH.value if action.tool_called else Intent.UNCLEAR.value)
        return action, None

    m.intent = decision.intent.value
    m.operation = decision.operation.value if decision.operation else None
    m.confidence = decision.confidence
    state.set_last_query_intent(m.intent)
    stored_facts = recall.capture_facts(decision.user_facts, message, state)  # side effect on every intent
    action, validated, rejected = await _decide_action(message, decision, state, budget, m, stored_facts)
    debug = {
        "intent": m.intent,
        "operation": m.operation,
        "slots": decision.slots.model_dump(),
        "validated": {k: v for k, v in validated.items() if v not in (None, [])},
        "rejected": rejected,
        "room_reference": decision.room_reference,
        "confidence": decision.confidence,
        "needs_clarification": decision.needs_clarification,
    }
    return action, debug


# ---- public entry point --------------------------------------------------


async def handle_turn(
    session_id: str, message: str, *, store: SessionStore | None = None
) -> TurnResult:
    started = time.perf_counter()
    m = metrics_mod.TurnMetrics()
    store = store or get_store()
    text = message if isinstance(message, str) else ""

    try:
        if not text.strip():
            m.action = "empty"
            return _result(EMPTY_REPLY, None, m, success=False, started=started)
        if len(text) > config.MAX_USER_MESSAGE_CHARS:
            m.action = "too_long"
            _log_withheld(store, session_id)
            return _result(TOO_LONG_REPLY, None, m, success=False, started=started)

        async with store.turn(session_id, user_message=text) as state:
            logged = False
            queries_before = state.query_count
            try:
                if _input_blocked(text):
                    m.action = "security_block"
                    state.record_query(None)  # placeholder, never the blocked text
                    state.record_turn(BLOCKED_PLACEHOLDER, extra_security.REFUSAL)
                    return _result(extra_security.REFUSAL, None, m, success=False, started=started)

                state.record_query(text)
                action, debug = await _run_turn(text, state, m)
                reply, rooms = action.reply, action.rooms
                m.action, m.tool_called = action.action, action.tool_called
                m.error_class = action.error_class

                if not await asyncio.to_thread(_output_ok, reply):
                    reply, rooms, m.error_class = OUTPUT_BLOCKED_REPLY, None, "output_blocked"
                    m.action = "output_blocked"
            except Exception as exc:
                logger.error("turn failed: %s", type(exc).__name__)
                m.error_class, m.action = type(exc).__name__, "error"
                reply, rooms, debug = ERROR_REPLY, None, None
                if state.query_count == queries_before:
                    state.record_query(None)
                state.record_turn(text, reply)
                return _result(reply, rooms, m, success=False, started=started)

            state.record_turn(text, reply)
            return _result(reply, rooms, m, debug=debug, started=started)
    except Exception as exc:  # bad session id, cancelled store, anything left
        logger.error("turn failed outside the session: %s", type(exc).__name__)
        m.error_class, m.action = type(exc).__name__, "error"
        return _result(ERROR_REPLY, None, m, success=False, started=started)
    finally:
        m.total_ms = m.total_ms or _ms(started)
        metrics_mod.record(m)
