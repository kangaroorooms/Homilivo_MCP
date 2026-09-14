# Homilivo Chatbot — Architecture Reference

This document reflects the code as it actually runs today. It is generated from a
direct read of the source (no docs, no plans) and is a reference, not a tutorial.

## 1. Overview

Homilivo Chatbot is a room-search conversational assistant that lets a user find
rooms/PGs/apartments via natural-language chat, backed by the Homilivo rooms API.
It is served by a FastAPI backend (REST `/chat` + WebSocket `/ws/chat`) wrapping a
deterministic LangGraph router in front of a hand-written conversational engine
(`app/experiments/agent_modes.py`), with four response modes: a templated FAST
mode (no LLM) and three real-LLM modes (Qwen/Gemma via local Ollama, Groq cloud).
A React/TypeScript demo frontend (`frontend-demo/`) is the reference client; there
is no voice input anywhere in the codebase — text only, WebSocket with REST fallback.

## 2. Complete Request Flow

```
User types in ChatWindow.tsx (frontend-demo/src/ChatWindow.tsx)
        │
        ▼
useChatSocket.sendMessage(text, mode)   [SESSION: appends local user ChatMessage]
        │
        ├── ensureSocket(): WebSocket already OPEN, or opens one
        │     │
        │     ├── SUCCESS → sendViaSocket(ws, text, mode)
        │     │      → ws.send({message, session_id, mode})
        │     │
        │     └── FAILURE (connect timeout/error) → status="rest-fallback"
        │           → sendViaRest() → POST /chat   (REST PATH, see below)
        │
        └── socket send not "handled" (socket closed mid-turn)
              → falls back to sendViaRest() → POST /chat

════════════════════ SERVER SIDE (shared by WS and REST) ════════════════════

REST /chat                                  WS /ws/chat
  │ Depends(enforce_rate_limit)                │ token required as query param
  │  → per-IP fixed-window limiter             │   (else close code 4401)
  │  → 429 + Retry-After if exceeded           │ → check_rate_limit(client_ip)
  │                                             │   (else close code 4429)
  │ Depends(bearer_scheme) — HTTPBearer         │ await websocket.accept()
  │  → Authorization: Bearer <token>            │ loop per turn:
  │  → _resolve_homilivo_token(): real BYOT     │   send {"event":"typing"}
  │    token OR redeem_demo_token() opaque      │   set_kangaroo_token(...)
  │    demo credential                          │   (same auth resolution as REST)
  │  → set_kangaroo_token(token) [per-request   │
  │    ContextVar, never a process-global]      │
  ▼                                             ▼
                    both converge on: app.api._run_chat_turn(session_id, mode, message, on_token)
        │
        ▼
[GUARDRAIL] is_security_blocked(message)   (app/api.py → app/guardrails/input_guardrail.py)
        │    Context-free regex match on BLOCKED_PATTERNS: api key/token,
        │    bearer/access/refresh token, password, secret, credentials,
        │    kangaroo_token, hack/malware/ransomware/exploit, sql injection
        │
        ├── MATCH → record_turn(status="blocked") → return fixed
        │           "I can only help with accommodation and room searches."
        │           [turn ends here — no session write, no graph entry]
        │
        └── NO MATCH → continue
              │
              ▼
        [SESSION] SESSIONS.get(session_id) → ConversationState
                  (in-memory or Redis, app/session_store.py)
                  → append {"role":"user", content} to conversation_history
              │
              ▼
        async with Client(mcp) as client:
        run_graph(user_query, mode, client, state, on_token)
        (app/graph/router.py — LangGraph, compiled once, cached)
              │
              ▼
        ┌────────────────────────────────────────────────────────┐
        │ NODE: fast_path_node (app/graph/nodes/fast_path.py)     │
        │   route = route_message(user_query, conversation_state)│
        │   (app/experiments/agent_modes.py — deterministic,     │
        │    NEVER calls an LLM, NEVER touches session state)    │
        └────────────────────────────────────────────────────────┘
              │
              │  route.message_type mapped to exactly one of 4 branches
              │  (full route_message() priority order in Section 4)
              │
     ┌────────┼──────────────┬───────────────────┬───────────────────┐
     ▼        ▼               ▼                    ▼
  BRANCH_FAQ  BRANCH_CONVERSATION  BRANCH_API_REQUEST   BRANCH_LLM_FALLBACK
  faq_node()  conversation_node()  api_request_node()   llm_fallback_node()
     │        │                    │                    │
     └────────┴────────────────────┴────────────────────┘
                              │
                              ▼
                run_graph() returns outcome["result"]
                              │
                              ▼
        [SESSION] _apply_result_to_state(): writes response into
        conversation_history, previous_rooms, previous_search,
        selected_room(_id), pending_search back onto ConversationState
        → SESSIONS.save(session_id, state)
                              │
                              ▼
                    record_turn() [observability only — no message
                    text or credentials logged]
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
             REST: return           WS: send {"event":"done", ...turn}
             ChatResponse JSON      (see STREAMING PATH below for how
                                    tokens got there)
```

### BRANCH_FAQ — `faq_node()`
```
faq_node
  │
  ▼
[EMBEDDING] answer_platform_question(user_query)  (app/knowledge/answer.py)
  │  retrieve() → Chroma vector search over app/knowledge/documents/*.md,
  │  cosine similarity, top_k=3, MIN_RELEVANCE_SCORE=0.35
  │
  ├── grounded=True → final_text = joined chunk texts
  │     → [GUARDRAIL] print_validated_response() → validate_ai_response()
  │         PASS → response=final_text, sources=doc titles
  │         FAIL → response stays None
  │
  └── grounded=False OR exception during retrieval
        → FALLBACK: process_message() — its own PLATFORM_KNOWLEDGE branch,
          which on no match returns the fixed "I'm sorry, I don't have
          verified information about that at the moment..." text
```

### BRANCH_CONVERSATION — `conversation_node()`
```
conversation_node
  │
  ├── route.message_type == GREETING
  │     → GREETING_RESPONSE = "Hello! I'm the Homilivo room-search
  │       assistant. How can I help you find a room today?"
  │     → [GUARDRAIL] print_validated_response()
  │
  └── CHATBOT_INFO / EXIT / UNSUPPORTED_ACTION
        → FALLBACK: process_message() static routes:
            CHATBOT_INFO       → CHATBOT_INFO_RESPONSE
            UNSUPPORTED_ACTION → UNSUPPORTED_ACTION_RESPONSE (no LLM, no MCP)
            EXIT               → FAREWELL_RESPONSE / ACKNOWLEDGMENT_RESPONSE
```

### BRANCH_API_REQUEST / BRANCH_LLM_FALLBACK — both call `process_message()` unchanged
```
process_message(user_query, mode, client, state, on_token, route)
  │
  ▼
Stage 1 — classify_conversational_closing() → FAREWELL/ACKNOWLEDGMENT/None
  │  MATCH → respond + validate + return (no guardrail run)
  ▼
Stage 2 — [GUARDRAIL] validate_user_input(user_query)
  │  Security pattern check, then
  │  [EMBEDDING] classify_user_intent() — SentenceTransformer cosine
  │  similarity vs ROOM_SEARCH/CHATBOT_INFO/GREETING/OUT_OF_SCOPE example
  │  sets (threshold 0.48, margin 0.03). Result deferred.
  ▼
Stage 3 — routing
  │  contextual_message = message_type in {FOLLOWUP, REFINEMENT,
  │  CLARIFICATION, PLATFORM_KNOWLEDGE, UNSUPPORTED_ACTION}
  │
  ├── guardrail REJECTED it, NOT contextual_message:
  │     → [LLM] fallback_llm_intent() via app/llm/gemma.py classify_intent()
  │       ollama.chat(model="gemma3:4b", format="json")
  │       ├── error/exception → None
  │       ├── confidence < 0.70 → treated as None
  │       ├── None → OUT_OF_SCOPE_RESPONSE, return
  │       └── GREETING/CHATBOT_INFO/ROOM_SEARCH → route overwritten, continue
  │
  ├── guardrail REJECTED it, contextual_message=True: block overridden
  │
  └── guardrail PASSED: continue
  ▼
Stage 4 — static response routes (dead-in-practice for GREETING/
  │        PLATFORM_KNOWLEDGE — execute.py intercepts those earlier in the
  │        real graph pipeline; reachable for CLI/tests/legacy callers)
  │
  ├── GREETING            → GREETING_RESPONSE
  ├── CHATBOT_INFO        → CHATBOT_INFO_RESPONSE
  ├── UNSUPPORTED_ACTION  → UNSUPPORTED_ACTION_RESPONSE
  ├── PLATFORM_KNOWLEDGE  → [EMBEDDING] answer_platform_question()
  └── UNKNOWN             → tries guardrail's own classification first,
        else [LLM] fallback_llm_intent() → GREETING/INFO (respond+return),
        NEW_SEARCH (fall through), else UNKNOWN_RESPONSE
  ▼
Stage 5 — ROOM_FOLLOWUP (no MCP call on this path — pure memory/LLM)
  │  resolve_room_reference() — ordinal/superlative/pronoun/explicit
  │  resolution → generate_response(is_followup=True) [see Section 5]
  │  → resolve_amenity_filter_matches() may set a per-turn displayed_rooms
  │  → [GUARDRAIL] print_validated_response()
  ▼
Stage 6 — CLARIFICATION / SEARCH_REFINEMENT / NEW_SEARCH → extraction
  │  extract_search_parameters() [regex, app/fast_parser.py]
  │    ├── exception → SILENT FAILURE (response=None, no message set)
  │    └── result "unhelpful" → [LLM] extract_search_parameters_via_llm():
  │          1. Gemma (ollama, JSON mode)
  │          2. on failure, Groq (JSON mode, model "openai/gpt-oss-20b")
  │          3. result force-validated via strict Pydantic
  │             app.schemas.chat.SearchParameters (extra="forbid")
  │          4. validation failure → discarded, treated as no LLM result
  ▼
Stage 7 — required city check
  │  city missing → pending_search set, respond "Sure. Which city are you
  │  looking for?", return [no MCP call]
  ▼
Stage 8 — [GUARDRAIL] validate_search_arguments(search_request)
  │  FAIL → SILENT FAILURE (response=None, no message set)
  ▼
Stage 9 — [EXTERNAL API] await client.call_tool("search_rooms", search_request)
  │  MCP tool → app/server.py search_rooms() → GET
  │  https://homilivo.onrender.com/api/rooms, Authorization: Bearer <token>,
  │  up to 3 attempts with exponential backoff on 5xx/timeout, 10s timeout
  │    ├── exception during call_tool() → HOMILIVO_UNAVAILABLE_RESPONSE
  │    └── tool returns success=False        → HOMILIVO_UNAVAILABLE_RESPONSE
  ▼
Stage 10 — parse mcp_result.content[0].text as JSON
  │  malformed/empty → HOMILIVO_UNAVAILABLE_RESPONSE
  ▼
Stage 11 — extract_rooms_from_response(api_data) → rooms list
  ▼
Stage 12 — [GUARDRAIL] validate_room_data(rooms)
  │  FAIL → SILENT FAILURE (response=None, no message set)
  ▼
Stage 13 — [SESSION] rooms/search saved, pending_search cleared,
  │  selected_room(_id) cleared, search_performed=True
  ▼
Stage 14 — generate_response(is_followup=False) [see Section 5]
  │  exception → SILENT FAILURE (response=None, no message set)
  ▼
Stage 15 — [GUARDRAIL] print_validated_response(final_text)
           PASS → response=final_text     FAIL → response stays None
```

### Streaming path — real vs simulated (WS `/ws/chat`)
```
relay = LiveTokenRelay()
turn_task = asyncio.create_task(_run_chat_turn(..., on_token=relay.on_token))
  │
  ├── QWEN/GEMMA/GROQ real model call reaches on_token per chunk
  │   [REAL STREAMING] → relayed live as {"event":"token","text":...}
  │   the instant each chunk is generated by the model
  │
  └── FAST mode, or any deterministic structured answer inside
      generate_response() (ROOM_TABLE, property-interest, room-fact,
      comparison, aggregate answers) — never calls on_token
      → [SIMULATED STREAMING] chunk_for_streaming() splits the ALREADY-
        FINISHED string into word-ish windows, sent with
        STREAM_CHUNK_DELAY_SECONDS (default 0.02s) pacing between chunks

Either way: {"event":"done", **turn} always carries the OUTPUT-GUARDRAIL-
VALIDATED text from print_validated_response() — even if raw, pre-
validation live tokens were already streamed to the client. If validation
had failed, "done" carries the safe fallback text regardless of what was
streamed live. Idle > WS_HEARTBEAT_INTERVAL_SECONDS (default 20s) sends
{"event":"ping"}.
```

### Output guardrail's markdown-table exemption
```
is_markdown_table = response.lstrip().startswith("| ")
  → True: contains_sensitive_information() / contains_internal_information()
    checks are SKIPPED (a room title/amenity containing "secret" must not
    block the whole listing table)
  → length / empty / control-character / 8+ blank-line checks STILL run
```

## 3. LangGraph Branch Table

| Branch | Condition (`route.message_type`) | What actually runs | Fallback |
|---|---|---|---|
| `faq_node` | `PLATFORM_KNOWLEDGE` | `answer_platform_question()` [EMBEDDING retrieval] | `process_message()`, only if ungrounded/exception |
| `conversation_node` | `GREETING` | `GREETING_RESPONSE` returned directly | `process_message()`, for `CHATBOT_INFO`/`EXIT`/`UNSUPPORTED_ACTION` |
| `api_request_node` | `NEW_SEARCH`, `SEARCH_REFINEMENT`, `ROOM_FOLLOWUP`, `CLARIFICATION_RESPONSE` | `process_message()` (full 15-stage pipeline) | none — this *is* the pipeline |
| `llm_fallback_node` | `UNKNOWN` | `process_message()` (which itself may call Gemma via `fallback_llm_intent()`) | none — this *is* the pipeline |

## 4. route_message() Decision Table

Checked in this exact order inside `route_message()`; first match wins.

| Priority | Message Type | Detected By | Branch |
|---|---|---|---|
| 1 | `EXIT` | `should_end_conversation()` — normalized text in farewell/acknowledgment command set | conversation |
| 2 | `GREETING` | `detect_simple_intent()` — exact-match greeting phrase | conversation |
| 2 | `CHATBOT_INFO` | `detect_simple_intent()` — substring-match info phrase | conversation |
| 2 | `NEW_SEARCH` (obvious) | `detect_simple_intent()` — substring-match ("find room", "show rooms"…) | api_request |
| 3 | `UNSUPPORTED_ACTION` | `is_unsupported_action_request()` — imperative booking/reservation/enquiry patterns | conversation |
| 4 | `CLARIFICATION_RESPONSE` | `is_clarification_response()` — `pending_search` set, city missing, message short/locational | api_request |
| 5 | `ROOM_FOLLOWUP` | `is_likely_followup()` (+ `detect_room_operation()` attached) | api_request |
| 6 | `SEARCH_REFINEMENT` | `is_likely_search_refinement()` — `previous_search` set + refinement keyword/bare price/BHK pattern | api_request |
| 7 | `SEARCH_REFINEMENT` (short value) | bare `\d+bhk` / `₹?\d{3,}` / students/family/furnished after a prior search | api_request |
| 8 | `NEW_SEARCH` (catch-all) | `looks_like_general_room_search()` — room vocabulary + `extract_search_parameters()` finds ≥1 field | api_request |
| 9 | `PLATFORM_KNOWLEDGE` | `is_platform_knowledge_question()` — fixed verification/fee/contact/booking keyword list | faq |
| 10 | `UNKNOWN` | none of the above matched (default) | llm_fallback |

## 5. generate_response() Branch Table

Followup dispatch (checked first when `is_followup=True`, shared by all 4 modes), then mode dispatch.

| Condition | Action | Output Type | LLM needed? |
|---|---|---|---|
| `detect_room_operation()=="ROOM_TABLE"` | `format_rooms_as_table()` | Markdown table | No |
| `detect_property_interest()` true | resolve verified city → `fetch_subscription_plans()` [EXTERNAL API] | pricing text, or "please search first" / "wasn't able to retrieve" | No |
| price comparison phrasing | `answer_room_comparison()` | deterministic lowest/highest/ranked-price sentence | No |
| `selected_room` set, room-fact question | `answer_room_fact()` | price/type/location/suitability/amenity sentence | No |
| no selection, set-wide question | `answer_*_across_rooms()` (amenity/prices/types/locations/suitability) | deterministic aggregate sentence | No |
| nothing above matched, `mode=="fast"` | `generate_fast_followup()` / `generate_fast_response()` | templated string | No |
| nothing above matched, `mode=="qwen"` | `generate_qwen_response()` | model text (streamable) | Yes — local Ollama `qwen3:4b` |
| nothing above matched, `mode=="gemma"` | `generate_gemma_response()` | model text (streamable) | Yes — local Ollama `gemma3:4b` |
| nothing above matched, `mode=="groq"` | `generate_groq_response()` | model text (streamable) | Yes — Groq `openai/gpt-oss-20b` |
| `generate_groq_response()` raises any exception | silent fallback to FAST-mode logic | templated string | No (degrades) |
| `mode` not in {fast,qwen,gemma,groq} | — | `raise ValueError("Unsupported mode: ...")` | n/a |

## 6. Component Responsibility Table

| File | Responsibility | Key Functions |
|---|---|---|
| `app/api.py` | FastAPI app, REST `/chat`, WS `/ws/chat`, session/turn orchestration, demo auth, rate limiting wiring | `_run_chat_turn()`, `chat_endpoint()`, `chat_websocket()`, `_resolve_homilivo_token()` |
| `app/graph/router.py` | Compiles/runs the LangGraph state machine wrapping the deterministic router | `build_graph()`, `run_graph()` |
| `app/graph/nodes/fast_path.py` | Maps `route_message()` output onto one of 4 graph branches | `fast_path_node()`, `classify_branch()` |
| `app/graph/nodes/execute.py` | Branch execution: FAQ/Conversation short-circuits, else delegates to `process_message()` | `faq_node()`, `conversation_node()`, `api_request_node()`, `llm_fallback_node()` |
| `app/graph/state.py` | TypedDict schema carried through the graph | `GraphState` |
| `app/experiments/agent_modes.py` | Core conversational engine — routing, extraction, resolution, response generation, all 4 modes, `ConversationState` | `route_message()`, `process_message()`, `generate_response()`, `resolve_room_reference()`, `ConversationState` |
| `app/guardrails/input_guardrail.py` | Pre-routing input validation + semantic intent classification | `validate_user_input()`, `classify_user_intent()`, `is_security_blocked()` |
| `app/guardrails/output_guardrail.py` | Final response validation before it reaches the user | `validate_ai_response()` |
| `app/guardrails/parameter_guardrail.py` | Validates structured search args before the MCP call | `validate_search_arguments()` |
| `app/guardrails/api_data_guardrail.py` | Validates room records returned by the Homilivo API | `validate_room_data()` |
| `app/knowledge/answer.py` | Grounds PLATFORM_KNOWLEDGE questions in retrieved docs | `answer_platform_question()` |
| `app/knowledge/store.py` | Chroma-backed RAG index over markdown docs | `retrieve()`, `ingest()`, `index_stats()` |
| `app/server.py` | MCP tool definition, outbound Homilivo API call, per-request token isolation | `search_rooms()`, `set_kangaroo_token()` |
| `app/fast_parser.py` | Regex-based search-parameter extraction | `extract_search_parameters()` |
| `app/streaming.py` | WS token relay + simulated-chunking helpers | `LiveTokenRelay`, `chunk_for_streaming()` |
| `app/session_store.py` | Per-session `ConversationState` storage (memory or Redis) | `SessionStore()`, `InMemorySessionStore`, `RedisSessionStore` |
| `app/rate_limit.py` | Fixed-window per-IP rate limiting (memory or Redis) | `check_rate_limit()`, `enforce_rate_limit()` |
| `app/experiments/interpreter.py` | Embedding-based fallback classifier for follow-up operation slots | `classify_operation()` |
| `app/embeddings.py` | Single shared SentenceTransformer instance | `get_shared_model()` |
| `app/demo_auth.py` | Opaque demo-token issuance/redemption for the bundled demo page | `issue_demo_token()`, `redeem_demo_token()` |
| `app/observability.py` | Structured logging + in-memory metrics registry | `record_turn()`, `get_metrics_snapshot()` |
| `app/feedback_store.py` | Append-only JSONL feedback log | `record_feedback()` |
| `app/llm/gemma.py` | Gemma-based intent classifier used as the guardrail-rejection fallback | `classify_intent()` |
| `app/schemas/chat.py` | Strict Pydantic schema validating LLM-extracted search params | `SearchParameters` |
| `frontend-demo/src/ChatWindow.tsx` | Chat UI: message list, mode selector, composer | `ChatWindow()` |
| `frontend-demo/src/useChatSocket.ts` | WS connection lifecycle + REST fallback | `useChatSocket()`, `sendMessage()` |

## 7. External Dependencies Table

| Service | URL | When Called | Fallback |
|---|---|---|---|
| Homilivo Rooms API | `https://homilivo.onrender.com/api/rooms` | Every completed room search (`search_rooms` MCP tool), up to 3 attempts with exponential backoff on 5xx/timeout | `HOMILIVO_UNAVAILABLE_RESPONSE` on exception, non-2xx, or `success:false` |
| Homilivo Pricing API | `https://kangaroo-rooms-backend.onrender.com/api/tenant-subscriptions/pricing` (overridable via `PRICING_API_URL`) | `detect_property_interest()` true + a verified city resolved, via `fetch_subscription_plans()` | Never raises — returns `None`, produces "I wasn't able to retrieve subscription plans..." text |
| Ollama (local) — Qwen | local Ollama daemon, model `qwen3:4b` | QWEN mode's model-generation step | Exception propagates as a normal turn error |
| Ollama (local) — Gemma (response gen) | local Ollama daemon, model `gemma3:4b` | GEMMA mode's model-generation step | Same as Qwen |
| Ollama (local) — Gemma (intent classifier) | local Ollama daemon, model `gemma3:4b` | `fallback_llm_intent()` when the input guardrail can't classify a message | Exception/error → treated as no result, degrades to `OUT_OF_SCOPE_RESPONSE` |
| Ollama (local) — Gemma (param extraction) | local Ollama daemon, model `gemma3:4b` | LLM parameter-extraction fallback, tier 1 | On failure, tries Groq next |
| Groq API | `chat.completions.create`, model `openai/gpt-oss-20b` (`GROQ_API_KEY` env) | GROQ mode's model-generation step; also parameter-extraction fallback tier 2 | Response generation: any exception → silent degrade to FAST mode. Extraction: any exception → LLM extraction treated as failed, regex-only result used |
| Chroma (embedded, local) | in-process, persisted under `CHROMA_PERSIST_DIR` (default `data/chroma`) | Every PLATFORM_KNOWLEDGE question, via `retrieve()` | Missing/failed client → empty retrieval, `answer_platform_question()` returns `grounded:False` |
| Redis (optional) | `REDIS_URL` env, active only when `STATE_BACKEND=redis` | Session store, rate limiter, demo-auth token store | Default backend is in-memory; Redis is opt-in |
| Sentence-Transformers (`all-MiniLM-L6-v2`) | loaded locally via `get_shared_model()`, no network call after first download | Input-guardrail intent classification, interpreter fallback classifier, knowledge-base retrieval embeddings | Load failure → each caller degrades independently (guardrail rejects input; interpreter/knowledge-store return `None`/empty) |

## 8. Known Gaps & Dead Code

- **`app/experiments/agent_modes_backup.py`** — untracked file, a duplicate/fork of `agent_modes.py`. Not imported anywhere under `app/` (confirmed by grep — zero references outside a docs markdown file). Pure dead weight.
- **`scripts/legacy_cli.py`** and **`scripts/load_test_ws.py`** — standalone scripts, not part of any request path. `legacy_cli.py` overlaps in purpose with `agent_modes.py`'s own CLI loop.
- **`conversational_main()` and `choose_mode()`** (`agent_modes.py`) plus its `if __name__ == "__main__":` block — the interactive terminal REPL. Only reachable via `python -m app.experiments.agent_modes`; the real API/graph path never calls it.
- **`process_message()`'s own `GREETING` and `PLATFORM_KNOWLEDGE` static branches** — explicitly noted in the code's own comments as dead-in-practice for the real API, because `conversation_node()`/`faq_node()` in `app/graph/nodes/execute.py` intercept those message types first. Still reachable for direct/CLI/test callers of `process_message()`.
- **`_extract_amenity_term()`** — a pure backward-compatibility alias for `resolve_amenity_term()`, kept only for old call sites/tests.
- **Voice input (Deepgram / VoiceInput)** — does not exist anywhere in this codebase. No microphone capture, no speech-to-text integration in `frontend-demo/` or `app/`.
- **`app/server.py`'s `if __name__ == "__main__": mcp.run()`** — standalone MCP server entry point, not used by the FastAPI request path (which imports `search_rooms` directly and drives it via `mcp.Client`).

## 9. Silent Failure Paths

Every place in `process_message()` (`app/experiments/agent_modes.py`) that returns
`result_state["response"] = None` with **no** dedicated user-facing message constant:

1. **Stage 6 — parameter extraction exception.** `extract_search_parameters()` (regex) raises → function returns immediately with `response=None`. No message is ever set for this case.
2. **Stage 8 — parameter guardrail failure.** `validate_search_arguments()` returns `False` → function returns with `response=None`. The guardrail's rejection reason is only printed to stdout, never surfaced to the user.
3. **Stage 12 — room-data guardrail failure.** `validate_room_data()` returns `False` on the Homilivo API's response → function returns with `response=None`. Same stdout-only logging, no user message.
4. **Stage 14 — response-generation exception.** Any exception raised inside `generate_response()` (search path, `is_followup=False`) → function returns with `response=None`, no fallback text set.

All four are distinct from the guardrail/API failure paths that *do* have dedicated
text (`HOMILIVO_UNAVAILABLE_RESPONSE`, `OUT_OF_SCOPE_RESPONSE`, etc.) — in each of
these four cases the caller (`app/api.py`'s `_run_chat_turn()`) receives
`response_text = None` and substitutes its own generic fallback string,
`"Sorry, I couldn't process that request. Please rephrase."`, rather than anything
`process_message()` itself produced.
