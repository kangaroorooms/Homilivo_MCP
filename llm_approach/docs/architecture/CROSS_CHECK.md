# Box-to-code cross-check (llm_approach v2, branch experiment/llm-approach_v2)

Evidence is the source and the trace test `llm_approach/tests/test_architecture_conformance.py` (run with `-s` to print every trace).
Earlier versions of this file, the README and V2_PLAN were not used as evidence.

The test runs real turns through `pipeline.handle_turn` (or `server_app.run_turn`, and the real `/ws/chat` and `POST /chat` routes
for the entry checks) with spies on the function behind each box. It asserts the exact order of the calls. The decision LLM is the
scripted stand-in from `conftest.pipe` (turn S uses the real `llm_client.decide` with a fake Gemini client), the rooms are the offline
fixture (turn A uses the real in-process MCP tool with a fake `urlopen`), pricing and the FAQ answer are fakes, and the Redis store is
fakeredis. Nothing in the source was edited.

## Test turns

| Turn | What it runs |
|---|---|
| A | new room search on the live path: MCP tool, rooms API (fake urlopen), API-data guardrail |
| B | refinement by budget |
| C1 / C2 / C3 | refinement to another city after a filtered search (the question); reply "same filters"; reply "new search" |
| D | follow-up on a displayed room |
| E | recall (name) |
| F | chat question: second LLM call |
| G | platform FAQ |
| H1 / H2 / H3 / H4 / H5 | owner contact: English, Hinglish, with PLATFORM_INFO, with GENERAL_CHAT (not owner contact), with low confidence (Clarify wins) |
| I1 to I5 | clarify: low confidence, UNCLEAR, rejected city slot, no city (search handler asks), unclear room reference |
| J1 / J2 | blocked by the app input guardrail / by extra_security |
| K | model gives no decision: rule-based fallback |
| L1 / L2 | reply refused by the output guardrail / link in a reply refused before it |
| M1 / M2 / M3 | over 500 characters; NFKC makes it longer; invisible characters do not count |
| N | turn over the 40 s limit (the sleep is replaced; no real waiting) |
| O1 to O6 | semantic gate: enforce blocks; shadow only records; off; rooms on screen; pending question; guardrails off |
| R | degraded-memory notice |
| S | real `llm_client.decide`: bad JSON, retry once, Pydantic parse |
| T1 to T4 | WebSocket turn from entry checks to streaming; IP limit twice; entry refusals; POST /chat |
| U | state load/save under the store, Redis store (fakeredis) |

## Box -> code -> proof

| Box / arrow | Code | Proved by turn |
|---|---|---|
| Chat widget (React) | website widget `newkangaroorooms/src/services/chatbot/chatbotWebSocket.ts` (outside this repo; named in PLAN.md) or `frontend-demo/src/useChatSocket.ts:26` (this repo, a reference client) | not testable here; the backend only sees `/ws/chat` (T1) |
| WebSocket, POST /chat | server_app.py `ws_chat`, `chat` | T1, T4 |
| FastAPI route `ws_chat -> run_turn` | server_app.py `run_turn` | T1, T2, T4 |
| 40 s limit (on the route, not an entry check) | server_app.py `run_turn` -> `turn_deadline.run(..., config.turn_deadline_s())` | N (seconds == 40.0), T1 |
| Entry: IP rate limit | `_rl.check_rate_limit` (on connect and per message) | T1, T2 |
| Entry: message 1 to 500, valid mode (WebSocket only) | `ws_chat` inline check, raw length | T3 (error event, pipeline not entered) |
| Entry: session-id format | `normalize_session_id` | T1, T4 |
| Entry: session limit, new-session limit | `check_session_rate_limit`, `_new_session_allowed` | T1, T3b, T4 |
| Entry: server ready (else 503 / error event) | `_startup["loading"]` | T3-loading |
| Input normalisation | pipeline.py `handle_turn`: `textclean.normalize_input`, empty, `MAX_USER_MESSAGE_CHARS` on the normalised text | M1, M2, M3 |
| Session state load | `store.turn` (lock, load) | every pipeline turn; U |
| Security filters | `_input_blocked`: `extra_security.is_extra_blocked`, then app guardrail | J1, J2 (block), all others (pass) |
| Semantic gate | `_semantic_gate`, `semantic_gate.evaluate` | O1 to O6 |
| Prompt build + LLM decision | `prompts.build_state_summary`, `llm_client.decide` | all; S for the real call |
| Retry once on bad JSON, Pydantic parse | `llm_client._decide_on`, `schema.parse_decision` | S |
| Validators | `validators.validate_slots`, first statement of `_decide_action` | A to I |
| Intent router | `_decide_action` | A to I |
| Platform FAQ, Grounded answer | `platform.handle_platform_info`, `reuse.get_answer_platform_question` (a fake here) | G |
| Chroma FAQ index | rag/store.py | UNVERIFIED (not reached by any turn) |
| State-based answers | `followup.handle_followup`, `recall.handle_recall`, `conversation.handle_*` | D, E, H4, L1 |
| Conversation answerer, Answer check | `conversation_answerer.answer_question`, `llm_client.generate_json`, `validators.validate_conversation_answer`, then `extra_security` and `find_unsafe` on the answer | F |
| Room search | `search.handle_new_search`, `handle_refinement` | A, B, I4 |
| Parameter guardrail | `search._check_arguments` | A, B, C2, K |
| MCP tool, rooms API, API-data guardrail | `rooms_client._mcp_request`, `urlopen` (fake), `validate_room_data` | A (the HTTP GET itself is a fake) |
| Owner contact (a post-LLM check, not an intent) | `_decide_action`: `intent in {UNSUPPORTED_ACTION, PLATFORM_INFO}` and `owner_contact.wants_owner_contact`, after the clarify decision | H1, H2, H3; H4 and H5 prove the conditions |
| Pricing API, Plans check | `pricing_client.fetch_plans` (fake httpx), `validate_plans` | H1 to H3 (the real HTTP GET is a fake) |
| Clarify | `clarify.handle_unclear` (UNCLEAR, low confidence), `_clarify_rejected` / `_reference_problem` inside the router | H5, I1, I2, I3, I5 |
| City-change question | `search._ask_city_change`, called by `handle_refinement` | C1 |
| City-change answer (top of the router) | `search.handle_city_change_answer` | C2, C3 |
| Rule-based fallback and its arrow to Response checks | `fallback.run_fallback`; its result goes on to `_output_ok` and the save | K |
| Response checks | `_output_ok`: `reply_safety.find_unsafe`, output guardrail | L1, L2, all others |
| State save, degraded notice | `record_turn`, store exit; notice added after the response check | R, U (Redis save) |
| Observability | `_record_observability`, `metrics.record` | all |
| WebSocket reply | `typing`, `token`, `done` | T1 |
| Session store (memory / Redis, 1 h, 1000) | config.py `SESSION_TTL_SECONDS`, `SESSION_MAX_COUNT` | U for the store path; TTL and cap are read from config, not asserted by a turn |
| Gemini API (limiter, 5xx retry, breaker, Groq fallback) | providers.py, gemini_limits.py, llm_resilience.py | S reaches the SDK with a fake client; limiter, breaker and Groq fallback are not traced |

Not in llm_approach (searched `langgraph|langchain` in all non-test .py files: no matches): LangGraph router, Pydantic extraction step.
Pydantic is used only to parse the model's JSON into `Decision` (schema.py) and for the API models in server_app.py.

## What was corrected in the diagram (this audit)

- React box named only `frontend-demo`; the website widget in `newkangaroorooms` is the other client, and its code is not in this repo.
- The 40 s limit was drawn as an entry check; it wraps the turn in `run_turn` (it now sits on the route box).
- Message length is checked twice: raw length (WebSocket only) at entry, and on the NFKC-normalised text in the pipeline.
- Validators are the first statement inside `_decide_action`, not a separate step before it.
- Owner contact is not an intent. It is a check inside `_decide_action` for two intents, after the clarify decision.
- City-change question is produced by the search handler (a refinement), not by Clarify. Its reply is read at the top of the router.
- The rule-based fallback now has an arrow into Response checks (its result is checked and saved like any other reply).
- Semantic gate label now states all conditions and the mode behaviour.
- Clarify box now lists what really reaches it; "missing city" on a search is asked by the Room search handler.
- The second LLM call is for CONVERSATION_QUESTION (and the recall safety net), noted in the legend.
- Degraded-memory notice added to State save. Bad JSON: retry once, then no decision, then the fallback.

## Real behaviour the diagram does not draw (kept out to stay readable)

Session busy (lock wait timeout -> fixed reply, no LLM call), the "yes" help message, stored-fact acknowledgements, rate-limit and
breaker replies, 5xx retry. They are in the code; a reader should not assume they are absent. The POST route checks the new-session limit
before the session limit; the WebSocket route does the reverse (T1, T4).
