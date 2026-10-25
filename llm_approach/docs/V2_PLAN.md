# Homilivo Chatbot v2: LLM Architecture and Implementation Plan

23 October 2026 · S. Mohamed Faizal · Branch: experiment/llm-approach_v2

v2 keeps the planned architecture stage for stage and replaces only the keyword-based LangGraph router with one validated LLM decision. Every guardrail, the authentication layer, the MCP tool path and the RAG answerer are copied from app/ into llm_approach/ unchanged where possible.

## Target architecture

The shape is the planned architecture: frontend, authentication, input guardrail, a router, three branches, response generation and the output guardrail. Only the router changes: one validated LLM decision picks the branch, and session state feeds it the conversation so far.

Legend: [copied] = code copied from app/ · [llm_approach] = new for the LLM approach

```
React chatbot and WebSocket                      (frontend, unchanged)
        |
FastAPI /ws/chat and authentication              [copied] demo_auth, rate_limit, secret check
        |
Input guardrail                                  [copied] security hard block + semantic gate
        |
LLM decision (replaces the LangGraph router)     [llm_approach] Gemini, one call: intent, search details, room
        |        <-->  Session state             [llm_approach] rooms, filters, pending question,
        |                                                       transcript, facts, per-session lock
Decision validation                              [llm_approach] allowed intent, room index, confidence >= 0.6
        |
        +-- PLATFORM_INFO ----------> FAQ / RAG                 [copied] platform answers
        |
        +-- chat, memory, unclear --> Conversation              [llm_approach] chat, recall, clarify,
        |                                                                      verified chat answers
        +-- search intents ---------> Parameter guardrail       [copied] slot rules
                                        -> MCP tool              [copied] rooms search with timeout
                                        -> Homilivo Rooms API    (external service)
                                        -> API data guardrail    [copied] checks every room
        |
   (all three branches join)
        |
Response generation                              [llm_approach] templates; LLM wording only for chat answers
        |
Output guardrail                                 [copied] validate_ai_response
        |
Save state and reply                             [llm_approach] memory, metrics, WebSocket back to the user
```

A message flows top to bottom. Search intents take the search branch through the parameter guardrail, the MCP tool, the Rooms API and the API data guardrail; platform questions go to FAQ/RAG; chat, memory and unclear messages go to Conversation. Unsafe messages stop at the input guardrail, before any LLM call.

## Stage by stage

Every guardrail and security stage of the plan already exists in app/ and is copied in. Only the router is replaced, and the memory and verified-answer parts are new, with no app/ equivalent. Source names come from the security-parity inventory of 21 October; Phase 0 confirms each file and function before anything is copied.

| # | Planned stage | llm_approach today | v2 plan | Source in app/ |
| --- | --- | --- | --- | --- |
| 1 | React Chatbot, WebSocket | Unchanged frontend, start-frontend-llm.ps1 | Unchanged | Frontend repo |
| 2 | FastAPI /ws/chat | Own server_app.py, which already has its own `_RedactTokenFilter` and `ChatRequest` (500-character cap) | Keep server_app.py; `_normalize_session_id` is copied to security/secrets_check.py (Phase 1) and wired in Phase 2 | api.py: _normalize_session_id (the filter and ChatRequest already match) |
| 3 | Authentication | demo_auth and rate_limit imported through reuse.py | Copy both (done in Phase 1); add the fail-fast secret check, requiring GEMINI_API_KEY (or GROQ_API_KEY when the provider is groq) and KANGAROO_TOKEN; wiring into server_app.py is Phase 2 | demo_auth.py, rate_limit.py, api.validate_secrets |
| 4 | Input Guardrail | Security hard block imported, plus extra_security.py; semantic gate not used | Copy input_guardrail.py whole; hard block and extra_security before the LLM as today. **The semantic gate has no conversation-context parameter and wrongly blocks follow-ups** ("does it have wifi", "show cheaper ones", "under 15k"). **Decided (D2) and built in Phase 3B**: the copy runs only when no rooms are on screen and no question is pending, in shadow mode (log only) by default | input_guardrail.py: is_security_blocked, validate_user_input |
| 5 | LangGraph Router | Keyword router replaced by one LLM decision, validators.py and a dispatcher in pipeline.py | **Replaced**: the router node becomes the LLM decision (prompt, schema-checked JSON, confidence threshold), routing by validated intent. **Decided (D1): keep the current dispatcher, no LangGraph** | Graph shell only; keyword routing not reused |
| 6 | FAQ/RAG | actions/platform.py answers PLATFORM_INFO through app/knowledge, reached by reuse.py | **Copy and decouple**: llm_approach already uses app/knowledge through reuse.py, so copy the retriever, answer path, embeddings.py and the documents (rag/documents/) and give v2 its own CHROMA_PERSIST_DIR | app/knowledge/, app/embeddings.py |
| 7 | Conversation | conversation.py, clarify.py, recall.py, conversation_answerer.py | **Keep** (memory, recall and verified chat answers are new); keep the reused fact-grounding and injection-defence prompt rules | agent_modes: FACT_GROUNDING_RULE, PROMPT_INJECTION_DEFENSE_RULE |
| 8 | API Request, Pydantic Extraction | Slots extracted inside the LLM decision (schema.py, Pydantic) | **Keep**: the decision call is the extraction step, so one call, not two | None |
| 9 | Parameter Guardrail | Re-implemented as validators.validate_slots | Copy parameter_guardrail.py as the single slot check; validators.py keeps only LLM-specific checks (allowed intent, room index, confidence) | parameter_guardrail.validate_search_arguments |
| 10 | MCP Tool | rooms_client.py calls the Rooms API directly | **DONE as MCP (team lead changed the Phase 3B decision, 25 Oct 2026)**: v2 uses the MCP tool for room search, like the main chatbot. `app/server.py` is copied to `tools/mcp_search.py` and `_call_search_rooms_with_timeout` to `tools/mcp_timeout.py`; `rooms_client` calls them through an in-process MCP client. The tool's 3 attempts are the only retry layer; the 15 s whole-search limit stays the outer limit. `LLM_APPROACH_SEARCH_TRANSPORT=direct` keeps the old httpx client (see Stage 10 status) | agent_modes._call_search_rooms_with_timeout, app/server.py |
| 11 | Homilivo API | Rooms API GET | Unchanged | None |
| 12 | API Guardrail | Not used; only a field allowlist and sanitiser in state.clean_room | Copy api_data_guardrail.py (done in Phase 1); run validate_room_data on every Rooms API response before rooms reach state or the prompt. **validate_room_data rejects the whole result on one bad room**, so log the failing field when it is wired in. `sanitize_room_for_llm_context` is in agent_modes, not api_data_guardrail, and is copied to prompt_safety.py | api_data_guardrail.validate_room_data; agent_modes.sanitize_room_for_llm_context |
| 13 | Response Generation | Templates per action; the LLM only drafts chat-memory answers, verified in code | **Keep**; delimiter neutraliser copied (already wrapped) | agent_modes._neutralize_prompt_delimiters |
| 14 | Output Guardrail | validate_ai_response imported through reuse.py | Copy output_guardrail.py unchanged | output_guardrail.validate_ai_response |
| 15 | Session state (not drawn) | state.py: rooms, filters, pending question, transcript, facts, queries, per-session lock | **Keep** | None |
| 16 | Observability (not drawn) | Own metrics.py and per-turn LLM budget | Keep; align field names with production's per-turn record | observability.record_turn |

## Reuse strategy

The app/ code is copied, not imported, so v2 runs without importing app/ (the clean start the team lead asked for) while app/ stays untouched. Six rules keep the copies honest:

1. **Copy the whole dependency closure.** A guardrail may import helpers (config, constants, an embedding model for the semantic gate). Phase 0 maps the imports; the copy includes everything the module needs, not just the file.
2. **Change as little as possible.** Only import paths and the lines the new pipeline needs. Every changed line carries a `# v2:` comment.
3. **Provenance header.** Each copied file starts with `Copied from app/<path> at commit <sha>. Changes: none` or the list of changes.
4. **Drift test.** A test compares each unchanged copy with its original (ignoring imports and the header) and fails if either side changes; for a changed copy it checks that the difference matches the recorded change list. A fix made in app/ can then never be silently missing from v2.
5. **Bring the tests along.** The app/ tests for each copied module are copied too and run against the copy.
6. **Retire the shims.** reuse.py is removed once every import points to a local copy.

Target folder layout:

```
llm_approach/
  guardrails/   common.py  input_guardrail.py  parameter_guardrail.py  api_data_guardrail.py  output_guardrail.py
  security/     demo_auth.py  rate_limit.py  secrets_check.py
  prompt_safety.py   (pure symbols extracted from agent_modes: neutraliser, sanitiser, rules, response strings)
  fast_parser.py  embeddings.py  observability.py
  rag/documents/     (knowledge-base documents; own CHROMA_PERSIST_DIR)
  tools/        mcp_search.py
  rag/          (the app/ FAQ/RAG module and its helpers)
  graph/        router.py   (LangGraph shell, if decision D1 is approved)
  kept as is:   pipeline.py  llm_client.py  providers.py  gemini_limits.py  prompts.py  schema.py
                validators.py  state.py  actions/  conversation_answerer.py  metrics.py  eval/  tests/
```

## Design decisions

Six choices shaped v2. On 25 October 2026 the team lead decided D1, D2, stage 10 and the 4/5 BHK rule (marked below). D4 was then decided by the developer as an ask-first rule and is implemented (see D4 below). Stage 10 was first decided as "keep rooms_client.py" (Phase 3B) and then changed by the team lead to "use the MCP tool" (see Stage 10 status).

| # | Decision | Options | Recommendation and why |
| --- | --- | --- | --- |
| D1 | Orchestration shell | LangGraph StateGraph, or the plain dispatcher in pipeline.py | **DECIDED: keep the plain dispatcher, no LangGraph.** (Original recommendation: LangGraph, matching the agreed diagram and production. Each node wraps an existing tested pipeline function, so behaviour does not change; the parity test must stay identical. Cost: a refactor of pipeline.py and a LangGraph dependency.) |
| D2 | Semantic input gate | Block before the LLM, or advise only | **DECIDED: copy it and run it in shadow mode, only when no rooms are on screen and no question is pending** (built in Phase 3B; `enforce` exists but is not the default). Original text: **Shadow mode first**: run it before the LLM with the conversation state passed in, log its verdict but do not block, for one full evaluation. Switch to blocking only if it raises 0 false positives on the 144 general messages and the 166 eval items. Reason: a follow-up such as "does it have wifi" looks off-topic without context. The LLM's OUT_OF_SCOPE intent stays as the second line. |
| D3 | Slot checks | Keep validators.validate_slots, or use the app/ parameter guardrail | **One source of truth: the app/ parameter guardrail.** Run both side by side in Phase 3; any rule that exists only in validators.py moves into the copy as a marked v2 change. |
| D4 | Different city while other filters are remembered | Carry over old filters, start fresh, or ask | **DECIDED and IMPLEMENTED (25 Oct 2026): ask first.** Original recommendation (start fresh) was replaced by the developer's behaviour, below. |
| D5 | LLM calls per message | As today, or more | **Unchanged**: one decision call; a second call only for questions about the chat; the RAG answer path counts inside the same per-turn budget. |
| D6 | Copy or import app/ code | Import through reuse.py, or copy | **Copy**, as the team lead's clean start asks, with the drift test so the copies cannot silently diverge. |

## Implementation phases

Seven phases, each ending in a check that must pass before the next starts. Durations are estimates in working days and depend on the Gemini quota (about 200 evaluation requests a day) and review time. Each phase is one Claude Code task and one commit.

| Phase | Work | Exit criterion | Estimate |
| --- | --- | --- | --- |
| 0. Branch, baseline, inventory | Commit the counter fix; cut experiment/llm-approach_v2; read-only inventory of every app/ component in the diagram (file, functions, imports, tests, config, external dependencies); record the v1 baseline (prompt 20f328e351bb: pooled 94.0%, ctxY 69.7%) | Inventory reviewed; no code changed | 1 day |
| 1. Copy unchanged (part A done: output, parameter and API data guardrails, demo_auth, rate_limit, fast_parser, observability, prompt_safety, secrets_check; input guardrail, embeddings and RAG wait for decisions) | Copy the input, output and parameter guardrails, the API data guardrail, demo_auth, rate_limit and the delimiter neutraliser into the new folders with provenance headers; switch imports away from reuse.py; copy their app/ tests; add the drift test. Nothing is wired in yet | All tests pass (1,369 + copied ones); safety 43/43, 0/144, 6/6; parity identical; none of these modules imported from app/ | 1 day |
| 2. Wire the missing protections (**done for secrets, API data check, parameter check, observability**; semantic gate still open) | API data guardrail on every Rooms API response; fail-fast secret check at startup; semantic gate in shadow mode | Robustness 7/7 plus new cases; new tests for each guard; shadow report with 0 would-be blocks on the 144 general and 166 eval messages | 2 days |
| 3. Parameter guardrail and MCP tool | Replace validate_slots with the copied parameter guardrail (side-by-side comparison first); route searches through the copied MCP tool call | Invalid-search rate 0; golden_slots at least 96.5% of fields; rooms tests pass | 1 day |
| 4. RAG and orchestration shell | Copy the RAG path for PLATFORM_INFO; LangGraph router (if D1); different-city rule (if D4) | Stub scenarios 300/300; parity on unchanged paths; platform answers come from RAG | 2 days |
| 5. Live validation | Five evaluation sets on v2; consistency (3 runs); live scenarios; held-out set (if approved) | Every quality gate in the next section | 2 days |
| 6. Docs and review | README, pipeline PDF, evaluation report and Overleaf report updated; draft PR from the v2 branch | Team lead and Rudrakash Sir approve | 1 day |

## Phase 2 status (wired, 1,481 tests passing)

| Stage | Wired | Where |
| --- | --- | --- |
| 3 Authentication | fail-fast secret check at startup | server_app._lifespan calls secrets_check.validate_secrets |
| 9 Parameter guardrail | **done** as the second line after validate_slots (D3) | actions/search._check_arguments, before the rooms API call |
| 12 API guardrail | **done**: one bad room rejects the whole result; failing field logged, counter kept | rooms_client._search_rooms |
| 16 Observability | **done**: one record_turn per turn, LLM latency and usage, rooms API latency | pipeline.handle_turn, llm_client, rooms_client |
| Production switch | pipeline.configure_production() at server start | server_app._lifespan |

Still open: the semantic input gate (D2), RAG copy (stage 6), MCP/search tool (stage 10), LangGraph (D1), the copied input guardrail. Known gaps from Phase 2: (1) the parameter guardrail accepts only PG, 1BHK, 2BHK, 3BHK, 1RK and SINGLE, so a 4 or 5 BHK search (allowed by validators.py) now asks again instead of searching (**closed in Phase 3B**: 4BHK and 5BHK added); (2) the API data guardrail's closed idealFor and availability lists may reject real rooms, so check a live sample before go-live; (3) eight older tests build rooms that lack required fields, so the two new checks are off by default and on only under configure_production().

## Phase 3A status (failure handling and RAG copy, 1,538 tests passing)

| Part | Status | Where |
| --- | --- | --- |
| 1 LLM failure handling | **done** (fakes only, no live call) | llm_resilience.py (kinds, breaker, fixed reply), llm_client.py (counting, fallback, breaker gate), pipeline.py (polite reply when the breaker is open), observability.py (failure and event counters), config.py (3 env variables), server_app.py (skip flag ignored in production) |
| Optional fallback provider | **done, OFF by default** (`LLM_APPROACH_FALLBACK_PROVIDER`); one retry per turn, for timeout, 429, 5xx, bad JSON after the retry; fallback answer quality is not evaluated | llm_client.py |
| Circuit breaker | **done** (`LLM_APPROACH_BREAKER_FAILURES` 5, `LLM_APPROACH_BREAKER_COOLDOWN_SECONDS` 60, one trial after the cooldown) | llm_resilience.py |
| 5xx retry | **done**: the active provider is retried once after a 500/502/503/504 (`LLM_APPROACH_5XX_RETRY_DELAY_SECONDS`, default 1.0, negative = off), counted by the Gemini limiter, skipped past the 25 s turn budget, one breaker failure per turn, event `recovered_5xx_retry`; not in the planner estimate | llm_client.py, llm_resilience.py, config.py |
| 2 RAG / platform questions | **done**: copied into llm_approach/rag/ (embeddings, store, answer, ingest, 2 documents), own storage `data/chroma_v2` and collection `homilivo_knowledge_v2`, drift-tested; reuse.py uses the local copies | llm_approach/rag/, tests/copied/test_knowledge.py (11 of the 17 original tests), tests/test_rag_storage.py |
| 3 Sessions in Redis | investigated in 3A, **implemented in Phase 4A** (next section) | session_store.py |
| 4 Commands | scripts/check_room_data.py written (not run); eval commands handed to the team lead | scripts/ |

Still open after Phase 3A:

- Live checks, run by hand: the 40-item golden and golden_slots evaluation with the current prompt fingerprint; scripts/check_room_data.py against one real rooms response.
- The semantic input gate (D2) came from app/ and loaded its own embedding model (**closed in Phase 3B**: copied, one shared model, no app/ import left).
- Load test, latency (p95), frontend wiring. (Redis sessions: Phase 4A, below.)
- Fallback provider quality: run the evaluation with `LLM_APPROACH_PROVIDER` set to the fallback before turning it on.
- homilivo_faq.md is a placeholder document (not official content): replace it before go-live.
- Add `data/chroma_v2/` to the repo-root .gitignore (llm_approach/ has no .gitignore of its own).
- MCP/search tool (stage 10), LangGraph (D1), the copied input guardrail, the 4 and 5 BHK parameter guardrail gap, and the other Phase 2 gaps above.

## Phase 4A status (sessions in Redis, 1,633 tests passing)

**Done** with fakeredis only (no real Redis, no LLM call). The default is unchanged: the in-memory `SessionStore`. Redis is used only when `STATE_BACKEND=redis`.

| Part | Status | Where |
| --- | --- | --- |
| Store protocol | **done**: `SessionStoreBackend` (get_or_create, peek, save, reset, count, turn, `is_shared`); the in-memory store gained `save` and `is_shared = False`, nothing else | session_store.py, state.py |
| Versioned state | **done**: `LLMConversationState.to_dict / from_dict`, JSON only, `schema_version` 1, a type check per field, selected room must be on screen. Bad, wrong-version or unreadable blobs give a fresh session and one WARNING without contents | state.py |
| RedisSessionStore | **done**: load before the turn, save after, idle TTL 3600 s, cap 1000 (LRU sorted set), copied from app/session_store.py (`_key`, `_evict_expired`, `_enforce_cap`; drift-tested); key prefix `homilivo:v2:` so v2 never touches app/ sessions | session_store.py |
| Redis down | **done**: the turn answers from a fresh in-memory state and saves nothing (so a fresh state can never overwrite the real session when Redis returns), error logged by class name only, counted as `session_store_events.redis_unavailable` in /metrics | session_store.py, observability.py |
| /ready | **done**: with Redis sessions, `session_store: {backend, redis_reachable}` is added. It is reported, not required: with Redis down turns still answer, so the load balancer is not told to drain the server | server_app.py |
| Cross-server lock | **done**: per-session `SET NX PX` with a random token, release and refresh only if the token still matches (WATCH, no Lua), the in-process `asyncio.Lock` kept, one wait bound (`LLM_APPROACH_SESSION_LOCK_WAIT_S`, 30 s) for both. At the bound the pipeline answers `RATE_LIMIT_REPLY` ("...please try again in a minute"), action `session_busy`, no LLM call | session_store.py, pipeline.py |
| 429 wait under the lock | **done by refreshing**: the lock lives 30 s (`LLM_APPROACH_SESSION_LOCK_TTL_MS`) and a background task renews it every TTL/3 while the turn runs, up to a hard cap of 120 s (`LLM_APPROACH_SESSION_LOCK_MAX_HOLD_S`, above the 60 s Gemini wait plus the 25 s call budget). A crashed server's lock lapses after 30 s. Lost lock and cap hits are counted (`lock_lost`, `lock_hold_cap`) | session_store.py |
| Reset during a turn | **done**: a reset while a turn holds the lock leaves a marker, so that turn does not save the session back | session_store.py |
| Config | `STATE_BACKEND` and `REDIS_URL` (same names and defaults as app/); new: `LLM_APPROACH_REDIS_TIMEOUT_S` (1.0), `LLM_APPROACH_SESSION_LOCK_TTL_MS`, `_WAIT_S`, `_MAX_HOLD_S`. A bad `STATE_BACKEND` stops the server at startup; Redis being down at startup does not | config.py, server_app.py |
| Reply without a second read | `TurnResult.state` carries the state the turn left, so `/chat` and the WebSocket do not read Redis again to build the reply | pipeline.py, server_app.py |

Open after Phase 4A:

- Run once against a real Redis (local or the production one): nothing here has touched one. Check `/ready`, a restart with a live session, and two servers on one Redis.
- Choose the busy wording: it reuses `RATE_LIMIT_REPLY` ("try again in a minute"), which is longer than the real wait.
- Decide whether `/ready` should return 503 when Redis is down (now: reported only).
- Add `redis` to the llm_approach requirements (it is in the repo-root requirements.txt).
- Redis memory per session: states hold up to 200 transcript entries and the rooms on screen; measure a long session before sizing Redis.
- The load test (about 20 concurrent users) should now run with `STATE_BACKEND=redis` and two server processes.

## Stage 10 status (MCP tool, 1,798 tests passing)

Built with fakes only: no Gemini, Groq or rooms API call. Nothing is committed. This replaces the Phase 3B choice (b) "keep rooms_client.py, no mcp package, no app/server.py copy".

| Part | Status | Where |
| --- | --- | --- |
| Copy of the MCP server and `search_rooms` tool | **done**: `app/server.py` copied whole (source commit `ed0406a`, its last change; the older copies cite `7dbcba3`). Marked `# v2:` changes: observability import, API URL from `config.ROOMS_API_URL`, a `page` argument, and the API's `meta` (total, hasNextPage) returned, which SHOW_MORE needs. Retries (3 attempts, 0.3 s backoff, 5xx and timeouts retried, 4xx not), the 10 s HTTP timeout, status handling and token handling are unchanged | tools/mcp_search.py |
| Copy of the 15 s wrapper | **done**: `_call_search_rooms_with_timeout` from `app/experiments/agent_modes.py` (commit `fca91ac`), unchanged | tools/mcp_timeout.py |
| Wiring | **done**: `rooms_client._search_rooms` calls the tool through `mcp.Client` in the search worker thread. Unchanged around it: offline fixture, `validate_room_data` right after the rooms are extracted, `clean_room`, paging meta, `room_data_rejections`, `search_timeouts`, request ids, error classes (`http_500`, `timeout`, `bad_payload`, `invalid_room_data`, `no_token`, `search_timeout`, ...) and the "Room search is unavailable right now" reply. The tool returns only an error string, so `mcp_failure` maps it back to the same error classes | rooms_client.py |
| One retry layer | **done**: on the MCP transport rooms_client's own retry is not used at all (its retry code is only in the direct path); the tool's 3 attempts are the only retries. The 15 s limit (`LLM_APPROACH_SEARCH_TIMEOUT_SECONDS`) stays the outer limit, and the copied 15 s wrapper is inside it | rooms_client._mcp_request |
| Switch | **done**: `LLM_APPROACH_SEARCH_TRANSPORT` = `mcp` (default) or `direct`; any other value means `mcp`. An injected `http_get` always means direct. `direct` is the old client with its single retry (timeout or connection error only) | config.py, rooms_client.py |
| Latency | **done**: the tool records API latency; `search_rooms` no longer records it a second time on the MCP path (it still does on a whole-search timeout and on the direct path) | rooms_client.search_rooms |
| Requirements | **done**: `requirements-mcp.txt` (mcp 2.0.0, mcp-types 2.0.0); the repo-root `requirements.txt` already pins the same versions | requirements-mcp.txt |
| Tests | **done**: 37 copied tests (tests/copied: api reliability, P2-12 failure handling, P2-11 MCP timeout, token context, MCP discovery) and 37 new ones (`test_mcp_transport.py`); drift entries for both copies | tests/ |

User-visible behaviour: unchanged replies for a normal search, no results, a 5xx, a timeout, a bad room and a missing token. Differences that are not replies: a 5xx is now retried (up to 3 attempts) instead of failing at once; a failing search can take a little longer; after the 15 s limit the tool's thread may still send its remaining retries (its result is dropped and nothing is saved; no cancel logic was added, by instruction); `rooms_client` no longer sends the 2 to 4 s jittered retry on the MCP path. The `mcp` package is loaded only when a live search runs (a test checks the pipeline and server import without it).

Existing tests: no assertion changed. Nine fixture lines in six test files that fake the direct httpx client now also set `LLM_APPROACH_SEARCH_TRANSPORT=direct` (marked `# v2 mcp:`): test_area, test_fallback, test_log_audit (2), test_paging, test_phase2_wiring, test_search_timeout (3). The offline runners (`run_scenarios`, `run_robustness`) now fake the rooms API for both transports through `eval/fake_rooms_api.py`.

Robustness (offline, `run_robustness`): 7 of 7 clean on the MCP transport and on `direct`, with identical rows. The 300-turn stub baseline still matches.

## Phase 3B status (decisions applied, 1,698 tests passing)

Built with fakes only: no Gemini, Groq or rooms API call. Nothing is committed.

| Part | Status | Where |
| --- | --- | --- |
| 1 4BHK / 5BHK | **done**: `VALID_ROOM_TYPES` also holds 4BHK and 5BHK (two lines marked `# v2:`); the Phase 2 test that pinned the difference now checks that validators and guardrail agree (1 to 5 pass, 6 fails) | guardrails/parameter_guardrail.py, tests/test_phase2_wiring.py |
| 2 Search timeout | **done**: the whole live search is bounded by `LLM_APPROACH_SEARCH_TIMEOUT_SECONDS` (default 15 s, same as `MCP_TOOL_TIMEOUT_SECONDS`). On timeout: `error_class=search_timeout`, the existing "Room search is unavailable right now" reply, the previous session state kept. The worker thread only talks to the rooms API and never touches state, and it sends no retry after the timeout. The existing retry (one retry, timeout or connection error only) is unchanged | rooms_client.py, config.py |
| 2 Token handling | **done**: the original binds the caller's token per request (a demo token is swapped for the server credential, any other bearer value is the caller's own Homilivo token). llm_approach used `KANGAROO_TOKEN` for everyone. Now `/chat` and `/ws/chat` bind the same value in a ContextVar; `KANGAROO_TOKEN` is the fallback. Never logged | rooms_client.set_rooms_token, server_app.py |
| 3 Input guardrail copy | **done**: `guardrails/input_guardrail.py`, one changed line (embedding import, marked `# v2:`), drift-tested. Uses `rag/embeddings.get_shared_model`, so one model copy is in memory; loaded lazily once, warmed at startup with the other preloads when the gate is not off | guardrails/input_guardrail.py, reuse.py, pipeline.preload |
| 3 No app/ imports | **done**: `reuse.get_is_security_blocked` is the local copy; a test fails if any llm_approach module imports from `app` | tests/test_no_app_imports.py |
| 3 Semantic gate | **done**: `LLM_APPROACH_SEMANTIC_GATE` = off / shadow / enforce. Default shadow under `configure_production()`, off in bare `handle_turn` and tests. Runs after the length and security checks and before the LLM call, only with no rooms on screen and no pending question. Shadow records `gate_checked`, `gate_would_block` and the label; enforce answers with the out-of-scope reply (status blocked, category `semantic_gate`, no LLM call). A classifier failure never blocks (`gate_error`). Logs: hashed session id, request id, verdict, label; never the text | semantic_gate.py, pipeline.py, switches.py, observability.py |
| 3 Report script | **written, not run**: `scripts/gate_report.py` counts the gate's log lines (`--log`) or reads `/metrics` if a server has one | scripts/gate_report.py |

Open after Phase 3B:

- **Gate latency and cold start**: measured with the real model on the developer machine (CPU): median about 19 ms and p95 about 29 ms per check after warm-up; first load about 70 s. Re-measure on the server hardware.
- **Shadow week**: set `LLM_APPROACH_SEMANTIC_GATE=enforce` only after a week of shadow logs show no legitimate message that would be blocked. Known risk seen with the real model: with no rooms on screen, short messages such as "2bhk in pune" or "my name is Rahul" are labelled ACKNOWLEDGMENT (allowed), but "what did I say my budget was" is labelled OUT_OF_SCOPE, so enforce mode would refuse memory questions that come before any search.
- **No /metrics route** (**closed in Phase 4B**): the route now exists, with a token.
- Original tests that stay in app/ only: `test_guardrail_wiring` (needs agent_modes and the LangGraph router) and sections 1 to 3 of `test_security_hardblock` (need app.api); not copied.
- The quality gates, the held-out set and every item of the Phase 3A and 4A "open" lists are unchanged.

## Phase 4B status (readiness and metrics)

Built with fakes only: no Gemini, Groq or rooms API call. Nothing is committed.

| Part | Status | Where |
| --- | --- | --- |
| Background startup | **done**: the lifespan no longer waits for the preload. The server listens at once and loads the guardrails (security model, output guardrail, semantic gate warm-up) and the knowledge base in a background task. A shutdown during the load waits for it to end | server_app._lifespan, _preload_background |
| /health | unchanged, no I/O: 200 from the first second | server_app |
| /ready while loading | 503 with the usual `checks` plus `loading` (names still loading) and a `message`. 200 once `guardrails` and `knowledge_base` are loaded. Redis down still leaves it at 200 (reported only, as decided in 4A) | server_app.ready |
| Chat before ready | `POST /chat` answers 503 `{success: false, status: "starting", message: "I'm just starting up. Please try again in a moment."}` with `Retry-After: 5`; a WebSocket message gets an `error` event with the same text and the connection stays open. No turn runs. If the preload **fails**, loading ends and chat is served as before (models load on first use); if the preload is skipped (tests), chat is never blocked | server_app |
| /metrics | **done**: `GET /metrics` returns `active_sessions` (a count) and `observability.get_metrics_snapshot()`. Needs `Authorization: Bearer` with `LLM_APPROACH_METRICS_TOKEN` (constant-time compare); wrong or other tokens 401, unset 404. Behind the per-IP rate limit | server_app.metrics |
| Missing counters | **added**: `parameter_rejections` (by field name) and `search_timeouts`. All the others already existed: `llm_failures` (kinds), `llm_events` (breaker_open / breaker_close / breaker_skip / recovered_5xx_retry / fallback_switch), `room_data_rejections`, `semantic_gate` (gate_checked, gate_would_block, gate_blocked, gate_error), `session_store_events` (redis_unavailable and the other redis events) | observability.py, actions/search.py, rooms_client.py |
| gate_report.py | reads `/metrics` with `--token` (or `HOMILIVO_METRICS_TOKEN`); the token goes in a header and is never printed | scripts/gate_report.py |

Open after Phase 4B:

- **/metrics auth (decided):** only the dedicated `LLM_APPROACH_METRICS_TOKEN` opens it (demo tokens and the server credential do not); unset means 404 and one startup WARNING, and `validate_secrets` does not require it. app/'s own route has no authentication.
- Startup timing on the real server hardware and a cold disk (about 70 s on the developer machine before; about 21 s with warm files): measure again, and set the host's health-check path to `/health` and its readiness gate to `/ready`.
- The frontend must show the 503 "starting up" reply as a retry, not as a failure.
- Redis check time at startup was not measured (no real Redis): the ping is bounded by `LLM_APPROACH_REDIS_TIMEOUT_S` (1.0 s).
- Metrics are per process and reset on restart, as in app/.

## Quality gates

v2 must hold the v1 experiment's accuracy while adding the protections. The gates below are proposed and need the team lead's confirmation.

| Metric | v1 experiment (21 Oct) | v2 gate (proposed) |
| --- | --- | --- |
| Intent accuracy, pooled 133 | 94.0% | at least 92.0% (no drop of more than 2 points) |
| Intent accuracy, ctxY | 69.7% | at least 67.7% |
| Held-out context set | not built | clearly above the current chatbot (needs approval) |
| Searches with an invalid filter | 0 of 112 | 0 |
| Unusable decisions | 0 of 186 | 0 |
| Safety: attacks blocked, false alarms, room-text injection | 43/43, 0/144, 6/6 | unchanged |
| Semantic gate false positives | not used | 0 on 144 general and 166 eval messages |
| Failure handling | 7 of 7 clean | 7 of 7, plus the new guardrail cases |
| Consistency across 3 runs | not measured | at least 95% identical decisions |
| Live scenarios | not run live | every turn passes, or each failure explained |
| LLM latency p95 per call | 3.3 s | 3 s target; reported per stage if above |
| Automated tests | 1,369 passing | all passing, including the copied app/ tests |

## Risks and mitigations

The biggest risk is the semantic gate blocking legitimate follow-ups; shadow mode exists to catch that before it reaches users.

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Semantic gate misreads follow-ups | "does it have wifi" refused as off-topic | Pass conversation state to the gate; shadow mode before blocking; gate on 0 false positives |
| Copies drift from app/ | A fix lands in one copy only | Provenance header with commit; drift test fails on any unrecorded difference |
| Dependency closure larger than expected | Bigger install, slower start (for example an embedding model) | Phase 0 maps imports first; load heavy models lazily, once per process |
| API data guardrail rejects real rooms | Searches come back empty | Run it against a live Rooms API sample in Phase 2 and log each rejection reason |
| LangGraph refactor changes behaviour | Silent regressions | Nodes wrap existing functions; parity and stub-scenario tests must stay identical |
| Added guards add latency | Slower replies | Measure each stage in metrics.py; keep guards that call no LLM |
| ctxY gain is partly tuned | Accuracy overstated | Held-out context set written before looking at model answers |
| Gemini free-tier quota | Runs stop partway | Rolling 24-hour counter, per-day run plan, resume after interruption |

## Production readiness beyond guardrails

Guardrails are not enough for production. Each item is marked with who must act.

| Item | Detail | Type |
| --- | --- | --- |
| LLM provider for production | The Gemini free tier allows 500 requests a day, and its prompts may be used by Google. Production needs a paid key, or Groq as primary | needs team-lead decision |
| Failure handling | **Done in Phase 3A** (fallback provider off by default, circuit breaker, fixed polite message, counters in /metrics). Open: an alert on the counters, and evaluating the fallback provider before enabling it | engineering task |
| Sessions in Redis | **Built in Phase 4A** (`STATE_BACKEND=redis`), tested with fakeredis only. Open: a run against a real Redis, the busy wording, the /ready rule | engineering task |
| Load test | About 20 concurrent users: check session locks and the rate limit | engineering task |
| Latency | p95 is 3.26 s against the 3 s target | engineering task |
| Frontend wiring | Point the frontend at the v2 server | engineering task |
| Evaluation after every phase | Rerun the full evaluation: pooled at least 92%, invalid searches 0, safety unchanged | rule |

## Handover checklist: switching to the paid Gemini key

Decided on 25 October 2026: production uses a paid Gemini key later. The free-tier limits and the evaluation results in this plan belong to the free key, so do these in order before go-live:

- [ ] **Paid key and data terms.** Get the paid key from the team lead. Read the paid tier's data terms (whether prompts are used for training or retained) and have them approved. Set `GEMINI_API_KEY` in the server environment only, never in the repo or logs.
- [ ] **Update the free-tier limits in config.py** to the paid tier's published limits: the 5 s spacing between requests, the 450 requests per day cap, and the 12 requests and 200K tokens per 60 s window. Change them through the existing environment settings or the constants in config.py and gemini_limits.py, and update the tests that pin the old numbers.
- [ ] **Rerun the full evaluation, the live scenarios and the consistency check** (3 runs) on the paid key. Compare with the quality gates above (pooled accuracy at least 92.0%, ctxY at least 67.7%, invalid searches 0, safety unchanged, at least 95% identical decisions across runs). Record the new prompt fingerprint.
- [ ] **Decide the Groq fallback.** It stays off (`LLM_APPROACH_FALLBACK_PROVIDER` unset) until its answers have been evaluated with the same sets. Switch it on only after that evaluation passes and the team lead agrees.
- [ ] **Set `LLM_APPROACH_SEMANTIC_GATE` from a week of shadow logs.** Keep `shadow` while the log shows would-blocks on legitimate messages; move to `enforce` only when a week shows none (see Phase 3B, open items). Use `scripts/gate_report.py --log <server log>`.

## Decisions needed from the team lead

- [x] Branch: experiment/llm-approach_v2, cut from experiment/llm-approach so the llm_approach/ work carries over (name left to us)
- [x] D1: keep the plain dispatcher, no LangGraph (decided 25 Oct 2026)
- [x] D2: semantic gate copied, shadow mode, applied only with no rooms on screen and no pending question (decided 25 Oct 2026); blocking only after a week of shadow logs show 0 false positives
- [x] D4: a different city while other filters are remembered asks "same filters or a new search?" (implemented 25 Oct 2026, see D4 status)
- [ ] The quality gates above, especially the 2-point tolerance on accuracy
- [ ] Approval to build the held-out context set
- [x] Stage 10: ~~keep rooms_client.py, add only the timeout and token handling~~ (Phase 3B) **changed by the team lead: v2 uses the MCP tool for room search; done, see Stage 10 status**
- [x] 4BHK and 5BHK are valid searches in the parameter guardrail (decided 25 Oct 2026)
- [x] LLM provider for production: a paid Gemini key later (decided 25 Oct 2026); see the handover checklist

## D4 status (different city while filters are remembered)

Implemented in `actions/city_change.py` (pure helpers), `actions/search.py` (the check and the answer handler), `pipeline._decide_action` (intercepts the reply), `state.py` (new pending type) and `observability.py` (counters). No prompt change (fingerprint 20f328e351bb unchanged), no new LLM call.

**Behaviour.** When the validated decision is a REFINEMENT whose city differs from the current search (aliases are canonical, so Bengaluru = Bangalore) and the current search has at least one filter to carry (BHK, min/max budget, ideal_for, amenity), the bot does not search. It asks, for example: "You were looking at 2 BHK rooms in Bangalore under ₹20,000 with wifi. Do you want to search in Jaipur with the same filters, or start a new search for Jaipur? You can also tell me your new requirements." Rupees are formatted like the room lines (₹20,000).
- No question when the search has only a city (an area alone is not a filter), when the city is the same, when only the area changes, or for NEW_SEARCH (which already starts fresh).
- The reply is read from the message text plus the slots of the one decision call already made. "same / keep / yes / ok / haan / wahi / thik hai" searches the new city with the old filters, with whatever the original message said (and anything said in the reply) on top. "new / fresh / start over / no / nahi / naya" searches the new city with only the original message (and the reply's slots). A reply with new requirements ("2bhk under 15k") is a fresh search for the new city. A reply that is unclear asks the question once more, then searches fresh. A reply with a different intent drops the question and is handled normally.
- State: one pending type `city_change` with context `{"new": <slots of the original message>, "asked": 1 or 2}`; the old filters stay in `current_search`. Cleared by any answer, a different intent, a reset, or the rule-based fallback (no model to read the reply: it searches fresh). `STATE_SCHEMA_VERSION` stays 1: the change is additive, old stored sessions load, and a `city_change` blob without a valid context is refused like any bad field. The model sees this state as a plain `clarify`, never the stored filters.
- Observability: `/metrics` has a `city_change` object with `city_change_question_asked` and outcomes `kept`, `fresh`, `unclear` (counted per unclear reply) and `abandoned`. Fixed words only, no user text.

**Intentional test changes.** Stub scenario S22 turn 4 ("in pune" after Bangalore, Koramangala, max 15000): was REFINEMENT / refinement with a rooms call, now REFINEMENT / ask_city_change with no rooms call (scenario file and baseline entry S22[3] updated; the other 299 entries are unchanged, still 300 turns). Unit tests updated to ask and then answer "same": `test_actions::test_refinement_keeps_bhk_and_price_when_only_city_changes`, `test_actions::test_amenity_is_kept_by_refinement`, `test_area::test_budget_refinement_keeps_area_and_city_then_new_city_clears_area`; `test_session_redis::test_pending_types_still_match_the_check` now gives `city_change` its context. New: `tests/test_city_change.py` (54 tests). Suite: 1800 before, 1854 after.

**Sort follow-up (25 Oct 2026).** `config.ROOMS_SORT` was a read-only constant (`"latest"`) that nothing ever wrote, so it could not leak between sessions; and there was no sort preference at all (the model has no sort slot, the guardrail accepts only `latest`). Now the preference is `current_search["sort"]` (`price_asc` "cheapest first", `price_desc` "most expensive first", `rating_desc` "highest rated first", or None), in `sorting.py`:
- Read from the user's words, not the model (no prompt change): cues such as "cheaper", "cheapest", "sabse sasta", "most expensive", "highest rated"; "newest first" clears it. A refinement that says one updates the session's sort and runs the search; later refinements keep it; a NEW_SEARCH resets it (its own words may set a new one).
- The rooms API still receives `latest` (`sorting.api_sort`, one place to change when the API gains price sorts). The order is applied to the rooms the search returned, so "cheapest first" orders that page of results; it does not re-rank the whole city.
- D4: the summary says "cheapest first"; a sort alone counts as a filter to ask about; "same filters" carries it, "new search" drops it (a sort in the original message is kept).
- State: additive, `STATE_SCHEMA_VERSION` stays 1; an old blob loads with sort None; an unknown stored value is dropped. The model never sees it in the state summary.
- Tests: `tests/test_sort_state.py` (26 new). The stub baseline is unchanged from the D4 state (only S22[3]). Suite: 1854 before this follow-up, 1880 after.

**Open.** With the model unavailable the fallback always searches fresh. Whether the real model returns usable slots for replies like "2bhk under 15k" while the pending state reads `clarify` is checked only with fakes; confirm in the next live scenario run. Sort cues are English and Hinglish keywords only.

## Hardening fixes and the limiter deadline (25 Oct 2026)

Five hardening fixes (tests in `tests/test_hardening.py`, fakes only):
1. **SHOW_MORE off the event loop.** The rooms call for "show more" runs in a worker thread, so a slow rooms API no longer blocks other turns.
2. **Rule-based fallback on 429, the daily cap and an open breaker.** All three use the same wording as any other LLM failure; the cause stays countable (`error_class` `RateLimitError` / `BreakerOpen`). The 300-turn stub baseline moved only for the no-decision rate-limit case.
3. **Redis rate limiter and demo-token store fail open** (`security/fail_open.py`). A Redis timeout or error allows the request (tokens fall back to process memory); one fixed-word counter (`rate_limit_redis_failopen`, `demo_token_redis_failopen`) and a log line with the exception class only. Redis clients get the short timeout from `config.redis_timeout_s()`.
4. **A blocked reply drops its cards.** A blocked search sends no cards and keeps none on screen.
5. **`/ready` reports the real knowledge-base state.** Not ready when the embedding model did not load or the collection is empty.

**Limiter deadline (new).** A turn used to be able to sit in the Gemini limiter for up to 60 s (window wait) plus 60 s (429 retry delay) while the user waited. Now `pipeline._run_turn` starts a deadline of 20 s (`LLM_APPROACH_GEMINI_TURN_DEADLINE_S`, default 20; unreadable or non-positive values use 20) when the LLM step begins, held in a context variable (`gemini_limits.start_turn_deadline`), so no function signature changed. Before **any** sleep in `gemini_limits.py` (the window wait in `_begin` and the 429 retry delay in `call`) the limiter checks `wait <= deadline - now`; if not, it raises `WaitTooLong` (a `GeminiLimitError`, `kind="wait_too_long"`, code 429) and does not sleep.
- It maps to `RateLimitError`, so the rule-based fallback (fix 2) answers at once. `llm_resilience.classify` calls it `rate_limit`.
- Nothing is counted against the daily cap for a request that was never sent: the window wait raises before the counter is touched, and a 429 retry delay raises before the second `_begin`.
- Short waits behave exactly as before. Without a deadline (calls outside the pipeline, e.g. `llm_client.decide` alone) nothing changes.
- The deadline is on the limiter's own clock, so tests with a fake clock stay deterministic.
- Tests: 11 new in `tests/test_gemini_limits.py` (short wait sleeps and succeeds; window wait over the deadline raises and does not sleep; 429 with a 55 s hint raises with no sleep and no retry; short hint retries once; counters unchanged; no sleep exceeds the remaining deadline; no deadline means old behaviour; env override; fallback reply at once).
- Live eval runs go through the pipeline too, so a window wait over 20 s there now ends that turn in the fallback instead of waiting. With the 5 s spacing the waits are short; raise the env value if a long eval needs the old behaviour.

**Flaky test fixed.** `test_the_refresh_stops_at_the_max_hold` used wall time. `RedisSessionStore` now takes an injectable `monotonic` (default `time.monotonic`, marked `# v2:`) used only by the lock-refresh loop; the test drives it with a fake clock and a fake sleep. 20 runs in a row passed.
