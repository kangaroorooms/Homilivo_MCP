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
| 4 | Input Guardrail | Security hard block imported, plus extra_security.py; semantic gate not used | Copy input_guardrail.py whole; hard block and extra_security before the LLM as today. **The semantic gate has no conversation-context parameter and wrongly blocks follow-ups** ("does it have wifi", "show cheaper ones", "under 15k"), so D2 must define a gating rule first, for example: gate only when no rooms are on screen and no question is pending; shadow mode first (decision D2) | input_guardrail.py: is_security_blocked, validate_user_input |
| 5 | LangGraph Router | Keyword router replaced by one LLM decision, validators.py and a dispatcher in pipeline.py | **Replaced**: the router node becomes the LLM decision (prompt, schema-checked JSON, confidence threshold), routing by validated intent (decision D1) | Graph shell only; keyword routing not reused |
| 6 | FAQ/RAG | actions/platform.py answers PLATFORM_INFO through app/knowledge, reached by reuse.py | **Copy and decouple**: llm_approach already uses app/knowledge through reuse.py, so copy the retriever, answer path, embeddings.py and the documents (rag/documents/) and give v2 its own CHROMA_PERSIST_DIR | app/knowledge/, app/embeddings.py |
| 7 | Conversation | conversation.py, clarify.py, recall.py, conversation_answerer.py | **Keep** (memory, recall and verified chat answers are new); keep the reused fact-grounding and injection-defence prompt rules | agent_modes: FACT_GROUNDING_RULE, PROMPT_INJECTION_DEFENSE_RULE |
| 8 | API Request, Pydantic Extraction | Slots extracted inside the LLM decision (schema.py, Pydantic) | **Keep**: the decision call is the extraction step, so one call, not two | None |
| 9 | Parameter Guardrail | Re-implemented as validators.validate_slots | Copy parameter_guardrail.py as the single slot check; validators.py keeps only LLM-specific checks (allowed intent, room index, confidence) | parameter_guardrail.validate_search_arguments |
| 10 | MCP Tool | rooms_client.py calls the Rooms API directly | **Needs a decision**: (a) copy app/server.py and add the mcp package, or (b) keep rooms_client.py and copy only the timeout and token behaviour (**recommended: b**) | agent_modes._call_search_rooms_with_timeout, app/server.py |
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

Six choices shape v2. Each has a recommendation; D1, D2 and D4 need the team lead's confirmation.

| # | Decision | Options | Recommendation and why |
| --- | --- | --- | --- |
| D1 | Orchestration shell | LangGraph StateGraph, or the plain dispatcher in pipeline.py | **LangGraph**, matching the agreed diagram and production. Each node wraps an existing tested pipeline function, so behaviour does not change; the parity test must stay identical. Cost: a refactor of pipeline.py and a LangGraph dependency. |
| D2 | Semantic input gate | Block before the LLM, or advise only | **Shadow mode first**: run it before the LLM with the conversation state passed in, log its verdict but do not block, for one full evaluation. Switch to blocking only if it raises 0 false positives on the 144 general messages and the 166 eval items. Reason: a follow-up such as "does it have wifi" looks off-topic without context. The LLM's OUT_OF_SCOPE intent stays as the second line. |
| D3 | Slot checks | Keep validators.validate_slots, or use the app/ parameter guardrail | **One source of truth: the app/ parameter guardrail.** Run both side by side in Phase 3; any rule that exists only in validators.py moves into the copy as a marked v2 change. |
| D4 | Different city with rooms on screen | Carry over old filters, or start fresh | **Start fresh**: a new city clears the old budget, BHK and area; short refinements ("under 15k") keep them. Today handle_refinement merges old filters, so "bhopal me 1bhk" can inherit a Bangalore budget. |
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

Still open: the semantic input gate (D2), RAG copy (stage 6), MCP/search tool (stage 10), LangGraph (D1), the copied input guardrail. Known gaps from Phase 2: (1) the parameter guardrail accepts only PG, 1BHK, 2BHK, 3BHK, 1RK and SINGLE, so a 4 or 5 BHK search (allowed by validators.py) now asks again instead of searching: team-lead decision; (2) the API data guardrail's closed idealFor and availability lists may reject real rooms, so check a live sample before go-live; (3) eight older tests build rooms that lack required fields, so the two new checks are off by default and on only under configure_production().

## Phase 3A status (failure handling and RAG copy, 1,538 tests passing)

| Part | Status | Where |
| --- | --- | --- |
| 1 LLM failure handling | **done** (fakes only, no live call) | llm_resilience.py (kinds, breaker, fixed reply), llm_client.py (counting, fallback, breaker gate), pipeline.py (polite reply when the breaker is open), observability.py (failure and event counters), config.py (3 env variables), server_app.py (skip flag ignored in production) |
| Optional fallback provider | **done, OFF by default** (`LLM_APPROACH_FALLBACK_PROVIDER`); one retry per turn, for timeout, 429, 5xx, bad JSON after the retry; fallback answer quality is not evaluated | llm_client.py |
| Circuit breaker | **done** (`LLM_APPROACH_BREAKER_FAILURES` 5, `LLM_APPROACH_BREAKER_COOLDOWN_SECONDS` 60, one trial after the cooldown) | llm_resilience.py |
| 2 RAG / platform questions | **done**: copied into llm_approach/rag/ (embeddings, store, answer, ingest, 2 documents), own storage `data/chroma_v2` and collection `homilivo_knowledge_v2`, drift-tested; reuse.py uses the local copies | llm_approach/rag/, tests/copied/test_knowledge.py (11 of the 17 original tests), tests/test_rag_storage.py |
| 3 Sessions in Redis | investigated, **not implemented** (plan in the Phase 3A report) | none |
| 4 Commands | scripts/check_room_data.py written (not run); eval commands handed to the team lead | scripts/ |

Still open after Phase 3A:

- Live checks, run by hand: the 40-item golden and golden_slots evaluation with the current prompt fingerprint; scripts/check_room_data.py against one real rooms response.
- The semantic input gate (D2) still comes from app/ (`reuse.get_is_security_blocked`, the only remaining app/ import); it also loads its own copy of the embedding model, so two copies are in memory until the gate is copied.
- Redis sessions, load test, latency (p95), frontend wiring.
- Fallback provider quality: run the evaluation with `LLM_APPROACH_PROVIDER` set to the fallback before turning it on.
- homilivo_faq.md is a placeholder document (not official content): replace it before go-live.
- Add `data/chroma_v2/` to the repo-root .gitignore (llm_approach/ has no .gitignore of its own).
- MCP/search tool (stage 10), LangGraph (D1), the copied input guardrail, the 4 and 5 BHK parameter guardrail gap, and the other Phase 2 gaps above.

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
| Sessions in Redis | Use Redis instead of memory (STATE_BACKEND exists in app/) so sessions survive restarts and work across several servers | engineering task |
| Load test | About 20 concurrent users: check session locks and the rate limit | engineering task |
| Latency | p95 is 3.26 s against the 3 s target | engineering task |
| Frontend wiring | Point the frontend at the v2 server | engineering task |
| Evaluation after every phase | Rerun the full evaluation: pooled at least 92%, invalid searches 0, safety unchanged | rule |

## Decisions needed from the team lead

- [x] Branch: experiment/llm-approach_v2, cut from experiment/llm-approach so the llm_approach/ work carries over (name left to us)
- [ ] D1: LangGraph shell matching the diagram, or keep the plain dispatcher
- [ ] D2: semantic gate in shadow mode first, blocking only after 0 false positives
- [ ] D4: a different city with rooms on screen starts a fresh search, without the old filters
- [ ] The quality gates above, especially the 2-point tolerance on accuracy
- [ ] Approval to build the held-out context set
- [ ] Stage 10: copy app/server.py and add the mcp package, or keep rooms_client.py and copy only the timeout and token behaviour (recommended)
- [ ] LLM provider for production: paid Gemini key, or Groq as primary
