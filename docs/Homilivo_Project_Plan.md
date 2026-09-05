# Homilivo Chatbot — Target Architecture Implementation Plan

> **What this is:** the authoritative plan to make the existing, working Homilivo chatbot **match the approved target architecture diagram** (React UI → WebSocket → FastAPI → Auth → Input Guardrail → LangGraph Router → {FAQ/RAG · Conversation · API-request → Pydantic Extraction → Parameter Guardrail → MCP Tool → Homilivo API → API Guardrail} → Response Generation → Output Guardrail → WebSocket → React).
> **This is a target-architecture implementation plan, not a generic improvement plan.** Every box in the diagram maps to a phase or an already-complete component.
> **Golden rule:** improve, don't rebuild. The chatbot works today (85% golden eval, 311 tests passing). Reuse and extend; wrap, don't replace; never regress.

**Verified against:** source, tests, and live runs on branch `parser-fixes`, commit `2a1560b` + uncommitted Phase-1 work, plus a full read-only architecture audit.

---

## 1. Goal

Implement the approved target architecture **exactly as diagrammed**, by adding the three things that make the current system literally match the diagram — a **LangGraph Router**, **Pydantic extraction**, and **true token streaming** — plus the supporting work (Redis state, real RAG, a React frontend, PII compliance). The existing engine that already performs the diagram's logic is preserved and wrapped, never rewritten.

Two outcomes are called out explicitly because they are your stated must-haves:
- **A real, stable WebSocket endpoint** that a separate Homilivo React frontend can connect to later (the endpoint is a *contract*).
- **Real streaming** (true token-by-token), not the current simulated "generate-then-chunk".

---

## 2. Design principles (non-negotiable)

1. **Improve, don't rebuild.** The existing engine (`app/experiments/agent_modes.py`), four guardrails, MCP tool, `fast_parser`, session store, and tests work and are tested. Adapt/wrap; replace only when the target requires it, and say why.
2. **Keep the chatbot working at every step.** Each phase ends with the full suite + golden eval green and the chatbot answering. No big-bang rewrite.
3. **Never game the eval.** Fixes are general. Never edit `tests/golden_set.jsonl` expected values or weaken tests. Golden eval stays ≥ 85%.
4. Hybrid by design. The deterministic router + regex parser remain the
fast, free, zero-hallucination DEFAULT decision/extraction tier.

LangGraph is the REQUIRED orchestration layer and wraps the existing
deterministic router. It does not replace or reimplement it.

LLM classification/extraction remains a fallback for the long tail.
An LLM call is never the critical path for common queries.An LLM call is never the critical path for the common case.
5. **Four guardrails at four trust boundaries:** input, parameter, api-data, output.
6. **No secrets in git; keys never reach the client.** `.env` only; `.env.example` with placeholders.
7. **The LLM never touches the DB / never writes SQL.** Data only via the MCP tool → Homilivo API.
8. **Untrusted content is data, not instructions** (prompt-injection safety).
9. **State to Redis = FOUR things:** session store, rate limiter, demo-auth token store, conversation memory.
10. **Don't guess external contracts.** Never invent Homilivo API fields, enum values, RAG content, or PII classifications. Mark UNKNOWN and ask.
11. **The WebSocket endpoint is a contract.** The basic demo React client and Homilivo's future real frontend both connect to the same `/ws/chat` contract. Do not change the contract casually — version it if it must change.

---

## 3. Current-state baseline (verified by audit)

**One turn, as actually implemented:** Browser (`chat_ui.html`) → WS `/ws/chat` (REST `/chat` fallback) → rate limit + per-request `ContextVar` token binding → **input guardrail** → deterministic `route_message()` (memory-aware) → branch {platform-knowledge via embedding retrieval · room follow-up from session cache · new-search/refinement → `extract_search_parameters()` regex → **parameter guardrail** → `search_rooms` MCP tool → Homilivo API → **api-data guardrail**} → `generate_response()` (fast|qwen|gemma|groq) → **output guardrail** → in-memory session → metrics → reply chunked over WS.

**Verified facts:** deterministic-first (Gemma is a fallback classifier only); 4 modes share one pipeline + four guardrails; one external dep (`search_rooms` → Homilivo API, 10s timeout / 3 retries); all state in-process memory; `app/agent.py` is a dead debug CLI; **311 passed, 1 xfail; golden eval 85% (17/20)**.

**WebSocket status (important):** the *connection* is real — `/ws/chat` genuinely works with a `typing → token×N → done → ping` event protocol, 20s heartbeat, close codes 4401/4429, and REST fallback. What is **not** real is the *streaming*: `process_message()` runs to completion, then `chunk_for_streaming()` drip-feeds the finished string word-by-word. So: connection = real ✅, streaming = simulated ⚠️ (Phase 8 makes it real).

---

## 4. Target diagram → status (every box accounted for)

| Diagram box | Status | Phase |
|---|---|---|
| React Chatbot (frontend) | Missing (HTML demo has parity) | **3a** basic demo now · 3b Homilivo integration later |
| WebSocket | Real connection; needs hardening + real streaming | **2.5** (harden) + **8** (streaming) |
| FastAPI `/ws/chat` | Complete | — |
| Authentication | Complete (BYOT + ContextVar) | — |
| Input Guardrail | Complete | — |
| **LangGraph Router** | Implemented differently (deterministic) → wrap in LangGraph | **5 (REQUIRED)** |
| FAQ / RAG | Partial (real math, placeholder content) | **4** |
| Conversation | Complete | — |
| API Request | Complete | — |
| Pydantic Extraction | Partial (dict regex; no LLM fallback) | **6** |
| Parameter Guardrail | Complete | — |
| MCP Tool | Complete | — |
| Homilivo API | Complete | — |
| API-data Guardrail | Complete | — |
| Response Generation | Complete (fast/qwen/gemma/groq) | — |
| Output Guardrail | Complete | — |
| (infra) Redis state | Missing | **2** |
| (infra) PII/DPDP | Missing | **7** |
| (infra) Observability / Eval | Complete | Phase 1 commits the CI gate |

Everything in the diagram is either already complete or has an owning phase. **LangGraph, Pydantic extraction, and true streaming are the three boxes that make the system literally match the diagram.**

---

## 5. What must NOT be changed

The deterministic router (`route_message()`, `detect_room_operation()`, `resolve_room_reference()`), the four guardrails, the MCP tool layer, the Homilivo client's timeout/retry logic, the per-request `ContextVar` isolation, the session-store/rate-limiter interfaces, the golden-eval harness, and the `/ws/chat` event contract. These are tested and reasoned; they may be **wrapped or extended**, never reimplemented.

---

## 6. LangGraph decision — REQUIRED (resolved)

Earlier this was "optional" because the deterministic router already performs the routing. **For this project the objective is to implement the approved target architecture, which explicitly contains a LangGraph Router — so Phase 5 is REQUIRED.**

The strategy is non-negotiable: **LangGraph wraps the existing deterministic router; it does not replace it.** The graph's fast-path node calls the existing `route_message()` chain as-is; explicit edges add the FAQ / Conversation / API-request branches. This gives diagram conformance *and* keeps the tested routing logic (follow-ups, room references, amenities, comparatives, city extraction, set-wide questions, unsupported actions). Deleting or reimplementing the router is forbidden.

---

## 7. Phase order (target-architecture roadmap)

**Recommended sequence:** 1 → 2 → 2.5 → 6 → 5 → 4 → 3a → 3b → 7 → 8.

| Phase | Title | Status | Blocked on |
|---|---|---|---|
| 1 | Foundation hardening — commit existing work + CI branch protection | mostly done | — |
| 2 | Redis-backed state (4 stores, swap-in) | ready | — |
| 2.5 | WebSocket hardening (real, stable, reusable endpoint) | ready | — |
| 6 | Pydantic-validated LLM extraction (fixes 3 messy eval misses) | ready | — |
| 5 | **LangGraph Router (REQUIRED; wrap existing)** | required | — |
| 4 | Real RAG on Chroma | port ready | real FAQ content |
| 3a | Basic standalone React demo client (Vite+TS, minimal) | ready | — |
| 3b | Integrate into Homilivo's real React app | spec ready | Homilivo React conventions |
| 7 | PII/DPDP + handoff hook (before real users) | spec ready | Homilivo PII field list |
| 8 | True token-by-token streaming + hardening | ready | — |

**Unblocked today:** 1, 2, 2.5, 6, 5, 3a, 8. **Blocked on external input:** 4 (content), 3b (Homilivo repo), 7 (PII fields).

---

## 8. WebSocket & streaming — the explicit end goal

Because these are your stated must-haves, they are first-class outcomes, not side-effects:

- **Phase 2.5 (WebSocket hardening):** verify and, if needed, fix `/ws/chat` so it is a genuine, stable endpoint a *separate* frontend can rely on — correct auth on connect, clean reconnect, heartbeat, documented and versioned event contract, graceful close codes. Deliverable includes a written **WS contract doc** so any frontend (yours or Homilivo's) can integrate against it.
- **Phase 8 (true streaming):** replace generate-then-chunk with real token-by-token streaming for the LLM modes, over the same event contract, so the React clients need no protocol change.
- **Phase 3a (basic React client):** a minimal standalone React app that connects to `/ws/chat` — proves the endpoint works end-to-end and serves as a reference client for whoever builds the real Homilivo frontend.

---

## 9. External inputs needed from the Homilivo team

1. **Real FAQ/policy content** — blocks Phase 4's value.
2. **Which room/owner fields are PII** — blocks Phase 7; must not be guessed.
3. **Homilivo React app conventions** — blocks Phase 3b (not 3a).
4. **Whether human-handoff is in scope**, and to which system — scopes Phase 7's hook.

---

## 10. Global Definition of Done

- [ ] The system matches the target diagram box-for-box: React UI → WS → FastAPI → Auth → Input Guardrail → LangGraph Router → {FAQ/RAG · Conversation · API-request → Pydantic Extraction → Parameter Guardrail → MCP → Homilivo API → API Guardrail} → Response Generation → Output Guardrail → WS → React.
- [ ] LangGraph router wraps the existing deterministic router (not reimplemented); three branches route correctly.
- [ ] All four guardrails invoked on every relevant turn — proven by a regression test.
- [ ] Pydantic-validated LLM extraction handles messy queries; golden eval > 85%, never regressed.
- [ ] `/ws/chat` is a real, stable, documented endpoint a separate frontend can connect to; reconnect + heartbeat + REST fallback work.
- [ ] Real token-by-token streaming for LLM modes; fast mode reliable and free.
- [ ] A basic standalone React client connects to `/ws/chat` and works.
- [ ] FAQ answers grounded in real Homilivo content via Chroma; honest "don't know" below floor.
- [ ] Sessions, rate limiting, demo-auth, memory on Redis; two instances behave as one.
- [ ] PII masked/audited; retention/deletion policy; handoff hook flagged off by default.
- [ ] CI runs pytest + golden eval on every PR and blocks regressions (branch-protected); feedback flowing.
- [ ] No secrets in git; `.env.example` documents all vars; full suite + eval green on main.

*Companion: `Homilivo_Implementation_Prompt.md` (phase-by-phase build instructions). If the two conflict, this plan wins — stop and ask.*
