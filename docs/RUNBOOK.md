# Homilivo Chatbot — Runbook

Operational reference: run it locally, run the test/eval gates, deploy
it, and handle the incidents that actually exist in this codebase.
Companion docs: `docs/WEBSOCKET_CONTRACT.md` (the WS/REST wire
contract) and `docs/Homilivo_Project_Plan.md` (architecture + phases).

---

## 1. Run locally

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate         # macOS/Linux

pip install -r requirements.txt
cp .env.example .env               # fill in real values - never commit .env
```

Minimum to get FAST mode working: nothing else — `mode="fast"` is a
deterministic templated responder with no external dependency beyond
the Homilivo API itself (BYOT: the caller supplies their own bearer
token per request, so no token needs to be in `.env` at all).

Optional, per mode/feature you want to exercise:

| Want to use... | Needs |
|---|---|
| `mode="qwen"` / `mode="gemma"` | A local [Ollama](https://ollama.com) daemon running, with `qwen3:4b` / `gemma3:4b` pulled (`ollama pull qwen3:4b`) |
| `mode="groq"` | `GROQ_API_KEY` in `.env` |
| `/chat-ui` demo page's one-click login | `KANGAROO_TOKEN` in `.env` (server-side fallback credential) |
| `STATE_BACKEND=redis` | A reachable Redis — `docker compose up -d redis` starts the bundled local one on `redis://localhost:6379/0` |

Start the server:

```bash
uvicorn app.api:app --reload --port 8000
```

- Demo chat UI: `http://localhost:8000/chat-ui`
- Interactive API docs (paste a Homilivo token into the green
  "Authorize" button): `http://localhost:8000/docs`
- Health checks: `GET /health`, `GET /knowledge/health`
- Metrics (JSON, not Prometheus format): `GET /metrics`

---

## 2. Run tests + eval

```bash
python -m pytest tests/ -q            # full suite - CI-safe, no live
                                       # Ollama/Groq/Redis required
                                       # (everything external is mocked)
python tests/run_golden_eval.py       # golden-set accuracy gate (>= 85%)
```

Both are exactly what `.github/workflows/ci.yml` runs on every PR and
every push to `main` — see `README.md`. If you changed
`app/fast_parser.py`, `app/graph/`, or anything in the guardrails,
also run the targeted regression suites directly for a faster loop:

```bash
python -m pytest tests/test_guardrail_wiring.py tests/test_graph_router.py \
    tests/test_golden_eval.py tests/test_streaming.py -v
```

**Never** edit `tests/golden_set.jsonl` expected values or weaken an
assertion to make a red run go green — see guardrail #3 in
`docs/Homilivo_Project_Plan.md`. A genuine regression gets fixed in the
pipeline, not in the test.

### Load test (dev tool, not part of CI/pytest)

```bash
# terminal 1
uvicorn app.api:app --port 8000

# terminal 2 - against a running server, real WebSocket traffic
python scripts/load_test_ws.py --url ws://localhost:8000/ws/chat \
    --token <a-real-or-demo-token> --sessions 20 --turns 5
```

Checks concurrent-session throughput/latency, that a deliberately idle
connection still receives a heartbeat `{"event":"ping"}`, and that a
dropped-and-reconnected session resumes its prior room/session context
(the same property that matters most once `STATE_BACKEND=redis` is in
play — see §4's Redis-down entry). Exits non-zero on any session error,
a missing heartbeat, or a failed reconnect. See the script's own
docstring for the full option list (`--mode`, `--heartbeat-wait`, etc.).

---

## 3. Deploy

There is no cloud-specific deploy config in this repo (no Dockerfile,
Procfile, or platform manifest) — this section is the generic shape any
target should follow; fill in your platform's specifics rather than
guessing them here.

1. **Process:** run the ASGI app with a real server, not `--reload`:
   `uvicorn app.api:app --host 0.0.0.0 --port 8000 --workers N` (or
   behind `gunicorn -k uvicorn.workers.UvicornWorker`). Multiple worker
   processes are exactly the case `STATE_BACKEND=redis` exists for —
   see below.
2. **Env vars:** every variable in `.env.example` should be set in the
   deploy environment (never committed). At minimum for a multi-worker
   or multi-instance deploy: `STATE_BACKEND=redis` and `REDIS_URL`
   pointing at a real Redis, so sessions/rate-limit/demo-auth/
   conversation-memory are shared instead of split per-process (Phase 2).
3. **Health checks:** point your platform's liveness/readiness probe at
   `GET /health` (process up, reports active session count) and
   optionally `GET /knowledge/health` (embedding model + doc index
   loaded — `"degraded"` if the embedding model failed to load, which
   breaks platform-knowledge answers but not room search).
4. **Observability:** `GET /metrics` returns a JSON snapshot (request/
   error/tool-failure/rate-limit counts, latency percentiles, and the
   Phase 8 `llm_usage`/`tier_summary` cost-tier counters — see §5). It
   is in-memory and resets per process; scrape it per-instance if you
   want a longer history, or swap `app/observability.py`'s registry for
   a real backend later (every call site already funnels through that
   one module).
5. **Secrets:** `KANGAROO_TOKEN` / `GROQ_API_KEY` never reach the
   browser (see `app/server.py`'s per-request `ContextVar` binding) —
   keep them as deploy-environment secrets, not build-time config baked
   into an image.

---

## 4. Incidents

### Homilivo API is down / unreachable

**Symptom:** room-search turns return the fixed
`HOMILIVO_UNAVAILABLE_RESPONSE` message (`app/experiments/agent_modes.py`)
instead of results; `/metrics`'s `tool_failures` counter climbs;
`api_latency` shows timeouts around `HOMILIVO_API_TIMEOUT_SECONDS`
(default 10s) × `HOMILIVO_API_MAX_ATTEMPTS` (default 3, with backoff).

**This is already handled gracefully** — the chatbot does not crash or
hang; it degrades to an honest "can't reach the room search service
right now" reply for search-shaped turns, while conversation, platform
FAQ, and previously-fetched room follow-ups keep working normally
(nothing else in the pipeline calls the Homilivo API).

**Action:** confirm it's really Homilivo and not a local network/DNS
issue; if it's their outage, there's nothing to fix on this side except
wait — do not raise `HOMILIVO_API_MAX_ATTEMPTS` or lower the timeout as
a workaround without understanding why first (that changes latency
characteristics for every user, not just during the outage).

### A model is down (Ollama unreachable, or Groq API error/outage)

**qwen/gemma (Ollama):** if the local Ollama daemon is down or the
model isn't pulled, `generate_qwen_response()`/`generate_gemma_response()`
raise. This propagates out of `process_message()` uncaught (there is no
automatic fast-mode fallback for these two modes, unlike Groq — see
next) and the turn surfaces as `{"event":"error"}` over WS or a 500 over
REST, with the fixed detail string `"Something went wrong processing
that message. Please try again."` — no internal error text or stack
trace reaches the client (see `app/api.py::chat_websocket`).

**Action:** restart/check the Ollama daemon (`ollama serve`), confirm
the model is pulled (`ollama list`), or tell affected users to retry
with `mode="fast"` or `mode="groq"` in the meantime.

**groq:** `generate_response()` already catches a Groq failure and
**automatically falls back to FAST mode** for that turn (see the GROQ
branch in `app/experiments/agent_modes.py::generate_response` — this
is existing, pre-Phase-8 behavior). If Groq is streaming live tokens to
a WebSocket client when it fails mid-stream, the client will have
already seen some raw partial text before the fallback's FAST-mode
reply replaces it in the `"done"` event — see §5.2's streaming note.
This is a rare failure path (a mid-generation network drop), not
something that needs an operational response beyond normal Groq
incident monitoring.

**Action:** check Groq's status page / your `GROQ_API_KEY` quota; no
immediate user-facing action needed since the fallback already covers
it, but investigate if the fallback rate climbs (visible as
`llm_usage.groq.calls` staying flat while overall turn volume in
`/metrics` doesn't).

### Redis is down (only relevant when `STATE_BACKEND=redis`)

**Symptom:** every session/rate-limit/demo-auth operation raises (there
is no automatic fallback to in-memory mid-flight — `STATE_BACKEND` is
a startup-time choice, not a runtime failover). Turns fail with the
same generic error response described above; `/health`'s process is
still up, but requests will 500/error.

**Action:**
1. Restore Redis first if at all possible — that's the only path that
   loses zero session state.
2. If Redis cannot be restored quickly and you must restore service
   immediately: set `STATE_BACKEND=memory` and restart the app
   instances. This is a deliberate, visible tradeoff — every active
   session, rate-limit window, and demo-auth token is lost (a fresh
   in-memory store starts empty), and if you're running more than one
   instance behind a load balancer they will each keep their own state
   again (no more cross-instance sharing) until Redis is back and
   `STATE_BACKEND=redis` is restored.
3. `docker compose up -d redis` brings back the bundled local Redis if
   that's what you're running; for a managed/hosted Redis, follow your
   provider's own recovery process.

### Output guardrail is blocking legitimate replies

**Symptom:** turns that should succeed instead return the fallback
`"Sorry, I couldn't process that request. Please rephrase."` message
(`success: false` in the REST/`"done"` payload), and `/metrics`'s
`status_counts.blocked` is elevated.

**Action:** check `app/guardrails/output_guardrail.py`'s pattern lists
(`SENSITIVE_OUTPUT_PATTERNS`, `INTERNAL_OUTPUT_PATTERNS`) for an
overly-broad match against legitimate room-listing text (e.g. a room
description that happens to contain a word like "secret" as part of an
amenity name). Never disable or weaken a check to "fix" this without
confirming it's a false positive first — this guardrail is the last
line of defense against a hijacked-model reply leaking a credential or
the system prompt (see `tests/test_prompt_injection.py` and
`tests/test_streaming.py::test_done_payload_is_guardrail_validated_even_if_streamed_tokens_leaked`).

---

## 5. Phase 8 specifics (true streaming, cost/tier controls)

### 5.1 What actually streams now

`mode="fast"` never called a model in the first place — it still
builds a complete templated string and hands it to
`app.streaming.chunk_for_streaming()`, which splits it into small
windows for the client. No change from before Phase 8.

`mode="qwen"` / `"gemma"` / `"groq"` now stream **real** tokens from
the model, as they're generated, whenever a turn actually reaches a
model call (some turns don't — e.g. "which rooms have AC?" is answered
deterministically from already-fetched room data regardless of mode,
per `generate_response()`'s "deterministic answer always wins" rule,
so no model call and no live tokens happen for those either — the
simulated chunking above covers them instead).

See `app/streaming.py`'s module docstring and `app/api.py::chat_websocket`
for the mechanics (a `LiveTokenRelay` bridges each provider SDK's
synchronous streaming callback, run off the event loop via
`asyncio.to_thread`, into the WebSocket send loop).

### 5.2 The OUTPUT guardrail stays authoritative

Live tokens are relayed to the client **before** the full reply has
been validated — that's what makes it real streaming rather than
"generate everything, validate, then trickle it out" (which is what
Phase 8 replaced). `validate_ai_response()` still runs on the complete
assembled text before the turn's `"done"` event is built, exactly as
it always has for the non-streaming REST path. The `"done"` payload —
never the live token preview — is the one authoritative, contract-
guaranteed result: if validation blocks the reply (a hijacked model
leaking a secret, an over-long response, etc.), `"done"` carries the
safe fallback text and `success: false`, even though the client may
already have seen some of the raw text stream by live before that
decision was made.

**This is a deliberate, documented tradeoff**, not an oversight: true
token-by-token streaming and "never send a single byte of unvalidated
text" are mutually exclusive (the guardrail's checks — length, secret/
internal-info patterns — need the complete text to evaluate). A client
must always treat the `"done"` event as authoritative and be prepared
to discard/replace whatever it rendered live during generation on the
rare turn where `success` comes back `false`. See
`tests/test_streaming.py::test_done_payload_is_guardrail_validated_even_if_streamed_tokens_leaked`
for the regression proving this holds, and the `CRITICAL` comment in
`app.api.chat_websocket` for where this is enforced.

### 5.3 Cost / model-tier controls

`GET /metrics` now includes:

- `llm_usage`: per-mode `{calls, estimated_tokens, estimated_cost_usd}`
  for qwen/gemma/groq only (fast mode never appears here — it makes no
  model call). Token counts are a `len(text) // 4` estimate, not a
  provider-reported figure — see `app/observability.py::_estimate_tokens`
  for why (provider usage fields differ in shape per SDK and don't
  exist on the mocked clients this project's tests use). Cost is
  likewise an estimate, computed only from the operator-supplied
  `LLM_COST_PER_1K_TOKENS_<MODE>` env vars (default 0) — this project
  does not hardcode a provider's real price (see `.env.example`).
- `tier_summary`: `{llm_tier_calls, fast_or_deterministic_tier_turns}`
  — a direct view of the hybrid-by-design split (guardrail #7): how
  many turns actually needed a real model call vs. how many were
  served by the free deterministic/fast tier. `fast_or_deterministic_tier_turns`
  is `request_count - llm_tier_calls`, i.e. every turn that never
  called `generate_qwen_response()`/`generate_gemma_response()`/
  `generate_groq_response()`.

Routing cheap/simple turns to the fast tier already happens structurally
— `generate_response()` tries deterministic answers first, and `mode`
is caller-chosen per turn — Phase 8 only adds these counters so that
split is visible instead of implicit.
