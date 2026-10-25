# llm_approach (v2)

An LLM-decides, code-validates chatbot for Homilivo room search. The model returns a structured decision
(intent, operation, slots); the code validates every slot, runs the rooms search, and writes the reply from
verified data. It is self-contained: nothing is imported from `app/` (`tests/test_no_app_imports.py` enforces it).

The earlier phase-by-phase notes (protocols, baselines, Phase 0 to 6b findings) are in
[docs/PHASE_LOG.md](docs/PHASE_LOG.md). Plans: [docs/V2_PLAN.md](docs/V2_PLAN.md), [PLAN.md](PLAN.md).

## Run

PowerShell, venv active, from the repo root. Variables are read from the repo-root `.env` (already-set
environment variables win). Keys are never printed.

| Command | What it does |
| --- | --- |
| `python -m llm_approach.cli [--offline-rooms] [--debug] [--skip-guardrail-load]` | Terminal chat. `--offline-rooms` uses `fixtures/rooms_bangalore.json`; `--debug` prints each decision; `--skip-guardrail-load` is faster but is not the full safety stack, so never use it for evaluation. |
| `llm_approach/scripts/start-server.ps1` | Prints provider, model and key presence, then `python -m uvicorn llm_approach.server_app:app --port 8001 --no-access-log`. |
| `python -m llm_approach.llm_info` | Which provider and model would answer, and whether its key is set. |

`--no-access-log` is required: the WebSocket token is in the URL query (`/ws/chat?token=...`).
Install: the repo `requirements.txt`; `requirements-gemini.txt` (Gemini provider, the default) and
`requirements-mcp.txt` (MCP search transport, the default) pin the extra packages.

### Environment variables

Secrets (names only): `GEMINI_API_KEY` or `GROQ_API_KEY` (the active provider's), `KANGAROO_TOKEN` (rooms API).

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_APPROACH_PROVIDER` | `gemini` | `gemini` or `groq`. |
| `LLM_APPROACH_GEMINI_MODEL` | `gemini-3.5-flash-lite` | Gemini model. |
| `LLM_APPROACH_GEMINI_THINKING` | `level:LOW` | `level:LOW/MINIMAL/MEDIUM/HIGH`, `budget:<int>` or `none`. |
| `LLM_APPROACH_GEMINI_MAX_OUTPUT_TOKENS` | 1024 | Thinking tokens count toward it. |
| `LLM_APPROACH_GEMINI_TIMEOUT_S` | 20 | Per call; minimum 10. |
| `LLM_APPROACH_FALLBACK_PROVIDER` | off | Second provider used when the first fails. Unset means no fallback (its quality is not evaluated). |
| `LLM_APPROACH_BREAKER_FAILURES` / `_BREAKER_COOLDOWN_SECONDS` | 5 / 60 | Circuit breaker per provider. |
| `LLM_APPROACH_5XX_RETRY_DELAY_SECONDS` | 1.0 | Wait before the one retry on a 5xx. |
| `LLM_APPROACH_GEMINI_SPACING_S` | 5 | Seconds between live (eval) calls; never below 4. |
| `LLM_APPROACH_GEMINI_DAILY_TOKEN_BUDGET` | unset | Live evals refuse to run without it. |
| `LLM_APPROACH_GEMINI_DAILY_REQUEST_BUDGET` | 450 | Requests per day cap for evals. |
| `LLM_APPROACH_GEMINI_DAY_RESET` | unset | Fixed UTC daily boundary, e.g. `07:00`. |
| `STATE_BACKEND` | `memory` | `memory` or `redis`; an unknown value stops start-up. |
| `REDIS_URL` | `redis://localhost:6379/0` | May hold a password; never logged. |
| `LLM_APPROACH_REDIS_TIMEOUT_S` | 1.0 | Connect/read timeout; a down Redis must not stall a turn. |
| `LLM_APPROACH_SESSION_LOCK_TTL_MS` / `_WAIT_S` / `_MAX_HOLD_S` | 30000 / 30 / 120 | Per-session Redis lock. |
| `SESSION_TTL_SECONDS`, `SESSION_MAX_COUNT`, `SESSION_MAX_HISTORY_MESSAGES` | 3600, 1000, 40 | Same names and defaults as `app/`. |
| `LLM_APPROACH_METRICS_TOKEN` | unset | Bearer token for `GET /metrics`. Unset means `/metrics` answers 404. |
| `LLM_APPROACH_SEMANTIC_GATE` | `shadow` (server) | `off`, `shadow` (record only) or `enforce` (block off-topic first messages). Bare `handle_turn` and tests run `off`. A typo stops start-up. |
| `LLM_APPROACH_SEARCH_TRANSPORT` | `mcp` | `mcp` (copied in-process MCP tool) or `direct` (`rooms_client`'s own httpx). Anything else means `mcp`. |
| `LLM_APPROACH_SEARCH_TIMEOUT_SECONDS` | 15 | Whole rooms search, retry included. |
| `LLM_APPROACH_ROOMS_API_URL` | `https://homilivo.onrender.com/api/rooms` | Rooms API. |
| `LLM_APPROACH_OFFLINE_ROOMS` | unset | `1` uses `fixtures/rooms_bangalore.json`, or a path. |
| `HOMILIVO_API_TIMEOUT_SECONDS`, `_MAX_ATTEMPTS`, `_RETRY_BACKOFF_SECONDS` | 10, 3, 0.3 | Per-attempt limits inside the MCP tool (copied from `app/server.py`). |
| `LLM_APPROACH_CHROMA_PERSIST_DIR` | `data/chroma_v2` | Knowledge-base index. The app's `CHROMA_PERSIST_DIR` is never read; a path equal to `data/chroma` is refused. |
| `LLM_APPROACH_CORS_ORIGINS` | off | Comma-separated origins. |
| `RATE_LIMIT_MAX_REQUESTS` / `_WINDOW_SECONDS` | 30 / 60 | Per client; `SESSION_RATE_LIMIT_*` (20 / 60) per session; `RATE_LIMIT_ENABLED=true`. |
| `LLM_APPROACH_METRICS` | off | `1` appends per-turn metric lines (fixed fields, no text) to `eval/results/metrics.jsonl`. |
| `LLM_APPROACH_SKIP_PRELOAD` | off | Skip the start-up model load. |
| `LLM_APPROACH_SKIP_SECRET_CHECK` | ignored | Honoured only by the test suite; the server always runs the check. |

**Secret check.** At start-up `validate_secrets()` refuses to start when the active provider's key or
`KANGAROO_TOKEN` is missing. It logs the secret's name only, never a value.

## Server behaviour

- **Start-up is non-blocking.** The server listens at once and loads the guardrails, the semantic-gate
  examples and the knowledge base in the background.
- `GET /health`: always `200` once the process is up.
- `GET /ready`: `200 ready` when the provider key, the guardrails and the knowledge base are loaded, else
  `503 not_ready` with a `loading` list while the background load runs. Values are booleans only. With
  Redis sessions it also reports `session_store.redis_reachable`; a down Redis does not make the server
  not ready.
- `POST /chat` and the WebSocket answer **503 `starting`** (with `Retry-After`) until the load ends. If the
  load fails, chat is served anyway and the models load on first use.
- `POST /chat/reset?session_id=...`, `GET /chat-ui/demo-token`, `WS /ws/chat?token=...` follow the current
  backend's protocol (details in the phase log).
- `GET /metrics`: counters and latency stats as JSON, opened only by `LLM_APPROACH_METRICS_TOKEN`
  (`Authorization: Bearer ...`); no message text, room data, keys or session ids.
- **Session ids.** `server_app.normalize_session_id` accepts a UUID (any case, lowercased) and replaces
  anything else, including `None`, empty, non-strings, over-long or email-like values, with a server-made
  UUID. It runs before the rate limiter, the session store and `handle_turn`. The raw value is never
  logged; logs carry a 12-character hash of the id. Redis keys use the normalised id only.

## Search path

Searches go through `tools/mcp_search.py`, a copy of the app's MCP room-search tool, called in-process
(`LLM_APPROACH_SEARCH_TRANSPORT=mcp`). Its retries and per-attempt timeouts are the app's. The whole search,
retry included, is bounded by **15 s** (`LLM_APPROACH_SEARCH_TIMEOUT_SECONDS`), as in `app/`. On failure the
user gets a polite message and no rooms are invented. `direct` uses `rooms_client`'s own client instead.
The API has no area filter, so an area search fetches up to 30 rooms and filters them locally.

## Sessions

Conversation state (`state.py`) is kept in memory by default. With `STATE_BACKEND=redis`
(`session_store.py`) it is stored in Redis with a TTL, so sessions survive a restart and work across
servers; a per-session lock serialises turns. With Redis down, turns still answer from a fresh state.
`scripts/redis_restart_check.py` demonstrates the restart case.

## City change (D4) and sorting

- **D4.** A refinement that names a *different city* while other filters are remembered (bhk, price,
  ideal-for, amenity, sort) does not search or merge silently. It asks whether to keep the same filters or
  start a new search. The reply is read from the message (keep / fresh / new requirements). An unclear
  reply is asked once more, then a fresh search runs. A reply about something else drops the question. The
  area is never carried to the new city.
- **Sort.** Cues such as "cheapest", "most expensive", "highest rated" set a per-session preference in
  `current_search["sort"]`; "newest" clears it. A new search starts without the old sort unless its own words
  set one. **One-page limitation:** the rooms API only knows `latest`, so the preference orders the rooms
  that one search returned (one page); it does not re-rank the whole city. Showing more pages orders each
  page separately.

## Tests

```powershell
python -m pytest llm_approach -q
```

Offline: no LLM, rooms API or network calls (the real guardrail model is stubbed except in the tests that
load it). `tests/copied/` holds tests copied from `app/`.

## Evaluation tooling

All are `python -m llm_approach.eval.<name>`; live runs call the provider and respect the daily budget.

| Module | Flags |
| --- | --- |
| `run_eval` | `--dataset NAME\|all`, `--runs N`, `--sample N`, `--spacing S`, `--daily-token-budget N`, `--dry-run` (no calls), `--only-missing`, `--resume`, `--fresh` |
| `run_scenarios` | `--stub` (offline, scripted) or `--live`; `--only S1,S4`, `--spacing`, `--daily-token-budget`, `--resume` |
| `run_robustness`, `run_safety` | `--out PATH`; `run_safety --no-real-guardrail` skips the torch guardrail |
| `report` | `--results-dir`, `--provider`, `--model`, `--prompt-fp` |
| `baseline_current` | measures the current router, read-only |

Also `scripts/gate_shadow_measure.py` (offline semantic-gate measurement), `scripts/gate_report.py`
(`--log FILE` or `--url`/`--token` for a running server), `scripts/check_room_data.py`. The prompt
fingerprint (`prompt_fingerprint.py`) ties results to the exact prompts used.

## Copy-from-app rules

Code taken from `app/` is **copied, never imported**; `reuse.py` is only a lazy loader over these local
copies (heavy ones such as torch load on first use).

- Every copied Python file starts with a provenance header: source file, commit, what changed, ending with
  `# ==== end of v2 provenance header ====`. `rag/PROVENANCE.md` lists the `rag/` copies; the two documents
  there are byte-for-byte copies.
- Every change inside a copy is marked `# v2:` (`# v2: was: <original>` where a line is replaced).
- `tests/test_copy_drift.py` compares each copy with its `app/` source as text. Ignoring headers and `# v2:`
  lines, the lines must be identical. When `app/` changes, re-copy and re-mark; never edit a copy silently.
- `tests/test_no_app_imports.py` fails on any `from app` / `import app` in code.
- v2 keeps its own storage (`data/chroma_v2`, collection `homilivo_knowledge_v2`).
