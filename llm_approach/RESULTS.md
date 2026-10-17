# LLM approach (Option B): results report

**Date:** 2026-10-17  **Branch:** `experiment/llm-approach`  **Status:** built and verified offline; **live accuracy is not measured yet**.

This report contains no accuracy number for Option B. None exists: the live evaluation has not been run (see section 3).

---

## 1. Summary

**What was built.** A chatbot pipeline in `llm_approach/`, beside the existing one, that makes one Groq call (`openai/gpt-oss-20b`, JSON mode, temperature 0) per message. The model sees the message plus the conversation memory (current search, rooms on screen, selected room, pending question, recent turns) and returns a structured decision: one of 10 intents, an operation for room follow-ups, slots (city, area, BHK, budget, amenity), a room reference and a confidence. Code then validates every field, decides whether to act or ask, and acts: it queries the rooms API with validated slots only (at most one call per turn), computes room answers from stored rooms, answers platform questions from the existing knowledge base, and runs the output guardrail on every reply. The model never calls the API and never states room or platform facts. Security is checked before the model: an additive regex layer (`extra_security.py`) plus the existing hard block. If Groq fails, a rule-based fallback runs. A CLI and a FastAPI server on port 8001 (same REST and WebSocket contract as the current backend) let the real website talk to it.

**Status.** Phases 0 to 6b and the offline part of Phase 7 are done. The live evaluation (and with it the prompt tuning and trimming of Phase 7) is deferred because the free Groq tier is too slow (PLAN.md section 15). The recommendation in section 5 is provisional.

---

## 2. What is verified (offline only)

Everything below ran with no Groq call and no rooms-API call (stub LLM, offline rooms fixtures).

| Check | Result |
|---|---|
| `python -m pytest llm_approach/tests -q` | **843 passed** (817 plus 26 added by the independent audit), about 10 s, no network, no torch |
| `run_safety` (real guardrail loaded) | **43/43 attacks blocked**; **0/144 false positives** on general messages; **room-text injection 6/6** neutralised |
| `run_robustness` | **7/7 clean** (Groq timeout, invalid JSON twice, 429, rooms API 500 with a Prisma message, rooms timeout, empty results, failing output guardrail): consistent history, no raw error text, rooms kept where required |
| `run_scenarios --stub` | **67/67 turns pass** across 28 scenarios (S1 to S28) |
| `run_eval --dry-run --dataset all` | 186 calls, about 688K tokens, about 93 min at 30 s spacing. Estimate only |
| Website test, checklist items 1 to 10 (PLAN section 10) | Run by hand against port 8001 by the project owner and reported as done. The per-item replies are not stored in this folder |

Added in this phase (Phase 7, offline):

- **Log audit** (`tests/test_log_audit.py`): a stub conversation with searches, follow-ups, a blocked message, a Groq error naming the key, a rate limit, invalid JSON and a rooms-API failure. No log line, metrics record or error reply contains user text, room text, a token or an env value. **It found one leak and the leak is fixed:** the rooms API's error `message` was logged verbatim, and Prisma-style messages echo query values and could echo a token. `rooms_client.scrub_message` now removes the search values, the token and token-shaped text before logging.
- **Blank Groq key:** an empty or whitespace key used to build a client and send a request with a blank key. `llm_client._get_client` now fails before any request (same `KeyError` as a missing key); the pipeline falls back and the user sees a normal reply.
- **Regression scenarios S24 to S28:** Groq rate limit (busy reply, no search, rooms kept), placeholder knowledge-base text (refused), closing-tag injection variants in the user message (blocked before the model), spaced or half fence tags that are not blocked (neutralised, turn stays normal), and the Koramangala "Which area?" loop (area reply, then a budget keeps city and area). Earlier regressions already in the set: S16 rooms 500, S19 to S22 area, S23 pagination through `meta`, S14 injection and credential requests.
- **Concurrency and restart** (`tests/test_concurrency_restart.py`): two sessions interleave while one session's turns stay in order; a cancelled turn releases its lock; a fresh `SessionStore` after a simulated restart starts clean (no stale rooms, no error) and the server keeps working.
- **Startup config** (`tests/test_startup_config.py`): the CLI with a missing or empty `GROQ_API_KEY` says so and exits with code 2; the server starts, reports the keys as missing on `/ready` and answers `/chat` with a friendly reply. No secret appears in any message.

Not changed in this phase: the system prompt text, the few-shot examples, `schema.py`, `validators.py`, `prompts.py`.

---

## 3. What is NOT verified yet

- **Live accuracy against the baseline.** The baseline is the current router measured with the real `run_graph` harness: **57.9% pooled** (77 of 133) and **27.3% on ctxY** (9 of 33). The figures from `evaluate.py` (53.4% and 21.2%) are the ones already quoted to the team lead. Option B has no measured number on either set.
- **Consistency across runs** (same decision in 3 runs of the same input).
- **Invalid-JSON rate** of the real model on real prompts. One early smoke call worked; three calls are not a measurement.
- **Token use per call in real traffic.** Only estimates exist (about 3.7K tokens per call).
- **Latency** (p50, p95 of the model call) on real calls. The tool exists (`python -m llm_approach.metrics`, reads `eval/results/metrics.jsonl` written when `LLM_APPROACH_METRICS=1`); no real traffic has been measured.
- **Groq prompt caching** (whether the usage object reports cached tokens).
- **Real-model behaviour on the scenarios.** The stub runs prove the plumbing (prompt, client, validation, actions, state); only the live run shows whether the model follows the examples.
- The rooms API through the pipeline for large cities (maximum page size unknown), a real uvicorn run, and website checklist items 11 and 12.

All evaluation data is synthetic (the POC sets). Even after the live run, results show direction, not production performance.

---

## 4. Expected benefit and risk (expectations, not results)

**Expected benefit.**
- The model sees the conversation memory, so short replies ("in udupi", "koramangala", "none of these are good", "the second one") should be understood in context. This is where the current router is weakest (27.3% on ctxY).
- Validation in code means a wrong model guess becomes a clarifying question, not a bad search. The invalid-search rate should stay 0 by construction.
- Safety should stay at least as good as today: the new layer blocks phrasings the current hard block misses, before any model call.

**Expected risk.**
- Latency: one hosted model call per message, probably 1 to 4 s added (unmeasured).
- Cost and quota: about 3.7K tokens per call. The free tier allows about 54 calls a day, so this cannot run on the free tier beyond testing.
- Model variance: the same input may not always give the same decision.
- Sending user messages to a hosted provider needs the team lead's approval before any real-user use.
- Results on synthetic data may look better or worse than real traffic.

---

## 5. Provisional recommendation

**Continue to the live evaluation.** The offline work is complete and nothing found so far argues against it. The decision to adopt Option B waits for the live numbers.

**Go / no-go rule, written before the runs.** PLAN.md section 8.3 gives targets but leaves "clearly above" undefined. The numbers marked *proposed* are mine and need the team lead's confirmation.

| Criterion | Go | Source |
|---|---|---|
| Safety block rate | 100% of attacks blocked or neutralised | PLAN 8.3 |
| Invalid-search rate | 0 | PLAN 8.3 |
| Stub and live scenarios (E1) | all turns pass, or each failure explained | PLAN 8.3 |
| Pooled intent accuracy (133 items) | at least 10 points above the baseline's 57.9%, i.e. 68% or more | threshold *proposed, to be confirmed by team lead* |
| ctxY intent accuracy (33 items) | at least 20 points above 27.3%, i.e. 47% or more | threshold *proposed, to be confirmed by team lead* |
| Both of the above | the lower end of the 95% interval must also beat the baseline | *proposed, to be confirmed by team lead* |
| Consistency across 3 runs | 95% or more identical decisions | PLAN 8.3 |
| p95 model latency | 3 s or less, excluding the rooms API | PLAN 8.3 |

**No-go** if any of these holds: safety below 100%; any invented-slot search; pooled or ctxY accuracy not above the baseline; consistency below 90%; p95 latency above 6 s (*proposed*) with no way to reduce it.

**If only part passes:**
- Accuracy passes but consistency or latency fails: trim the prompt, check prompt caching, try a smaller or faster model.
- Accuracy fails on a few intent categories: tune the examples for those categories only, then re-run the affected sets (the prompt is frozen until the first full run is done).
- ctxY passes but pooled fails (or the reverse): look at the paired table in `comparison.md` to see where the two systems disagree before deciding; a hybrid (current router first, model for context-dependent turns) is an option that needs separate approval.

---

## 6. Limitations

- **`reuse.py` extracts source from `agent_modes.py`.** To keep imports light, constants and helpers (model name, grounding rule, room sanitiser, delimiter neutraliser) are loaded by parsing that file with `ast`. If those definitions start depending on other module-level names, extraction fails and it falls back to a normal (slow) import; the tests still pass in that case.
- **`poc_classifier/` is not in the repo** (untracked). `eval/baseline_current.py` and the dataset loaders read it, so nobody else can re-run the baseline or the live evaluation without that folder.
- **Rooms API.** Render cold starts make the first request slow (a timeout was seen once). The database has a connection limit of 1 and fails intermittently, so there are no parallel calls and no retry on 5xx. The API has no amenity or area parameter, so both are filtered client side, on at most 30 fetched rooms; the maximum `limit` for large cities is unverified.
- **Groq free tier.** About 200K tokens a day and 8,000 tokens per minute; one call is about 3.7K tokens. Rate-limit misses are reported separately and are not counted as wrong.
- **Knowledge base quality.** Retrieved answers can contain placeholder text or be loosely related. This pipeline refuses placeholder text; the content itself needs review (PENDING.md).
- **Synthetic, Bangalore-only fixtures.** Evaluation sets are synthetic; offline rooms are a 10-room Bangalore fixture and a 25-room synthetic one. Some items cannot be represented faithfully (selected room, pending question, gold written against the old harness's rooms); `comparison.md` flags them.
- **Fallback is rule-based and limited:** no area extraction during a Groq outage.
- **No streaming from the model:** one non-streaming JSON call, so `token` events are the finished reply split into chunks.
- **Not implemented:** `/feedback`, `/metrics`, `/knowledge/health`, the legacy `/api/chat`.
- **E5 Hinglish set** is skipped (no data file).

---

## 7. How to run everything

PowerShell, from the repo root, venv active. Needs `GROQ_API_KEY` and `KANGAROO_TOKEN` in `.env` (never printed).

```powershell
cd "C:\Faizal\AI intern\Homilivo chatbot"
.\venv\Scripts\Activate.ps1
```

**Tests and offline checks (no Groq, no rooms API):**

```powershell
python -m pytest llm_approach/tests -q
python -m llm_approach.eval.run_safety
python -m llm_approach.eval.run_robustness
python -m llm_approach.eval.run_scenarios --stub
python -m llm_approach.eval.run_eval --dry-run --dataset all
```

**CLI** (real Groq; guardrails load takes about 40 s):

```powershell
python -m llm_approach.cli --offline-rooms --debug     # fixture rooms
python -m llm_approach.cli --debug                     # real rooms API
```

Commands inside: `/state`, `/reset`, `/debug on|off`, `/quit`.

**Server on port 8001** (always with `--no-access-log`, the WebSocket token is in the URL):

```powershell
.\llm_approach\scripts\start-server.ps1
# runs: python -m uvicorn llm_approach.server_app:app --port 8001 --no-access-log
```

**Frontend script** (points the local website at 8001, writes only the git-excluded `.env.development.local`):

```powershell
Copy-Item llm_approach\scripts\start-frontend-llm.ps1.template ..\start-frontend-llm.ps1   # once
..\start-frontend-llm.ps1
```

To go back to the old bot, point `VITE_CHATBOT_WS_URL`, `VITE_API_BASE_URL` and the token at port 8000 (README Phase 5).

**Resume the live evaluation** (each command stops at the daily token budget and prints how to continue):

```powershell
python -m llm_approach.eval.run_eval --dataset ctxY
python -m llm_approach.eval.run_eval --dataset matrix55
python -m llm_approach.eval.run_eval --dataset fresh
python -m llm_approach.eval.run_eval --dataset golden
python -m llm_approach.eval.run_eval --dataset golden_slots
python -m llm_approach.eval.run_scenarios --live --spacing 30
python -m llm_approach.eval.run_eval --dataset all --sample 30 --runs 3
python -m llm_approach.eval.report
```

---

## 8. Integration notes (listed only, nothing was done)

To switch the real backend to this pipeline later, these would be needed. Every item edits code outside `llm_approach/` and needs separate approval.

1. Decide the cutover shape: run shadow mode first (call both, log the difference, answer from the old one), then switch, or switch `app/api.py` to call `llm_approach.pipeline.handle_turn` behind a flag.
2. Wire `handle_turn` into `app/api.py`'s `_run_chat_turn` (REST and WebSocket), mapping `TurnResult` to the existing `ChatResponse` (the server already builds it in `llm_approach/server_app.py`).
3. Move or share the session store: the new in-memory `SessionStore` is single-process; Redis-backed state would be needed for more than one worker (the old code has `STATE_BACKEND`).
4. Add `GROQ_API_KEY` handling and a startup check to the deployment, and a quota or paid plan sized for the traffic (about 3.7K tokens per message).
5. Replace the `ast` extraction in `reuse.py` with normal imports, or move the shared constants into a small module that both sides import.
6. Merge `extra_security.py` patterns into `app/guardrails/input_guardrail.py` (or adopt the PR #25 patterns) so the old path is covered too.
7. Fix the knowledge base content and retrieval threshold (placeholder text, loose matches).
8. Promote the evaluation tooling and datasets into the repo (`poc_classifier/` data is untracked).
9. Approval for sending user messages to Groq, a data-retention check, and log review (this pipeline logs no message text).
10. Add `/feedback`, `/metrics` and `/knowledge/health` if the frontend or operations use them.
11. Update the frontend default `mode` or drop the field; the new server accepts and ignores it.
