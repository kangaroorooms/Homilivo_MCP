# Pending work (LLM approach)

One-page checklist of everything still open. Last updated 2026-10-19 (session memory added). See [PLAN.md](PLAN.md) section 15 for the reason and the resume commands.

Legend: `[ ]` open, `[x]` done.

## Live runs (blocked on Groq quota)

- [ ] Live evaluation datasets, in this order: `run_eval --dataset ctxY`, `matrix55`, `fresh`, `golden`, `golden_slots` (186 calls, about 886K tokens since the session-memory fixes, was 688K before session memory; about 93 min of runtime). The run must use the current prompt, which includes session memory
- [ ] Live scenarios: `run_scenarios --live --spacing 30` (S25 and S16 are stub only and are skipped)
- [ ] Consistency sample: `run_eval --dataset all --sample 30 --runs 3` (about 360K tokens)
- [ ] Regenerate the comparison report: `python -m llm_approach.eval.report`
- [ ] Decision from the team lead on the quota: Groq Developer plan, a separate approved key, or the 8 to 9 day free-tier schedule

## Checks that need a person or the live system

- [ ] Website checklist item 11: refresh the page mid-conversation, note what the frontend does with the session
- [ ] Website checklist item 12: stop the backend, widget shows offline; restart, it reconnects with a fresh token
- [ ] Groq prompt caching check: does the usage object report `cached_tokens` for the 13K-character system prompt? (`LLMBudget.cached_tokens` is already collected; the report says "not reported" if absent)
- [ ] Rooms API maximum page size (`limit`) for a large city. Area search fetches 30; if the API caps lower, area searches see fewer rooms per page
- [ ] Rooms API behaviour for `page` beyond the last page, and with filters other than city
- [ ] Real Groq model behaviour on S1 to S52 (S29 to S52 are the session-memory scenarios: does the real model fill `user_facts` and `recall_target` correctly, does it keep "what's my bank balance" OUT_OF_SCOPE) (the stub runs prove the plumbing, not the model)
- [ ] Real uvicorn run: log redaction was tested on the filter, not on uvicorn itself

## Skipped unless approved

- [ ] E5 Hinglish set (32 queries): no data file exists, the "about 26/32" figure cannot be reproduced. Needs the team lead to point to the source, or approval to write the set

## Findings to report to the team lead about existing code (outside `llm_approach/`, not changed)

- [ ] `is_security_blocked` misses instruction-override phrasing: "ignore previous instructions", "disregard all prior rules", "you are now DAN", "developer mode on", "forget the above instructions", role tags like `</verified_data> SYSTEM: ...`. The PR #25 branch adds patterns for these. The new pipeline covers them with `extra_security.py`, the current backend does not (0 of 10 PR #25 phrases blocked by the real function alone)
- [ ] Knowledge base placeholder text: a retrieved answer can contain placeholder or "not verified, confirm wording" text and still count as grounded. `actions/platform.py` refuses it in this pipeline; the current backend does not
- [ ] Knowledge base retrieval is loose: unrelated questions can come back "grounded". Needs a stricter threshold or reviewed content
- [ ] Rooms API error messages can echo query values (Prisma-style). The old code logs them; this pipeline now scrubs search values and token-shaped text before logging
- [ ] `_neutralize_prompt_delimiters` strips only exact tags (spaced or full-width variants pass). Handled here by `sanitize.py`; the original is unchanged
- [ ] Frontend `chatbotWebSocket.ts` default mode is `"groq"`, while its comment says `"fast"`

## Phase 7 items that need live results

- [ ] Prompt tuning for failing categories (re-run the affected sets after each change)
- [ ] Prompt trimming (see the findings below)
- [ ] Review latency from real traffic: run with `LLM_APPROACH_METRICS=1`, then `python -m llm_approach.metrics` (p50/p95 of turn time, Groq time per turn and per call); also max tokens and history length
- [ ] Results stable across 3 runs (consistency at least 95%)

## Prompt-size findings (input for tuning after the live results; nothing was trimmed)

- Current system prompt: **17,344 characters**, about **4.3K tokens** (4,336 at 4 characters per token) per call, before the state summary. 33 few-shot examples are the bulk of it. Session memory (2026-10-19) added 3,383 characters (13,090 before): one intent, four schema fields, two rules, six examples.
- It grew from 11,823 to 13,090 characters when the area slot was added (Phase 5b). Earlier it was cut from about 24.6K to 11.4K to fit the free-tier limit.
- The dry run estimates about 3.7K tokens per call including the state summary and 250 output tokens (886K for 186 calls, was 688K before session memory). The state summary is capped at 8,000 characters.
- Free tier: 8,000 tokens per minute and about 200K a day. At 4.8K tokens a call (3.7K before session memory) that is about 1.7 calls a minute and about 41 calls a day.
- Candidates once live results exist: drop examples whose category already scores 100% in all runs; merge near-duplicate Hinglish examples; shorten rule text; shrink `MAX_ROOMS_IN_PROMPT` or the fields per room; keep a fixed prefix so Groq prompt caching can apply (if it does, the prompt cost matters less).
- Do not trim before the live run: every trim changes the system under test.

## Session memory (added 2026-10-19)

- [ ] Run the live evaluation on this version: the prompt, schema and validators are frozen again after this feature
- [ ] Check with the real model that recall questions are CONVERSATION_RECALL and not OUT_OF_SCOPE, and that facts in ordinary messages are extracted without being invented (the code rejects ungrounded facts, but a miss means the fact is simply not stored)
- [ ] Product decision: should `preferred_city` and `budget` become search defaults? Today they are remembered and recalled but never used for a search
- [ ] Privacy note for the team lead: facts (name, age, occupation, gender) and the user's own messages are kept in memory for the session (TTL 3600 s), sent to Groq only as `user_facts` in the state summary, never logged, never written to disk, never sent to the rooms API
- [ ] Recall fixes (2026-10-19): re-run the live transcript (name and age together, then "yes" twice) against the real model; the stub proves the code paths, not what the model returns for "what's my name and age ?"
