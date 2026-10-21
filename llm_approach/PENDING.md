# Pending work (LLM approach)

One-page checklist of everything still open. Last updated 2026-10-19 (session memory added; Gemini provider switch added, see the last section). See [PLAN.md](PLAN.md) section 15 for the reason and the resume commands.

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


## Conversation memory (added 2026-10-19)

- [ ] Run the live evaluation and the scenarios live on this version (S53 to S69 are stub-only today): the prompt, schema and validators changed again (prompt 17,344 -> 18,570 characters, 12 intents)
- [ ] Check with the real model that "what's my college", "why did you show me that room", "summarise our chat" and "what did you say about the second room" are labelled CONVERSATION_QUESTION (not OUT_OF_SCOPE, not CONVERSATION_RECALL), and that its answers are grounded
- [ ] Decision for the team lead: phone numbers and e-mail addresses. They are refused as facts but stay in the in-session transcript and can be quoted back to the same session. Keep, or refuse them in the transcript too?
- [ ] Decision for the team lead: the model now writes the text of a conversation answer (the one exception to "the model never writes the reply"). The checks verify grounding, not meaning
- [ ] A conversation-memory evaluation set (questions with gold answers, wrong-answer and injection cases) needs the team lead's approval before it is made
- [ ] Cost: a conversation question is two Groq calls, about 5,300 to 11,000 tokens (about 4,800 for a normal turn); on the free tier a long chat can hit the per-minute limit. Consider a smaller model or a shorter record for the answer call
- [ ] Privacy note for the team lead: the whole transcript (both roles, up to 200 messages) is kept in memory for the session (TTL 3600 s) and sent to Groq only in the answer call of a conversation question; never logged, never in metrics, never written to disk, never sent to the rooms API


## Gemini provider switch (added 2026-10-19)

The default provider is now Gemini `gemini-3.5-flash-lite` (the team lead approved it for the live evaluation); Groq stays available with `LLM_APPROACH_PROVIDER=groq`. Items above that say "Groq" now apply to the active provider; the items about Groq's quota apply only when Groq is selected.

- [ ] Install the SDK and run the live smoke test once: `pip install -r llm_approach/requirements-gemini.txt`, `GEMINI_API_KEY` in `.env`, then `python -m llm_approach.smoke_gemini`. Confirm it prints `OK`, a valid decision and token counts (this also confirms the live model accepts the thinking setting for its family: `thinking_level=LOW` on `gemini-3.x`)
- [ ] Add to `.env`: `LLM_APPROACH_GEMINI_SPACING_S=5` and `LLM_APPROACH_GEMINI_DAILY_TOKEN_BUDGET=2000000` (our own cap: Google lists no daily token limit). Without the budget `run_eval` and `run_scenarios --live` refuse to start. Limits of `gemini-3.5-flash-lite` on the free tier: 15 requests/min, 250K tokens/min, 500 requests/day (the binding one). The code keeps a Pacific-day request counter (`LLM_APPROACH_GEMINI_DAILY_REQUEST_BUDGET`, default 450), a 12 requests / 200K tokens per 60 s window and a 5 s spacing (minimum 4 s)
- [ ] Plan the days first: `python -m llm_approach.eval.plan` (estimated requests and minutes per dataset, suggested order per day). `python -m llm_approach.llm_info` shows requests used today. A live run prints its own plan and refuses to start if the worst case does not fit in what is left today
- [ ] If a run prints `STOPPED EARLY ... Resume with: <command>`, run exactly that command (it ends in `--resume`; only the same provider, model and dataset can be resumed). The report refuses to compare an unfinished run
- [ ] The first real prompt of a live run is the connectivity check; a wrong key, model or quota stops the run after one call with a clear message
- [ ] Run the live evaluation on Gemini (same order and commands as above); regenerate the report with `python -m llm_approach.eval.report --provider gemini --model gemini-3.5-flash-lite`. Old Groq results, if any, stay in the same files and are never merged into a Gemini table
- [ ] Phase 7: the prompt was written for `gpt-oss-20b` and may need tuning on Gemini after the first live results. Prompts, schema and validators stay frozen during the evaluation
- [ ] Check whether Gemini reports cached tokens for the 18K-character system prompt (`cached_content_token_count`; the report says "not reported" if absent) and measure real tokens per call (the dry run is an estimate; tokenizers differ)
- [ ] Check the real Gemini error shapes: a 429 must give the "lot of requests" reply, a blocked or empty reply must retry once and fall back (tested with fakes shaped like the SDK's `ClientError`, not live)
- [ ] Privacy: free-tier content may be used by Google under its terms (check the current terms); the evaluation data is synthetic. Do not send real user messages through a free-tier key
- [ ] Check whether the frontend or any dashboard reads `/ready` -> `checks.groq_key_present`; it is now `checks.llm_key_present` (with `provider` and `model` at the top level)
- [ ] `eval/results/comparison.md` and `comparison.json` were generated before the labels existed and have not been regenerated
- [ ] Thinking `LOW` (default on `gemini-3.x`) may add latency and output tokens, and the 1024 output cap is a guess. The live results will show the real cost (check the `thinking` token count in the smoke test and the latency/token tables); adjust with `LLM_APPROACH_GEMINI_THINKING` (`level:MINIMAL`, `none`) and `LLM_APPROACH_GEMINI_MAX_OUTPUT_TOKENS` if needed
