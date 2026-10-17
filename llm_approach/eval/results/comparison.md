# LLM approach: evaluation and comparison

> **PARTIAL RESULTS: 0 of 186 items evaluated.** Every number below covers only the items done so far.

| Dataset | Items done | Total | Rate-limit misses |
|---|---|---|---|
| ctxY | 0 | 33 | 0 |
| matrix55 | 0 | 54 | 0 |
| fresh | 0 | 59 | 0 |
| golden | 0 | 20 | 0 |
| golden_slots | 0 | 20 | 0 |

## What is compared, and how

- **All data is synthetic** (the POC test sets). Results show direction, not production accuracy.
- **Level of comparison.** The new decision layer is scored at the **intent level** (10 intents, `label_map.py`). The old router's published figures are **end-to-end routing** against the 23 POC labels. The paired table below maps the old router's per-item prediction to the same 10 intents, so it is like for like; the stored label-level figures are shown only for reference. Mapping to 10 intents merges GREETING, ACKNOWLEDGMENT and CLOSING into GENERAL_CHAT, which makes intent-level scores slightly easier than label-level ones.
- **Baseline used.** The real `run_graph` harness (setup A): **57.9% pooled (133: golden + fresh + matrix55)** and **27.3% on ctxY (33)**. For reference, the `evaluate.py` figures quoted earlier: 53.4% pooled and 21.2% ctxY. The harness golden rows had 12 replay misses, so its golden figure is less certain.
- **Pooled sets.** *pooled 133* = golden + fresh + matrix55 (the baseline's pooled set); *pooled 166* adds the 33 ctxY items. Each table says which one it uses.
- **Strict / lenient.** Strict: predicted intent equals the gold label's intent. Lenient: the gold intent or the intent of an `acceptable` label.
- **Brackets** are Wilson 95% confidence intervals. Headline numbers use the first run of every item.
- **ROOM_TABLE** is not in the PLAN 6.4 mapping. It was added as ROOM_FOLLOWUP / ROOM_DETAILS (decision recorded in the README).
- **Rooms.** Items with rooms on screen use the 10 fixture rooms (Bangalore, nothing selected, no history).
- Never stored or shown here: keys, tokens, message text, model reasoning.

## Intent accuracy (new decision layer)

| Set | Strict | Lenient |
|---|---|---|
| ctxY | n/a | n/a |
| matrix55 | n/a | n/a |
| fresh | n/a | n/a |
| golden | n/a | n/a |
| pooled_133 | n/a | n/a |
| pooled_166 | n/a | n/a |

### Paired comparison with the old router (same items, intent level, strict)

| Set | n | New | Old | New right, old wrong | New wrong, old right | Both right | Both wrong |
|---|---|---|---|---|---|---|---|
| ctxY | 0 | - | - | - | - | - | - |
| matrix55 | 0 | - | - | - | - | - | - |
| fresh | 0 | - | - | - | - | - | - |
| golden | 0 | - | - | - | - | - | - |
| pooled_133 | 0 | - | - | - | - | - | - |
| pooled_166 | 0 | - | - | - | - | - | - |

Lenient version (pooled sets):

_no data yet_

### Rooms on screen versus none (pooled 166, strict)

| Category | Accuracy |
|---|---|
| - | n/a |

### By gold label (pooled 166, strict)

| Gold label | Accuracy |
|---|---|
| - | n/a |

## Operation, clarification, validity

| Metric | Value |
|---|---|
| Operation accuracy, ROOM_FOLLOWUP items (pooled 166, strict) | n/a |
| Operation accuracy (lenient) | n/a |
| Clarification precision (gold ask = UNCLEAR) | n/a |
| Clarification recall | n/a |
| Invalid-slot rate (search intents, validators rejected a slot) | n/a |
| Invalid-search rate (a search would carry a rejected slot; target 0) | n/a |
| No usable decision (invalid JSON twice, timeout, API error) | n/a |
| Consistency across runs (items with 2+ runs, identical decision) | n/a |

The clarification gold is only the UNCLEAR label; the sets have no other 'should ask' labels, so read these as a proxy. The invalid-search rate is evaluated with the pipeline's own rule (a rejected city, BHK or price makes it ask instead of searching); the end-to-end check of that rule is in the pipeline tests and the robustness report.

### Calibration (pooled 166, strict)

| Confidence | Accuracy |
|---|---|
| 0.0-0.6 | n/a |
| 0.6-0.8 | n/a |
| 0.8-0.9 | n/a |
| 0.9-1.0 | n/a |

## Cost

| Metric | Value |
|---|---|
| Latency p50 / p95 (LLM call incl. retry) | n/a / n/a (n=0) |
| Tokens in / out per call | n/a |
| Cached tokens per call | not reported |

## Slot accuracy (`golden_slots`, reported separately from intent accuracy)

_no data yet_

## Items that could not be represented faithfully

They were still run. The data records only a yes/no flag for rooms on screen, so the following cannot be rebuilt exactly:
- a **selected room** (follow-ups such as 'does it have wifi' may assume one);
- a **pending bot question** (a bare 'pune' or 'ok' may answer one);
- **room-dependent gold** written against the old harness's recorded rooms, while the new run uses the 10 fixture rooms.

| Dataset | Items | Reason |
|---|---|---|
| matrix55 | 17 | gold written against the old harness's rooms; fixture rooms used |
| matrix55 | 3 | gold written against the old harness's rooms; fixture rooms used; short reply: gold may assume a pending bot question or selected room that the data does not record |
| matrix55 | 3 | short reply: gold may assume a pending bot question or selected room that the data does not record |
| fresh | 6 | short reply: gold may assume a pending bot question or selected room that the data does not record |
| fresh | 19 | gold written against the old harness's rooms; fixture rooms used |

Intent accuracy on rooms-on-screen items: flagged n/a; not flagged n/a. A large gap means part of the result depends on the missing context rather than on the model.

## Scenario pass/fail (live, `scenarios_live.jsonl`)

_no live scenario results yet_

## Safety (offline, `safety.json`)

| Check | Result |
|---|---|
| Real guardrail used | True |
| Attack block rate (s1-s6, s8, PR #25 phrases, other attacks) | 43/43 |
| s1-s8 attacks not blocked | none |
| s7 (payment request, not a security block) | not blocked (handled as UNSUPPORTED_ACTION) |
| False positives on general messages | 0/144  |
| Room-text injection neutralised | 6/6 |

The prompt keeps only index, title, price, type and location of a room. 'neutralised' means the room text cannot add or close a <verified_data>/<user_message> fence; instruction-like words inside a title or location can still appear as data (the system prompt tells the model it is data).

## Robustness (offline mocks, `robustness.json`)

7 of 7 cases clean.

| Case | Reply type | History consistent | State kept where required | Raw error leaked |
|---|---|---|---|---|
| groq_timeout | fallback:fallback_new_search | yes | yes | no |
| groq_invalid_json_twice | fallback:fallback_new_search | yes | yes | no |
| groq_rate_limit | rate_limit_busy | yes | yes | no |
| rooms_api_500_prisma | rooms_unavailable | yes | yes | no |
| rooms_api_timeout | rooms_unavailable | yes | yes | no |
| empty_results | new_search | yes | yes | no |
| output_guardrail_blocks | output_blocked | yes | yes | no |

## Proposed targets (PLAN section 8.3)

| Metric | Proposed target | Current |
|---|---|---|
| Safety block rate | 100% | 43/43 |
| Invalid-search rate | 0 | n/a |
| E1 scenario pass rate | 100% of turns, or each failure explained | not run live |
| Intent accuracy, E2 pooled (133) | clearly above the current pipeline's 53.4% (harness: 57.9%) | n/a |
| Intent accuracy, E2 ctxY | clearly above 21.2% (harness: 27.3%) | n/a |
| Consistency across 3 runs | >= 95% identical decisions | n/a |
| p95 LLM latency | <= 3 s (rooms API excluded) | n/a |

Robustness: 7/7 cases clean.
