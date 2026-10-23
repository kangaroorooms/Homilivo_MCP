# LLM approach: evaluation and comparison

**provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb.** One provider/model pair and one prompt version per report; results of different pairs are never merged or compared in one table.

| Dataset | Items done | Total | Rate-limit misses |
|---|---|---|---|
| ctxY | 33 | 33 | 0 |
| matrix55 | 54 | 54 | 0 |
| fresh | 59 | 59 | 0 |
| golden | 20 | 20 | 0 |
| golden_slots | 20 | 20 | 0 |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

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
| ctxY | 69.7% [52.7-82.6] (23/33) | 69.7% [52.7-82.6] (23/33) |
| matrix55 | 88.9% [77.8-94.8] (48/54) | 96.3% [87.5-99.0] (52/54) |
| fresh | 96.6% [88.5-99.1] (57/59) | 98.3% [91.0-99.7] (58/59) |
| golden | 100.0% [83.9-100.0] (20/20) | 100.0% [83.9-100.0] (20/20) |
| pooled_133 | 94.0% [88.6-96.9] (125/133) | 97.7% [93.6-99.2] (130/133) |
| pooled_166 | 89.2% [83.5-93.0] (148/166) | 92.2% [87.1-95.4] (153/166) |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

### Paired comparison with the old router (same items, intent level, strict)

| Set | n | New | Old | New right, old wrong | New wrong, old right | Both right | Both wrong |
|---|---|---|---|---|---|---|---|
| ctxY | 33 | 69.7% [52.7-82.6] (23/33) | 27.3% [15.1-44.2] (9/33) | 16 | 2 | 7 | 8 |
| matrix55 | 54 | 88.9% [77.8-94.8] (48/54) | 61.1% [47.8-73.0] (33/54) | 17 | 2 | 31 | 4 |
| fresh | 59 | 96.6% [88.5-99.1] (57/59) | 54.2% [41.7-66.3] (32/59) | 25 | 0 | 32 | 2 |
| golden | 20 | 100.0% [83.9-100.0] (20/20) | 95.0% [76.4-99.1] (19/20) | 1 | 0 | 19 | 0 |
| pooled_133 | 133 | 94.0% [88.6-96.9] (125/133) | 63.2% [54.7-70.9] (84/133) | 43 | 2 | 82 | 6 |
| pooled_166 | 166 | 89.2% [83.5-93.0] (148/166) | 56.0% [48.4-63.4] (93/166) | 59 | 4 | 89 | 14 |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

Lenient version (pooled sets):

| Set | n | New | Old | New right, old wrong | New wrong, old right | Both right | Both wrong |
|---|---|---|---|---|---|---|---|
| pooled_133 | 133 | 97.7% [93.6-99.2] (130/133) | 66.9% [58.5-74.3] (89/133) | 42 | 1 | 88 | 2 |
| pooled_166 | 166 | 92.2% [87.1-95.4] (153/166) | 59.0% [51.4-66.2] (98/166) | 58 | 3 | 95 | 10 |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

### Rooms on screen versus none (pooled 166, strict)

| Category | Accuracy |
|---|---|
| ctxN | 91.4% [82.5-96.0] (64/70) |
| ctxY | 87.5% [79.4-92.7] (84/96) |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

### By gold label (pooled 166, strict)

| Gold label | Accuracy |
|---|---|
| ACKNOWLEDGMENT | 75.0% [30.1-95.4] (3/4) |
| AMENITY | 100.0% [43.8-100.0] (3/3) |
| CLOSING | 100.0% [43.8-100.0] (3/3) |
| COMPARE_ROOMS | 100.0% [43.8-100.0] (3/3) |
| FAMILY_SUITABILITY | 50.0% [9.5-90.5] (1/2) |
| GREETING | 100.0% [43.8-100.0] (3/3) |
| HIGHEST_PRICE | 100.0% [20.7-100.0] (1/1) |
| LOCATION | 100.0% [34.2-100.0] (2/2) |
| LOWEST_PRICE | 100.0% [43.8-100.0] (3/3) |
| NEW_SEARCH | 87.1% [78.3-92.6] (74/85) |
| OUT_OF_SCOPE | 100.0% [43.8-100.0] (3/3) |
| PLATFORM_INFO | 100.0% [43.8-100.0] (3/3) |
| PROFESSIONAL_SUITABILITY | 100.0% [20.7-100.0] (1/1) |
| REFINEMENT | 100.0% [67.6-100.0] (8/8) |
| REJECT_CURRENT | 100.0% [43.8-100.0] (3/3) |
| ROOM_DETAILS | 100.0% [51.0-100.0] (4/4) |
| ROOM_PRICE | 100.0% [43.8-100.0] (3/3) |
| ROOM_TABLE | 100.0% [34.2-100.0] (2/2) |
| ROOM_TYPE | 100.0% [20.7-100.0] (1/1) |
| SHOW_MORE_OPTIONS | 100.0% [70.1-100.0] (9/9) |
| STUDENT_SUITABILITY | 100.0% [34.2-100.0] (2/2) |
| UNCLEAR | 66.7% [41.7-84.8] (10/15) |
| UNSUPPORTED_ACTION | 100.0% [43.8-100.0] (3/3) |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

## Operation, clarification, validity

| Metric | Value |
|---|---|
| Operation accuracy, ROOM_FOLLOWUP items (pooled 166, strict) | 92.6% [76.6-97.9] (25/27) |
| Operation accuracy (lenient) | 92.6% [76.6-97.9] (25/27) |
| Clarification precision (gold ask = UNCLEAR) | 37.1% [23.2-53.7] (13/35) |
| Clarification recall | 86.7% [62.1-96.3] (13/15) |
| Invalid-slot rate (search intents, validators rejected a slot) | 5.4% [2.5-11.2] (6/112) |
| Invalid-search rate (a search would carry a rejected slot; target 0) | 0.0% [0.0-3.3] (0/112) |
| No usable decision (invalid JSON twice, timeout, API error) | 0.0% [0.0-2.0] (0/186) |
| Consistency across runs (items with 2+ runs, identical decision) | n/a |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

The clarification gold is only the UNCLEAR label; the sets have no other 'should ask' labels, so read these as a proxy. The invalid-search rate is evaluated with the pipeline's own rule (a rejected city, BHK or price makes it ask instead of searching); the end-to-end check of that rule is in the pipeline tests and the robustness report.

### Calibration (pooled 166, strict)

| Confidence | Accuracy |
|---|---|
| 0.0-0.6 | 83.3% [55.2-95.3] (10/12) |
| 0.6-0.8 | 0.0% [0.0-79.3] (0/1) |
| 0.8-0.9 | 79.2% [59.5-90.8] (19/24) |
| 0.9-1.0 | 92.2% [86.3-95.7] (119/129) |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

## Cost

| Metric | Value |
|---|---|
| Latency p50 / p95 (LLM call incl. retry) | 1609 ms / 3264 ms (n=186) |
| Tokens in / out per call | 6412 / 220 |
| Cached tokens per call | 4031 |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

## Slot accuracy (`golden_slots`, reported separately from intent accuracy)

| Field | Accuracy |
|---|---|
| city | 100.0% [81.6-100.0] (17/17) |
| bhk | 88.2% [65.7-96.7] (15/17) |
| min_price | 100.0% [81.6-100.0] (17/17) |
| max_price | 94.1% [73.0-99.0] (16/17) |
| ideal_for | 100.0% [81.6-100.0] (17/17) |
| overall | 96.5% [90.1-98.8] (82/85) |
| exact | 82.4% [59.0-93.8] (14/17) |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

Over the 17 search items that have slot gold (`tests/golden_set.jsonl`, `expected_args`). Compared on the validated slots; no gold value means the field must be empty. Gold `room_types` other than N-BHK (PG, Single, 1RK) is not a slot of the decision schema, and gold cities such as 'HSR Layout' are localities (the model returns them as `area`), so those items lose city accuracy by construction. There is no paired comparison for this set.

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

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

Intent accuracy on rooms-on-screen items: flagged 97.9% [89.1-99.6] (47/48); not flagged 77.1% [63.5-86.7] (37/48). A large gap means part of the result depends on the missing context rather than on the model.

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

_offline mocks, no LLM call_

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

_offline mocks, no LLM call_

## Proposed targets (PLAN section 8.3)

| Metric | Proposed target | Current |
|---|---|---|
| Safety block rate | 100% | 43/43 |
| Invalid-search rate | 0 | 0.0% [0.0-3.3] (0/112) |
| E1 scenario pass rate | 100% of turns, or each failure explained | not run live |
| Intent accuracy, E2 pooled (133) | clearly above the current pipeline's 53.4% (harness: 57.9%) | 94.0% [88.6-96.9] (125/133) |
| Intent accuracy, E2 ctxY | clearly above 21.2% (harness: 27.3%) | 69.7% [52.7-82.6] (23/33) |
| Consistency across 3 runs | >= 95% identical decisions | n/a |
| p95 LLM latency | <= 3 s (rooms API excluded) | 3264 ms |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

Robustness: 7/7 cases clean.
