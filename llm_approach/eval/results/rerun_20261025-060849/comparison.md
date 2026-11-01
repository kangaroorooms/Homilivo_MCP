# LLM approach: evaluation and comparison

**provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb.** One provider/model pair and one prompt version per report; results of different pairs are never merged or compared in one table.

> **PARTIAL RESULTS: 20 of 186 items evaluated.** Every number below covers only the items done so far.

| Dataset | Items done | Total | Failed calls (rate limit, timeout, other) |
|---|---|---|---|
| ctxY | 0 | 33 | 0 |
| matrix55 | 20 | 54 | 1 |
| fresh | 0 | 59 | 0 |
| golden | 0 | 20 | 0 |
| golden_slots | 0 | 20 | 0 |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

Failed calls: 1 (rate limit 0, timeout 1, server error 0, other 0). They are not counted as wrong answers anywhere. Result files from before the kind was recorded are classified from their error fields; a record with none shows as other. Re-run with `--only-missing` to retry them.

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
| matrix55 | 90.0% [69.9-97.2] (18/20) | 100.0% [83.9-100.0] (20/20) |
| fresh | n/a | n/a |
| golden | n/a | n/a |
| pooled_133 | 90.0% [69.9-97.2] (18/20) | 100.0% [83.9-100.0] (20/20) |
| pooled_166 | 90.0% [69.9-97.2] (18/20) | 100.0% [83.9-100.0] (20/20) |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

### Paired comparison with the old router (same items, intent level, strict)

| Set | n | New | Old | New right, old wrong | New wrong, old right | Both right | Both wrong |
|---|---|---|---|---|---|---|---|
| ctxY | 0 | - | - | - | - | - | - |
| matrix55 | 20 | 90.0% [69.9-97.2] (18/20) | 65.0% [43.3-81.9] (13/20) | 6 | 1 | 12 | 1 |
| fresh | 0 | - | - | - | - | - | - |
| golden | 0 | - | - | - | - | - | - |
| pooled_133 | 20 | 90.0% [69.9-97.2] (18/20) | 65.0% [43.3-81.9] (13/20) | 6 | 1 | 12 | 1 |
| pooled_166 | 20 | 90.0% [69.9-97.2] (18/20) | 65.0% [43.3-81.9] (13/20) | 6 | 1 | 12 | 1 |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

Lenient version (pooled sets):

| Set | n | New | Old | New right, old wrong | New wrong, old right | Both right | Both wrong |
|---|---|---|---|---|---|---|---|
| pooled_133 | 20 | 100.0% [83.9-100.0] (20/20) | 65.0% [43.3-81.9] (13/20) | 7 | 0 | 13 | 0 |
| pooled_166 | 20 | 100.0% [83.9-100.0] (20/20) | 65.0% [43.3-81.9] (13/20) | 7 | 0 | 13 | 0 |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

### Rooms on screen versus none (pooled 166, strict)

| Category | Accuracy |
|---|---|
| ctxN | 90.9% [62.3-98.4] (10/11) |
| ctxY | 88.9% [56.5-98.0] (8/9) |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

### By gold label (pooled 166, strict)

| Gold label | Accuracy |
|---|---|
| FAMILY_SUITABILITY | 0.0% [0.0-79.3] (0/1) |
| LOWEST_PRICE | 100.0% [20.7-100.0] (1/1) |
| NEW_SEARCH | 100.0% [72.2-100.0] (10/10) |
| REFINEMENT | 100.0% [34.2-100.0] (2/2) |
| ROOM_DETAILS | 100.0% [34.2-100.0] (2/2) |
| ROOM_TABLE | 100.0% [20.7-100.0] (1/1) |
| SHOW_MORE_OPTIONS | 100.0% [34.2-100.0] (2/2) |
| UNCLEAR | 0.0% [0.0-79.3] (0/1) |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

## Operation, clarification, validity

| Metric | Value |
|---|---|
| Operation accuracy, ROOM_FOLLOWUP items (pooled 166, strict) | 80.0% [37.6-96.4] (4/5) |
| Operation accuracy (lenient) | 80.0% [37.6-96.4] (4/5) |
| Clarification precision (gold ask = UNCLEAR) | 16.7% [3.0-56.4] (1/6) |
| Clarification recall | 100.0% [20.7-100.0] (1/1) |
| Invalid-slot rate (search intents, validators rejected a slot) | 0.0% [0.0-21.5] (0/14) |
| Invalid-search rate (a search would carry a rejected slot; target 0) | 0.0% [0.0-21.5] (0/14) |
| No usable decision (invalid JSON twice, timeout, API error) | 0.0% [0.0-6.1] (0/59) |
| Consistency across runs (strict: identical intent, operation, room and slots) | 90.0% [69.9-97.2] (18/20) |
| Same action across runs (same user-visible action and filters) | 100.0% [83.9-100.0] (20/20) |
| Strict consistency, only items with all 3 runs | 89.5% [68.6-97.1] (17/19) |
| Same action, only items with all 3 runs | 100.0% [83.2-100.0] (19/19) |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

**Consistency.** *Strict* needs identical intent, operation, room reference and slots in every run (confidence is ignored). *Same action* needs the runs to lead to the same user-visible action: search (with the filters sent to the rooms API), clarifying question, follow-up (operation and room), show more, platform answer, chat or blocked. A different label or confidence that ends in the same action does not count against it. Both use the items with at least 2 usable runs; an item with a failed call is judged on its remaining runs (1 item(s) have fewer than 3). The last two rows count only items with every planned run usable.

The clarification gold is only the UNCLEAR label; the sets have no other 'should ask' labels, so read these as a proxy. The invalid-search rate is evaluated with the pipeline's own rule (a rejected city, BHK or price makes it ask instead of searching); the end-to-end check of that rule is in the pipeline tests and the robustness report.

### Calibration (pooled 166, strict)

| Confidence | Accuracy |
|---|---|
| 0.0-0.6 | n/a |
| 0.6-0.8 | n/a |
| 0.8-0.9 | 83.3% [43.6-97.0] (5/6) |
| 0.9-1.0 | 92.9% [68.5-98.7] (13/14) |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

## Cost

| Metric | Value |
|---|---|
| Latency p50 / p95 (LLM call incl. retry) | 1523 ms / 3024 ms (n=59) |
| Tokens in / out per call | 6380 / 208 |
| Cached tokens per call | 4031 |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

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

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

Intent accuracy on rooms-on-screen items: flagged 85.7% [48.7-97.4] (6/7); not flagged 100.0% [34.2-100.0] (2/2). A large gap means part of the result depends on the missing context rather than on the model.

## Scenario pass/fail (live, `scenarios_live.jsonl`)

_no live scenario results yet_

## Safety (offline, `safety.json`)

_not run yet: `python -m llm_approach.eval.run_safety`_

## Robustness (offline mocks, `robustness.json`)

_not run yet: `python -m llm_approach.eval.run_robustness`_

## Proposed targets (PLAN section 8.3)

| Metric | Proposed target | Current |
|---|---|---|
| Safety block rate | 100% | not run |
| Invalid-search rate | 0 | 0.0% [0.0-21.5] (0/14) |
| E1 scenario pass rate | 100% of turns, or each failure explained | not run live |
| Intent accuracy, E2 pooled (133) | clearly above the current pipeline's 53.4% (harness: 57.9%) | 90.0% [69.9-97.2] (18/20) |
| Intent accuracy, E2 ctxY | clearly above 21.2% (harness: 27.3%) | n/a |
| Consistency across 3 runs | >= 95% identical decisions (strict: intent, operation, room, slots) | 90.0% [69.9-97.2] (18/20) |
| Same action across 3 runs | >= 95% of items lead to the same user-visible action and filters (proposed) | 100.0% [83.9-100.0] (20/20) |
| p95 LLM latency | <= 3 s (rooms API excluded) | 3024 ms |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_
