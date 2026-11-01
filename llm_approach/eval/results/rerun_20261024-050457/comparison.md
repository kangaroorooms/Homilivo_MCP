# LLM approach: evaluation and comparison

**provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb.** One provider/model pair and one prompt version per report; results of different pairs are never merged or compared in one table.

> **PARTIAL RESULTS: 20 of 186 items evaluated.** Every number below covers only the items done so far.

| Dataset | Items done | Total | Rate-limit misses |
|---|---|---|---|
| ctxY | 0 | 33 | 0 |
| matrix55 | 0 | 54 | 0 |
| fresh | 0 | 59 | 0 |
| golden | 0 | 20 | 0 |
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
| ctxY | n/a | n/a |
| matrix55 | n/a | n/a |
| fresh | n/a | n/a |
| golden | n/a | n/a |
| pooled_133 | n/a | n/a |
| pooled_166 | n/a | n/a |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

### Paired comparison with the old router (same items, intent level, strict)

| Set | n | New | Old | New right, old wrong | New wrong, old right | Both right | Both wrong |
|---|---|---|---|---|---|---|---|
| ctxY | 0 | - | - | - | - | - | - |
| matrix55 | 0 | - | - | - | - | - | - |
| fresh | 0 | - | - | - | - | - | - |
| golden | 0 | - | - | - | - | - | - |
| pooled_133 | 0 | - | - | - | - | - | - |
| pooled_166 | 0 | - | - | - | - | - | - |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

Lenient version (pooled sets):

_no data yet_

### Rooms on screen versus none (pooled 166, strict)

| Category | Accuracy |
|---|---|
| - | n/a |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

### By gold label (pooled 166, strict)

| Gold label | Accuracy |
|---|---|
| - | n/a |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

## Operation, clarification, validity

| Metric | Value |
|---|---|
| Operation accuracy, ROOM_FOLLOWUP items (pooled 166, strict) | n/a |
| Operation accuracy (lenient) | n/a |
| Clarification precision (gold ask = UNCLEAR) | n/a |
| Clarification recall | n/a |
| Invalid-slot rate (search intents, validators rejected a slot) | 5.9% [1.0-27.0] (1/17) |
| Invalid-search rate (a search would carry a rejected slot; target 0) | 0.0% [0.0-18.4] (0/17) |
| No usable decision (invalid JSON twice, timeout, API error) | 0.0% [0.0-16.1] (0/20) |
| Consistency across runs (items with 2+ runs, identical decision) | n/a |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

The clarification gold is only the UNCLEAR label; the sets have no other 'should ask' labels, so read these as a proxy. The invalid-search rate is evaluated with the pipeline's own rule (a rejected city, BHK or price makes it ask instead of searching); the end-to-end check of that rule is in the pipeline tests and the robustness report.

### Calibration (pooled 166, strict)

| Confidence | Accuracy |
|---|---|
| 0.0-0.6 | n/a |
| 0.6-0.8 | n/a |
| 0.8-0.9 | n/a |
| 0.9-1.0 | n/a |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

## Cost

| Metric | Value |
|---|---|
| Latency p50 / p95 (LLM call incl. retry) | 1579 ms / 2780 ms (n=20) |
| Tokens in / out per call | 6195 / 218 |
| Cached tokens per call | not reported |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_

## Slot accuracy (`golden_slots`, reported separately from intent accuracy)

| Field | Accuracy |
|---|---|
| city | 100.0% [81.6-100.0] (17/17) |
| bhk | 94.1% [73.0-99.0] (16/17) |
| min_price | 100.0% [81.6-100.0] (17/17) |
| max_price | 94.1% [73.0-99.0] (16/17) |
| ideal_for | 100.0% [81.6-100.0] (17/17) |
| overall | 97.6% [91.8-99.4] (83/85) |
| exact | 88.2% [65.7-96.7] (15/17) |

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

Intent accuracy on rooms-on-screen items: flagged n/a; not flagged n/a. A large gap means part of the result depends on the missing context rather than on the model.

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
| Invalid-search rate | 0 | 0.0% [0.0-18.4] (0/17) |
| E1 scenario pass rate | 100% of turns, or each failure explained | not run live |
| Intent accuracy, E2 pooled (133) | clearly above the current pipeline's 53.4% (harness: 57.9%) | n/a |
| Intent accuracy, E2 ctxY | clearly above 21.2% (harness: 27.3%) | n/a |
| Consistency across 3 runs | >= 95% identical decisions | n/a |
| p95 LLM latency | <= 3 s (rooms API excluded) | 2780 ms |

_provider gemini, model gemini-3.5-flash-lite, prompt 20f328e351bb_
