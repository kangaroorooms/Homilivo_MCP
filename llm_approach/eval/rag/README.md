# Knowledge-base evaluation

Scripts from the RAG production-readiness audit. They measure retrieval and answer quality on the documents in
`llm_approach/rag/documents`. They are not tests: pytest does not collect them (no `test_` names). Run them by hand from the
repository root, with the project's virtual environment.

They build a temporary index in the system temp folder (`data/chroma_v2` is never touched), load the embedding model from the
local Hugging Face cache only (`HF_HUB_OFFLINE=1`), and make no Gemini or other network calls. The first run needs the model in
the cache; `python -m llm_approach.rag.check_docs` or a server start puts it there.

| Command | What it does |
|---|---|
| `python -m llm_approach.eval.rag.list_chunks` | Every chunk with its id (`<doc_id>#<index in doc>`), as `eval_set.py` refers to them |
| `python -m llm_approach.eval.rag.eval_rag` | Top-1, top-3 and MRR per category, out-of-domain false positives, a floor table (0.35 to 0.60), the worst misses and the top-1/top-2 margin. Writes the full results to `llm_approach_rag_eval_results.json` in the temp folder |
| `python -m llm_approach.eval.rag.probe_near_domain` | 20 questions the documents do not cover, and what the production answer function returns for them. Run `eval_rag` first for the score spread |
| `python -m llm_approach.eval.rag.check_content` | Every chunk's display text: openers, references, table leftovers, markup, guardrail blocks, static amounts, short chunks, and where the key facts appear |

`eval_set.py` holds the questions: 111 in-domain (plain, paraphrase, short, typo, Hinglish, Hindi, with a city or room) and 20
out-of-domain. Each question lists the chunk ids accepted as a correct answer; an empty list means no knowledge answer is
expected.
Labels updated when the owner-listing FAQ entry (`homilivo_faq#20`) was added: the two owner-listing questions now expect it instead of `terms_and_conditions#8`.
Batch C added the cancel/subscription FAQ entry (`homilivo_faq#23`) and two new eval questions for it; the existing "How do I cancel my subscription?" question now also accepts `homilivo_faq#23` alongside `refund_and_cancellation_policy#4`.

## When the final documents arrive

1. Replace the files in `llm_approach/rag/documents` and set `verified: true` in their front matter.
2. Run `list_chunks` and update the chunk ids in `eval_set.py` (the questions stay; the ids change with the paragraphs).
3. Run `python -m llm_approach.rag.check_docs` with `LLM_APPROACH_CHECK_DOCS_STRICT=1`, so wording warnings fail.
4. Run `eval_rag`, `probe_near_domain` and `check_content`, and compare with the audit baseline: top-1 59.5% overall, 1 of 20
   out-of-domain answered at the 0.35 floor, 15 of 20 near-domain gaps answered.
