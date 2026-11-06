"""Near-domain questions the documents do NOT answer: what does the production answer function return for them? Also prints the
top-1 score spread of correct and wrong answers from the last eval_rag run.

    python -m llm_approach.eval.rag.eval_rag            (first: writes the results this reads)
    python -m llm_approach.eval.rag.probe_near_domain
"""

from __future__ import annotations

import json

from llm_approach.eval.rag import common

GAPS = [
    "do you charge brokerage", "how are owners verified", "how long does a plan last", "what is the average rent in Bangalore",
    "how do I write a rent agreement", "is a 2bhk better than a 1bhk", "can I get a refund on my security deposit from the owner",
    "what is a PG", "do you have rooms for students", "can I pay with UPI", "is there a mobile app", "what is your phone number",
    "what is your office address", "can I talk to a human", "do plans auto renew", "can I transfer my plan to another city",
    "how many unlocks do I have left", "what is the price of the gold plan in Bangalore", "who is the CEO of Homilivo",
    "is Homilivo approved by RERA",
]


def main() -> None:
    store = common.use_temp_index()
    from llm_approach.rag import answer as answer_mod

    _, by_text = common.chunk_ids()
    if common.RESULTS.exists():
        res = json.loads(common.RESULTS.read_text(encoding="utf-8"))
        correct = sorted(r["top_scores"][0] for r in res["rows"] if r["acc"] and r["rank"] == 1)
        wrong = sorted(r["top_scores"][0] for r in res["rows"] if r["acc"] and r["rank"] != 1)
        if correct:
            print("correct top-1 scores: min %.3f p10 %.3f median %.3f" % (correct[0], correct[len(correct) // 10], correct[len(correct) // 2]))
        if wrong:
            print("wrong top-1 scores:   min %.3f median %.3f max %.3f" % (wrong[0], wrong[len(wrong) // 2], wrong[-1]))
        print()
    answered = 0
    for q in GAPS:
        top = store.retrieve(q, top_k=1, min_score=-2)
        a = answer_mod.answer_platform_question(q)
        answered += a["grounded"]
        cid = f"{top[0].chunk.doc_id}#{by_text[(top[0].chunk.doc_id, top[0].chunk.text)]}" if top else "-"
        score = top[0].score if top else 0.0
        print(f"{score:.3f} {'ANSWERED' if a['grounded'] else 'refused '} {cid:34s} {q}\n      -> {a['message'][:150]!r}")
    print(f"\nanswered {answered} of {len(GAPS)} questions the documents do not cover")


if __name__ == "__main__":
    main()
