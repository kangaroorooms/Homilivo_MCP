"""Retrieval quality of the knowledge base (top-1, top-3, MRR per category, out-of-domain false positives, a floor table, the
worst misses and the top-1/top-2 margin), plus the cold model load and index build times. Temporary index only; no Gemini, no
network (the embedding model comes from the local cache).

    python -m llm_approach.eval.rag.eval_rag

Writes the full results to common.RESULTS (the system temp folder), which probe_near_domain and check_content read.
"""

from __future__ import annotations

import json
import shutil
import statistics
import time
from collections import defaultdict

from llm_approach.eval.rag import common

FLOORS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def _summary(rs):
    n = len(rs)
    return dict(n=n, top1=sum(r["rank"] == 1 for r in rs) / n, top3=sum(bool(r["rank"]) and r["rank"] <= 3 for r in rs) / n,
                mrr=sum(1 / r["rank"] for r in rs if r["rank"]) / n,
                prod_correct=sum(r["used"] in r["acc"] for r in rs) / n,
                prod_refused=sum(not r["grounded"] for r in rs) / n)


def _rounded(d):
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()}


def main() -> None:
    t0 = time.perf_counter()
    from llm_approach.rag.embeddings import get_shared_model
    get_shared_model()
    cold_model_s = time.perf_counter() - t0

    shutil.rmtree(common.TMP_INDEX, ignore_errors=True)
    store = common.use_temp_index()
    t0 = time.perf_counter()
    stats = store.index_stats()  # builds the temporary index (the model is already loaded: this is the build cost alone)
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    store._forget_ready()
    store.index_stats()  # a restart that reuses the index (fingerprint match)
    reuse_s = time.perf_counter() - t0

    from llm_approach.eval.rag.eval_set import EVAL
    from llm_approach.rag import answer as answer_mod
    from llm_approach.rag.reply import clean_text

    chunks, by_text = common.chunk_ids()
    n_chunks = len(chunks)

    def cid(r):
        return f"{r.chunk.doc_id}#{by_text.get((r.chunk.doc_id, r.chunk.text), '?')}"

    rows = []
    for cat, q, acc in EVAL:
        ranked = store.retrieve(q, top_k=n_chunks, min_score=-2.0)
        ids = [cid(r) for r in ranked]
        scores = [r.score for r in ranked]
        if not ids:  # a query the store does not retrieve at all (fewer than 2 letters)
            ids, scores = ["-"], [0.0]
        ids, scores = ids + ["-"] * 3, scores + [0.0] * 3
        rank = next((i + 1 for i, x in enumerate(ids) if x in acc), None) if acc else None
        exp_id = ids[rank - 1] if rank else (acc[0] if acc else None)
        exp_score = scores[rank - 1] if rank else None
        prod = answer_mod.answer_platform_question(q)  # the production answer function (raw query only, no rewrite)
        used = None
        if prod["grounded"]:
            for d, i, c in chunks:
                if clean_text(c.text, d) == prod["message"]:
                    used = f"{d}#{i}"
                    break
        rows.append(dict(cat=cat, q=q, acc=acc, top=ids[:3], top_scores=[round(s, 4) for s in scores[:3]], rank=rank,
                         exp_id=exp_id, exp_score=exp_score and round(exp_score, 4), grounded=prod["grounded"], used=used,
                         reply=prod["message"][:160]))

    ind = [r for r in rows if r["acc"]]
    ood = [r for r in rows if not r["acc"]]
    by_cat = defaultdict(list)
    for r in ind:
        by_cat[r["cat"]].append(r)
    per_cat = {c: _summary(rs) for c, rs in by_cat.items()}
    overall = _summary(ind)
    mismatch = [r["q"] for r in rows if r["grounded"] and r["used"] != r["top"][0]]  # the reply must be the top-1 chunk

    floor_table = []
    for f in FLOORS:
        answered_in = [r for r in ind if r["top_scores"][0] >= f]
        correct = [r for r in answered_in if r["rank"] == 1]
        answered_ood = [r for r in ood if r["top_scores"][0] >= f]
        answered = len(answered_in) + len(answered_ood)
        floor_table.append(dict(floor=f, in_answered=len(answered_in), correct=len(correct), wrong_in=len(answered_in) - len(correct),
                                ood_answered=len(answered_ood), precision=len(correct) / answered if answered else 1.0,
                                recall=len(correct) / len(ind), ood_fp=len(answered_ood) / len(ood)))

    misses = sorted([r for r in ind if r["rank"] != 1], key=lambda r: -(r["top_scores"][0] - (r["exp_score"] or 0)))
    margins_ok = [r["top_scores"][0] - r["top_scores"][1] for r in ind if r["rank"] == 1]
    margins_bad = [r["top_scores"][0] - r["top_scores"][1] for r in ind if r["rank"] != 1]
    margins_ood = [r["top_scores"][0] - r["top_scores"][1] for r in ood]
    margin_rules = [dict(margin=m, wrong_caught=sum(x < m for x in margins_bad), wrong_total=len(margins_bad),
                         correct_lost=sum(x < m for x in margins_ok), correct_total=len(margins_ok),
                         ood_caught=sum(x < m for x in margins_ood)) for m in (0.01, 0.02, 0.03, 0.05)]
    same_doc_bad = sum(r["top"][0].split("#")[0] == (r["exp_id"] or "").split("#")[0] for r in misses)

    out = dict(stats=stats, cold_model_s=cold_model_s, build_s=build_s, reuse_s=reuse_s, chunks=n_chunks, n_in=len(ind), n_ood=len(ood),
               overall=overall, per_cat=per_cat, floor_table=floor_table, prod_top1_mismatch=mismatch,
               ood_rows=[dict(q=r["q"], top=r["top"][0], score=r["top_scores"][0]) for r in ood],
               misses=[dict(q=r["q"], expected=r["exp_id"], exp_score=r["exp_score"], exp_rank=r["rank"], returned=r["top"][0],
                            ret_score=r["top_scores"][0]) for r in misses],
               margin=dict(ok_median=statistics.median(margins_ok) if margins_ok else None,
                           bad_median=statistics.median(margins_bad) if margins_bad else None,
                           ood_median=statistics.median(margins_ood) if margins_ood else None, rules=margin_rules,
                           same_doc_misses=same_doc_bad),
               rows=rows)
    common.RESULTS.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"model cold load {cold_model_s:.1f}s, index build {build_s:.2f}s, reuse-on-restart {reuse_s:.2f}s, stats {stats}")
    print(f"in-domain {len(ind)}, ood {len(ood)}")
    print("overall", _rounded(overall))
    for c, s in per_cat.items():
        print(f"  {c:10s}", _rounded(s))
    print("prod reply != top-1 chunk:", mismatch)
    print("floor table:")
    for t in floor_table:
        print("  ", _rounded(t))
    print("OOD top scores:")
    for r in sorted(ood, key=lambda r: -r["top_scores"][0]):
        print(f"   {r['top_scores'][0]:.3f} {r['top'][0]:36s} {r['q']}")
    print(f"misses: {len(misses)}")
    for r in misses:
        print(f"   exp {r['exp_id']} ({r['exp_score']}, rank {r['rank']})  got {r['top'][0]} ({r['top_scores'][0]})  | {r['q']}")
    print("margins", json.dumps(out["margin"]))
    print(f"results: {common.RESULTS}")


if __name__ == "__main__":
    main()
