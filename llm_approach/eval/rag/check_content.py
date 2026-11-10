"""Every chunk through the production display transform, then readability, safety and consistency checks (a wider net than
rag/check_docs: also static amounts, short chunks, references to other documents), the chunks most often returned as the top-1
answer (from the last eval_rag run) and where the key facts appear.

    python -m llm_approach.eval.rag.check_content
"""

from __future__ import annotations

import json
import re
from collections import Counter

from llm_approach.eval.rag import common

OPENERS = re.compile(r"^(Yes|No|It|This|They|Also|These|That|Use|To|Include|Business changes)\b")
REFS = re.compile(r"see below|as above|see section|section \d+|our (Refund|Privacy) Policy|Terms & Conditions|this policy|these Terms", re.I)
TABLE_LEFTOVER = re.compile(r"What happens|Examples and when we collect it|What you get|Includes ₹|\bTime (Within|Usually)\b|\b\d\. \w+: ")
LABEL_LEFT = re.compile(r"^[A-Z][^:.]{0,40}: ")
MARKUP = re.compile(r"\*\*|__|^#|<[a-z/]|\[|\]|`", re.M)
RUPEE = re.compile(r"₹\s?\d[\d,]*|Rs\.?\s?\d+|\d+\s?(?:rupees|INR)", re.I)
FACTS = ("5-7 business days", "25", "3 ", "unlimited", "support@homilivo.com", "Razorpay", "18")


def main() -> None:
    from llm_approach import reply_safety, reuse
    from llm_approach.actions.platform import has_internal_marker
    from llm_approach.rag.reply import clean_text

    chunks, _ = common.chunk_ids()
    guard = reuse.get_output_guardrail()
    problems = []
    for d, i, c in chunks:
        shown = clean_text(c.text, d)
        p = []
        if OPENERS.match(shown):
            p.append("dangling opener")
        if REFS.search(shown):
            p.append("reference: " + REFS.search(shown).group(0))
        if TABLE_LEFTOVER.search(shown):
            p.append("table leftover: " + TABLE_LEFTOVER.search(shown).group(0))
        if LABEL_LEFT.match(shown) and not TABLE_LEFTOVER.search(shown):
            p.append("label leftover: " + LABEL_LEFT.match(shown).group(0))
        if MARKUP.search(shown):
            p.append("markup")
        if not shown.rstrip().endswith((".", "?", "!")):
            p.append("no final punctuation (truncated?)")
        if has_internal_marker(shown):
            p.append("placeholder marker")
        if reply_safety.find_unsafe(shown):
            p.append("reply_safety: " + reply_safety.find_unsafe(shown))
        if not guard(shown)[0]:
            p.append("OUTPUT GUARDRAIL BLOCKS")
        if RUPEE.search(shown):
            p.append("static amount: " + RUPEE.search(shown).group(0))
        if len(shown) < 80:
            p.append(f"short ({len(shown)} chars)")
        if p:
            problems.append((f"{d}#{i}", p, shown))

    for cid, p, shown in problems:
        print(f"{cid:34s} {'; '.join(p)}\n      {shown[:200]!r}")
    print(f"\n{len(problems)} of {len(chunks)} chunks flagged")

    if common.RESULTS.exists():  # attractor chunks: how often a chunk is the top-1 across the eval set (incl. misses)
        res = json.loads(common.RESULTS.read_text(encoding="utf-8"))
        top = Counter(r["top"][0] for r in res["rows"])
        wrong_top = Counter(r["top"][0] for r in res["rows"] if r["acc"] and r["rank"] != 1)
        print("\nmost frequent top-1 chunks (all / as a wrong answer):")
        for cid, n in top.most_common(8):
            print(f"   {cid:34s} {n:3d} / {wrong_top.get(cid, 0)}")

    for fact in FACTS:  # the same fact in several documents: they must agree
        where = [f"{d}#{i}" for d, i, c in chunks if fact.lower() in c.text.lower()]
        print(f"fact {fact!r}: {len(where)} chunks: {', '.join(where)}")


if __name__ == "__main__":
    main()
