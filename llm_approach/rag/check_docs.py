"""Offline check of the knowledge documents.

    python -m llm_approach.rag.check_docs

Lists every chunk of rag/documents with its source file and length and flags the ones that must not ship
(heading-only or separator-only, over the chunk limit, marker words, links, HTML, a domain other than the bare
homilivo.com, a missing title or source). Then it builds the index for the shipped documents in a temporary folder
(the real embedding model only; data/chroma_v2 is never touched), runs sample questions through retrieval and says,
for each, whether the reply would be refused by the score floor, the marker guard, the link/HTML check or the output
guardrail. Those four run through the real functions. No network service is called; only the local model is loaded.
Exit code 1 when a question that should be answerable is not.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_approach import reply_safety, reuse  # noqa: E402
from llm_approach.actions.platform import has_internal_marker  # noqa: E402
from llm_approach.rag import answer as answer_module  # noqa: E402
from llm_approach.rag import store  # noqa: E402

CHUNK_LIMIT = store.CHUNK_MAX_CHARS
_HEADING_ONLY = re.compile(r"^\s*#+\s*[^\n]*$")
_SEPARATOR = re.compile(r"^[\s\-=*_|:]*$")
_MARKDOWN = re.compile(r"^\s*#|\*\*|\||\[\[|\]\]", re.MULTILINE)

# (question, words of which at least one must appear in the answer for it to count as on topic, document that should
# hold the answer). A missing document is reported as WEAK, not as a failure: the text is still on topic.
ANSWERABLE = [
    ("what are the plans", ("free", "gold", "platinum"), ('homilivo_faq',)),
    ("how much does gold cost", ("gold", "₹"), ('homilivo_faq',)),
    ("how do I pay", ("razorpay", "payment"), ('homilivo_faq',)),
    ("what payment methods do you accept", ("razorpay", "payment"), ('homilivo_faq',)),
    ("refund policy", ("refund",), ('refund_and_cancellation_policy',)),
    ("my payment failed", ("payment", "deducted", "refund"), ('homilivo_faq', 'refund_and_cancellation_policy')),
    ("how do I delete my account", ("delete account", "delete"), ('homilivo_faq',)),
    ("how do I contact support", ("support@homilivo.com",), ('homilivo_faq',)),
    ("is it safe to pay the owner in advance", ("advance", "pay", "deposit"), ('safety_tips_for_renting',)),
    ("what is homilivo", ("homilivo is",), ('homilivo_faq', 'about_homilivo')),
    ("how does homilivo work", ("how homilivo works", "connects tenants"), ('how_homilivo_works',)),
]
# Not in the temporary documents yet: reported, never counted as a failure.
EXPECTED_GAPS = [
    "do you charge brokerage",
    "how are owners verified",
    "which cities do you cover",
    "how long does a plan last",
]


def chunk_problems(chunk: store.KnowledgeChunk) -> list[str]:
    """Reasons a chunk must not ship (empty list = fine)."""
    text, problems = chunk.text, []
    if _HEADING_ONLY.match(text) or _SEPARATOR.match(text):
        problems.append("heading-only or separator")
    if len(text) > CHUNK_LIMIT:
        problems.append(f"over {CHUNK_LIMIT} characters")
    if has_internal_marker(text):
        problems.append("marker word")
    if _MARKDOWN.search(text):
        problems.append("markdown heading, bold, table or [[ ]]")
    unsafe = reply_safety.find_unsafe(text)  # links, HTML and every domain except bare homilivo.com
    if unsafe:
        problems.append(f"link/HTML/domain ({unsafe})")
    if not str(chunk.title or "").strip():
        problems.append("missing title")
    if not str(chunk.source or "").strip() or chunk.source == "unknown":
        problems.append("missing source")
    return problems


def refusals(reply: str) -> list[str]:
    """Which of the reply checks would refuse this candidate reply (real functions, same order as the pipeline)."""
    out = []
    if has_internal_marker(reply):
        out.append("marker guard (actions/platform.py)")
    if reply_safety.find_unsafe(reply):
        out.append(f"link/HTML check (reply_safety: {reply_safety.find_unsafe(reply)})")
    guarded = "." + reply if reply.lstrip().startswith("|") else reply  # same guard step as pipeline._guard
    if not reuse.get_output_guardrail()(guarded)[0]:
        out.append("output guardrail")
    return out


def _check_chunks() -> int:
    chunks = store._load_documents()
    bad = 0
    print(f"== Chunks ({len(chunks)} in {len({c.doc_id for c in chunks})} documents) ==")
    for i, chunk in enumerate(chunks):
        problems = chunk_problems(chunk)
        bad += bool(problems)
        flag = "  <-- " + "; ".join(problems) if problems else ""
        print(f"{chunk.doc_id}.md #{i:02d} len={len(chunk.text):3d}{flag}")
    print(f"chunks flagged: {bad}\n")
    return bad


def _run_question(question: str, docs: tuple[str, ...] = ()) -> tuple[bool, str]:
    """Print the top chunks and refusal status; return (answered_ok, reason_if_not)."""
    results = store.retrieve(question, top_k=store.TOP_K, min_score=0.0)  # all three, to show scores below the floor too
    print(f"Q: {question}")
    for r in results:
        floor = "" if r.score >= store.MIN_RELEVANCE_SCORE else "  (below the 0.35 floor)"
        print(f"   {r.score:.3f} [{r.chunk.doc_id}] {r.chunk.text[:110]!r}{floor}")
    if docs and not any(r.chunk.doc_id in docs and r.score >= store.MIN_RELEVANCE_SCORE for r in results):
        print(f"   !! WEAK: none of the top chunks comes from {', '.join(docs)}")
    final = answer_module.answer_platform_question(question)  # the real retrieval path (0.35 floor, top 3)
    if not final["grounded"]:
        print("   -> REFUSED by (a) score floor / no match: reply would be the support message\n")
        return False, "no match"
    reply = final["message"]
    found = refusals(reply)
    print("   -> refused by: " + (", ".join(found) if found else "(a) no  (b) no  (c) no  (d) no"))
    if found:
        print()
        return False, "refused: " + ", ".join(found)
    return True, reply


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # the rupee sign on a Windows console
    flagged = _check_chunks()
    saved = os.environ.get(store.PERSIST_DIR_ENV)
    with tempfile.TemporaryDirectory(prefix="check_docs_chroma_", ignore_cleanup_errors=True) as tmp:  # Windows keeps index files open
        os.environ[store.PERSIST_DIR_ENV] = tmp
        store._reset_caches()
        store._forget_ready()
        try:
            stats = store.index_stats()
            print(f"== Index (temporary folder): {stats} ==\n")
            if not stats["embedding_model_available"] or not stats["chunks_indexed"]:
                print("FAIL: the embedding model could not be loaded or nothing was indexed")
                return 1
            failures = []
            print("== Sample questions ==")
            for question, words, docs in ANSWERABLE:
                ok, detail = _run_question(question, docs)
                if ok and not any(w.lower() in detail.lower() for w in words):
                    print(f"   -> off topic: none of {words} in the reply\n")
                    ok, detail = False, "off topic"
                elif ok:
                    print()
                if not ok:
                    failures.append((question, detail))
            print("== Expected gaps (no good answer in the temporary documents yet; not counted as failures) ==")
            for question in EXPECTED_GAPS:
                ok, _ = _run_question(question)
                print("   (expected gap: " + ("it still returned text, check it is not misleading)" if ok else "no answer, as expected)") + "\n")
        finally:
            store._reset_caches()
            store._forget_ready()
            if saved is None:
                os.environ.pop(store.PERSIST_DIR_ENV, None)
            else:
                os.environ[store.PERSIST_DIR_ENV] = saved
    print(f"== Summary: {flagged} flagged chunks, {len(failures)} failed questions ==")
    for question, detail in failures:
        print(f"FAIL: {question!r}: {detail}")
    return 1 if flagged or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
