"""Every chunk with its id ('<doc_id>#<index within doc>'), as eval_set.py refers to them.

    python -m llm_approach.eval.rag.list_chunks
"""

from __future__ import annotations

from llm_approach.eval.rag import common


def main() -> None:
    for d, i, c in common.chunk_ids()[0]:
        print(f"{d}#{i} len={len(c.text)} | {c.text[:90]}")


if __name__ == "__main__":
    main()
