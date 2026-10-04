"""Exact + lowercased/whitespace-normalised overlap of a data file against matrix55 and golden_set."""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
def rd(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
protected = [r["text"] for r in rd(os.path.join(HERE, "data", "test_matrix55.jsonl"))]
protected += [r["query"] for r in rd(os.path.join(ROOT, "tests", "golden_set.jsonl"))]
P_exact, P_norm = set(protected), {norm(p) for p in protected}
for fn in sys.argv[1:]:
    rows = rd(os.path.join(HERE, "data", fn))
    ex = [r["text"] for r in rows if r["text"] in P_exact]
    nm = [r["text"] for r in rows if norm(r["text"]) in P_norm]
    print(f"{fn}: rows={len(rows)} exact_hits={len(ex)} normalized_hits={len(nm)}", nm)
    td = [r["text"] for r in rows]
    dup = len(td) - len({norm(t) for t in td}); print(f"  internal duplicates (normalised): {dup}")
