"""Derives data/test_golden.jsonl from tests/golden_set.jsonl (read-only)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
special = {"hi": "GREETING", "thanks": "ACKNOWLEDGMENT", "how do I verify a listing on Homilivo": "PLATFORM_INFO"}
out = []
for l in open(os.path.join(ROOT, "tests", "golden_set.jsonl"), encoding="utf-8"):
    d = json.loads(l)
    lab = "NEW_SEARCH" if d["expected_tool"] == "search_rooms" else special.get(d["query"])
    if lab is None: continue
    out.append({"text": d["query"], "label": lab, "has_room_context": False})
with open(os.path.join(HERE, "data", "test_golden.jsonl"), "w", encoding="utf-8") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(out))
