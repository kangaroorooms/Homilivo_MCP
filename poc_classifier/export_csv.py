"""Export frozen JSONL data files to CSV (read-only on the sources)."""
import csv, hashlib, json, sys
from pathlib import Path

DATA = Path(__file__).parent / "data"
BASE = ["text", "label", "has_room_context"]
EXTRA = ["acceptable", "score"]
FILES = {
    "train": BASE,
    "test_matrix55": BASE + EXTRA,
    "test_fresh": BASE + EXTRA,
    "test_newsearch_ctxY": BASE + EXTRA,
    "test_golden": BASE,
}


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def cell(row, col):
    v = row[col]
    if col == "acceptable":
        return "|".join(v)
    if col in ("has_room_context", "score"):
        return json.dumps(v)  # same literal as in JSONL (true/false)
    return v


def export(name, cols):
    rows = read_jsonl(DATA / f"{name}.jsonl")
    with open(DATA / f"{name}.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(cols)
        for r in rows:
            w.writerow([cell(r, c) for c in cols])
    return rows


def verify(name, cols, rows):
    with open(DATA / f"{name}.csv", encoding="utf-8-sig", newline="") as f:
        back = list(csv.DictReader(f))
    assert len(back) == len(rows), f"{name}: {len(back)} != {len(rows)}"
    for i, (b, r) in enumerate(zip(back, rows)):
        assert b["text"] == r["text"], f"{name} row {i} text"
        assert b["label"] == r["label"], f"{name} row {i} label"
        assert json.loads(b["has_room_context"]) == r["has_room_context"], f"{name} row {i} ctx"
        if "acceptable" in cols:
            assert b["acceptable"] == "|".join(r["acceptable"]), f"{name} row {i} acceptable"
            assert json.loads(b["score"]) == r["score"], f"{name} row {i} score"
    return back


def main():
    for name, cols in FILES.items():
        rows = export(name, cols)
        back = verify(name, cols, rows)
        print(f"{name}.csv: {len(back)} rows, verified against JSONL")
        if name == "train":
            assert len(back) == 712
            for r in back[:5]:
                print("  ", r)
    h = hashlib.sha256((DATA / "train.jsonl").read_bytes()).hexdigest()
    expected = next(l.split()[0] for l in (DATA / "HASHES.txt").read_text(encoding="utf-8").splitlines()
                    if l.split()[1:2] == ["train.jsonl"])
    print("train.jsonl sha256:", h)
    print("matches HASHES.txt:", h == expected)
    sys.exit(0 if h == expected else 1)


if __name__ == "__main__":
    main()
