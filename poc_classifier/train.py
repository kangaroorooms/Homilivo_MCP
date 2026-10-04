"""PoC step 4: train LogisticRegression intent classifiers on train.jsonl ONLY.
Two variants (with / without has_room_context). C chosen by stratified 5-fold CV on train only."""
import sys, os, json, hashlib, datetime, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
import numpy as np, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from app.embeddings import get_shared_model

SEED = 0
rows = [json.loads(l) for l in open(os.path.join(HERE, "data", "train.jsonl"), encoding="utf-8") if l.strip()]
texts = [r["text"] for r in rows]; y = np.array([r["label"] for r in rows])
ctx = np.array([[1.0 if r["has_room_context"] else 0.0] for r in rows])
E = get_shared_model().encode(texts, normalize_embeddings=True, batch_size=64)
X = {"with_ctx": np.hstack([E, ctx]), "no_ctx": E}
Cs = [0.1, 1, 10, 100]
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
mk = lambda C: LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=SEED)  # lbfgs -> multinomial
os.makedirs(os.path.join(HERE, "artifacts"), exist_ok=True)
hl = []
for name, Xv in X.items():
    res = {C: cross_val_score(mk(C), Xv, y, cv=cv, scoring="accuracy").mean() for C in Cs}
    print(name, " ".join(f"C={C}: {a:.4f}" for C, a in res.items()))
    best = max(Cs, key=lambda C: (round(res[C], 6), -C))   # ties -> smaller C
    print(f"  chosen C={best}")
    m = mk(best).fit(Xv, y)
    p = os.path.join(HERE, "artifacts", f"clf_{name}.joblib")
    joblib.dump({"model": m, "variant": name, "C": best, "labels": list(m.classes_)}, p)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    hl.append(f"{h}  artifacts/clf_{name}.joblib  frozen {datetime.datetime.now().isoformat(timespec='seconds')}")
    print(f"  saved {p} {os.path.getsize(p)} bytes")
with open(os.path.join(HERE, "data", "HASHES.txt"), "a", encoding="utf-8") as f:
    f.write("# classifier artifacts trained on train.jsonl only (CV-selected C); frozen before any test scoring\n" + "\n".join(hl) + "\n")
print("\n".join(hl))
