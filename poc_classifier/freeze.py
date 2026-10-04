"""Append SHA-256 of given data files to data/HASHES.txt (frozen moment recorded)."""
import hashlib, os, sys, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "data", "HASHES.txt"), "a", encoding="utf-8") as h:
    for fn in sys.argv[1:]:
        d = hashlib.sha256(open(os.path.join(HERE, "data", fn), "rb").read()).hexdigest()
        line = f"{d}  {fn}  frozen {datetime.datetime.now().isoformat(timespec='seconds')}"
        h.write(line + "\n"); print(line)
