"""Semantic gate report (read-only, offline: no model, no LLM, no rooms API).

    python llm_approach/scripts/gate_report.py --log server.log      count the gate's INFO lines in a log file
    python llm_approach/scripts/gate_report.py [--url http://127.0.0.1:8000]   read /metrics, if the server has one

The gate (``LLM_APPROACH_SEMANTIC_GATE=shadow``) writes one INFO line per check:
``semantic gate mode=shadow verdict=would_block label=OUT_OF_SCOPE request_id=... session=...``
and counts the same in ``observability.get_metrics_snapshot()["semantic_gate"]`` (gate_checked, gate_would_block,
gate_blocked, gate_error, labels, latency). Those lines hold no message text, so they are safe to keep for a week.

llm_approach/server_app.py has no /metrics route today. Without one, use --log, or read the counters in process:
``from llm_approach import observability; observability.get_metrics_snapshot()["semantic_gate"]``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter

LINE = re.compile(r"semantic gate mode=(\w+) verdict=(\w+) label=(\w+) ")
FAILED = re.compile(r"semantic gate failed: (\w+)")


def report_log(path: str) -> int:
    verdicts: Counter[str] = Counter()
    would_block_labels: Counter[str] = Counter()
    errors = 0
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = LINE.search(line)
            if match:
                verdicts[match.group(2)] += 1
                if match.group(2) != "pass":
                    would_block_labels[match.group(3)] += 1
            elif FAILED.search(line):
                errors += 1
    checked = sum(verdicts.values())
    print(f"gate checks: {checked}   would_block: {verdicts['would_block']}   blocked: {verdicts['blocked']}   errors: {errors}")
    if checked:
        print(f"would-block rate: {100 * (verdicts['would_block'] + verdicts['blocked']) / checked:.1f}%")
    for label, count in would_block_labels.most_common():
        print(f"  label {label}: {count}")
    return 0


def report_url(base: str) -> int:
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=5) as response:
            data = json.load(response)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"No metrics snapshot at {base}/metrics ({type(exc).__name__}).")
        print(__doc__)
        return 1
    gate = data.get("semantic_gate")
    if gate is None:
        print("The server answered, but its snapshot has no semantic_gate section.")
        return 1
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Semantic gate report")
    parser.add_argument("--log", help="a server log file with the gate's INFO lines")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="a running server (reads GET /metrics)")
    args = parser.parse_args(argv)
    return report_log(args.log) if args.log else report_url(args.url)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
