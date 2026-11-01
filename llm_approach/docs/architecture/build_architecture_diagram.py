"""Generates architecture.svg (and architecture.html for rendering) for llm_approach v2.

Edit the data below (BANDS, BRANCHES, SIDE) and run:  python build_architecture_diagram.py
Render the PNG with a headless Chromium, see render_architecture.ps1 or the command at the bottom of this file.
Standard library only. Every box is annotated with the code that proves it in CROSS_CHECK.md.
"""

from __future__ import annotations

import html
from pathlib import Path

OUT = Path(__file__).resolve().parent
W, H = 2400, 4308
CX, BW = 1060, 560  # centre column
BAND_X, BAND_W = 320, 1480  # dashed lane rectangles
TEXT = "#1F2937"
SUB = "#374151"
ARROW = "#4B5563"
FONT = "'Segoe UI', Arial, sans-serif"

KIND = {  # fill, stroke, lane name, short description
    "fe": ("#DCEBFA", "#3B82C4", "Frontend (React)", "Chat UI in the browser"),
    "be": ("#DDF2E2", "#3E9B5C", "Backend API layer (FastAPI)", "Routes, token, rate limits"),
    "gr": ("#FFF1BF", "#C79A12", "Guardrails and Security", "Input and output safety"),
    "ai": ("#DDD0F6", "#6B46C1", "AI Orchestration", "LLM decision, validators, action modules"),
    "ex": ("#CFEFF3", "#1E9AAE", "External APIs", "Gemini, rooms, pricing"),
    "tp": ("#F1E4F7", "#A45BBE", "Tools and Processing", "Session store, index, metrics"),
}

# ---- data --------------------------------------------------------------------------------------------------------

# (kind, title, lines)
USER = ("fe", "User", ["types a message"])
REACT = ("fe", "Chat widget (React)", ["website widget (newkangaroorooms) or", "frontend-demo (useChatSocket)", "shows reply and room cards"])
WS_IN = ("be", "WebSocket", ["/ws/chat?token=...", "(POST /chat is the same turn)"])
ROUTE = ("be", "FastAPI route", ["ws_chat  ->  run_turn", "run_turn: the whole turn runs under a 40 s limit"])
ENTRY = ("be", "Entry checks", [
    "token present  -  IP rate limit",
    "message 1 to 500 chars, valid mode (WebSocket)",
    "session-id format  -  session limit",
    "new-session limit  -  server ready (else 503)",
])
PREP = ("gr", "Input normalisation", ["NFKC, invisible chars removed", "empty or over 500 chars after this: fixed reply"])
LOAD = ("tp", "Session state load", ["per-session lock, then read"])
SEC = ("gr", "Security filters", ["extra_security + app input guardrail"])
GATE = ("gr", "Semantic gate", ["MiniLM classifier; server default: shadow (records only)", "runs only with no rooms on screen, no pending question", "enforce mode (env): off-topic -> fixed reply, no LLM"])
LLM = ("ai", "Prompt build + LLM decision", ["one JSON call: intent + slots", "Pydantic Decision model parses it", "bad JSON: retry once, then no decision -> fallback"])
VALID = ("ai", "Validators", ["first step of _decide_action: city, BHK,", "price, room ref; bad slot -> ask, no search"])
ROUTER = ("ai", "Intent router", ["_decide_action: 12 intents -> 5 paths", "+ pending city-change reply, owner-contact check"])
RESP = ("gr", "Response checks", ["links / HTML / markdown refused; output guardrail", "every path incl. fallback; else a fixed reply"])
SAVE = ("tp", "State save", ["history + rooms on screen; saved at lock release", "store down: fixed notice added to the reply"])
WS_OUT = ("be", "WebSocket reply", ["typing, token chunks, done"])
REACT2 = ("fe", "Chat widget (React)", ["renders reply and room cards"])
USER2 = ("fe", "User", ["reads the answer"])

BANDS = [  # (lane kind, boxes)
    ("fe", [USER, REACT]),
    ("be", [WS_IN, ROUTE, ENTRY]),
    ("gr", [PREP]),
    ("tp", [LOAD]),
    ("gr", [SEC, GATE]),
    ("ai", [LLM, VALID, ROUTER]),
]
BANDS_AFTER = [
    ("gr", [RESP]),
    ("tp", [SAVE]),
    ("be", [WS_OUT]),
    ("fe", [REACT2, USER2]),
]

BRANCHES = [  # columns of boxes, in drawing order
    [("ai", "Platform FAQ", ["PLATFORM_INFO intent", "answer is not LLM-written"]),
     ("tp", "Chroma FAQ index", ["MiniLM embeddings", "similarity search"]),
     ("gr", "Grounded answer", ["retrieved text only,", "else fixed 'no info'"])],
    [("ai", "State-based answers", ["room follow-up, recall,", "chat, out of scope,", "unsupported (fixed text)"]),
     ("ai", "Conversation answerer", ["chat question: 2nd LLM", "call over this chat's record"]),
     ("gr", "Answer check", ["must match transcript,", "else 'not in this chat'"])],
    [("ai", "Owner contact", ["check after the LLM call:", "UNSUPPORTED_ACTION or", "PLATFORM_INFO, and owner", "wording -> city's plans"]),
     ("ex", "Pricing API", ["HTTP GET per city", "2 attempts, 15 s limit"]),
     ("gr", "Plans check", ["validate, clean, format"])],
    [("ai", "Clarify", ["UNCLEAR, low confidence,", "bad city/BHK/budget slot,", "unclear room reference,", "reject ('not this one')"]),
     ("ai", "City-change question", ["asked by Room search: new", "city + saved filters; no", "search yet. Reply is read at", "the top of the router"])],
    [("ai", "Room search", ["search, refine, sort,", "show more; asks for a city"]),
     ("gr", "Parameter guardrail", ["city, price, type, sort"]),
     ("tp", "MCP tool (in process)", ["search_rooms", "15 s limit, retries"]),
     ("ex", "Homilivo rooms API", ["HTTP GET"]),
     ("gr", "API-data guardrail", ["room data check, text", "clean, off-criteria drop"])],
]
COL_W, COL_GAP, COL_X0 = 262, 36, 335

# side boxes: (key, kind, title, lines, width)
SIDE_X, SIDE_W = 1900, 440

LEGEND_X, LEGEND_Y, LEGEND_W = 1850, 130, 520

# ---- drawing helpers ---------------------------------------------------------------------------------------------

parts: list[str] = []
checks: list[tuple[float, str]] = []  # (max text width, text id) for the overflow check


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def box_h(lines: list[str], minimum: int = 0) -> int:
    return max(minimum, 20 + 28 + 24 * len(lines) + 12)


def text(x: float, y: float, s: str, size: int, weight: str, fill: str, anchor: str = "middle", maxw: float | None = None) -> None:
    tid = f"t{len(checks)}"
    if maxw:
        checks.append((maxw, tid))
    parts.append(
        f'<text id="{tid}" x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{esc(s)}</text>'
    )


def box(x: float, y: float, w: float, h: float, spec: tuple, extra: str = "") -> None:
    kind, title, lines = spec
    fill, stroke = KIND[kind][0], KIND[kind][1]
    parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="16" fill="{fill}" stroke="{stroke}" stroke-width="2.5"{extra}/>')
    block = 28 + 24 * len(lines)
    top = y + (h - block) / 2
    text(x + w / 2, top + 21, title, 22, "700", TEXT, maxw=w - 20)
    for i, line in enumerate(lines):
        text(x + w / 2, top + 28 + 24 * (i + 1) - 4, line, 18, "400", SUB, maxw=w - 20)


def arrow(points: list[tuple[float, float]], dashed: bool = False, both: bool = False) -> None:
    d = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in points)
    dash = ' stroke-dasharray="9 7"' if dashed else ""
    start = ' marker-start="url(#head-start)"' if both else ""
    parts.append(f'<path d="{d}" fill="none" stroke="{ARROW}" stroke-width="3"{dash}{start} marker-end="url(#head)"/>')


def line(points: list[tuple[float, float]]) -> None:
    d = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in points)
    parts.append(f'<path d="{d}" fill="none" stroke="{ARROW}" stroke-width="3"/>')


def icon(kind: str, x: float, y: float) -> None:
    s = KIND[kind][1]
    g = f'<g transform="translate({x:.1f},{y:.1f})" fill="none" stroke="{s}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round">'
    if kind == "fe":  # monitor
        g += '<rect x="3" y="5" width="30" height="20" rx="3"/><path d="M12 31h12M18 25v6"/>'
    elif kind == "be":  # server stack
        g += '<rect x="4" y="4" width="28" height="10" rx="3"/><rect x="4" y="19" width="28" height="10" rx="3"/><circle cx="10" cy="9" r="1.2"/><circle cx="10" cy="24" r="1.2"/>'
    elif kind == "gr":  # shield
        g += '<path d="M18 3l13 5v9c0 8-6 13-13 16C11 30 5 25 5 17V8z"/><path d="M12 17l4 4 8-8"/>'
    elif kind == "ai":  # node graph
        g += '<circle cx="8" cy="9" r="4"/><circle cx="28" cy="9" r="4"/><circle cx="18" cy="28" r="4"/><path d="M12 10h12M10 13l6 12M26 13l-6 12"/>'
    elif kind == "ex":  # cloud
        g += '<path d="M10 28h17a6 6 0 0 0 1-12 9 9 0 0 0-17-2 7 7 0 0 0-1 14z"/>'
    elif kind == "tp":  # gear
        g += '<circle cx="18" cy="18" r="11" stroke-width="5" stroke-dasharray="5 3.4"/><circle cx="18" cy="18" r="5"/>'
    g += "</g>"
    parts.append(g)


def band(kind: str, top: float, bottom: float, label: str | None = None, desc: str | None = None) -> None:
    fill, stroke, name, short = KIND[kind]
    parts.append(
        f'<rect x="{BAND_X}" y="{top:.1f}" width="{BAND_W}" height="{bottom - top:.1f}" rx="26" fill="{fill}" fill-opacity="0.22" '
        f'stroke="{stroke}" stroke-width="2.4" stroke-dasharray="14 9"/>'
    )
    h = max(bottom - top, 140)
    parts.append(f'<rect x="30" y="{top:.1f}" width="270" height="{h:.1f}" rx="20" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>')
    icon(kind, 46, top + 18)
    name = label or name
    short = desc or short
    # name: wrap at ~19 chars across two lines
    words, lines_, cur = name.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 19 and cur:
            lines_.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines_.append(cur)
    for i, ln in enumerate(lines_):
        text(94, top + 38 + 25 * i, ln, 21, "700", TEXT, "start", maxw=196)
    base = top + 38 + 25 * len(lines_) + 8
    words, dl, cur = short.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 29 and cur:
            dl.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    dl.append(cur)
    for i, ln in enumerate(dl):
        text(46, base + 22 * i + 14, ln, 17, "400", SUB, "start", maxw=240)


# ---- layout ------------------------------------------------------------------------------------------------------

def layout() -> dict:
    placed: dict[str, tuple[float, float, float, float]] = {}  # title -> x, y, w, h  (first occurrence per section key)
    y = 150.0
    prev = None  # (bottom y) of the previous main-column box

    def place_band(kind: str, boxes: list[tuple], tag: str, gap_before: float = 64) -> None:
        nonlocal y, prev
        top = y
        cy = top + 20
        for i, spec in enumerate(boxes):
            h = box_h(spec[2], 96 if len(boxes) == 1 else 0)
            if prev is not None:
                arrow([(CX, prev), (CX, cy)])
            box(CX - BW / 2, cy, BW, h, spec)
            placed[f"{tag}{i}"] = (CX - BW / 2, cy, BW, h)
            prev = cy + h
            cy += h + 44
        bottom = cy - 44 + 20
        band(kind, top, bottom)
        y = bottom + gap_before

    for n, (kind, boxes) in enumerate(BANDS):
        place_band(kind, boxes, f"a{n}_")

    # ---- branch region
    router = placed["a5_2"]
    router_bottom = router[1] + router[3]
    bus_y = router_bottom + 58
    y = router_bottom + 20 + 20 + 90 + 20  # band gap (the bus lives in it)
    branch_top = y
    cols_x = [COL_X0 + i * (COL_W + COL_GAP) for i in range(5)]
    col_bottoms = []
    for ci, col in enumerate(BRANCHES):
        cy = branch_top + 20
        for bi, spec in enumerate(col):
            h = box_h(spec[2], 96)
            if bi == 0:
                arrow([(cols_x[ci] + COL_W / 2, bus_y), (cols_x[ci] + COL_W / 2, cy)])
            elif not (ci == 3 and bi == 1):  # the city-change question is not a step after Clarify: it comes from Room search
                arrow([(cols_x[ci] + COL_W / 2, cy - 44), (cols_x[ci] + COL_W / 2, cy)])
            box(cols_x[ci], cy, COL_W, h, spec)
            placed[f"b{ci}_{bi}"] = (cols_x[ci], cy, COL_W, h)
            cy += h + 44
        col_bottoms.append(cy - 44)
    rs, cc = placed["b4_0"], placed["b3_1"]
    gap_x = rs[0] - COL_GAP / 2
    arrow([(rs[0], rs[1] + rs[3] / 2), (gap_x, rs[1] + rs[3] / 2), (gap_x, cc[1] + cc[3] / 2), (cc[0] + cc[2], cc[1] + cc[3] / 2)])
    branch_bottom = max(col_bottoms) + 20
    band("ai", branch_top, branch_bottom, "AI Orchestration", "Action modules, colours show the layer")
    # split: router -> bus
    centres = [cx + COL_W / 2 for cx in cols_x]
    line([(CX, router_bottom), (CX, bus_y)])
    line([(centres[0], bus_y), (centres[-1], bus_y)])
    # merge bus
    merge_y = branch_bottom + 58
    for ci, bottom in enumerate(col_bottoms):
        line([(centres[ci], bottom), (centres[ci], merge_y)])
    line([(centres[0], merge_y), (centres[-1], merge_y)])
    cl = placed["b3_0"]  # Clarify's own way to the merge line, down the gap on its left
    line([(cl[0], cl[1] + cl[3] / 2), (cl[0] - COL_GAP / 2, cl[1] + cl[3] / 2), (cl[0] - COL_GAP / 2, merge_y)])
    y = branch_bottom + 20 + 20 + 90 + 20
    prev = None
    first_after = True
    for n, (kind, boxes) in enumerate(BANDS_AFTER):
        if first_after:
            top = y
            cy = top + 20
            spec = boxes[0]
            h = box_h(spec[2], 96)
            arrow([(CX, merge_y), (CX, cy)])
            box(CX - BW / 2, cy, BW, h, spec)
            placed[f"c{n}_0"] = (CX - BW / 2, cy, BW, h)
            prev = cy + h
            bottom = cy + h + 20
            band(kind, top, bottom)
            y = bottom + 76
            first_after = False
        else:
            place_band(kind, boxes, f"c{n}_")
    global H
    H = max(int(y) - 76 + 90, 4308)  # same canvas as the first version unless the content needs more
    return placed


# ---- side boxes --------------------------------------------------------------------------------------------------

def side(placed: dict) -> None:
    def put(key_anchor: str, kind: str, title: str, lines: list[str], dy: float, joint_dy: float, dashed=False, both=False, elbow_x=None, h=None):
        ax, ay, aw, ah = placed[key_anchor]
        sh = h or box_h(lines, 0)
        sy = ay + dy
        y_main = ay + joint_dy  # where the connector leaves the main box
        box(SIDE_X, sy, SIDE_W, sh, (kind, title, lines))
        y_side = sy + sh / 2
        x0 = ax + aw
        if abs(y_main - y_side) < 1:
            arrow([(x0, y_main), (SIDE_X, y_side)], dashed=dashed, both=both)
        else:
            ex = elbow_x or 1850
            arrow([(x0, y_main), (ex, y_main), (ex, y_side), (SIDE_X, y_side)], dashed=dashed, both=both)

    ax, ay, aw, ah = placed["a3_0"]
    put("a3_0", "tp", "Session store", ["memory (default) or Redis", "idle TTL 1 h, max 1000 sessions"], 0, ah / 2, both=True)
    ax, ay, aw, ah = placed["a5_0"]
    put("a5_0", "ex", "Gemini API", ["gemini-3.5-flash-lite (default)", "limiter, 5xx retry, circuit breaker", "optional fallback provider: Groq", "also used by the answerer"], -8, 40, both=True, h=130)
    put("a5_0", "tp", "Rule-based fallback", ["fast_parser slots -> room search", "used when there is no decision", "its result goes to Response checks"], 160, 100, dashed=True, elbow_x=1835)
    fx, fy, fh = SIDE_X + SIDE_W / 2, placed["a5_0"][1] + 160, box_h(["x"] * 3, 0)
    rx, ry, rw, rh = placed["c0_0"]
    arrow([(fx, fy + fh), (fx, ry + rh / 2), (rx + rw, ry + rh / 2)], dashed=True)
    text(1580, ry + rh / 2 - 10, "no decision: fallback result", 17, "400", SUB, "middle", maxw=300)
    ax, ay, aw, ah = placed["c1_0"]
    put("c1_0", "tp", "Session store", ["same store, state saved"], 0, 30, h=72)
    put("c1_0", "tp", "Observability", ["turn counters, latency, request", "id; no message text; /metrics"], 90, 70, elbow_x=1850)


# ---- legend ------------------------------------------------------------------------------------------------------

def legend() -> None:
    x, y = LEGEND_X, LEGEND_Y
    items = list(KIND.items())
    h = 70 + 38 * len(items) + 4 * 38 + 120
    parts.append(f'<rect x="{x}" y="{y}" width="{LEGEND_W}" height="{h}" rx="18" fill="#FFFFFF" stroke="#9CA3AF" stroke-width="2.4"/>')
    text(x + 24, y + 38, "Legend", 24, "700", TEXT, "start")
    yy = y + 62
    for kind, (fill, stroke, name, _short) in items:
        parts.append(f'<rect x="{x + 24}" y="{yy}" width="46" height="26" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="2.4"/>')
        text(x + 86, yy + 20, name, 19, "400", TEXT, "start", maxw=LEGEND_W - 110)
        yy += 38
    yy += 6
    arrow([(x + 24, yy + 14), (x + 70, yy + 14)])
    text(x + 86, yy + 20, "request flow", 19, "400", TEXT, "start")
    yy += 38
    arrow([(x + 24, yy + 14), (x + 70, yy + 14)], dashed=True)
    text(x + 86, yy + 20, "fallback path", 19, "400", TEXT, "start")
    yy += 38
    arrow([(x + 24, yy + 14), (x + 70, yy + 14)], both=True)
    text(x + 86, yy + 20, "two-way call to a service or store", 19, "400", TEXT, "start", maxw=LEGEND_W - 110)
    yy += 38
    parts.append(f'<rect x="{x + 24}" y="{yy}" width="46" height="26" rx="9" fill="none" stroke="{ARROW}" stroke-width="2.4" stroke-dasharray="8 5"/>')
    text(x + 86, yy + 20, "dashed lane = layer group", 19, "400", TEXT, "start", maxw=LEGEND_W - 110)
    yy += 52
    for ln in ("No LangGraph router. One LLM call returns", "intent + slots (Pydantic parses it);", "chat questions add a 2nd call."):
        text(x + 24, yy, ln, 17, "400", SUB, "start", maxw=LEGEND_W - 40)
        yy += 22


# ---- build -------------------------------------------------------------------------------------------------------

def build() -> str:
    placed = layout()
    side(placed)
    legend()
    text(1010, 62, "HOMILIVO Chatbot Architecture (LLM Approach v2)", 46, "700", "#111827")
    text(1010, 98, "One chat turn in llm_approach: from the user's message to the reply", 22, "400", SUB)
    text(W / 2, H - 28, "2026-10-27   |   branch: experiment/llm-approach_v2", 20, "400", "#6B7280")
    defs = (
        '<defs>'
        f'<marker id="head" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="9" markerHeight="9" orient="auto"><path d="M1,1 L11,6 L1,11 z" fill="{ARROW}"/></marker>'
        f'<marker id="head-start" viewBox="0 0 12 12" refX="2" refY="6" markerWidth="9" markerHeight="9" orient="auto-start-reverse"><path d="M1,1 L11,6 L1,11 z" fill="{ARROW}"/></marker>'
        '</defs>'
    )
    body = "\n".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">\n'
        f'<title>HOMILIVO Chatbot Architecture (LLM Approach v2)</title>\n{defs}\n'
        f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>\n{body}\n</svg>\n'
    )


if __name__ == "__main__":
    svg = build()
    (OUT / "architecture.svg").write_text(svg, encoding="utf-8")
    checks_js = ",".join(f'["{tid}",{maxw:.0f}]' for maxw, tid in checks)
    page = (
        f'<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;background:#fff}}</style></head><body>{svg}'
        '<pre id="res" style="position:absolute;left:0;top:0;font-size:1px;color:#fff"></pre>'
        f'<script>window.onload=()=>{{const bad=[];for(const [id,m] of [{checks_js}]){{const e=document.getElementById(id);'
        'const w=e.getComputedTextLength();if(w>m)bad.push("OVERFLOW "+e.textContent+" "+Math.round(w)+">"+m);}'
        'document.getElementById("res").textContent="CHECKDONE "+bad.join(" | ");}</script></body></html>'
    )
    (OUT / "architecture_render.html").write_text(page, encoding="utf-8")
    print("wrote architecture.svg and architecture_render.html", W, H)

# Render (PowerShell), 1.5x scale gives a 3600 px wide PNG:
#   & "$env:LOCALAPPDATA\ms-playwright\chromium_headless_shell-1248\chrome-headless-shell-win64\chrome-headless-shell.exe" `
#     --screenshot=architecture.png --window-size=2400,4308 --force-device-scale-factor=1.5 --hide-scrollbars `
#     file:///<path>/architecture_render.html
