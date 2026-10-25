"""Drift control: every file copied from app/ must still match its source.

Header lines and lines marked ``# v2:`` are ignored. Every other line of a copy must be identical
to the source (a source line may only disappear where a ``# v2:`` line replaces it). The test only
reads app/ source as text; it never imports from app/.
"""

from __future__ import annotations

import ast
import difflib
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HEADER_END = "# ==== end of v2 provenance header ===="
V2_MARK = "# v2:"
WAS_RE = re.compile(r"# v2: was: (.*)$")

# (source in app/, copy in llm_approach/)
WHOLE_FILE_COPIES = [
    ("app/guardrails/common.py", "llm_approach/guardrails/common.py"),
    ("app/guardrails/output_guardrail.py", "llm_approach/guardrails/output_guardrail.py"),
    ("app/guardrails/parameter_guardrail.py", "llm_approach/guardrails/parameter_guardrail.py"),
    ("app/guardrails/input_guardrail.py", "llm_approach/guardrails/input_guardrail.py"),
    ("app/guardrails/api_data_guardrail.py", "llm_approach/guardrails/api_data_guardrail.py"),
    ("app/fast_parser.py", "llm_approach/fast_parser.py"),
    ("app/demo_auth.py", "llm_approach/security/demo_auth.py"),
    ("app/observability.py", "llm_approach/observability.py"),
    ("app/rate_limit.py", "llm_approach/security/rate_limit.py"),
    ("app/embeddings.py", "llm_approach/rag/embeddings.py"),
    ("app/knowledge/store.py", "llm_approach/rag/store.py"),
    ("app/knowledge/answer.py", "llm_approach/rag/answer.py"),
    ("app/knowledge/ingest.py", "llm_approach/rag/ingest.py"),
]

# Documents carry no header (every *.md in the documents folder is indexed): they must match byte for byte.
DOCUMENT_COPIES = [
    ("app/knowledge/documents/about_this_assistant.md", "llm_approach/rag/documents/about_this_assistant.md"),
    ("app/knowledge/documents/homilivo_faq.md", "llm_approach/rag/documents/homilivo_faq.md"),
]

PROMPT_SAFETY_NAMES = (
    "GROQ_MODEL",
    "FACT_GROUNDING_RULE",
    "PROMPT_INJECTION_DEFENSE_RULE",
    "_PROMPT_FENCE_DELIMITERS",
    "_neutralize_prompt_delimiters",
    "_SANITIZE_FIELD_MAX_LENGTHS",
    "_SANITIZE_DEFAULT_MAX_LENGTH",
    "_CONTROL_CHAR_RE",
    "_MULTI_SPACE_RE",
    "_MULTI_NEWLINE_RE",
    "_sanitize_string_value",
    "sanitize_room_for_llm_context",
    "UNSUPPORTED_ACTION_RESPONSE",
    "OUT_OF_SCOPE_RESPONSE",
    "SHOW_ALTERNATIVES_RESPONSE",
    "NOT_VERIFIED_AMENITY_RESPONSE",
)

# (source, copy, names): only these definitions are compared
PARTIAL_COPIES = [
    ("app/experiments/agent_modes.py", "llm_approach/prompt_safety.py", PROMPT_SAFETY_NAMES),
    (
        "app/api.py",
        "llm_approach/security/secrets_check.py",
        ("_SESSION_ID_RE", "_normalize_session_id", "validate_secrets"),
    ),
]


# (source, copy, class name, members): class members (methods and class attributes) copied into a v2 class
CLASS_MEMBER_COPIES = [
    (
        "app/session_store.py",
        "llm_approach/session_store.py",
        "RedisSessionStore",
        ("_KEY_PREFIX", "_LRU_KEY", "_key", "_evict_expired", "_enforce_cap"),
    ),
]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _drift(source_lines: list[str], copy_lines: list[str], label: str) -> list[str]:
    """Problems found comparing source lines with copy lines (empty list == no drift)."""
    v2_lines = [ln for ln in copy_lines if V2_MARK in ln]
    kept = [ln for ln in copy_lines if V2_MARK not in ln]
    problems: list[str] = []
    removed = 0
    matcher = difflib.SequenceMatcher(a=source_lines, b=kept, autojunk=False)
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            removed += a1 - a0
            continue
        if tag == "insert" and not any(ln.strip() for ln in kept[b0:b1]):
            continue  # blank lines around a block added with '# v2:' lines
        problems.append(f"{label}: source lines {a0 + 1}-{a1} differ from copy: {source_lines[a0:a1]!r} vs {kept[b0:b1]!r}")
    if removed > len(v2_lines):
        problems.append(f"{label}: {removed} source lines are missing from the copy but only {len(v2_lines)} are marked '# v2:'")
    source_text = {ln.strip() for ln in source_lines}
    for ln in v2_lines:
        match = WAS_RE.search(ln)
        if match and match.group(1).strip() not in source_text:
            problems.append(f"{label}: '# v2: was:' text not found in the source any more: {match.group(1)!r}")
    return problems


def _body_after_header(text: str) -> list[str]:
    lines = text.split("\n")
    assert HEADER_END in lines, "copy has no provenance header"
    return lines[lines.index(HEADER_END) + 1:]


def _segments(text: str, names) -> dict[str, list[str]]:
    lines = text.split("\n")
    found: dict[str, list[str]] = {}
    for node in ast.parse(text).body:
        name = getattr(node, "name", None)
        if name is None and isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            found[name] = lines[node.lineno - 1:node.end_lineno]
    return found


@pytest.mark.parametrize("source,copy", WHOLE_FILE_COPIES, ids=[c for _, c in WHOLE_FILE_COPIES])
def test_whole_file_copy_matches_source(source, copy):
    copy_text = _read(copy)
    header = copy_text.split(HEADER_END)[0]
    assert source in header and "7dbcba3" in header and "copied, not imported" in header
    problems = _drift(_read(source).split("\n"), _body_after_header(copy_text), copy)
    assert not problems, f"{copy} drifted from {source}:\n" + "\n".join(problems)


@pytest.mark.parametrize("source,copy", DOCUMENT_COPIES, ids=[c for _, c in DOCUMENT_COPIES])
def test_document_copy_is_identical_to_source(source, copy):
    assert (REPO / source).read_bytes() == (REPO / copy).read_bytes()


def test_rag_folder_holds_only_the_copied_documents():
    names = sorted(p.name for p in (REPO / "llm_approach/rag/documents").iterdir())
    assert names == sorted(Path(c).name for _, c in DOCUMENT_COPIES)  # a stray file here would be indexed


@pytest.mark.parametrize("source,copy,names", PARTIAL_COPIES, ids=[c for _, c, _ in PARTIAL_COPIES])
def test_extracted_definitions_match_source(source, copy, names):
    copy_text = _read(copy)
    header = copy_text.split(HEADER_END)[0]
    assert source in header and "7dbcba3" in header and "copied, not imported" in header
    src_segs = _segments(_read(source), names)
    copy_segs = _segments(copy_text, names)
    problems: list[str] = []
    for name in names:
        if name not in src_segs:
            problems.append(f"{copy}: {name} no longer exists in {source}")
        elif name not in copy_segs:
            problems.append(f"{copy}: {name} is missing from the copy")
        else:
            problems += _drift(src_segs[name], copy_segs[name], f"{copy}:{name}")
    assert not problems, f"{copy} drifted from {source}:\n" + "\n".join(problems)


# Copies made after the first batch cite the last commit that changed their source (not 7dbcba3).
# (source, copy, sha in the header)
MCP_WHOLE_FILE_COPIES = [
    ("app/server.py", "llm_approach/tools/mcp_search.py", "ed0406a"),
]
# (source, copy, sha in the header, names)
MCP_PARTIAL_COPIES = [
    ("app/experiments/agent_modes.py", "llm_approach/tools/mcp_timeout.py", "fca91ac", ("_call_search_rooms_with_timeout",)),
]


@pytest.mark.parametrize("source,copy,sha", MCP_WHOLE_FILE_COPIES, ids=[c for _, c, _ in MCP_WHOLE_FILE_COPIES])
def test_mcp_whole_file_copy_matches_source(source, copy, sha):
    copy_text = _read(copy)
    header = copy_text.split(HEADER_END)[0]
    assert source in header and sha in header and "copied, not imported" in header
    problems = _drift(_read(source).split("\n"), _body_after_header(copy_text), copy)
    assert not problems, f"{copy} drifted from {source}:\n" + "\n".join(problems)


@pytest.mark.parametrize("source,copy,sha,names", MCP_PARTIAL_COPIES, ids=[c for _, c, _, _ in MCP_PARTIAL_COPIES])
def test_mcp_extracted_definitions_match_source(source, copy, sha, names):
    copy_text = _read(copy)
    header = copy_text.split(HEADER_END)[0]
    assert source in header and sha in header and "copied, not imported" in header
    src_segs = _segments(_read(source), names)
    copy_segs = _segments(copy_text, names)
    problems: list[str] = []
    for name in names:
        if name not in src_segs:
            problems.append(f"{copy}: {name} no longer exists in {source}")
        elif name not in copy_segs:
            problems.append(f"{copy}: {name} is missing from the copy")
        else:
            problems += _drift(src_segs[name], copy_segs[name], f"{copy}:{name}")
    assert not problems, f"{copy} drifted from {source}:\n" + "\n".join(problems)


def _member_segments(text: str, class_name: str, names) -> dict[str, list[str]]:
    lines = text.split("\n")
    found: dict[str, list[str]] = {}
    for cls in ast.parse(text).body:
        if isinstance(cls, ast.ClassDef) and cls.name == class_name:
            for node in cls.body:
                name = getattr(node, "name", None)
                if name is None and isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                if name in names:
                    found[name] = lines[node.lineno - 1:node.end_lineno]
    return found


@pytest.mark.parametrize("source,copy,class_name,names", CLASS_MEMBER_COPIES, ids=[c for _, c, _, _ in CLASS_MEMBER_COPIES])
def test_copied_class_members_match_source(source, copy, class_name, names):
    copy_text = _read(copy)
    header = copy_text.split(HEADER_END)[0]
    assert source in header and "7dbcba3" in header and "copied, not imported" in header
    src_segs = _member_segments(_read(source), class_name, names)
    copy_segs = _member_segments(copy_text, class_name, names)
    problems: list[str] = []
    for name in names:
        if name not in src_segs:
            problems.append(f"{copy}: {class_name}.{name} no longer exists in {source}")
        elif name not in copy_segs:
            problems.append(f"{copy}: {class_name}.{name} is missing from the copy")
        else:
            problems += _drift(src_segs[name], copy_segs[name], f"{copy}:{class_name}.{name}")
    assert not problems, f"{copy} drifted from {source}:\n" + "\n".join(problems)


def test_class_member_drift_check_notices_a_changed_method():
    src = "class A:\n    def m(self):\n        return 1\n"
    assert _member_segments(src, "A", ("m",))["m"] == ["    def m(self):", "        return 1"]
    assert _drift(_member_segments(src, "A", ("m",))["m"], ["    def m(self):", "        return 2"], "x")


def test_drift_check_detects_a_changed_line():
    source = ["a = 1", "b = 2", "c = 3"]
    assert _drift(source, ["a = 1", "b = 2", "c = 3"], "x") == []
    assert _drift(source, ["a = 1", "b = 99", "c = 3"], "x")
    assert _drift(source, ["a = 1", "c = 3"], "x")  # unmarked deletion
    assert _drift(source, ["a = 1", "b = 20  # v2: was: b = 2", "c = 3"], "x") == []
    assert _drift(source, ["a = 1", "b = 20  # v2: was: b = 5", "c = 3"], "x")
