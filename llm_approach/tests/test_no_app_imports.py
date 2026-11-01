"""llm_approach must not import anything from app/: the code is copied, not imported (Phase 3B).

Comments, docstrings and the provenance headers are ignored: only real import statements and
``importlib.import_module`` / ``__import__`` calls with a literal module name count.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _is_app(name: str | None) -> bool:
    return bool(name) and (name == "app" or name.startswith("app."))


def app_imports(source: str) -> list[int]:
    """Line numbers of imports of the ``app`` package in ``source``."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import) and any(_is_app(a.name) for a in node.names):
            hits.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and _is_app(node.module):
            hits.append(node.lineno)
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in ("import_module", "__import__") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and _is_app(arg.value):
                    hits.append(node.lineno)
    return hits


def test_no_llm_approach_module_imports_from_app():
    offenders = {}
    for path in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        lines = app_imports(path.read_text(encoding="utf-8"))
        if lines:
            offenders[str(path.relative_to(ROOT))] = lines
    assert not offenders, f"llm_approach modules import from app/: {offenders}"


def test_the_scan_finds_real_imports_and_ignores_comments():
    assert app_imports("from app.guardrails import x\n") == [1]
    assert app_imports("import app.embeddings\n") == [1]
    assert app_imports("import app\n") == [1]
    assert app_imports("def f():\n    import importlib\n    importlib.import_module('app.server')\n") == [3]
    assert app_imports("# from app.guardrails import x\nx = 1  # v2: was: from app.embeddings import y\n") == []
    assert app_imports('"""from app.x import y"""\nname = "app.server"\n') == []
    assert app_imports("from llm_approach.application import x\nimport apple\nfrom . import app\n") == []
