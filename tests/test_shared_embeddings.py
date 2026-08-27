"""
Regression tests for the shared SentenceTransformer loader.

Before app/embeddings.py existed, app/guardrails/input_guardrail.py and
app/experiments/interpreter.py each instantiated their own independent
SentenceTransformer("all-MiniLM-L6-v2"), doubling memory and duplicating
the one-time load cost. These tests prove both call sites now resolve
to the exact same model instance.
"""

import app.embeddings as embeddings
from app.guardrails.input_guardrail import get_semantic_model
from app.experiments import interpreter


def test_shared_loader_returns_the_same_instance_on_repeat_calls():
    first = embeddings.get_shared_model()
    second = embeddings.get_shared_model()

    assert first is second


def test_input_guardrail_and_interpreter_share_one_model_instance():
    guardrail_model = get_semantic_model()

    assert interpreter._build_anchor_index()
    interpreter_model = interpreter._load_model()

    assert guardrail_model is interpreter_model
    assert guardrail_model is embeddings.get_shared_model()


def test_interpreter_still_degrades_gracefully_when_loading_fails(monkeypatch):
    """
    interpreter.py must keep working deterministically (returning None
    from classify_operation) if the shared model can't be loaded - the
    sharing change must not weaken this existing safety behaviour.
    """
    monkeypatch.setattr(interpreter, "_model", None)
    monkeypatch.setattr(interpreter, "_model_failed", False)
    monkeypatch.setattr(interpreter, "_anchor_matrix", None)
    monkeypatch.setattr(interpreter, "_anchor_labels", None)

    def _boom():
        raise RuntimeError("simulated model load failure")

    monkeypatch.setattr(embeddings, "get_shared_model", _boom)

    assert interpreter._load_model() is None
    assert interpreter._model_failed is True
    assert interpreter.classify_operation("which one is cheapest") is None
