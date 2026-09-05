"""
Tests for POST /feedback (app/api.py) and its storage (app/feedback_store.py).

Covers: up/down submission, tagging with a real request_id returned by
/chat, a missing request_id being rejected, an optional note round-
tripping, and append-only persistence across multiple submissions.
"""

from __future__ import annotations

import pytest

import app.feedback_store as feedback_store


@pytest.fixture(autouse=True)
def _isolated_feedback_log(tmp_path, monkeypatch):
    """Redirect feedback storage to a throwaway file per test."""
    log_path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback_store, "FEEDBACK_LOG_PATH", log_path)
    yield log_path


# ============================================================
# POST /feedback
# ============================================================

def test_feedback_up_is_recorded(client, auth_headers, mock_homilivo_success, _isolated_feedback_log):
    turn = client.post(
        "/chat", json={"message": "find rooms in Jaipur", "mode": "fast"}, headers=auth_headers,
    ).json()
    request_id = turn["request_id"]
    assert request_id

    response = client.post(
        "/feedback", json={"request_id": request_id, "rating": "up"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["request_id"] == request_id

    records = feedback_store.read_feedback(_isolated_feedback_log)
    assert len(records) == 1
    assert records[0]["request_id"] == request_id
    assert records[0]["rating"] == "up"
    assert records[0]["note"] is None


def test_feedback_down_with_note_is_recorded(client, auth_headers, mock_homilivo_success, _isolated_feedback_log):
    turn = client.post(
        "/chat", json={"message": "find rooms in Jaipur", "mode": "fast"}, headers=auth_headers,
    ).json()
    request_id = turn["request_id"]

    response = client.post(
        "/feedback",
        json={"request_id": request_id, "rating": "down", "note": "wrong city returned"},
    )

    assert response.status_code == 200

    records = feedback_store.read_feedback(_isolated_feedback_log)
    assert len(records) == 1
    assert records[0]["rating"] == "down"
    assert records[0]["note"] == "wrong city returned"


def test_feedback_missing_request_id_is_rejected(client, _isolated_feedback_log):
    response = client.post("/feedback", json={"rating": "up"})

    assert response.status_code == 422
    assert feedback_store.read_feedback(_isolated_feedback_log) == []


def test_feedback_empty_request_id_is_rejected(client, _isolated_feedback_log):
    response = client.post("/feedback", json={"request_id": "", "rating": "up"})

    assert response.status_code == 422
    assert feedback_store.read_feedback(_isolated_feedback_log) == []


def test_feedback_invalid_rating_is_rejected(client, _isolated_feedback_log):
    response = client.post("/feedback", json={"request_id": "some-id", "rating": "sideways"})

    assert response.status_code == 422
    assert feedback_store.read_feedback(_isolated_feedback_log) == []


def test_feedback_is_append_only_across_multiple_submissions(client, _isolated_feedback_log):
    client.post("/feedback", json={"request_id": "r1", "rating": "up"})
    client.post("/feedback", json={"request_id": "r2", "rating": "down"})
    client.post("/feedback", json={"request_id": "r3", "rating": "up"})

    records = feedback_store.read_feedback(_isolated_feedback_log)
    assert [r["request_id"] for r in records] == ["r1", "r2", "r3"]


# ============================================================
# app/feedback_store.py unit coverage
# ============================================================

def test_record_feedback_writes_one_json_line(tmp_path, monkeypatch):
    log_path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback_store, "FEEDBACK_LOG_PATH", log_path)

    feedback_store.record_feedback("req-abc", "up", None)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_read_feedback_returns_empty_list_when_no_file_exists(tmp_path):
    missing_path = tmp_path / "does-not-exist.jsonl"

    assert feedback_store.read_feedback(missing_path) == []
