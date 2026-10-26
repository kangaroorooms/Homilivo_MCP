# Adapted from tests/test_p3_18_observability.py (tests 8: session-id handling) at commit 7dbcba3.
# The originals drive /chat and /ws/chat through app.api, which is not copied. These call the copied
# _normalize_session_id directly, so the same rules are checked without app.api. All lines are v2.
# Not copied: the TestClient/app.api cases, the log-field cases and the MCP/pricing cases.

import logging
import uuid

import pytest

from llm_approach.security.secrets_check import _normalize_session_id  # v2: was: through app.api /chat


def test_invalid_session_id_replaced_with_uuid(caplog):  # v2: was test_invalid_session_id_replaced_with_uuid via /chat
    bad_session_id = "user@email.com"  # v2

    with caplog.at_level(logging.WARNING):  # v2
        used = _normalize_session_id(bad_session_id)  # v2

    assert used != bad_session_id  # v2
    assert str(uuid.UUID(used)) == used  # v2
    assert bad_session_id not in caplog.text  # v2: the invalid value is never logged
    assert "Invalid session_id format received" in caplog.text  # v2


@pytest.mark.parametrize("case", [str.lower, str.upper])  # v2
def test_valid_uuid_session_id_is_accepted_and_lowercased(case):  # v2
    canonical = str(uuid.uuid4())  # v2

    assert _normalize_session_id(case(canonical)) == canonical  # v2


def test_upper_and_lower_case_uuid_resolve_to_one_session():  # v2
    canonical = str(uuid.uuid4())  # v2

    assert _normalize_session_id(canonical.upper()) == _normalize_session_id(canonical.lower()) == canonical  # v2


@pytest.mark.parametrize("empty", [None, ""])  # v2
def test_missing_session_id_gets_a_fresh_uuid(empty):  # v2
    first, second = _normalize_session_id(empty), _normalize_session_id(empty)  # v2
    assert first != second  # v2
    assert str(uuid.UUID(first)) == first  # v2
