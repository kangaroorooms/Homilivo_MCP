"""App configuration tests: lock in safe FastAPI settings."""

from app.api import app


def test_app_debug_is_false():
    assert app.debug is False


def test_app_has_title():
    assert app.title is not None
    assert app.title.strip() != ""
