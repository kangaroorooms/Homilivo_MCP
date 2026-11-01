"""The logic of scripts/redis_restart_check.py against fakeredis: no network, no real Redis, no LLM."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import fakeredis
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "redis_restart_check.py"


@pytest.fixture
def check():
    spec = importlib.util.spec_from_file_location("redis_restart_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_steps_pass_and_only_own_keys_are_deleted(check):
    server = fakeredis.FakeServer()

    def new_client():
        return fakeredis.FakeRedis(server=server, decode_responses=True)

    bystander = new_client()
    bystander.set("homilivo:v2:session:someone-else", "x")
    lines: list[str] = []

    assert check.run_checks(new_client, emit=lines.append) is True

    assert [line.split()[0] for line in lines] == ["PASS"] * 6
    assert sorted(bystander.keys("*")) == ["homilivo:v2:session:someone-else"]
    assert not any(check.SEARCH_MSG in line or check.WIFI_MSG in line for line in lines)  # no message text printed


def test_a_store_that_forgets_fails_the_restart_step(check, monkeypatch):
    server = fakeredis.FakeServer()

    def new_client():
        return fakeredis.FakeRedis(server=server, decode_responses=True)

    original = check._turn
    calls = {"n": 0}

    def forgetful(store, sid, message):
        calls["n"] += 1
        if calls["n"] == 3:  # the wifi turn after the "restart": wipe Redis first
            store._redis.flushall()
        return original(store, sid, message)

    monkeypatch.setattr(check, "_turn", forgetful)
    lines: list[str] = []
    assert check.run_checks(new_client, emit=lines.append) is False
    assert any(line.startswith("FAIL  3") for line in lines)
