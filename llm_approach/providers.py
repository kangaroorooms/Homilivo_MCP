"""The one place that knows which LLM provider is active.

``complete(messages, max_tokens)`` sends the same OpenAI-style messages to Groq or Gemini and returns the
reply text plus token usage. Nothing else in ``llm_approach`` looks at the provider. Both SDKs are imported
lazily, so the offline tests and the other provider never need them.

Errors are not caught here: timeouts and API errors propagate to ``llm_client``, which logs the class name,
HTTP code and status only (``error_details``). Neither the key, the prompt nor the reply is ever logged from this module.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from typing import Any

from llm_approach import config, gemini_limits


@dataclass(frozen=True)
class Completion:
    text: str  # "" when the model returned nothing (empty or blocked reply)
    tokens_in: int
    tokens_out: int
    cached: int | None  # None: the usage data never reported cached tokens
    finish_reason: str | None = None  # Gemini only, e.g. "STOP" or "MAX_TOKENS"
    thinking_tokens: int = 0  # Gemini only; already included in tokens_out


# Finish reasons whose text must not be used: a cut-off or blocked reply counts as empty (invalid JSON).
_UNUSABLE_FINISH = {"MAX_TOKENS", "SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "RECITATION"}


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


async def complete(
    messages: list[dict[str, str]], max_tokens: int | None = None, provider: str | None = None
) -> Completion:
    """``provider`` names the fallback provider for one call; the default is the active provider."""
    max_tokens = config.LLM_MAX_TOKENS if max_tokens is None else max_tokens
    if (provider or config.provider()) == "gemini":
        return await _complete_gemini(messages, config.gemini_max_output_tokens())  # own cap, see config
    return await _complete_groq(messages, max_tokens)


# ---- Groq (unchanged behaviour) -------------------------------------------------

_groq_client: Any = None


def _get_groq_client() -> Any:
    """Create the async Groq client on first use; the key comes from the environment."""
    global _groq_client
    if _groq_client is None:
        from groq import AsyncGroq

        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:  # missing or blank: fail here, before any request is made
            raise KeyError("GROQ_API_KEY")
        _groq_client = AsyncGroq(api_key=key, max_retries=0)  # fail fast; we retry once ourselves
    return _groq_client


async def _complete_groq(messages: list[dict[str, str]], max_tokens: int) -> Completion:
    kwargs: dict[str, Any] = dict(
        model=config.groq_model(),
        messages=messages,
        temperature=config.TEMPERATURE,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    if config.REASONING_EFFORT:
        kwargs["reasoning_effort"] = config.REASONING_EFFORT
    response = await asyncio.wait_for(
        _get_groq_client().chat.completions.create(**kwargs), timeout=config.LLM_TIMEOUT_S
    )
    usage = getattr(response, "usage", None)
    details = getattr(usage, "prompt_tokens_details", None)
    return Completion(
        response.choices[0].message.content or "",
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
        _count(getattr(details, "cached_tokens", None)),
    )


# ---- Gemini ----------------------------------------------------------------------

_gemini_client: Any = None


def _get_gemini_client() -> Any:
    """Create the google-genai client on first use. SDK-side retries are off (one attempt): our code
    owns the single JSON retry. The key comes from the environment only."""
    global _gemini_client
    if _gemini_client is None:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:  # missing or blank: fail here, before any request is made
            raise KeyError("GEMINI_API_KEY")
        _gemini_client = _new_gemini_client(gemini_timeout_ms())
    return _gemini_client


def gemini_timeout_ms() -> int:
    """The SDK's ``HttpOptions.timeout`` is in milliseconds; ``config.gemini_timeout_s()`` is in seconds."""
    return int(config.gemini_timeout_s() * 1000)


def _new_gemini_client(timeout_ms: int | None) -> Any:
    """A fresh google-genai client; ``timeout_ms=None`` leaves the timeout option out (smoke narrowing)."""
    from google import genai

    http_options: dict[str, Any] = {"retry_options": {"attempts": 1}}
    if timeout_ms is not None:
        http_options["timeout"] = timeout_ms
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "").strip(), http_options=http_options)


_STATUS_RE = re.compile(r"^[A-Z][A-Z_]{1,39}$")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_.\-]{20,}")


def error_details(exc: BaseException) -> tuple[int | None, str | None]:
    """HTTP code and status string (e.g. ``400``, ``INVALID_ARGUMENT``) of a google-genai API error, else
    ``(None, None)``. Both are validated, so nothing free-form can slip through."""
    code = getattr(exc, "code", None)
    code = code if isinstance(code, int) and not isinstance(code, bool) and 100 <= code <= 599 else None
    status = getattr(exc, "status", None)
    status = status if isinstance(status, str) and _STATUS_RE.match(status) else None
    return code, status


def redact(text: str, limit: int = 200) -> str:
    """``text`` with the Gemini key and any 20+ character token-like string replaced by ``***``,
    then cut to ``limit`` characters."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        text = text.replace(key, "***")
    return _TOKEN_RE.sub("***", text)[:limit]


def to_gemini_request(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, Any]]]:
    """OpenAI-style messages -> (system instruction, contents). ``system`` becomes the system
    instruction, ``user`` stays ``user``, ``assistant`` becomes ``model``; the order is kept."""
    system: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role, text = message.get("role"), message.get("content") or ""
        if not text.strip():  # the API rejects an empty text part
            continue
        if role == "system":
            system.append(text)
        else:
            contents.append({"role": "model" if role == "assistant" else "user", "parts": [{"text": text}]})
    return ("\n\n".join(system) or None), contents


def gemini_config(system: str | None, max_tokens: int) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "temperature": config.TEMPERATURE,
        "max_output_tokens": max_tokens,
        "response_mime_type": "application/json",  # JSON response mode
    }
    thinking = config.gemini_thinking()  # per model family, or the env override; None sends nothing
    if thinking:
        cfg["thinking_config"] = thinking
    if system:
        cfg["system_instruction"] = system
    return cfg


def _gemini_text(response: Any) -> str:
    """The reply text, or "" for an empty or blocked reply (no candidates, safety block, no text part)."""
    try:
        text = response.text
    except Exception:  # noqa: BLE001 - a blocked reply may make the SDK raise; treat it as empty
        return ""
    return text if isinstance(text, str) else ""


def build_gemini_request(messages: list[dict[str, str]], max_tokens: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(contents, config) exactly as sent to the SDK."""
    system, contents = to_gemini_request(messages)
    return contents, gemini_config(system, max_tokens)


def request_shape(contents: list[dict[str, Any]], cfg: dict[str, Any]) -> str:
    """One line describing the request's structure: counts, roles, lengths and non-text config values.
    Never any prompt or reply text."""
    system = cfg.get("system_instruction")
    total = sum(len(part.get("text", "")) for c in contents for part in c.get("parts", [])) + len(system or "")
    values = {k: v for k, v in cfg.items() if k != "system_instruction"}
    return (
        f"messages {len(contents)} (roles: {', '.join(c['role'] for c in contents)}); "
        f"system instruction {'yes, ' + str(len(system)) + ' chars' if system else 'no'}; total {total} chars; "
        f"config {values}; timeout {gemini_timeout_ms()} ms (http_options.timeout, milliseconds)"
    )


async def send_gemini(client: Any, contents: list[dict[str, Any]], cfg: dict[str, Any]) -> Any:
    return await asyncio.wait_for(
        client.aio.models.generate_content(model=config.model("gemini"), contents=contents, config=cfg),
        timeout=config.gemini_timeout_s(),
    )


async def _complete_gemini(messages: list[dict[str, str]], max_tokens: int) -> Completion:
    contents, cfg = build_gemini_request(messages, max_tokens)
    client = _get_gemini_client()
    chars = sum(len(part.get("text", "")) for c in contents for part in c.get("parts", [])) + len(cfg.get("system_instruction") or "")
    response = await gemini_limits.get_limiter().call(  # daily cap, 60 s window, one 429 retry
        lambda: send_gemini(client, contents, cfg),
        budget=config.daily_request_budget(),
        prompt_estimate=gemini_limits.estimate_prompt_tokens(chars),
        tokens_of=_response_tokens,
    )
    usage = getattr(response, "usage_metadata", None)
    thinking = _count(getattr(usage, "thoughts_token_count", None)) or 0  # billed as output
    out = (_count(getattr(usage, "candidates_token_count", None)) or 0) + thinking
    finish = _finish_reason(response)
    text = "" if finish in _UNUSABLE_FINISH else _gemini_text(response)
    return Completion(
        text,
        _count(getattr(usage, "prompt_token_count", None)) or 0,
        out,
        _count(getattr(usage, "cached_content_token_count", None)),
        finish,
        thinking,
    )


def _response_tokens(response: Any) -> int | None:
    usage = getattr(response, "usage_metadata", None)
    total = (_count(getattr(usage, "prompt_token_count", None)) or 0) + (
        _count(getattr(usage, "candidates_token_count", None)) or 0) + (_count(getattr(usage, "thoughts_token_count", None)) or 0)
    return total or None


def _finish_reason(response: Any) -> str | None:
    """Finish reason of the first candidate as a plain name ("STOP", "MAX_TOKENS"), or None."""
    try:
        reason = response.candidates[0].finish_reason
    except Exception:  # noqa: BLE001 - no candidates (blocked or empty) or a fake without them
        return None
    if reason is None:
        return None
    name = getattr(reason, "name", None) or str(reason)
    return name.rsplit(".", 1)[-1].upper()
