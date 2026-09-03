# -*- coding: utf-8 -*-
"""Tests for the Codex OAuth generation backend."""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import pytest

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from src.llm import codex_oauth  # noqa: E402
from src.llm.backend_factory import create_generation_backend  # noqa: E402
from src.llm.backend_registry import (  # noqa: E402
    CODEX_OAUTH_BACKEND_ID,
    GENERATION_ONLY_BACKEND_IDS,
    SUPPORTED_GENERATION_BACKENDS,
    resolve_generation_backend_id,
)
from src.llm.codex_oauth_backend import CodexOAuthGenerationBackend  # noqa: E402
from src.llm.generation_backend import GenerationError, GenerationErrorCode  # noqa: E402


def _credential(**overrides):
    credential = {
        "type": "codex",
        "access_token": "header.payload.signature",
        "refresh_token": "rt.test",
        "id_token": "",
        "account_id": "account-123",
        "plan_type": "prolite",
        "email": "tester@example.com",
        # Comfortably in the future so no refresh is attempted.
        "expires_at": time.time() + 86400,
    }
    credential.update(overrides)
    return credential


def _config(auth_file, **overrides):
    defaults = {
        "codex_oauth_auth_file": str(auth_file),
        "codex_oauth_model": "gpt-5.6-terra",
        "codex_oauth_reasoning_effort": "medium",
        "generation_backend_timeout_seconds": 30,
        "generation_backend_max_output_bytes": 1048576,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture()
def auth_file(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(_credential()), encoding="utf-8")
    return path


class _FakeResponse:
    """Minimal stand-in for a streamed requests.Response."""

    def __init__(self, status_code=200, lines=(), text=""):
        self.status_code = status_code
        self._lines = list(lines)
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def iter_lines(self, decode_unicode=False):
        # requests hands back bytes for this endpoint: it declares no charset.
        return iter(self._lines)


def _sse(events):
    return [f"data: {json.dumps(event)}".encode("utf-8") for event in events]


COMPLETED_EVENTS = [
    {"type": "response.output_text.delta", "delta": "分析"},
    {"type": "response.output_text.delta", "delta": "结果"},
    {
        "type": "response.completed",
        "response": {
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 18,
                "output_tokens_details": {"reasoning_tokens": 3},
            }
        },
    },
]


# --- registration -----------------------------------------------------------


def test_backend_is_registered_and_generation_only():
    assert CODEX_OAUTH_BACKEND_ID in SUPPORTED_GENERATION_BACKENDS
    # Tool calling is not implemented, so the Agent path must not accept it.
    assert CODEX_OAUTH_BACKEND_ID in GENERATION_ONLY_BACKEND_IDS


def test_resolver_accepts_configured_backend():
    config = SimpleNamespace(generation_backend=CODEX_OAUTH_BACKEND_ID)
    assert resolve_generation_backend_id(config) == CODEX_OAUTH_BACKEND_ID


def test_factory_builds_backend(auth_file):
    backend = create_generation_backend(
        CODEX_OAUTH_BACKEND_ID, config=_config(auth_file)
    )
    assert isinstance(backend, CodexOAuthGenerationBackend)
    assert backend.backend_id == CODEX_OAUTH_BACKEND_ID
    assert backend.capabilities.supports_tools is False


# --- credential handling ----------------------------------------------------


def test_missing_credential_reports_login_required(tmp_path):
    backend = CodexOAuthGenerationBackend(_config(tmp_path / "absent.json"))
    error = backend.get_config_error()
    assert error is not None
    assert error.error_code is GenerationErrorCode.LOGIN_REQUIRED


def test_saved_credential_is_owner_only(tmp_path):
    path = tmp_path / "nested" / "auth.json"
    codex_oauth.save_credential(str(path), _credential())
    assert oct(os.stat(path).st_mode & 0o777) == "0o600"


class _TokenResponse:
    """Stand-in for the OAuth token endpoint's JSON response."""

    def __init__(self, access_token):
        self.status_code = 200
        self.text = ""
        self._access_token = access_token

    def json(self):
        return {
            "access_token": self._access_token,
            "refresh_token": f"rotated-for-{self._access_token}",
            "expires_in": 3600,
        }


def _patch_token_endpoint(monkeypatch, calls):
    def fake_post(url, **kwargs):
        assert url == codex_oauth.OAUTH_TOKEN_URL
        calls.append((kwargs.get("data") or {}).get("refresh_token"))
        return _TokenResponse(f"refreshed-{len(calls)}.token.sig")

    monkeypatch.setattr(codex_oauth.requests, "post", fake_post)


def test_expired_credential_triggers_refresh(monkeypatch, tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(_credential(expires_at=time.time() - 10)), encoding="utf-8")
    calls = []
    _patch_token_endpoint(monkeypatch, calls)

    result = codex_oauth.ensure_fresh_credential(
        codex_oauth.load_credential(str(path)), str(path)
    )

    assert calls == ["rt.test"]
    assert result["access_token"] == "refreshed-1.token.sig"
    # The rotated pair is persisted, or the next process would replay a dead token.
    assert json.loads(path.read_text())["access_token"] == "refreshed-1.token.sig"


def test_expired_credential_reuses_a_refresh_another_worker_already_did(monkeypatch, tmp_path):
    """OpenAI rotates the refresh token, so a second rotation kills the first."""
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(_credential(access_token="fresh.token.sig")), encoding="utf-8")
    calls = []
    _patch_token_endpoint(monkeypatch, calls)

    stale = _credential(access_token="stale.token.sig", expires_at=time.time() - 10)
    result = codex_oauth.ensure_fresh_credential(stale, str(path))

    assert calls == []
    assert result["access_token"] == "fresh.token.sig"


def test_forced_refresh_reuses_a_newer_credential_from_disk(monkeypatch, tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(_credential(access_token="fresh.token.sig")), encoding="utf-8")
    calls = []
    _patch_token_endpoint(monkeypatch, calls)

    stale = _credential(access_token="stale.token.sig")
    result = codex_oauth.refresh_credential(stale, str(path))

    assert calls == []
    assert result["access_token"] == "fresh.token.sig"


def test_concurrent_expiry_rotates_the_token_exactly_once(monkeypatch, tmp_path):
    import threading

    path = tmp_path / "auth.json"
    path.write_text(json.dumps(_credential(expires_at=time.time() - 10)), encoding="utf-8")
    calls = []
    lock = threading.Lock()

    def fake_post(url, **kwargs):
        with lock:
            calls.append((kwargs.get("data") or {}).get("refresh_token"))
            index = len(calls)
        # Hold long enough that an unserialized implementation would let a
        # second worker in with the same (already spent) refresh token.
        time.sleep(0.05)
        return _TokenResponse(f"refreshed-{index}.token.sig")

    monkeypatch.setattr(codex_oauth.requests, "post", fake_post)

    barrier = threading.Barrier(4)
    results = [None] * 4

    def worker(index):
        barrier.wait()
        results[index] = codex_oauth.ensure_fresh_credential(
            codex_oauth.load_credential(str(path)), str(path)
        )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == ["rt.test"]
    assert {r["access_token"] for r in results} == {"refreshed-1.token.sig"}


def test_credential_write_leaves_no_temp_file_and_stays_owner_only(tmp_path):
    path = tmp_path / "auth.json"
    codex_oauth.save_credential(str(path), _credential())

    assert oct(os.stat(path).st_mode & 0o777) == "0o600"
    # The temp file is unique per write and lands in the same directory, so an
    # interrupted write can never be mistaken for the credential itself.
    assert [p.name for p in tmp_path.glob(".auth.json.*")] == []


# --- request construction ---------------------------------------------------


def test_request_body_shape():
    body = codex_oauth.build_request_body(
        "问题", model="gpt-5.6-sol", effort="high", instructions="系统指令"
    )
    assert body["model"] == "gpt-5.6-sol"
    assert body["instructions"] == "系统指令"
    # The endpoint rejects non-streamed requests.
    assert body["stream"] is True
    assert body["reasoning"] == {"effort": "high", "summary": "auto"}
    assert body["input"][0]["content"][0]["type"] == "input_text"


def test_effort_none_omits_reasoning():
    body = codex_oauth.build_request_body("问题", model="m", effort="none")
    assert "reasoning" not in body


def test_invalid_effort_falls_back_to_default():
    assert codex_oauth.normalize_effort("wild") == codex_oauth.DEFAULT_EFFORT
    assert codex_oauth.normalize_effort("high") == "high"


def test_requests_carry_explicit_user_agent(monkeypatch, auth_file):
    """Cloudflare 530s the default python-requests UA, so this is load bearing."""
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs.get("headers") or {})
        return _FakeResponse(lines=_sse(COMPLETED_EVENTS))

    monkeypatch.setattr(codex_oauth.requests, "post", fake_post)
    codex_oauth.generate(_credential(), "问题")
    assert captured["User-Agent"] == codex_oauth.USER_AGENT
    assert captured["Originator"] == codex_oauth.ORIGINATOR
    assert captured["Chatgpt-Account-Id"] == "account-123"


# --- generation -------------------------------------------------------------


def test_generate_returns_text_and_usage(monkeypatch, auth_file):
    monkeypatch.setattr(
        codex_oauth.requests,
        "post",
        lambda url, **kwargs: _FakeResponse(lines=_sse(COMPLETED_EVENTS)),
    )
    backend = CodexOAuthGenerationBackend(_config(auth_file))
    result = backend.generate("问题", {})

    assert result.text == "分析结果"
    assert result.backend == CODEX_OAUTH_BACKEND_ID
    assert result.model == "gpt-5.6-terra"
    # Real provider usage, unlike the local CLI backends.
    assert result.usage["prompt_tokens"] == 11
    assert result.usage["completion_tokens"] == 7
    assert result.usage["usage_available"] is True


def test_generation_config_overrides_model(monkeypatch, auth_file):
    seen = {}

    def fake_post(url, **kwargs):
        seen.update(kwargs.get("json") or {})
        return _FakeResponse(lines=_sse(COMPLETED_EVENTS))

    monkeypatch.setattr(codex_oauth.requests, "post", fake_post)
    backend = CodexOAuthGenerationBackend(_config(auth_file))
    result = backend.generate("问题", {"model": "gpt-5.6-luna", "reasoning_effort": "low"})

    assert seen["model"] == "gpt-5.6-luna"
    assert seen["reasoning"]["effort"] == "low"
    assert result.model == "gpt-5.6-luna"


def test_unauthorized_refreshes_once_then_succeeds(monkeypatch, auth_file):
    attempts = {"post": 0, "refresh": 0}

    def fake_post(url, **kwargs):
        attempts["post"] += 1
        if attempts["post"] == 1:
            return _FakeResponse(status_code=401, text="token expired")
        return _FakeResponse(lines=_sse(COMPLETED_EVENTS))

    def fake_refresh(credential, path):
        attempts["refresh"] += 1
        return _credential(access_token="new.token.sig")

    monkeypatch.setattr(codex_oauth.requests, "post", fake_post)
    monkeypatch.setattr(codex_oauth, "refresh_credential", fake_refresh)

    backend = CodexOAuthGenerationBackend(_config(auth_file))
    result = backend.generate("问题", {})

    assert attempts == {"post": 2, "refresh": 1}
    assert result.text == "分析结果"
    assert result.diagnostics["refreshed_mid_request"] is True


def test_persistent_unauthorized_raises_login_required(monkeypatch, auth_file):
    monkeypatch.setattr(
        codex_oauth.requests,
        "post",
        lambda url, **kwargs: _FakeResponse(status_code=401, text="nope"),
    )
    monkeypatch.setattr(
        codex_oauth, "refresh_credential", lambda credential, path: _credential()
    )
    backend = CodexOAuthGenerationBackend(_config(auth_file))

    with pytest.raises(GenerationError) as excinfo:
        backend.generate("问题", {})
    assert excinfo.value.error_code is GenerationErrorCode.LOGIN_REQUIRED


def test_output_over_limit_is_structured(monkeypatch, auth_file):
    events = [{"type": "response.output_text.delta", "delta": "x" * 64}] * 4
    monkeypatch.setattr(
        codex_oauth.requests,
        "post",
        lambda url, **kwargs: _FakeResponse(lines=_sse(events)),
    )
    backend = CodexOAuthGenerationBackend(
        _config(auth_file, generation_backend_max_output_bytes=100)
    )

    with pytest.raises(GenerationError) as excinfo:
        backend.generate("问题", {})
    assert excinfo.value.error_code is GenerationErrorCode.OUTPUT_TOO_LARGE
    assert excinfo.value.fallbackable is True


def test_upstream_failure_event_is_structured(monkeypatch, auth_file):
    events = [{"type": "response.failed", "response": {"error": {"message": "boom"}}}]
    monkeypatch.setattr(
        codex_oauth.requests,
        "post",
        lambda url, **kwargs: _FakeResponse(lines=_sse(events)),
    )
    backend = CodexOAuthGenerationBackend(_config(auth_file))

    with pytest.raises(GenerationError) as excinfo:
        backend.generate("问题", {})
    assert excinfo.value.details["detail"] == "boom"


def test_validator_failure_maps_to_invalid_json(monkeypatch, auth_file):
    monkeypatch.setattr(
        codex_oauth.requests,
        "post",
        lambda url, **kwargs: _FakeResponse(lines=_sse(COMPLETED_EVENTS)),
    )
    backend = CodexOAuthGenerationBackend(_config(auth_file))

    def validator(_text):
        raise ValueError("not json")

    with pytest.raises(GenerationError) as excinfo:
        backend.generate("问题", {}, response_validator=validator)
    assert excinfo.value.error_code is GenerationErrorCode.INVALID_JSON


def test_empty_response_is_structured(monkeypatch, auth_file):
    events = [{"type": "response.completed", "response": {"usage": {}}}]
    monkeypatch.setattr(
        codex_oauth.requests,
        "post",
        lambda url, **kwargs: _FakeResponse(lines=_sse(events)),
    )
    backend = CodexOAuthGenerationBackend(_config(auth_file))

    with pytest.raises(GenerationError) as excinfo:
        backend.generate("问题", {})
    assert excinfo.value.error_code is GenerationErrorCode.EMPTY_OUTPUT


# --- device login -----------------------------------------------------------


def test_device_login_persists_credential(monkeypatch, tmp_path):
    path = tmp_path / "auth.json"
    prompts = []

    monkeypatch.setattr(
        codex_oauth,
        "request_device_code",
        lambda: {
            "device_auth_id": "dev-1",
            "user_code": "ABCD-1234",
            "interval": 1,
            "verification_url": codex_oauth.DEVICE_VERIFICATION_URL,
        },
    )
    monkeypatch.setattr(
        codex_oauth,
        "poll_device_token",
        lambda *args, **kwargs: {
            "authorization_code": "code",
            "code_verifier": "verifier",
        },
    )
    monkeypatch.setattr(
        codex_oauth, "exchange_device_code", lambda payload: _credential()
    )

    credential = codex_oauth.device_login(str(path), on_prompt=prompts.append)

    assert prompts[0]["user_code"] == "ABCD-1234"
    assert credential["email"] == "tester@example.com"
    assert json.loads(path.read_text(encoding="utf-8"))["account_id"] == "account-123"


def test_device_poll_waits_while_pending(monkeypatch):
    responses = [
        _FakeResponse(status_code=403),
        _FakeResponse(status_code=404),
        _FakeResponse(status_code=200),
    ]
    responses[2].json = lambda: {"authorization_code": "c", "code_verifier": "v"}
    slept = []

    monkeypatch.setattr(
        codex_oauth.requests, "post", lambda url, **kwargs: responses.pop(0)
    )
    payload = codex_oauth.poll_device_token(
        "dev-1", "ABCD-1234", 1, sleep=slept.append
    )

    # 403 and 404 both mean "waiting on the user", not failure.
    assert slept == [1, 1]
    assert payload["authorization_code"] == "c"


# --- tool-capable path (Agent) ----------------------------------------------


TOOL_CALL_EVENTS = [
    # The argument deltas also stream past; the completed item is the source of
    # truth, so parsing must not depend on accumulating them.
    {"type": "response.function_call_arguments.delta", "delta": '{"tic'},
    {"type": "response.function_call_arguments.delta", "delta": 'ker":"NVDA"}'},
    {
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_realtime_quote",
            "arguments": '{"ticker":"NVDA"}',
        },
    },
    {
        "type": "response.completed",
        "response": {"usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}},
    },
]


def test_generate_with_tools_returns_function_calls(monkeypatch):
    monkeypatch.setattr(
        codex_oauth.requests,
        "post",
        lambda url, **kwargs: _FakeResponse(lines=_sse(TOOL_CALL_EVENTS)),
    )

    text, tool_calls, usage, diagnostics = codex_oauth.generate_with_tools(
        _credential(), [{"type": "message", "role": "user", "content": []}], tools=[{"type": "function", "name": "t"}]
    )

    # An answer-free turn is normal when the model decides to call a tool, so
    # unlike generate() this must not raise empty_output.
    assert text == ""
    assert len(tool_calls) == 1
    assert tool_calls[0]["call_id"] == "call_1"
    assert tool_calls[0]["arguments"] == '{"ticker":"NVDA"}'
    assert usage["total_tokens"] == 8
    assert diagnostics["tool_calls"] == 1


def test_generate_with_tools_sends_tools_and_full_conversation(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs.get("json") or {})
        return _FakeResponse(lines=_sse(TOOL_CALL_EVENTS))

    monkeypatch.setattr(codex_oauth.requests, "post", fake_post)
    items = [
        {"type": "message", "role": "user", "content": []},
        {"type": "function_call", "call_id": "c", "name": "t", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c", "output": "{}"},
    ]
    codex_oauth.generate_with_tools(
        _credential(), items, tools=[{"type": "function", "name": "t"}], instructions="sys"
    )

    assert captured["input"] == items
    assert captured["tool_choice"] == "auto"
    assert captured["tools"] == [{"type": "function", "name": "t"}]
    assert captured["instructions"] == "sys"
    assert captured["stream"] is True


def test_single_prompt_body_still_omits_tools():
    body = codex_oauth.build_request_body("hi", model="m", effort="medium")
    assert "tools" not in body
    assert "tool_choice" not in body
    assert body["input"][0]["content"] == [{"type": "input_text", "text": "hi"}]
