# -*- coding: utf-8 -*-
"""Tests for the Codex OAuth device-login session service."""

from __future__ import annotations

import json
import time

import pytest

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from src.llm import codex_oauth  # noqa: E402
from src.services.codex_oauth_login_service import (  # noqa: E402
    STATE_AUTHORIZED,
    STATE_CANCELLED,
    STATE_FAILED,
    STATE_PENDING,
    CodexOAuthLoginService,
)


def _credential(**overrides):
    credential = {
        "type": "codex",
        "access_token": "header.payload.signature",
        "refresh_token": "rt.test",
        "account_id": "account-123",
        "plan_type": "prolite",
        "email": "tester@example.com",
        "expires_at": time.time() + 86400,
        "last_refresh": "2026-08-27T00:00:00+00:00",
    }
    credential.update(overrides)
    return credential


def _service(tmp_path):
    return CodexOAuthLoginService(auth_file_provider=lambda: str(tmp_path / "auth.json"))


def _wait_for(predicate, timeout=5.0):
    """Wait for the background polling thread to reach a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _stub_device_code(monkeypatch, interval=1):
    monkeypatch.setattr(
        codex_oauth,
        "request_device_code",
        lambda: {
            "device_auth_id": "dev-1",
            "user_code": "ABCD-1234",
            "interval": interval,
            "verification_url": codex_oauth.DEVICE_VERIFICATION_URL,
        },
    )


# --- credential status ------------------------------------------------------


def test_status_reports_unauthorized_when_missing(tmp_path):
    status = _service(tmp_path).get_credential_status()
    assert status["authorized"] is False
    assert status["reason"] == "login_required"


def test_status_reports_account_without_leaking_tokens(tmp_path):
    (tmp_path / "auth.json").write_text(json.dumps(_credential()), encoding="utf-8")
    status = _service(tmp_path).get_credential_status()

    assert status["authorized"] is True
    assert status["email"] == "tester@example.com"
    assert status["plan_type"] == "prolite"
    assert status["refreshable"] is True
    # Token material must never reach the browser.
    assert "access_token" not in status
    assert "refresh_token" not in status
    assert "id_token" not in status


def test_status_expired_token_is_still_authorized_when_refreshable(tmp_path):
    (tmp_path / "auth.json").write_text(
        json.dumps(_credential(expires_at=time.time() - 10)), encoding="utf-8"
    )
    status = _service(tmp_path).get_credential_status()

    # An expired access token is fine: the backend refreshes it on next use.
    assert status["authorized"] is True
    assert status["expires_in_seconds"] == 0
    assert status["refreshable"] is True


# --- login sessions ---------------------------------------------------------


def test_start_login_returns_device_code(monkeypatch, tmp_path):
    _stub_device_code(monkeypatch)
    monkeypatch.setattr(
        codex_oauth, "poll_device_token", lambda *a, **k: {"authorization_code": "c", "code_verifier": "v"}
    )
    monkeypatch.setattr(codex_oauth, "exchange_device_code", lambda payload: _credential())

    service = _service(tmp_path)
    started = service.start_login()

    assert started["user_code"] == "ABCD-1234"
    assert started["verification_url"] == codex_oauth.DEVICE_VERIFICATION_URL
    assert started["state"] == STATE_PENDING
    assert started["session_id"]


def test_successful_login_persists_credential(monkeypatch, tmp_path):
    _stub_device_code(monkeypatch)
    monkeypatch.setattr(
        codex_oauth, "poll_device_token", lambda *a, **k: {"authorization_code": "c", "code_verifier": "v"}
    )
    monkeypatch.setattr(codex_oauth, "exchange_device_code", lambda payload: _credential())

    service = _service(tmp_path)
    session_id = service.start_login()["session_id"]

    assert _wait_for(lambda: service.get_session(session_id)["state"] == STATE_AUTHORIZED)
    session = service.get_session(session_id)
    assert session["email"] == "tester@example.com"
    assert session["plan_type"] == "prolite"

    stored = json.loads((tmp_path / "auth.json").read_text(encoding="utf-8"))
    assert stored["access_token"] == "header.payload.signature"
    assert service.get_credential_status()["authorized"] is True


def test_failed_login_reports_structured_reason(monkeypatch, tmp_path):
    _stub_device_code(monkeypatch)

    def boom(*args, **kwargs):
        raise codex_oauth.CodexOAuthError("device_timeout", "等待授权超时（15 分钟）")

    monkeypatch.setattr(codex_oauth, "poll_device_token", boom)

    service = _service(tmp_path)
    session_id = service.start_login()["session_id"]

    assert _wait_for(lambda: service.get_session(session_id)["state"] == STATE_FAILED)
    session = service.get_session(session_id)
    assert session["reason"] == "device_timeout"
    assert "超时" in session["message"]
    # A failed login must not leave a half-written credential behind.
    assert not (tmp_path / "auth.json").exists()


def test_start_login_propagates_upstream_failure(monkeypatch, tmp_path):
    def boom():
        raise codex_oauth.CodexOAuthError("device_code_failed", "cf_route_error", 530)

    monkeypatch.setattr(codex_oauth, "request_device_code", boom)

    with pytest.raises(codex_oauth.CodexOAuthError) as excinfo:
        _service(tmp_path).start_login()
    assert excinfo.value.reason == "device_code_failed"


def test_cancel_stops_pending_session(monkeypatch, tmp_path):
    _stub_device_code(monkeypatch, interval=60)

    def never_returns(device_auth_id, user_code, interval, timeout_seconds=None, sleep=None):
        # Mirrors the real poll loop: sleeps between attempts, wakes on cancel.
        sleep(interval)
        raise codex_oauth.CodexOAuthError("device_timeout", "timed out")

    monkeypatch.setattr(codex_oauth, "poll_device_token", never_returns)

    service = _service(tmp_path)
    session_id = service.start_login()["session_id"]
    result = service.cancel_login(session_id)

    assert result["state"] == STATE_CANCELLED
    # Cancelling must not later flip the session into failed.
    time.sleep(0.2)
    assert service.get_session(session_id)["state"] == STATE_CANCELLED


def test_unknown_session_is_reported(tmp_path):
    assert _service(tmp_path).get_session("nope")["state"] == "unknown"
    assert _service(tmp_path).cancel_login("nope")["state"] == "unknown"


def test_auth_file_provider_is_read_per_call(tmp_path):
    """The configured path can change at runtime; the service must not pin it."""
    current = {"path": str(tmp_path / "first.json")}
    service = CodexOAuthLoginService(auth_file_provider=lambda: current["path"])

    assert service.get_credential_status()["auth_file"].endswith("first.json")
    current["path"] = str(tmp_path / "second.json")
    assert service.get_credential_status()["auth_file"].endswith("second.json")
