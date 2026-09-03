# -*- coding: utf-8 -*-
"""Codex OAuth device-login sessions for the Web settings page.

The device flow is inherently multi-request: the browser needs the user code
immediately, then polls while the user confirms in another tab. This service
owns that in-between state — one background thread per session polls OpenAI and
writes the credential on success.

Sessions live in this process only. That matches how DSA serves the settings
page (single uvicorn process) and keeps OAuth material out of any shared store;
a restart mid-login just means the user starts over.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.llm import codex_oauth

logger = logging.getLogger(__name__)

# Sessions are dropped this long after they finish or expire, so a browser that
# polls a little late still gets a real answer instead of "unknown session".
SESSION_RETENTION_SECONDS = 300

STATE_PENDING = "pending"
STATE_AUTHORIZED = "authorized"
STATE_FAILED = "failed"
STATE_CANCELLED = "cancelled"


@dataclass
class _LoginSession:
    session_id: str
    user_code: str
    verification_url: str
    interval: int
    device_auth_id: str
    started_at: float
    expires_at: float
    state: str = STATE_PENDING
    error_reason: str = ""
    error_detail: str = ""
    credential_summary: Dict[str, Any] = field(default_factory=dict)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    finished_at: Optional[float] = None


def _summarize(credential: Dict[str, Any]) -> Dict[str, Any]:
    """Expose only non-secret identity fields to the browser."""
    return {
        "email": credential.get("email", ""),
        "plan_type": credential.get("plan_type", ""),
        "expires_at": float(credential.get("expires_at") or 0),
        "last_refresh": credential.get("last_refresh", ""),
    }


class CodexOAuthLoginService:
    """Start and track Codex device-login sessions."""

    def __init__(self, auth_file_provider):
        # Callable so the service always resolves the currently configured path
        # rather than pinning whatever it was at construction time.
        self._auth_file_provider = auth_file_provider
        self._sessions: Dict[str, _LoginSession] = {}
        self._lock = threading.Lock()

    # --- credential status --------------------------------------------------

    def get_credential_status(self) -> Dict[str, Any]:
        """Report whether a usable credential is already stored."""
        auth_file = self._auth_file_provider()
        try:
            credential = codex_oauth.load_credential(auth_file)
        except codex_oauth.CodexOAuthError as exc:
            return {
                "authorized": False,
                "auth_file": auth_file,
                "reason": exc.reason,
                "message": exc.detail or exc.reason,
            }

        summary = _summarize(credential)
        expires_at = summary["expires_at"]
        return {
            "authorized": True,
            "auth_file": auth_file,
            "email": summary["email"],
            "plan_type": summary["plan_type"],
            "expires_at": expires_at,
            "expires_in_seconds": max(0, int(expires_at - time.time())),
            # An expired access token is not a problem: the backend refreshes it
            # on the next call, as long as a refresh token is present.
            "refreshable": bool(credential.get("refresh_token")),
        }

    # --- login sessions -----------------------------------------------------

    def start_login(self) -> Dict[str, Any]:
        """Begin a device-code login and return what the browser must display."""
        self._prune()
        device = codex_oauth.request_device_code()

        session = _LoginSession(
            session_id=uuid.uuid4().hex,
            user_code=device["user_code"],
            verification_url=device["verification_url"],
            interval=int(device["interval"]),
            device_auth_id=device["device_auth_id"],
            started_at=time.time(),
            expires_at=time.time() + codex_oauth.DEVICE_POLL_TIMEOUT_SECONDS,
        )
        with self._lock:
            self._sessions[session.session_id] = session

        thread = threading.Thread(
            target=self._poll_until_done,
            args=(session,),
            name=f"codex-oauth-login-{session.session_id[:8]}",
            daemon=True,
        )
        thread.start()

        return {
            "session_id": session.session_id,
            "user_code": session.user_code,
            "verification_url": session.verification_url,
            "interval_seconds": session.interval,
            "expires_at": session.expires_at,
            # The start response describes "a login has begun", not the live
            # state: the polling thread may already have raced ahead.
            "state": STATE_PENDING,
        }

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Return the current state of one login session."""
        self._prune()
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return {"state": "unknown", "session_id": session_id}

        payload: Dict[str, Any] = {
            "session_id": session.session_id,
            "state": session.state,
            "user_code": session.user_code,
            "verification_url": session.verification_url,
            "expires_at": session.expires_at,
        }
        if session.state == STATE_AUTHORIZED:
            payload.update(session.credential_summary)
        elif session.state == STATE_FAILED:
            payload["reason"] = session.error_reason
            payload["message"] = session.error_detail
        return payload

    def cancel_login(self, session_id: str) -> Dict[str, Any]:
        """Stop polling for a session the user abandoned."""
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return {"state": "unknown", "session_id": session_id}
        if session.state == STATE_PENDING:
            session.cancel_event.set()
            session.state = STATE_CANCELLED
            session.finished_at = time.time()
        return {"session_id": session_id, "state": session.state}

    # --- internals ----------------------------------------------------------

    def _poll_until_done(self, session: _LoginSession) -> None:
        auth_file = self._auth_file_provider()
        try:
            token_payload = codex_oauth.poll_device_token(
                session.device_auth_id,
                session.user_code,
                session.interval,
                timeout_seconds=max(
                    1, int(session.expires_at - time.time())
                ),
                sleep=self._interruptible_sleep(session),
            )
            if session.cancel_event.is_set():
                return
            credential = codex_oauth.exchange_device_code(token_payload)
            codex_oauth.save_credential(auth_file, credential)
        except codex_oauth.CodexOAuthError as exc:
            if session.cancel_event.is_set():
                return
            logger.warning("Codex OAuth device login failed: %s", exc.message)
            session.error_reason = exc.reason
            session.error_detail = exc.detail or exc.reason
            session.state = STATE_FAILED
            session.finished_at = time.time()
            return
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Codex OAuth device login crashed: %s", exc, exc_info=True)
            session.error_reason = "unknown_error"
            session.error_detail = str(exc)
            session.state = STATE_FAILED
            session.finished_at = time.time()
            return

        session.credential_summary = _summarize(credential)
        session.state = STATE_AUTHORIZED
        session.finished_at = time.time()
        logger.info(
            "Codex OAuth device login succeeded for %s",
            session.credential_summary.get("email", "unknown"),
        )

    @staticmethod
    def _interruptible_sleep(session: _LoginSession):
        """Sleep between polls but wake immediately when the user cancels."""

        def _sleep(seconds: float) -> None:
            if session.cancel_event.wait(timeout=seconds):
                # Abort the shared poll loop instead of returning immediately and
                # letting it hammer the upstream endpoint until the deadline.
                raise codex_oauth.CodexOAuthError("device_cancelled", "用户已取消授权")

        return _sleep

    def _prune(self) -> None:
        now = time.time()
        with self._lock:
            for session_id, session in list(self._sessions.items()):
                finished = session.finished_at is not None
                if finished and now - session.finished_at > SESSION_RETENTION_SECONDS:
                    del self._sessions[session_id]
                elif not finished and now > session.expires_at + SESSION_RETENTION_SECONDS:
                    del self._sessions[session_id]
