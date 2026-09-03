# -*- coding: utf-8 -*-
"""Codex OAuth credential handling and ChatGPT Responses transport.

This module talks to OpenAI's Codex device-code endpoints and to the ChatGPT
backend Responses API using a ChatGPT/Codex subscription. No metered
``OPENAI_API_KEY`` is involved: the only credential is an OAuth token bundle
obtained through the device flow and refreshed in place.

The protocol is OpenAI's private Codex CLI contract, so two details are load
bearing and easy to get wrong:

* every request must send an explicit ``User-Agent`` -- the Cloudflare edge in
  front of ``auth.openai.com`` answers ``530 cf_route_error`` for the default
  ``python-requests/x.y``;
* the PKCE ``code_verifier`` is issued by the server in the device-token poll
  response, not generated locally, so RFC 8628 style clients fail at exchange.
"""

from __future__ import annotations

import base64
import json
import os
import stat
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

try:  # pragma: no cover - Windows uses the in-process lock fallback.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

import requests


CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
DEVICE_USERCODE_URL = "https://auth.openai.com/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = "https://auth.openai.com/api/accounts/deviceauth/token"
DEVICE_VERIFICATION_URL = "https://auth.openai.com/codex/device"
DEVICE_REDIRECT_URI = "https://auth.openai.com/deviceauth/callback"
OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"

# Codex CLI identity. The ChatGPT backend keys off these; do not drift them
# casually -- a mismatch shows up as 400/403 rather than as a clear error.
ORIGINATOR = "codex-tui"
USER_AGENT = (
    "codex-tui/0.146.0 (Mac OS 26.5.0; arm64) iTerm.app/3.6.10 (codex-tui; 0.146.0)"
)

# Relative to the DSA working directory. ``data/`` is the mounted volume in the
# Docker deployment, so a login survives container rebuilds.
DEFAULT_AUTH_FILE = "data/codex_oauth/auth.json"
DEFAULT_MODEL = "gpt-5.6-terra"
SUPPORTED_EFFORTS = ("none", "low", "medium", "high")
DEFAULT_EFFORT = "medium"

DEVICE_POLL_TIMEOUT_SECONDS = 15 * 60
DEVICE_DEFAULT_POLL_INTERVAL = 5
# Refresh this far ahead of expiry so a long generation cannot straddle it.
REFRESH_SKEW_SECONDS = 120
AUTH_REQUEST_TIMEOUT_SECONDS = 60

_CREDENTIAL_LOCKS_GUARD = threading.Lock()
_CREDENTIAL_LOCKS: Dict[str, threading.Lock] = {}


@dataclass
class CodexOAuthError(Exception):
    """Structured Codex OAuth / transport failure.

    ``reason`` is a stable snake_case code the generation backend maps onto
    ``GenerationErrorCode``; ``detail`` carries a already-truncated upstream
    message safe to place in diagnostics.
    """

    reason: str
    detail: str = ""
    http_status: Optional[int] = None

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    @property
    def message(self) -> str:
        if self.http_status is not None:
            return f"{self.reason} (HTTP {self.http_status}): {self.detail}"
        return f"{self.reason}: {self.detail}" if self.detail else self.reason


def _auth_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    headers.update(extra or {})
    return headers


def _truncate(text: Any, limit: int = 300) -> str:
    return str(text or "").strip()[:limit]


def decode_jwt_claims(token: str) -> Dict[str, Any]:
    """Read a JWT payload without verifying it.

    Signature verification is OpenAI's job; these claims are only used for
    display and for the ``Chatgpt-Account-Id`` routing header.
    """
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return {}


# --- credential storage -----------------------------------------------------


def _credential_path(path: str) -> str:
    if not path:
        raise CodexOAuthError("login_required", "未配置 CODEX_OAUTH_AUTH_FILE")
    return os.path.abspath(os.path.expanduser(path))


@contextmanager
def _credential_lock(path: str) -> Iterator[None]:
    """Serialize token rotation across Agent/generation threads and processes."""
    expanded = _credential_path(path)
    directory = os.path.dirname(expanded) or "."
    os.makedirs(directory, mode=stat.S_IRWXU, exist_ok=True)

    with _CREDENTIAL_LOCKS_GUARD:
        thread_lock = _CREDENTIAL_LOCKS.setdefault(expanded, threading.Lock())

    lock_path = f"{expanded}.lock"
    with thread_lock:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, stat.S_IRUSR | stat.S_IWUSR)
            os.fchmod(lock_fd, stat.S_IRUSR | stat.S_IWUSR)
        except OSError as exc:
            raise CodexOAuthError("invalid_credential", f"无法创建凭证锁: {exc}") from exc
        try:
            if fcntl is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(lock_fd)


def load_credential(path: str) -> Dict[str, Any]:
    """Load the OAuth bundle, or raise ``login_required`` when absent."""
    expanded = _credential_path(path)
    if not os.path.exists(expanded):
        raise CodexOAuthError(
            "login_required",
            f"未找到 Codex OAuth 凭证（{expanded}），请先运行 scripts/codex_oauth_login.py",
        )
    try:
        with open(expanded, "r", encoding="utf-8") as handle:
            credential = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CodexOAuthError("invalid_credential", f"凭证文件无法解析: {exc}") from exc
    if not credential.get("access_token"):
        raise CodexOAuthError("invalid_credential", "凭证文件缺少 access_token")
    return credential


def _save_credential_unlocked(path: str, credential: Dict[str, Any]) -> str:
    expanded = _credential_path(path)
    directory = os.path.dirname(expanded) or "."
    os.makedirs(directory, mode=stat.S_IRWXU, exist_ok=True)

    fd = -1
    tmp_path = ""
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(expanded)}.",
            dir=directory,
        )
        os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            json.dump(credential, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, expanded)
    except OSError as exc:
        if fd >= 0:
            os.close(fd)
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
        raise CodexOAuthError("invalid_credential", f"凭证文件无法写入: {exc}") from exc
    return expanded


def save_credential(path: str, credential: Dict[str, Any]) -> str:
    """Persist the OAuth bundle atomically with owner-only permissions."""
    with _credential_lock(path):
        return _save_credential_unlocked(path, credential)


def build_credential(
    token_response: Dict[str, Any],
    previous: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize an OAuth token response into the stored credential shape."""
    previous = previous or {}
    access_token = token_response.get("access_token", "")
    access_claims = decode_jwt_claims(access_token)
    id_token = token_response.get("id_token") or previous.get("id_token", "")
    id_claims = decode_jwt_claims(id_token) if id_token else {}
    auth_info = id_claims.get("https://api.openai.com/auth", {})

    # The access token's own ``exp`` is authoritative; expires_in is a fallback.
    expires_at = access_claims.get("exp")
    if not expires_at:
        expires_at = time.time() + float(token_response.get("expires_in") or 3600)

    return {
        "type": "codex",
        "access_token": access_token,
        # Refresh responses do not always re-issue a refresh token.
        "refresh_token": token_response.get("refresh_token")
        or previous.get("refresh_token", ""),
        "id_token": id_token,
        "account_id": auth_info.get("chatgpt_account_id") or previous.get("account_id", ""),
        "plan_type": auth_info.get("chatgpt_plan_type") or previous.get("plan_type", ""),
        "email": id_claims.get("email") or previous.get("email", ""),
        "expires_at": float(expires_at),
        "last_refresh": datetime.now(timezone.utc).isoformat(),
    }


# --- device-code login ------------------------------------------------------


def request_device_code() -> Dict[str, Any]:
    """Start the device flow and return device_auth_id / user_code / interval."""
    try:
        response = requests.post(
            DEVICE_USERCODE_URL,
            json={"client_id": CLIENT_ID},
            headers=_auth_headers({"Content-Type": "application/json"}),
            timeout=AUTH_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise CodexOAuthError("network_error", f"申请设备码失败: {exc}") from exc

    if response.status_code != 200:
        raise CodexOAuthError(
            "device_code_failed", _truncate(response.text), response.status_code
        )

    payload = response.json()
    user_code = (payload.get("user_code") or payload.get("usercode") or "").strip()
    device_auth_id = (payload.get("device_auth_id") or "").strip()
    if not user_code or not device_auth_id:
        raise CodexOAuthError("device_code_failed", "设备码响应缺少必要字段")

    try:
        interval = max(1, int(str(payload.get("interval") or DEVICE_DEFAULT_POLL_INTERVAL)))
    except (TypeError, ValueError):
        interval = DEVICE_DEFAULT_POLL_INTERVAL

    return {
        "device_auth_id": device_auth_id,
        "user_code": user_code,
        "interval": interval,
        "verification_url": DEVICE_VERIFICATION_URL,
    }


def poll_device_token(
    device_auth_id: str,
    user_code: str,
    interval: int,
    *,
    timeout_seconds: int = DEVICE_POLL_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    """Poll until the user approves in the browser, then return the auth code."""
    deadline = time.time() + timeout_seconds
    body = {"device_auth_id": device_auth_id, "user_code": user_code}

    while True:
        if time.time() > deadline:
            raise CodexOAuthError("device_timeout", "等待授权超时（15 分钟）")

        try:
            response = requests.post(
                DEVICE_TOKEN_URL,
                json=body,
                headers=_auth_headers({"Content-Type": "application/json"}),
                timeout=AUTH_REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise CodexOAuthError("network_error", f"轮询授权状态失败: {exc}") from exc

        if 200 <= response.status_code < 300:
            return response.json()
        # 403/404 both mean "the user has not confirmed yet".
        if response.status_code in (403, 404):
            sleep(interval)
            continue
        raise CodexOAuthError(
            "device_poll_failed", _truncate(response.text), response.status_code
        )


def exchange_device_code(token_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Exchange the device authorization code for an OAuth token bundle."""
    auth_code = (token_payload.get("authorization_code") or "").strip()
    code_verifier = (token_payload.get("code_verifier") or "").strip()
    if not auth_code or not code_verifier:
        raise CodexOAuthError("device_exchange_failed", "设备授权响应缺少必要字段")

    try:
        response = requests.post(
            OAUTH_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": auth_code,
                "redirect_uri": DEVICE_REDIRECT_URI,
                "code_verifier": code_verifier,
            },
            headers=_auth_headers(
                {"Content-Type": "application/x-www-form-urlencoded"}
            ),
            timeout=AUTH_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise CodexOAuthError("network_error", f"换取 token 失败: {exc}") from exc

    if response.status_code != 200:
        raise CodexOAuthError(
            "device_exchange_failed", _truncate(response.text), response.status_code
        )
    return build_credential(response.json())


def device_login(
    path: str,
    *,
    on_prompt: Optional[Callable[[Dict[str, Any]], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    """Run the whole device flow and persist the resulting credential.

    ``on_prompt`` receives the device-code payload so callers can render the
    verification URL and user code however they like.
    """
    device = request_device_code()
    if on_prompt is not None:
        on_prompt(device)

    token_payload = poll_device_token(
        device["device_auth_id"],
        device["user_code"],
        device["interval"],
        sleep=sleep,
    )
    credential = exchange_device_code(token_payload)
    if not credential.get("access_token"):
        raise CodexOAuthError("device_exchange_failed", "token 响应缺少 access_token")
    save_credential(path, credential)
    return credential


# --- refresh ----------------------------------------------------------------


def _credential_is_fresh(credential: Dict[str, Any]) -> bool:
    try:
        expires_at = float(credential.get("expires_at") or 0)
    except (TypeError, ValueError):
        return False
    return expires_at - time.time() > REFRESH_SKEW_SECONDS


def _latest_credential(path: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    expanded = _credential_path(path)
    return load_credential(expanded) if os.path.exists(expanded) else fallback


def _refresh_credential_unlocked(
    credential: Dict[str, Any], path: str
) -> Dict[str, Any]:
    refresh_token = (credential.get("refresh_token") or "").strip()
    if not refresh_token:
        raise CodexOAuthError("login_required", "凭证缺少 refresh_token，需要重新登录")

    try:
        response = requests.post(
            OAUTH_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": refresh_token,
                "scope": "openid profile email",
            },
            headers=_auth_headers(
                {"Content-Type": "application/x-www-form-urlencoded"}
            ),
            timeout=AUTH_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise CodexOAuthError("network_error", f"刷新 token 失败: {exc}") from exc

    if response.status_code != 200:
        raise CodexOAuthError(
            "refresh_failed", _truncate(response.text), response.status_code
        )
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise CodexOAuthError("refresh_failed", "刷新响应不是有效 JSON") from exc
    if not isinstance(payload, dict) or not payload.get("access_token"):
        raise CodexOAuthError("refresh_failed", "刷新响应缺少 access_token")

    refreshed = build_credential(payload, previous=credential)
    _save_credential_unlocked(path, refreshed)
    return refreshed


def refresh_credential(credential: Dict[str, Any], path: str) -> Dict[str, Any]:
    """Rotate and persist a token once across concurrent workers."""
    with _credential_lock(path):
        latest = _latest_credential(path, credential)
        # Another worker may have refreshed while this request was waiting.
        if (
            latest.get("access_token") != credential.get("access_token")
            and _credential_is_fresh(latest)
        ):
            return latest
        return _refresh_credential_unlocked(latest, path)


def ensure_fresh_credential(credential: Dict[str, Any], path: str) -> Dict[str, Any]:
    """Refresh an expiring credential once across all local workers."""
    if _credential_is_fresh(credential):
        return credential
    with _credential_lock(path):
        latest = _latest_credential(path, credential)
        if _credential_is_fresh(latest):
            return latest
        return _refresh_credential_unlocked(latest, path)


# --- generation transport ---------------------------------------------------


def normalize_effort(value: Any) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in SUPPORTED_EFFORTS else DEFAULT_EFFORT


def build_conversation_request_body(
    input_items: List[Dict[str, Any]],
    *,
    model: str,
    effort: str,
    instructions: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the Responses payload the ChatGPT Codex backend accepts.

    ``input_items`` is the full conversation in Responses item form, so the
    same builder serves a one-shot prompt and a multi-turn agent exchange that
    replays ``function_call`` / ``function_call_output`` items.
    """
    body: Dict[str, Any] = {
        "model": model,
        # ``instructions`` must be present; the backend rejects a null field.
        "instructions": instructions or "",
        "input": list(input_items),
        # This endpoint only serves streamed responses; non-stream is rejected.
        "stream": True,
        "store": False,
    }
    if effort != "none":
        body["reasoning"] = {"effort": effort, "summary": "auto"}
    if tools:
        body["tools"] = list(tools)
        body["tool_choice"] = "auto"
    return body


def build_request_body(
    prompt: str,
    *,
    model: str,
    effort: str,
    instructions: str = "",
) -> Dict[str, Any]:
    """Build the payload for a single-prompt generation."""
    return build_conversation_request_body(
        [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        model=model,
        effort=effort,
        instructions=instructions,
    )


def _normalize_usage(raw_usage: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Map Responses-style token counts onto DSA's canonical usage keys."""
    prompt_tokens = int(raw_usage.get("input_tokens") or 0)
    completion_tokens = int(raw_usage.get("output_tokens") or 0)
    total_tokens = int(raw_usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    reasoning_tokens = int(
        (raw_usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0
    )
    cached_tokens = int(
        (raw_usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": cached_tokens,
        "model": model,
        "provider": "codex_oauth",
        "usage_available": True,
        "usage_source": "provider",
    }


def _raise_for_status(status_code: int, detail: str) -> None:
    if status_code == 401:
        raise CodexOAuthError("unauthorized", detail, status_code)
    if status_code == 403:
        raise CodexOAuthError("forbidden", detail, status_code)
    if status_code == 429:
        raise CodexOAuthError("rate_limited", detail, status_code)
    if status_code == 400:
        raise CodexOAuthError("bad_request", detail, status_code)
    raise CodexOAuthError("upstream_error", detail, status_code)


def _build_diagnostics(
    credential: Dict[str, Any], *, model: str, effort: str
) -> Dict[str, Any]:
    return {
        "backend": "codex_oauth",
        "model": model,
        "reasoning_effort": effort,
        "endpoint": RESPONSES_URL,
        "plan_type": credential.get("plan_type", ""),
        "stream_consumed_internally": True,
    }


def _post_stream(
    credential: Dict[str, Any],
    body: Dict[str, Any],
    *,
    timeout_seconds: int,
    session_id: Optional[str] = None,
) -> Any:
    """POST one Responses request and hand back the streaming response."""
    headers = {
        "Authorization": "Bearer " + credential.get("access_token", ""),
        "Chatgpt-Account-Id": credential.get("account_id", ""),
        "Originator": ORIGINATOR,
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if session_id:
        headers["Session-Id"] = session_id
    try:
        return requests.post(
            RESPONSES_URL,
            json=body,
            headers=headers,
            stream=True,
            timeout=timeout_seconds,
        )
    except requests.Timeout as exc:
        raise CodexOAuthError("timeout", f"请求超时（{timeout_seconds}s）") from exc
    except requests.RequestException as exc:
        raise CodexOAuthError("network_error", f"调用失败: {exc}") from exc


def generate_with_tools(
    credential: Dict[str, Any],
    input_items: List[Dict[str, Any]],
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    instructions: str = "",
    timeout_seconds: int = 300,
    session_id: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Run one Agent turn and return ``(text, tool_calls, usage, diagnostics)``.

    Unlike :func:`generate` this takes the whole conversation as Responses
    items and may come back with ``function_call`` items instead of text; an
    empty answer is a valid turn there, so no ``empty_output`` is raised.
    """
    body = build_conversation_request_body(
        input_items, model=model, effort=effort, instructions=instructions, tools=tools
    )
    diagnostics = _build_diagnostics(credential, model=model, effort=effort)
    diagnostics["tool_count"] = len(tools or [])
    response = _post_stream(
        credential, body, timeout_seconds=timeout_seconds, session_id=session_id
    )
    text, tool_calls, raw_usage, output_bytes = _consume_stream(
        response,
        max_output_bytes=None,
        progress_callback=None,
        diagnostics=diagnostics,
    )
    diagnostics["output_bytes"] = output_bytes
    diagnostics["tool_calls"] = len(tool_calls)
    return text, tool_calls, _normalize_usage(raw_usage, model), diagnostics


def generate(
    credential: Dict[str, Any],
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    instructions: str = "",
    timeout_seconds: int = 300,
    max_output_bytes: Optional[int] = None,
    session_id: Optional[str] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Run one generation and return ``(text, usage, diagnostics)``.

    The stream is consumed internally and accumulated: DSA's analysis path
    wants one complete response, not incremental tokens.
    """
    body = build_request_body(prompt, model=model, effort=effort, instructions=instructions)
    diagnostics = _build_diagnostics(credential, model=model, effort=effort)
    response = _post_stream(
        credential, body, timeout_seconds=timeout_seconds, session_id=session_id
    )

    text, tool_calls, raw_usage, output_bytes = _consume_stream(
        response,
        max_output_bytes=max_output_bytes,
        progress_callback=progress_callback,
        diagnostics=diagnostics,
    )
    del tool_calls  # the single-prompt path never declares tools

    diagnostics["output_bytes"] = output_bytes
    if not text:
        raise CodexOAuthError("empty_output", "模型返回空内容")

    return text, _normalize_usage(raw_usage, model), diagnostics


def _consume_stream(
    response: Any,
    *,
    max_output_bytes: Optional[int],
    progress_callback: Optional[Callable[[int], None]],
    diagnostics: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any], int]:
    """Drain one SSE response into ``(text, tool_calls, raw_usage, output_bytes)``.

    Shared by the single-prompt generation path and the Agent tool path so both
    read the same events and raise the same structured errors.
    """
    with response:
        if response.status_code < 200 or response.status_code >= 300:
            _raise_for_status(response.status_code, _truncate(response.text, 400))

        chunks: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        raw_usage: Dict[str, Any] = {}
        output_bytes = 0

        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            # requests only decodes when the response declares a charset, and
            # this endpoint does not, so the lines arrive as bytes.
            line = (
                raw_line.decode("utf-8", errors="replace")
                if isinstance(raw_line, bytes)
                else raw_line
            )
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")
            if event_type == "response.output_text.delta":
                delta = event.get("delta", "")
                output_bytes += len(delta.encode("utf-8"))
                if max_output_bytes is not None and output_bytes > max_output_bytes:
                    diagnostics["output_bytes"] = output_bytes
                    raise CodexOAuthError(
                        "output_too_large",
                        f"输出超过上限 {max_output_bytes} 字节",
                    )
                chunks.append(delta)
                if progress_callback is not None:
                    progress_callback(1)
            elif event_type == "response.output_item.done":
                # The completed item carries the whole call (call_id, name and
                # fully assembled arguments), so the argument deltas that also
                # stream past can be ignored.
                item = event.get("item") or {}
                if item.get("type") == "function_call":
                    tool_calls.append(item)
            elif event_type == "response.completed":
                raw_usage = (event.get("response") or {}).get("usage") or {}
            elif event_type in ("response.failed", "error"):
                message = (
                    (event.get("response") or {}).get("error", {}).get("message")
                    or event.get("message")
                    or _truncate(json.dumps(event, ensure_ascii=False))
                )
                raise CodexOAuthError("upstream_error", _truncate(message))

    return "".join(chunks).strip(), tool_calls, raw_usage, output_bytes
