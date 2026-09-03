# -*- coding: utf-8 -*-
"""Codex OAuth generation backend.

Generation-only backend backed by a ChatGPT/Codex subscription over OAuth.
Unlike the local CLI backends it needs no executable on the host: DSA holds the
OAuth credential itself and speaks the ChatGPT Responses protocol directly, so
it reports real token usage and picks its own model.

Tool calling is not implemented, so this stays out of ``AGENT_CAPABLE_BACKEND_IDS``
and the Agent path keeps running on LiteLLM.
"""

from __future__ import annotations

import threading
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

from src.llm import codex_oauth
from src.llm.backend_registry import CODEX_OAUTH_BACKEND_ID
from src.llm.generation_backend import (
    GenerationBackend,
    GenerationCapabilities,
    GenerationError,
    GenerationErrorCode,
    GenerationResult,
)

DEFAULT_TIMEOUT_SECONDS = 300
MAX_TIMEOUT_SECONDS = 3600
DEFAULT_MAX_OUTPUT_BYTES = 1048576
MAX_OUTPUT_BYTES = 33554432
DEFAULT_MAX_CONCURRENCY = 1
MAX_CONCURRENCY = 16


class _GlobalGenerationGate:
    """Process-wide gate shared by every Codex OAuth backend instance."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active = 0

    @contextmanager
    def slot(self, limit: int) -> Iterator[None]:
        with self._condition:
            while self._active >= limit:
                self._condition.wait()
            self._active += 1
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()


_GENERATION_GATE = _GlobalGenerationGate()

# Reason codes from src.llm.codex_oauth mapped onto DSA's structured error
# contract. ``retryable`` / ``fallbackable`` follow the local CLI backend's
# conventions: configuration problems are terminal, transport hiccups are not.
_ERROR_MAPPING: Dict[str, tuple] = {
    "login_required": (GenerationErrorCode.LOGIN_REQUIRED, "configuration", False, True),
    "invalid_credential": (GenerationErrorCode.LOGIN_REQUIRED, "configuration", False, True),
    "unauthorized": (GenerationErrorCode.LOGIN_REQUIRED, "execution", False, True),
    "forbidden": (GenerationErrorCode.CAPABILITY_UNSUPPORTED, "execution", False, True),
    "refresh_failed": (GenerationErrorCode.LOGIN_REQUIRED, "configuration", False, True),
    "device_timeout": (GenerationErrorCode.LOGIN_REQUIRED, "configuration", False, True),
    "rate_limited": (GenerationErrorCode.NON_ZERO_EXIT, "execution", True, True),
    "timeout": (GenerationErrorCode.TIMEOUT, "execution", True, True),
    "network_error": (GenerationErrorCode.UNKNOWN_BACKEND_ERROR, "execution", True, True),
    "output_too_large": (GenerationErrorCode.OUTPUT_TOO_LARGE, "execution", False, True),
    "empty_output": (GenerationErrorCode.EMPTY_OUTPUT, "execution", True, True),
    "bad_request": (GenerationErrorCode.CAPABILITY_UNSUPPORTED, "execution", False, True),
    "upstream_error": (GenerationErrorCode.UNKNOWN_BACKEND_ERROR, "execution", True, True),
}


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


class CodexOAuthGenerationBackend(GenerationBackend):
    """Subscription-backed generation backend using Codex OAuth credentials."""

    backend_id = CODEX_OAUTH_BACKEND_ID
    capabilities = GenerationCapabilities(
        supports_json=True,
        supports_tools=False,
        # The upstream endpoint is stream-only, but DSA consumes the stream
        # internally and returns one complete response, so no streaming
        # capability is exposed to callers.
        supports_stream=False,
        supports_vision=False,
        supports_health_check=False,
        supports_smoke_test=False,
    )

    def __init__(self, config: Any) -> None:
        self._config = config

    # --- configuration ------------------------------------------------------

    @property
    def _auth_file(self) -> str:
        return str(getattr(self._config, "codex_oauth_auth_file", "") or "")

    @property
    def _model(self) -> str:
        return str(
            getattr(self._config, "codex_oauth_model", "") or codex_oauth.DEFAULT_MODEL
        )

    @property
    def _effort(self) -> str:
        return codex_oauth.normalize_effort(
            getattr(self._config, "codex_oauth_reasoning_effort", None)
        )

    @property
    def _timeout_seconds(self) -> int:
        return min(
            _positive_int(
                getattr(self._config, "generation_backend_timeout_seconds", None),
                DEFAULT_TIMEOUT_SECONDS,
            ),
            MAX_TIMEOUT_SECONDS,
        )

    @property
    def _max_output_bytes(self) -> int:
        return min(
            _positive_int(
                getattr(self._config, "generation_backend_max_output_bytes", None),
                DEFAULT_MAX_OUTPUT_BYTES,
            ),
            MAX_OUTPUT_BYTES,
        )

    @property
    def _max_concurrency(self) -> int:
        return min(
            _positive_int(
                getattr(self._config, "generation_backend_max_concurrency", None),
                DEFAULT_MAX_CONCURRENCY,
            ),
            MAX_CONCURRENCY,
        )

    def get_config_error(self) -> Optional[GenerationError]:
        """Validate the credential without spending a real generation call."""
        try:
            codex_oauth.load_credential(self._auth_file)
        except codex_oauth.CodexOAuthError as exc:
            return self._map_error(exc, {"auth_file": self._auth_file})
        return None

    # --- generation ---------------------------------------------------------

    def generate(
        self,
        prompt: str,
        generation_config: Dict[str, Any],
        *,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        stream_progress_callback: Optional[Callable[[int], None]] = None,
        response_validator: Optional[Callable[[str], None]] = None,
        audit_context: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        with _GENERATION_GATE.slot(self._max_concurrency):
            return self._generate_once(
                prompt,
                generation_config,
                system_prompt=system_prompt,
                stream=stream,
                stream_progress_callback=stream_progress_callback,
                response_validator=response_validator,
                audit_context=audit_context,
            )

    def _generate_once(
        self,
        prompt: str,
        generation_config: Dict[str, Any],
        *,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        stream_progress_callback: Optional[Callable[[int], None]] = None,
        response_validator: Optional[Callable[[str], None]] = None,
        audit_context: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        auth_file = self._auth_file
        # Per-request overrides win over the configured defaults so callers can
        # pin a model for one analysis without touching .env.
        model = str((generation_config or {}).get("model") or self._model)
        effort = codex_oauth.normalize_effort(
            (generation_config or {}).get("reasoning_effort") or self._effort
        )

        diagnostics: Dict[str, Any] = {
            "backend": self.backend_id,
            "auth_file": auth_file,
            "model": model,
            "reasoning_effort": effort,
            "stream_degraded": bool(stream),
            "timeout_seconds": self._timeout_seconds,
            "max_output_bytes": self._max_output_bytes,
            "max_concurrency": self._max_concurrency,
        }

        self._emit_progress(stream_progress_callback, 0)

        try:
            credential = codex_oauth.ensure_fresh_credential(
                codex_oauth.load_credential(auth_file), auth_file
            )
        except codex_oauth.CodexOAuthError as exc:
            raise self._map_error(exc, diagnostics) from exc

        diagnostics["account_email"] = credential.get("email", "")
        diagnostics["plan_type"] = credential.get("plan_type", "")
        self._emit_progress(stream_progress_callback, 1)

        try:
            text, usage, call_diagnostics = codex_oauth.generate(
                credential,
                prompt,
                model=model,
                effort=effort,
                instructions=(system_prompt or "").strip(),
                timeout_seconds=self._timeout_seconds,
                max_output_bytes=self._max_output_bytes,
                session_id=str(uuid.uuid4()),
            )
        except codex_oauth.CodexOAuthError as exc:
            # A 401 here means the token died mid-flight despite the pre-check;
            # refresh once and retry before giving up.
            if exc.reason != "unauthorized":
                raise self._map_error(exc, diagnostics) from exc
            try:
                credential = codex_oauth.refresh_credential(credential, auth_file)
                text, usage, call_diagnostics = codex_oauth.generate(
                    credential,
                    prompt,
                    model=model,
                    effort=effort,
                    instructions=(system_prompt or "").strip(),
                    timeout_seconds=self._timeout_seconds,
                    max_output_bytes=self._max_output_bytes,
                    session_id=str(uuid.uuid4()),
                )
                diagnostics["refreshed_mid_request"] = True
            except codex_oauth.CodexOAuthError as retry_exc:
                raise self._map_error(retry_exc, diagnostics) from retry_exc

        diagnostics.update(call_diagnostics)
        self._emit_progress(stream_progress_callback, 2)

        if response_validator is not None:
            try:
                response_validator(text)
            except GenerationError:
                raise
            except Exception as exc:
                raise GenerationError(
                    error_code=GenerationErrorCode.INVALID_JSON,
                    stage="validation",
                    retryable=True,
                    fallbackable=True,
                    backend=self.backend_id,
                    provider=self.backend_id,
                    details={**diagnostics, "reason": str(exc) or "invalid_json"},
                ) from exc

        return GenerationResult(
            text=text,
            model=model,
            provider=self.backend_id,
            backend=self.backend_id,
            usage=usage,
            raw=None,
            diagnostics=diagnostics,
        )

    # --- helpers ------------------------------------------------------------

    def _map_error(
        self,
        exc: codex_oauth.CodexOAuthError,
        diagnostics: Dict[str, Any],
    ) -> GenerationError:
        error_code, stage, retryable, fallbackable = _ERROR_MAPPING.get(
            exc.reason,
            (GenerationErrorCode.UNKNOWN_BACKEND_ERROR, "execution", True, True),
        )
        return GenerationError(
            error_code=error_code,
            stage=stage,
            retryable=retryable,
            fallbackable=fallbackable,
            backend=self.backend_id,
            provider=self.backend_id,
            details={
                **diagnostics,
                "reason": exc.reason,
                "detail": exc.detail,
                "http_status": exc.http_status,
            },
        )

    @staticmethod
    def _emit_progress(callback: Optional[Callable[[int], None]], value: int) -> None:
        if callback is None:
            return
        try:
            callback(value)
        except Exception:
            # Progress reporting must never break a generation.
            pass
