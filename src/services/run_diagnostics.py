# -*- coding: utf-8 -*-
"""Lightweight run diagnostic context for one analysis trace.

This module intentionally keeps Phase 1 diagnostics in memory and fail-open.
Persistence can reuse existing analysis context snapshots until a dedicated
diagnostic store is introduced.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_CURRENT_CONTEXT: ContextVar[Optional["RunDiagnosticContext"]] = ContextVar(
    "run_diagnostic_context",
    default=None,
)

_SECRET_REDACTIONS = (
    (
        re.compile(r"(?i)\b(authorization)\s*[:=]\s*(?:(?:Bearer|Basic|Token)\s+)?[^\s,&;]+"),
        lambda match: f"{match.group(1)}=<redacted>",
    ),
    (
        re.compile(r"(https?://)([^/\s:@]+):([^@\s/]+)@"),
        r"\1<redacted>:<redacted>@",
    ),
    (
        re.compile(r"https?://[^\s]+?(?:token|key|secret|webhook)[^\s]*", re.IGNORECASE),
        "<redacted-url>",
    ),
    (
        re.compile(
            r"(?i)([\"']?)"
            r"([A-Z0-9_]*?(?:api[_-]?key|access[_-]?token|token|secret|password|passwd|cookie))"
            r"\1\s*:\s*([\"'])([^\"']+)\3"
        ),
        lambda match: f"{match.group(1)}{match.group(2)}{match.group(1)}: {match.group(3)}<redacted>{match.group(3)}",
    ),
    (
        re.compile(
            r"(?i)\b([A-Z0-9_]*?(?:api[_-]?key|access[_-]?token|token|secret|password|passwd|cookie))"
            r"\s*=\s*([^\s,&;]+)"
        ),
        lambda match: f"{match.group(1)}=<redacted>",
    ),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|access[_-]?token|token|secret|password|passwd|cookie)"
            r"\s*:\s*([^\s,&;]+)"
        ),
        lambda match: f"{match.group(1)}=<redacted>",
    ),
    (
        re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
        "Bearer <redacted>",
    ),
)
_SENSITIVE_KEY_RE = re.compile(
    r"(?i)(authorization|api[_-]?key|access[_-]?token|(?:^|[_-])(?:auth|refresh|session|bearer)?[_-]?token$|secret|password|passwd|cookie|"
    r"webhook|sendkey|prompt|raw[_-]?prompt|raw[_-]?response|headers?|proxy)"
)
_WEBHOOK_URL_RE = re.compile(r"https?://[^\s]+?(?:webhook|token|key|secret|sendkey)[^\s]*", re.IGNORECASE)
_LOCAL_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\w:/.-])(?:/(?:home|Users|root|var|tmp|opt|etc)/[^\s,;]+|[A-Za-z]:\\[^\s,;]+)"
)
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|token|secret|password|passwd|cookie|webhook|sendkey|"
    r"prompt|raw[_-]?prompt|raw[_-]?response)\s*[:=]\s*([^\s,&;]+)"
)


def build_trace_id() -> str:
    """Build a compact trace id suitable for logs, API responses, and SSE."""
    return uuid.uuid4().hex


def sanitize_diagnostic_text(value: Any, *, max_length: int = 300) -> Optional[str]:
    """Return a short diagnostic string with sensitive details redacted."""
    if value is None:
        return None

    text = " ".join(str(value).split())
    if not text:
        return None

    for pattern, replacement in _SECRET_REDACTIONS:
        text = pattern.sub(replacement, text)
    text = _WEBHOOK_URL_RE.sub("<redacted-url>", text)
    text = _LOCAL_ABSOLUTE_PATH_RE.sub("<redacted-path>", text)
    text = _SENSITIVE_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=<redacted>", text)

    if len(text) > max_length:
        return f"{text[:max_length].rstrip()}..."
    return text


def safe_diagnostic_key(value: Any) -> str:
    """Normalize a diagnostic object key after applying text redaction."""
    text = sanitize_diagnostic_text(value, max_length=80) or ""
    return re.sub(r"[^A-Za-z0-9_]+", "_", text.strip().lower()).strip("_")[:80]


def sanitize_diagnostic_metadata(value: Any, *, depth: int = 0) -> Any:
    """Recursively redact diagnostic metadata before it reaches API/SSE payloads."""
    if depth > 3:
        return "<truncated>"
    if isinstance(value, Mapping):
        sanitized: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 20:
                sanitized["truncated"] = True
                break
            safe_key = safe_diagnostic_key(key)
            if not safe_key:
                continue
            if _SENSITIVE_KEY_RE.search(str(key)):
                sanitized[safe_key] = "<redacted>"
                continue
            safe_value = sanitize_diagnostic_metadata(item, depth=depth + 1)
            if safe_value not in (None, "", [], {}):
                sanitized[safe_key] = safe_value
        return sanitized
    if isinstance(value, list):
        items = [sanitize_diagnostic_metadata(item, depth=depth + 1) for item in value[:8]]
        return [item for item in items if item not in (None, "", [], {})]
    if isinstance(value, tuple):
        return sanitize_diagnostic_metadata(list(value), depth=depth)
    if isinstance(value, (int, float, bool)):
        return value
    return sanitize_diagnostic_text(value, max_length=160)


@dataclass
class ProviderRun:
    """One provider attempt in a trace."""

    trace_id: str
    data_type: str
    provider: str
    operation: str
    success: bool
    latency_ms: Optional[int] = None
    error_type: Optional[str] = None
    error_message_sanitized: Optional[str] = None
    fallback_from: Optional[str] = None
    fallback_to: Optional[str] = None
    cache_hit: Optional[bool] = None
    stale_seconds: Optional[int] = None
    record_count: Optional[int] = None
    data_date: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "trace_id": self.trace_id,
            "data_type": self.data_type,
            "provider": self.provider,
            "operation": self.operation,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "error_type": self.error_type,
            "error_message_sanitized": self.error_message_sanitized,
            "fallback_from": self.fallback_from,
            "fallback_to": self.fallback_to,
            "cache_hit": self.cache_hit,
            "stale_seconds": self.stale_seconds,
            "record_count": self.record_count,
            "data_date": self.data_date,
            "created_at": self.created_at,
        }
        return {key: value for key, value in payload.items() if value is not None}


@dataclass
class LLMRun:
    """One LLM call result in a trace."""

    trace_id: str
    provider: Optional[str] = None
    model: Optional[str] = None
    call_type: str = "analysis"
    success: bool = True
    tokens: Optional[int] = None
    duration_ms: Optional[int] = None
    fallback_model: Optional[str] = None
    error_type: Optional[str] = None
    error_message_sanitized: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "trace_id": self.trace_id,
            "provider": self.provider,
            "model": self.model,
            "call_type": self.call_type,
            "success": self.success,
            "tokens": self.tokens,
            "duration_ms": self.duration_ms,
            "fallback_model": self.fallback_model,
            "error_type": self.error_type,
            "error_message_sanitized": self.error_message_sanitized,
            "created_at": self.created_at,
        }
        return {key: value for key, value in payload.items() if value is not None}


@dataclass
class NotificationRun:
    """Notification dispatch result in a trace."""

    trace_id: str
    channel: str
    status: str
    success: bool
    attempts: int = 1
    error_message_sanitized: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "trace_id": self.trace_id,
            "channel": self.channel,
            "status": self.status,
            "success": self.success,
            "attempts": self.attempts,
            "error_message_sanitized": self.error_message_sanitized,
            "created_at": self.created_at,
        }
        return {key: value for key, value in payload.items() if value is not None}


@dataclass
class HistoryRun:
    """History persistence result in a trace."""

    trace_id: str
    report_saved: bool
    metadata_saved: Optional[bool] = None
    analysis_history_id: Optional[int] = None
    error_message_sanitized: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "trace_id": self.trace_id,
            "report_saved": self.report_saved,
            "metadata_saved": self.metadata_saved,
            "analysis_history_id": self.analysis_history_id,
            "error_message_sanitized": self.error_message_sanitized,
            "created_at": self.created_at,
        }
        return {key: value for key, value in payload.items() if value is not None}


@dataclass
class RunDiagnosticComponent:
    """User-facing status for one diagnostic component."""

    key: str
    label: str
    status: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "key": self.key,
            "label": self.label,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}


@dataclass
class RunDiagnosticSummary:
    """User-facing diagnostic summary for one analysis run."""

    status: str
    status_label: str
    reason: str
    trace_id: Optional[str] = None
    task_id: Optional[str] = None
    query_id: Optional[str] = None
    stock_code: Optional[str] = None
    trigger_source: Optional[str] = None
    components: Dict[str, RunDiagnosticComponent] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "query_id": self.query_id,
            "stock_code": self.stock_code,
            "trigger_source": self.trigger_source,
            "status": self.status,
            "status_label": self.status_label,
            "reason": self.reason,
            "components": {
                key: component.to_dict()
                for key, component in self.components.items()
            },
        }
        payload["copy_text"] = format_copyable_diagnostics(payload)
        return {key: value for key, value in payload.items() if value is not None}


@dataclass
class RunDiagnosticContext:
    """Diagnostic state for one analysis run."""

    trace_id: str
    task_id: Optional[str] = None
    query_id: Optional[str] = None
    stock_code: Optional[str] = None
    trigger_source: Optional[str] = None
    scope: Optional[str] = None
    provider_runs: List[ProviderRun] = field(default_factory=list)
    llm_runs: List[LLMRun] = field(default_factory=list)
    notification_runs: List[NotificationRun] = field(default_factory=list)
    history_runs: List[HistoryRun] = field(default_factory=list)
    event_sink: Optional[Callable[[Dict[str, Any]], None]] = None
    flow_event_index: int = 0
    provider_attempt_index_by_type: Dict[str, int] = field(default_factory=dict)
    provider_pending_attempt_index_by_key: Dict[str, List[int]] = field(default_factory=dict)
    llm_attempt_index_by_type: Dict[str, int] = field(default_factory=dict)
    llm_pending_attempt_index_by_key: Dict[str, List[int]] = field(default_factory=dict)
    llm_pending_attempt_index_by_call_type: Dict[str, List[int]] = field(default_factory=dict)

    def record_provider_run(self, provider_run: ProviderRun) -> None:
        self.provider_runs.append(provider_run)
        data_type_key = _safe_event_key(provider_run.data_type) or "provider"
        pending_key = _provider_pending_key(
            provider_run.data_type,
            provider_run.provider,
            provider_run.operation,
        )
        pending_indexes = self.provider_pending_attempt_index_by_key.get(pending_key) or []
        if pending_indexes:
            attempt_index = pending_indexes.pop(0)
            if pending_indexes:
                self.provider_pending_attempt_index_by_key[pending_key] = pending_indexes
            else:
                self.provider_pending_attempt_index_by_key.pop(pending_key, None)
        else:
            attempt_index = self.provider_attempt_index_by_type.get(data_type_key, 0) + 1
            self.provider_attempt_index_by_type[data_type_key] = attempt_index
        self._emit_flow_event(_provider_flow_event(self, provider_run, attempt_index))

    def record_provider_run_started(
        self,
        *,
        data_type: str,
        provider: str,
        operation: str,
    ) -> None:
        data_type_key = _safe_event_key(data_type) or "provider"
        attempt_index = self.provider_attempt_index_by_type.get(data_type_key, 0) + 1
        self.provider_attempt_index_by_type[data_type_key] = attempt_index
        pending_key = _provider_pending_key(data_type, provider, operation)
        pending_indexes = self.provider_pending_attempt_index_by_key.get(pending_key) or []
        pending_indexes.append(attempt_index)
        self.provider_pending_attempt_index_by_key[pending_key] = pending_indexes
        self._emit_flow_event(
            _provider_started_flow_event(
                self,
                data_type=data_type,
                provider=provider,
                operation=operation,
                index=attempt_index,
            )
        )

    def record_llm_run(self, llm_run: LLMRun) -> None:
        self.llm_runs.append(llm_run)
        call_type_key = _safe_event_key(llm_run.call_type) or "analysis"
        pending_key = _llm_pending_key(llm_run.call_type, llm_run.provider, llm_run.model)
        pending_indexes = self.llm_pending_attempt_index_by_key.get(pending_key) or []
        if pending_indexes:
            attempt_index = pending_indexes.pop(0)
            if pending_indexes:
                self.llm_pending_attempt_index_by_key[pending_key] = pending_indexes
            else:
                self.llm_pending_attempt_index_by_key.pop(pending_key, None)
            self._remove_llm_pending_call_type_index(call_type_key, attempt_index)
        else:
            call_type_pending_indexes = self.llm_pending_attempt_index_by_call_type.get(call_type_key) or []
            if call_type_pending_indexes:
                attempt_index = call_type_pending_indexes.pop(0)
                if call_type_pending_indexes:
                    self.llm_pending_attempt_index_by_call_type[call_type_key] = call_type_pending_indexes
                else:
                    self.llm_pending_attempt_index_by_call_type.pop(call_type_key, None)
                self._remove_llm_pending_exact_index(attempt_index)
            else:
                attempt_index = self.llm_attempt_index_by_type.get(call_type_key, 0) + 1
                self.llm_attempt_index_by_type[call_type_key] = attempt_index
        self._emit_flow_event(_llm_flow_event(self, llm_run, attempt_index))

    def _remove_llm_pending_call_type_index(self, call_type_key: str, attempt_index: int) -> None:
        pending_indexes = self.llm_pending_attempt_index_by_call_type.get(call_type_key) or []
        if attempt_index not in pending_indexes:
            return
        pending_indexes = [index for index in pending_indexes if index != attempt_index]
        if pending_indexes:
            self.llm_pending_attempt_index_by_call_type[call_type_key] = pending_indexes
        else:
            self.llm_pending_attempt_index_by_call_type.pop(call_type_key, None)

    def _remove_llm_pending_exact_index(self, attempt_index: int) -> None:
        for pending_key, pending_indexes in list(self.llm_pending_attempt_index_by_key.items()):
            if attempt_index not in pending_indexes:
                continue
            pending_indexes = [index for index in pending_indexes if index != attempt_index]
            if pending_indexes:
                self.llm_pending_attempt_index_by_key[pending_key] = pending_indexes
            else:
                self.llm_pending_attempt_index_by_key.pop(pending_key, None)

    def record_llm_run_started(
        self,
        *,
        call_type: str = "analysis",
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        call_type_key = _safe_event_key(call_type) or "analysis"
        attempt_index = self.llm_attempt_index_by_type.get(call_type_key, 0) + 1
        self.llm_attempt_index_by_type[call_type_key] = attempt_index
        pending_key = _llm_pending_key(call_type, provider, model)
        pending_indexes = self.llm_pending_attempt_index_by_key.get(pending_key) or []
        pending_indexes.append(attempt_index)
        self.llm_pending_attempt_index_by_key[pending_key] = pending_indexes
        call_type_pending_indexes = self.llm_pending_attempt_index_by_call_type.get(call_type_key) or []
        call_type_pending_indexes.append(attempt_index)
        self.llm_pending_attempt_index_by_call_type[call_type_key] = call_type_pending_indexes
        self._emit_flow_event(
            _llm_started_flow_event(
                self,
                call_type=call_type,
                provider=provider,
                model=model,
                index=attempt_index,
            )
        )

    def record_notification_run(self, notification_run: NotificationRun) -> None:
        self.notification_runs.append(notification_run)
        self._emit_flow_event(_notification_flow_event(self, notification_run, len(self.notification_runs)))

    def record_history_run(self, history_run: HistoryRun) -> None:
        self.history_runs.append(history_run)
        self._emit_flow_event(_history_flow_event(self, history_run, len(self.history_runs)))

    def _emit_flow_event(self, event: Dict[str, Any]) -> None:
        if self.event_sink is None:
            return
        try:
            self.flow_event_index += 1
            event_payload = sanitize_diagnostic_metadata(event)
            event_payload = dict(event_payload) if isinstance(event_payload, Mapping) else {}
            event_payload["id"] = event_payload.get("id") or f"flow_{self.flow_event_index:04d}"
            self.event_sink(event_payload)
        except Exception as exc:  # pragma: no cover - defensive fail-open guard
            logger.warning("run-flow event sink failed: %s", exc)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "query_id": self.query_id,
            "stock_code": self.stock_code,
            "trigger_source": self.trigger_source,
            "scope": self.scope,
            "provider_runs": [run.to_dict() for run in self.provider_runs],
            "llm_runs": [run.to_dict() for run in self.llm_runs],
            "notification_runs": [run.to_dict() for run in self.notification_runs],
            "history_runs": [run.to_dict() for run in self.history_runs],
        }


def get_current_diagnostic_context() -> Optional[RunDiagnosticContext]:
    return _CURRENT_CONTEXT.get()


def activate_run_diagnostic_context(
    *,
    trace_id: Optional[str] = None,
    task_id: Optional[str] = None,
    query_id: Optional[str] = None,
    stock_code: Optional[str] = None,
    trigger_source: Optional[str] = None,
    scope: Optional[str] = None,
    event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Token:
    """Activate a diagnostic context and return its reset token."""
    context = RunDiagnosticContext(
        trace_id=trace_id or query_id or task_id or build_trace_id(),
        task_id=task_id,
        query_id=query_id,
        stock_code=stock_code,
        trigger_source=trigger_source,
        scope=scope,
        event_sink=event_sink,
    )
    return _CURRENT_CONTEXT.set(context)


def reset_run_diagnostic_context(token: Optional[Token]) -> None:
    if token is None:
        return
    try:
        _CURRENT_CONTEXT.reset(token)
    except Exception as exc:  # pragma: no cover - defensive fail-open guard
        logger.warning("run diagnostic context reset failed: %s", exc)


def current_diagnostic_snapshot() -> Optional[Dict[str, Any]]:
    context = get_current_diagnostic_context()
    if context is None:
        return None
    try:
        return context.snapshot()
    except Exception as exc:  # pragma: no cover - defensive fail-open guard
        logger.warning("run diagnostic snapshot failed: %s", exc)
        return None


_DATA_TYPE_LABELS = {
    "realtime_quote": "实时行情",
    "daily_data": "日线K线",
    "daily_bars": "日线K线",
    "technical": "技术指标",
    "news": "新闻舆情",
    "news_search": "新闻舆情",
    "fundamental": "基本面",
    "fundamentals": "基本面",
    "belong_boards": "所属板块",
    "chip": "筹码结构",
}


def _safe_event_key(value: Any) -> str:
    return safe_diagnostic_key(value)


def _clean_metadata(value: Dict[str, Any]) -> Dict[str, Any]:
    return {
        str(key): item
        for key, item in value.items()
        if item not in (None, "", [], {})
    }


def _provider_pending_key(data_type: Any, provider: Any, operation: Any) -> str:
    return "|".join(
        (
            _safe_event_key(data_type) or "provider",
            _safe_event_key(provider) or "unknown",
            _safe_event_key(operation) or "operation",
        )
    )


def _llm_pending_key(call_type: Any, provider: Any, model: Any) -> str:
    _ = (provider, model)
    return _safe_event_key(call_type) or "analysis"


def _flow_status_for_success(success: bool, *, fallback: bool = False, skipped: bool = False) -> str:
    if skipped:
        return "skipped"
    if success:
        return "fallback" if fallback else "success"
    return "failed"


def _started_at_from_end_and_duration(end: Any, duration_ms: Optional[int]) -> Optional[str]:
    if duration_ms is None or duration_ms < 0:
        return None
    if isinstance(end, datetime):
        parsed = end
    elif isinstance(end, str) and "T" in end:
        normalized = end[:-1] + "+00:00" if end.endswith("Z") else end
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
    else:
        return None
    return (parsed - timedelta(milliseconds=duration_ms)).isoformat()


def _provider_started_flow_event(
    context: RunDiagnosticContext,
    *,
    data_type: str,
    provider: str,
    operation: str,
    index: int,
) -> Dict[str, Any]:
    data_type_key = _safe_event_key(data_type) or "provider"
    provider_key = _safe_event_key(provider) or "unknown"
    label = _DATA_TYPE_LABELS.get(data_type_key, data_type_key)
    node_id = f"provider_{data_type_key}_{provider_key}_{index}"
    timestamp = datetime.now().isoformat()
    message = f"{label} {provider} 调用中"
    return {
        "timestamp": timestamp,
        "severity": "info",
        "type": "provider_run_started",
        "node_id": node_id,
        "title": f"{label}开始",
        "message": sanitize_diagnostic_text(message, max_length=220),
        "metadata": _clean_metadata(
            {
                "trace_id": context.trace_id,
                "provider": provider,
                "data_type": data_type,
                "operation": operation,
                "node": {
                    "id": node_id,
                    "lane": "data_source",
                    "kind": "data_source",
                    "label": f"{label} · {provider}",
                    "status": "running",
                    "provider": provider,
                    "started_at": timestamp,
                    "attempts": 1,
                    "message": message,
                },
            }
        ),
    }


def _provider_flow_event(
    context: RunDiagnosticContext,
    run: ProviderRun,
    index: int,
) -> Dict[str, Any]:
    data_type = _safe_event_key(run.data_type) or "provider"
    provider_key = _safe_event_key(run.provider) or "unknown"
    label = _DATA_TYPE_LABELS.get(data_type, data_type)
    same_type_runs = [
        item
        for item in context.provider_runs
        if (_safe_event_key(item.data_type) or "provider") == data_type
    ]
    # One trace may execute the same provider chain multiple times (for example
    # a market review requests three separate news queries).  A provider repeat
    # after a successful primary starts a new chain; do not classify all later
    # queries as supplements of the first query.
    current_chain: List[ProviderRun] = []
    seen_providers: set[str] = set()
    chain_has_success = False
    for item in same_type_runs:
        item_provider = _safe_event_key(item.provider) or "unknown"
        if item_provider in seen_providers and chain_has_success:
            current_chain = []
            seen_providers = set()
            chain_has_success = False
        current_chain.append(item)
        seen_providers.add(item_provider)
        chain_has_success = chain_has_success or bool(item.success)
    preceding_runs = current_chain[:-1]
    primary_already_succeeded = any(item.success for item in preceding_runs)
    role = "supplement" if primary_already_succeeded else "primary"
    run_payload = run.to_dict()
    skipped = _is_skipped_provider_run(run_payload)
    had_real_failure_before_success = any(
        _is_real_provider_failure(item.to_dict()) for item in preceding_runs
    )
    if run.success:
        fallback = not primary_already_succeeded and bool(
            run.fallback_from or had_real_failure_before_success
        )
        status = _flow_status_for_success(True, fallback=fallback)
    elif skipped:
        status = _flow_status_for_success(False, skipped=True)
    elif "timeout" in str(run.error_type or "").lower():
        status = "timeout"
    else:
        status = "failed"
    node_id = f"provider_{data_type}_{provider_key}_{index}"
    started_at = _started_at_from_end_and_duration(run.created_at, run.latency_ms)
    if run.success:
        outcome = "补充成功" if role == "supplement" else "成功"
        message = f"{label} {run.provider} {outcome}"
        title = f"{label}{'补充完成' if role == 'supplement' else '成功'}"
        severity = "success"
    elif skipped:
        message = f"{label} {run.provider} 跳过：{run.error_message_sanitized or run.error_type or '当前未配置或不适用'}"
        title = f"{label}跳过"
        severity = "info"
    else:
        message = f"{label} {run.provider} 失败：{run.error_message_sanitized or run.error_type or '未知错误'}"
        title = f"{label}失败"
        severity = "warning"
    return {
        "timestamp": run.created_at,
        "severity": severity,
        "type": "provider_run",
        "node_id": node_id,
        "title": title,
        "message": sanitize_diagnostic_text(message, max_length=220),
        "metadata": _clean_metadata(
            {
                "trace_id": context.trace_id,
                "provider": run.provider,
                "data_type": run.data_type,
                "operation": run.operation,
                "duration_ms": run.latency_ms,
                "record_count": run.record_count,
                "role": role,
                "fallback_from": run.fallback_from,
                "fallback_to": run.fallback_to,
                "error_type": run.error_type,
                "node": {
                    "id": node_id,
                    "lane": "data_source",
                    "kind": "data_source",
                    "label": f"{label} · {run.provider}",
                    "status": status,
                    "provider": run.provider,
                    "started_at": started_at,
                    "ended_at": run.created_at,
                    "duration_ms": run.latency_ms,
                    "record_count": run.record_count,
                    "message": message,
                    "metadata": {
                        "data_type": run.data_type,
                        "operation": run.operation,
                        "role": role,
                        "error_type": run.error_type,
                    },
                },
            }
        ),
    }


def _llm_started_flow_event(
    context: RunDiagnosticContext,
    *,
    call_type: str,
    provider: Optional[str],
    model: Optional[str],
    index: int,
) -> Dict[str, Any]:
    call_type_key = _safe_event_key(call_type) or "analysis"
    display_model = model or provider or "unknown"
    node_id = f"llm_{call_type_key}_{index}"
    timestamp = datetime.now().isoformat()
    message = f"LLM {display_model} 调用中"
    return {
        "timestamp": timestamp,
        "severity": "info",
        "type": "llm_run_started",
        "node_id": node_id,
        "title": "LLM 开始",
        "message": sanitize_diagnostic_text(message, max_length=220),
        "metadata": _clean_metadata(
            {
                "trace_id": context.trace_id,
                "provider": provider,
                "model": model,
                "call_type": call_type,
                "node": {
                    "id": node_id,
                    "lane": "analysis",
                    "kind": "model",
                    "label": "LLM 生成",
                    "status": "running",
                    "provider": display_model,
                    "started_at": timestamp,
                    "attempts": 1,
                    "message": message,
                },
            }
        ),
    }


def _llm_flow_event(
    context: RunDiagnosticContext,
    run: LLMRun,
    index: int,
) -> Dict[str, Any]:
    call_type = _safe_event_key(run.call_type) or "analysis"
    model = run.model or run.provider or "unknown"
    status = _flow_status_for_success(run.success, fallback=bool(run.fallback_model or index > 1))
    node_id = f"llm_{call_type}_{index}"
    started_at = _started_at_from_end_and_duration(run.created_at, run.duration_ms)
    message = (
        f"LLM {model} 成功"
        if run.success
        else f"LLM {model} 失败：{run.error_message_sanitized or run.error_type or '未知错误'}"
    )
    return {
        "timestamp": run.created_at,
        "severity": "success" if run.success else "danger",
        "type": "llm_run",
        "node_id": node_id,
        "title": f"LLM {'成功' if run.success else '失败'}",
        "message": sanitize_diagnostic_text(message, max_length=220),
        "metadata": _clean_metadata(
            {
                "trace_id": context.trace_id,
                "provider": run.provider,
                "model": run.model,
                "call_type": run.call_type,
                "duration_ms": run.duration_ms,
                "fallback_model": run.fallback_model,
                "error_type": run.error_type,
                "node": {
                    "id": node_id,
                    "lane": "analysis",
                    "kind": "model",
                    "label": "LLM 生成",
                    "status": status,
                    "provider": model,
                    "started_at": started_at,
                    "ended_at": run.created_at,
                    "duration_ms": run.duration_ms,
                    "message": message,
                },
            }
        ),
    }


def _history_flow_event(
    context: RunDiagnosticContext,
    run: HistoryRun,
    index: int,
) -> Dict[str, Any]:
    node_id = "history_save" if index == 1 else f"history_save_{index}"
    status = "success" if run.report_saved else "failed"
    message = "报告历史已保存" if run.report_saved else f"报告历史保存失败：{run.error_message_sanitized or '未知错误'}"
    return {
        "timestamp": run.created_at,
        "severity": "success" if run.report_saved else "danger",
        "type": "history_run",
        "node_id": node_id,
        "title": "历史保存成功" if run.report_saved else "历史保存失败",
        "message": sanitize_diagnostic_text(message, max_length=220),
        "metadata": _clean_metadata(
            {
                "trace_id": context.trace_id,
                "metadata_saved": run.metadata_saved,
                "analysis_history_id": run.analysis_history_id,
                "node": {
                    "id": node_id,
                    "lane": "artifact",
                    "kind": "artifact",
                    "label": "保存报告",
                    "status": status,
                    "message": message,
                },
            }
        ),
    }


def _notification_flow_event(
    context: RunDiagnosticContext,
    run: NotificationRun,
    index: int,
) -> Dict[str, Any]:
    channel = run.channel or "unknown"
    channel_key = _safe_event_key(channel) or "unknown"
    skipped = run.status in {"skipped", "not_configured"}
    status = _flow_status_for_success(run.success, skipped=skipped)
    node_id = f"notification_{channel_key}_{index}"
    if status == "success":
        title = "通知发送成功"
        message = f"{channel} 通知发送成功"
    elif status == "skipped":
        title = "通知跳过"
        message = f"{channel} 通知跳过"
    else:
        title = "通知失败"
        message = f"{channel} 通知失败：{run.error_message_sanitized or run.status or '未知错误'}"
    return {
        "timestamp": run.created_at,
        "severity": "success" if status == "success" else ("warning" if status == "skipped" else "danger"),
        "type": "notification_run",
        "node_id": node_id,
        "title": title,
        "message": sanitize_diagnostic_text(message, max_length=220),
        "metadata": _clean_metadata(
            {
                "trace_id": context.trace_id,
                "channel": channel,
                "status": run.status,
                "attempts": run.attempts,
                "node": {
                    "id": node_id,
                    "lane": "artifact",
                    "kind": "notification",
                    "label": f"推送通知 · {channel}",
                    "status": status,
                    "provider": channel,
                    "attempts": run.attempts,
                    "message": message,
                },
            }
        ),
    }


def record_provider_run(
    *,
    data_type: str,
    provider: str,
    operation: str,
    success: bool,
    latency_ms: Optional[int] = None,
    error_type: Optional[str] = None,
    error_message: Optional[Any] = None,
    fallback_from: Optional[str] = None,
    fallback_to: Optional[str] = None,
    cache_hit: Optional[bool] = None,
    stale_seconds: Optional[int] = None,
    record_count: Optional[int] = None,
    data_date: Optional[str] = None,
) -> None:
    """Append a provider attempt to the active context without affecting callers."""
    context = get_current_diagnostic_context()
    if context is None:
        return

    try:
        context.record_provider_run(
            ProviderRun(
                trace_id=context.trace_id,
                data_type=data_type,
                provider=provider,
                operation=operation,
                success=success,
                latency_ms=latency_ms,
                error_type=error_type,
                error_message_sanitized=sanitize_diagnostic_text(error_message),
                fallback_from=fallback_from,
                fallback_to=fallback_to,
                cache_hit=cache_hit,
                stale_seconds=stale_seconds,
                record_count=record_count,
                data_date=data_date,
            )
        )
    except Exception as exc:  # pragma: no cover - defensive fail-open guard
        logger.warning("provider diagnostic record failed: %s", exc)


def record_provider_run_started(
    *,
    data_type: str,
    provider: str,
    operation: str,
) -> None:
    """Emit a live provider-start event without changing persisted diagnostics."""
    context = get_current_diagnostic_context()
    if context is None:
        return

    try:
        context.record_provider_run_started(
            data_type=data_type,
            provider=provider,
            operation=operation,
        )
    except Exception as exc:  # pragma: no cover - defensive fail-open guard
        logger.warning("provider started diagnostic record failed: %s", exc)


def record_llm_run(
    *,
    success: bool,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    call_type: str = "analysis",
    tokens: Optional[int] = None,
    duration_ms: Optional[int] = None,
    fallback_model: Optional[str] = None,
    error_type: Optional[str] = None,
    error_message: Optional[Any] = None,
) -> None:
    """Append an LLM call result to the active context without affecting callers."""
    context = get_current_diagnostic_context()
    if context is None:
        return

    try:
        context.record_llm_run(
            LLMRun(
                trace_id=context.trace_id,
                provider=provider,
                model=model,
                call_type=call_type,
                success=success,
                tokens=tokens,
                duration_ms=duration_ms,
                fallback_model=fallback_model,
                error_type=error_type,
                error_message_sanitized=sanitize_diagnostic_text(error_message),
            )
        )
    except Exception as exc:  # pragma: no cover - defensive fail-open guard
        logger.warning("llm diagnostic record failed: %s", exc)


def record_llm_run_started(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    call_type: str = "analysis",
) -> None:
    """Emit a live LLM-start event without changing persisted diagnostics."""
    context = get_current_diagnostic_context()
    if context is None:
        return

    try:
        context.record_llm_run_started(
            provider=provider,
            model=model,
            call_type=call_type,
        )
    except Exception as exc:  # pragma: no cover - defensive fail-open guard
        logger.warning("llm started diagnostic record failed: %s", exc)


def record_notification_run(
    *,
    channel: str,
    status: str,
    success: bool,
    attempts: int = 1,
    error_message: Optional[Any] = None,
) -> None:
    """Append a notification result to the active context without affecting callers."""
    context = get_current_diagnostic_context()
    if context is None:
        return

    try:
        context.record_notification_run(
            NotificationRun(
                trace_id=context.trace_id,
                channel=channel,
                status=status,
                success=success,
                attempts=attempts,
                error_message_sanitized=sanitize_diagnostic_text(error_message),
            )
        )
    except Exception as exc:  # pragma: no cover - defensive fail-open guard
        logger.warning("notification diagnostic record failed: %s", exc)


def record_history_run(
    *,
    report_saved: bool,
    metadata_saved: Optional[bool] = None,
    analysis_history_id: Optional[int] = None,
    error_message: Optional[Any] = None,
) -> None:
    """Append a history persistence result to the active context without affecting callers."""
    context = get_current_diagnostic_context()
    if context is None:
        return

    try:
        context.record_history_run(
            HistoryRun(
                trace_id=context.trace_id,
                report_saved=report_saved,
                metadata_saved=metadata_saved,
                analysis_history_id=analysis_history_id,
                error_message_sanitized=sanitize_diagnostic_text(error_message),
            )
        )
    except Exception as exc:  # pragma: no cover - defensive fail-open guard
        logger.warning("history diagnostic record failed: %s", exc)


_SUMMARY_STATUS_LABELS = {
    "normal": "正常",
    "degraded": "部分降级",
    "failed": "失败",
    "unknown": "未知",
}
_ANALYSIS_INPUT_STATUS_MESSAGES = {
    "missing": "未进入本次分析输入",
    "partial": "本次分析输入仅部分可用",
    "fallback": "本次分析输入使用降级数据",
    "stale": "本次分析输入使用过期数据",
    "estimated": "本次分析输入使用估算数据",
    "fetch_failed": "输入块显示抓取失败",
    "not_supported": "输入块标记为不支持",
}
# 数据源“未配置/请求时不可用”，属于跳过而非失败，见 `_is_skipped_provider_run`。
_SKIPPED_PROVIDER_ERROR_TYPES = {
    "unavailable",
    "not_available",
    "not_configured",
    "not_applicable",
    "unsupported",
    "skipped",
}


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _component(
    key: str,
    label: str,
    status: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> RunDiagnosticComponent:
    clean_details = {
        key: value
        for key, value in (details or {}).items()
        if value is not None
    }
    return RunDiagnosticComponent(
        key=key,
        label=label,
        status=status,
        message=message,
        details=clean_details,
    )


def _analysis_context_overview(context_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    overview = context_snapshot.get("analysis_context_pack_overview")
    if not isinstance(overview, dict):
        overview = context_snapshot.get("analysisContextPackOverview")
    return overview if isinstance(overview, dict) else {}


def _analysis_input_block(
    context_snapshot: Dict[str, Any],
    block_key: str,
) -> Dict[str, Any]:
    blocks = _analysis_context_overview(context_snapshot).get("blocks")
    if isinstance(blocks, list):
        for block in blocks:
            if isinstance(block, dict) and block.get("key") == block_key:
                return block
    if isinstance(blocks, dict):
        block = blocks.get(block_key)
        if isinstance(block, dict):
            return block
    return {}


def _analysis_input_status_message(block: Dict[str, Any]) -> Optional[str]:
    status = str(block.get("status") or "").strip()
    if status == "available" or not status:
        return None
    return _ANALYSIS_INPUT_STATUS_MESSAGES.get(status, f"输入块状态为 {status}")


def _analysis_input_component(
    context_snapshot: Dict[str, Any],
) -> RunDiagnosticComponent:
    """Summarize the same context-pack quality boundary shown by run flow."""
    label = "分析输入"
    overview = _analysis_context_overview(context_snapshot)
    raw_blocks = overview.get("blocks")
    if isinstance(raw_blocks, dict):
        blocks = [block for block in raw_blocks.values() if isinstance(block, dict)]
    elif isinstance(raw_blocks, list):
        blocks = [block for block in raw_blocks if isinstance(block, dict)]
    else:
        blocks = []
    if not blocks:
        return _component("analysis_input", label, "unknown", "未记录分析输入上下文质量")

    degraded_statuses = {
        "fetch_failed",
        "fallback",
        "partial",
        "stale",
        "estimated",
        "missing",
    }
    affected_blocks: List[Dict[str, Any]] = []
    unsupported_blocks: List[Dict[str, Any]] = []
    unknown_blocks: List[Dict[str, Any]] = []
    available_count = 0
    for block in blocks:
        status = str(block.get("status") or "").strip()
        item = {
            "key": block.get("key"),
            "label": block.get("label") or block.get("key"),
            "status": status or "unknown",
            "source": block.get("source"),
            "missing_reasons": _list_text(block.get("missing_reasons")),
        }
        item = {key: value for key, value in item.items() if value not in (None, [], "")}
        if status == "available":
            available_count += 1
        elif status in degraded_statuses:
            affected_blocks.append(item)
        elif status == "not_supported":
            unsupported_blocks.append(item)
        else:
            unknown_blocks.append(item)

    quality = _as_dict(overview.get("data_quality"))
    details = {
        "available_count": available_count,
        "block_count": len(blocks),
        "overall_score": quality.get("overall_score"),
        "quality_level": quality.get("level"),
        "counts": overview.get("counts") if isinstance(overview.get("counts"), dict) else None,
        "affected_blocks": affected_blocks or None,
        "not_supported_blocks": unsupported_blocks or None,
        "unknown_blocks": unknown_blocks or None,
    }
    if affected_blocks:
        affected_text = "、".join(
            f"{item.get('label') or item.get('key')}（{_ANALYSIS_INPUT_STATUS_MESSAGES.get(str(item.get('status')), str(item.get('status')))}）"
            for item in affected_blocks[:3]
        )
        return _component(
            "analysis_input",
            label,
            "degraded",
            f"分析输入存在 {len(affected_blocks)} 个降级块：{affected_text}",
            details,
        )
    if unknown_blocks:
        return _component(
            "analysis_input",
            label,
            "unknown",
            f"分析输入有 {len(unknown_blocks)} 个块状态未知",
            details,
        )

    message = f"分析输入已组装，{available_count} 个块可用"
    if unsupported_blocks:
        message += f"，{len(unsupported_blocks)} 个块结构性不支持"
    return _component("analysis_input", label, "ok", message, details)


def _list_text(value: Any, *, limit: int = 5) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        text = str(item).strip() if item is not None else ""
        if text and text not in result:
            result.append(text)
    return result[:limit]


def _reconcile_daily_provider_with_analysis_input(
    component: RunDiagnosticComponent,
    context_snapshot: Dict[str, Any],
) -> RunDiagnosticComponent:
    input_block = _analysis_input_block(context_snapshot, "daily_bars")
    input_message = _analysis_input_status_message(input_block)
    if not input_message or component.status not in {"ok", "degraded"}:
        return component

    details = dict(component.details or {})
    details.update(
        {
            "provider_run_status": component.status,
            "analysis_input_block": "daily_bars",
            "analysis_input_status": input_block.get("status"),
            "analysis_input_source": input_block.get("source"),
            "analysis_input_missing_reasons": _list_text(
                input_block.get("missing_reasons")
            ),
            "evidence_scope": "provider_run_vs_analysis_input",
        }
    )
    provider = details.get("provider") or "unknown"
    return _component(
        component.key,
        component.label,
        "degraded",
        f"{component.label}{provider} 成功，但{input_message}",
        details,
    )


def _is_skipped_provider_run(run: Dict[str, Any]) -> bool:
    """未配置/未启用的数据源：被跳过，而不是“失败”。

    `data_provider` 在数据源缺少凭据、未安装依赖或请求时不可用时，会记录
    `error_type="unavailable"` 的尝试。这类数据源根本没有发起请求，把它算成
    失败会让“没有配置长桥/富途/Tushare”的部署永久显示降级。
    """
    if run.get("success") is not False:
        return False
    return str(run.get("error_type") or "").strip().lower() in _SKIPPED_PROVIDER_ERROR_TYPES


def _is_real_provider_failure(run: Dict[str, Any]) -> bool:
    """真实失败：数据源确实被调用过并且没有拿到可用数据。"""
    return run.get("success") is False and not _is_skipped_provider_run(run)


def _provider_names(runs: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for run in runs:
        name = str(run.get("provider") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _partition_repeated_provider_chains(
    runs: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """Split repeated searches into independent provider chains.

    Market review performs several news queries in one diagnostic trace.  Once a
    chain has succeeded, seeing one of its providers again marks the next query;
    later failures must not be misclassified as supplements of the first query.
    """
    chains: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    seen_providers: set[str] = set()
    chain_has_success = False
    for run in runs:
        provider = str(run.get("provider") or "unknown").strip().lower() or "unknown"
        if provider in seen_providers and chain_has_success:
            if current:
                chains.append(current)
            current = []
            seen_providers = set()
            chain_has_success = False
        current.append(run)
        seen_providers.add(provider)
        chain_has_success = chain_has_success or run.get("success") is True
    if current:
        chains.append(current)
    return chains


def _market_review_news_count(context_snapshot: Dict[str, Any]) -> Optional[int]:
    payload = context_snapshot.get("market_review_payload")
    if not isinstance(payload, dict):
        return None
    markets = payload.get("markets")
    payloads = (
        [item for item in markets.values() if isinstance(item, dict)]
        if isinstance(markets, dict)
        else [payload]
    )
    found = False
    count = 0
    for item in payloads:
        news = item.get("news")
        if isinstance(news, list):
            found = True
            count += len(news)
    return count if found else None


def _news_provider_details(
    provider_runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    runs = [
        run for run in provider_runs
        if isinstance(run, dict) and run.get("data_type") == "news_search"
    ]
    if not runs:
        return {}

    chains = _partition_repeated_provider_chains(runs)
    fallback_count = 0
    failed_attempts = 0
    primary_successes: List[Dict[str, Any]] = []
    for chain in chains:
        first_success_index = next(
            (index for index, run in enumerate(chain) if run.get("success") is True),
            None,
        )
        if first_success_index is None:
            failed_attempts += sum(1 for run in chain if _is_real_provider_failure(run))
            continue
        primary_successes.append(chain[first_success_index])
        preceding_failures = [
            run for run in chain[:first_success_index] if _is_real_provider_failure(run)
        ]
        if preceding_failures:
            fallback_count += 1
            failed_attempts += len(preceding_failures)

    details: Dict[str, Any] = {
        "attempts": len(runs),
        "chain_count": len(chains),
        "fallback_count": fallback_count,
        "failed_attempts": failed_attempts,
        "providers": _provider_names(primary_successes) or None,
        "failed_providers": _provider_names(
            [run for run in runs if _is_real_provider_failure(run)]
        ) or None,
        "skipped_providers": _provider_names(
            [run for run in runs if _is_skipped_provider_run(run)]
        ) or None,
        "provider_record_count": sum(
            int(run.get("record_count") or 0)
            for run in primary_successes
            if not isinstance(run.get("record_count"), bool)
        ),
    }
    return {key: value for key, value in details.items() if value is not None}


def _provider_component(
    *,
    key: str,
    label: str,
    data_type: str,
    provider_runs: List[Dict[str, Any]],
) -> RunDiagnosticComponent:
    runs = [
        run for run in provider_runs
        if isinstance(run, dict) and run.get("data_type") == data_type
    ]
    if not runs:
        return _component(key, label, "unknown", f"{label}未记录诊断信息")

    # provider_runs 按完成顺序追加（`record_provider_run_started` 只发实时事件，
    # 不进入该列表），因此列表下标即真实尝试顺序。
    first_success_index = next(
        (index for index, run in enumerate(runs) if run.get("success") is True),
        None,
    )
    last_run = runs[-1]
    if first_success_index is not None:
        success_run = runs[first_success_index]
        provider = success_run.get("provider") or "unknown"
        # 只有“首个成功之前”的真实失败才是 fallback（降级）；首个成功之后的尝试
        # 是 `_supplement_quote` 之类的字段补充，失败不影响本次数据来源与质量。
        preceding_failures = [
            run for run in runs[:first_success_index] if _is_real_provider_failure(run)
        ]
        skipped_providers = _provider_names(
            [run for run in runs if _is_skipped_provider_run(run)]
        )
        cache_hit = success_run.get("cache_hit") is True
        data_date = success_run.get("data_date")
        details = {
            "provider": provider,
            "attempts": len(runs),
            "record_count": success_run.get("record_count"),
            "fallback_to": next(
                (
                    run.get("fallback_to")
                    for run in preceding_failures
                    if run.get("fallback_to")
                ),
                None,
            ),
            "cache_hit": True if cache_hit else None,
            "data_date": data_date,
            "skipped_providers": skipped_providers or None,
        }
        if preceding_failures:
            details["failed_providers"] = _provider_names(preceding_failures)
            details = {
                key_: value for key_, value in details.items() if value is not None
            }
            return _component(
                key,
                label,
                "degraded",
                f"{label}{provider} 成功，前置数据源失败后已继续",
                details,
            )

        details = {key_: value for key_, value in details.items() if value is not None}
        if cache_hit:
            date_text = f"（数据日期 {data_date}）" if data_date else ""
            return _component(
                key,
                label,
                "ok",
                f"{label}来自本地存储缓存{date_text}，本次未请求外部数据源",
                details,
            )
        return _component(
            key,
            label,
            "ok",
            f"{label}{provider} 成功",
            details,
        )

    message = (
        last_run.get("error_message_sanitized")
        or last_run.get("error_type")
        or "所有数据源尝试失败"
    )
    return _component(
        key,
        label,
        "failed",
        f"{label}失败：{message}",
        {
            "attempts": len(runs),
            "provider": last_run.get("provider"),
            "error_type": last_run.get("error_type"),
        },
    )


def _news_component(
    context_snapshot: Dict[str, Any],
    raw_result: Dict[str, Any],
    provider_runs: List[Dict[str, Any]],
) -> RunDiagnosticComponent:
    label = "新闻搜索"
    input_block = _analysis_input_block(context_snapshot, "news")
    input_message = _analysis_input_status_message(input_block)
    has_retrieval_news = "news_retrieval_content" in context_snapshot
    has_snapshot_news = has_retrieval_news or "news_content" in context_snapshot
    provider_details = _news_provider_details(provider_runs)

    news_result_count = context_snapshot.get("news_result_count")
    if not isinstance(news_result_count, int) or isinstance(news_result_count, bool):
        raw_count = raw_result.get("news_result_count")
        news_result_count = (
            raw_count
            if isinstance(raw_count, int) and not isinstance(raw_count, bool)
            else _market_review_news_count(context_snapshot)
        )

    fallback_count = int(provider_details.get("fallback_count") or 0)
    if isinstance(news_result_count, int):
        details = {"record_count": news_result_count, **provider_details}
        if news_result_count > 0:
            if input_message:
                details.update(
                    {
                        "analysis_input_block": "news",
                        "analysis_input_status": input_block.get("status"),
                        "analysis_input_missing_reasons": _list_text(
                            input_block.get("missing_reasons")
                        ),
                        "evidence_scope": "retrieval_vs_analysis_input",
                    }
                )
                return _component(
                    "news",
                    label,
                    "degraded",
                    f"新闻检索返回 {news_result_count} 条结果，但新闻{input_message}；报告页相关资讯可能来自后续检索或历史持久化",
                    details,
                )
            if fallback_count:
                return _component(
                    "news",
                    label,
                    "degraded",
                    f"新闻检索返回 {news_result_count} 条结果，{fallback_count} 次检索在前置数据源失败后降级成功",
                    details,
                )
            return _component(
                "news",
                label,
                "ok",
                f"新闻检索返回 {news_result_count} 条结果",
                details,
            )
        return _component("news", label, "degraded", "新闻搜索无结果", details)

    if provider_details:
        provider_record_count = int(provider_details.get("provider_record_count") or 0)
        if provider_record_count > 0:
            status = "degraded" if fallback_count else "ok"
            message = f"新闻数据源返回 {provider_record_count} 条结果，最终去重条数未记录"
            if fallback_count:
                message += f"；其中 {fallback_count} 次检索发生真实降级"
            return _component("news", label, status, message, provider_details)
        return _component(
            "news",
            label,
            "degraded",
            "新闻数据源已尝试，但未记录可用结果",
            provider_details,
        )

    if input_message:
        return _component(
            "news",
            label,
            "unknown",
            f"新闻{input_message}；报告页相关资讯可能来自后续检索或历史持久化",
            {
                "analysis_input_block": "news",
                "analysis_input_status": input_block.get("status"),
                "analysis_input_missing_reasons": _list_text(
                    input_block.get("missing_reasons")
                ),
                "evidence_scope": "analysis_input_only",
            },
        )
    if has_snapshot_news and not has_retrieval_news:
        return _component("news", label, "unknown", "新闻检索未记录原始证据，可能未尝试或未启用")
    return _component("news", label, "unknown", "新闻搜索未记录诊断信息")


def _llm_component(diagnostics: Dict[str, Any], raw_result: Dict[str, Any]) -> RunDiagnosticComponent:
    label = "LLM"
    runs = [
        run for run in _as_list(diagnostics.get("llm_runs"))
        if isinstance(run, dict)
    ]
    if runs:
        successes = [run for run in runs if run.get("success") is True]
        failures = [run for run in runs if run.get("success") is False]
        last_run = runs[-1]
        if successes:
            success_run = successes[-1]
            model = success_run.get("model") or raw_result.get("model_used") or "unknown"
            status = "degraded" if failures or success_run.get("fallback_model") else "ok"
            message = f"LLM {model} 成功"
            if status == "degraded":
                message = f"LLM {model} 成功，期间发生过失败或模型切换"
            return _component(
                "llm",
                label,
                status,
                message,
                {
                    "model": model,
                    "tokens": success_run.get("tokens"),
                    "duration_ms": success_run.get("duration_ms"),
                    "fallback_model": success_run.get("fallback_model"),
                },
            )
        return _component(
            "llm",
            label,
            "failed",
            f"LLM 失败：{last_run.get('error_message_sanitized') or last_run.get('error_type') or '未知错误'}",
            {"model": last_run.get("model"), "error_type": last_run.get("error_type")},
        )

    if raw_result:
        if raw_result.get("success") is False:
            return _component(
                "llm",
                label,
                "failed",
                f"LLM 失败：{sanitize_diagnostic_text(raw_result.get('error_message')) or '未知错误'}",
            )
        model = raw_result.get("model_used")
        if model:
            return _component("llm", label, "ok", f"LLM {model} 成功", {"model": model})
        if raw_result.get("analysis_summary"):
            return _component("llm", label, "ok", "LLM 成功，模型未记录")
    return _component("llm", label, "unknown", "LLM 未记录诊断信息")


def _notification_component(diagnostics: Dict[str, Any]) -> RunDiagnosticComponent:
    label = "通知"
    runs = [
        run for run in _as_list(diagnostics.get("notification_runs"))
        if isinstance(run, dict)
    ]
    if not runs:
        return _component("notification", label, "unknown", "通知结果未记录")

    skipped = [run for run in runs if run.get("status") in {"skipped", "not_configured"}]
    successes = [run for run in runs if run.get("success") is True]
    failures = [run for run in runs if run.get("success") is False and run not in skipped]
    channels = [run.get("channel") for run in runs if run.get("channel")]
    if successes and failures:
        return _component(
            "notification",
            label,
            "degraded",
            "部分通知渠道失败，其余渠道已发送",
            {"channels": channels, "failed": [run.get("channel") for run in failures]},
        )
    if successes:
        return _component(
            "notification",
            label,
            "ok",
            "通知发送成功",
            {"channels": channels},
        )
    if skipped and not failures:
        status = "not_configured" if any(run.get("status") == "not_configured" for run in skipped) else "skipped"
        return _component(
            "notification",
            label,
            status,
            "通知未配置或本次跳过",
            {"channels": channels},
        )
    last_failure = failures[-1] if failures else runs[-1]
    return _component(
        "notification",
        label,
        "failed",
        f"通知失败：{last_failure.get('error_message_sanitized') or last_failure.get('status') or '未知错误'}",
        {"channels": channels},
    )


def _history_component(
    diagnostics: Dict[str, Any],
    report_saved: Optional[bool],
) -> RunDiagnosticComponent:
    label = "历史保存"
    runs = [
        run for run in _as_list(diagnostics.get("history_runs"))
        if isinstance(run, dict)
    ]
    if runs:
        last_run = runs[-1]
        if last_run.get("report_saved") is True:
            return _component(
                "history",
                label,
                "ok",
                "报告历史已保存",
                {"analysis_history_id": last_run.get("analysis_history_id")},
            )
        return _component(
            "history",
            label,
            "failed",
            f"报告历史保存失败：{last_run.get('error_message_sanitized') or '未知错误'}",
        )
    if report_saved is True:
        return _component("history", label, "ok", "报告历史已保存")
    if report_saved is False:
        return _component("history", label, "failed", "报告历史保存失败")
    return _component("history", label, "unknown", "历史保存未记录诊断信息")


def build_run_diagnostic_summary(
    *,
    context_snapshot: Optional[Any] = None,
    raw_result: Optional[Any] = None,
    report_saved: Optional[bool] = None,
    query_id: Optional[str] = None,
    stock_code: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a user-facing diagnostic summary from persisted or in-memory evidence."""
    snapshot = _as_dict(context_snapshot)
    raw = _as_dict(raw_result)
    diagnostics = _as_dict(snapshot.get("diagnostics"))
    provider_runs = [
        run for run in _as_list(diagnostics.get("provider_runs"))
        if isinstance(run, dict)
    ]
    llm_runs = [
        run for run in _as_list(diagnostics.get("llm_runs"))
        if isinstance(run, dict)
    ]

    daily_data_component = _provider_component(
        key="daily_data",
        label="日线数据",
        data_type="daily_data",
        provider_runs=provider_runs,
    )
    components = {
        "analysis_input": _analysis_input_component(snapshot),
        "realtime_quote": _provider_component(
            key="realtime_quote",
            label="实时行情",
            data_type="realtime_quote",
            provider_runs=provider_runs,
        ),
        "daily_data": _reconcile_daily_provider_with_analysis_input(
            daily_data_component,
            snapshot,
        ),
        "news": _news_component(snapshot, raw, provider_runs),
        "llm": _llm_component(diagnostics, raw),
        "notification": _notification_component(diagnostics),
        "history": _history_component(diagnostics, report_saved),
    }

    has_evidence = bool(snapshot or raw or diagnostics or report_saved is not None)
    has_core_diagnostic_runs = bool(provider_runs or llm_runs)
    if not has_evidence or not diagnostics:
        status = "unknown"
    elif components["llm"].status == "failed" or components["history"].status == "failed":
        status = "failed"
    elif any(component.status in {"failed", "degraded"} for component in components.values()):
        status = "degraded"
    elif all(component.status == "unknown" for component in components.values()):
        status = "unknown"
    elif not has_core_diagnostic_runs:
        status = "unknown"
    else:
        status = "normal"

    if status == "unknown":
        reason = "旧报告或诊断证据不足，无法判断本次运行状态"
    else:
        reason = next(
            (
                component.message
                for component in components.values()
                if component.status == "failed"
            ),
            next(
                (
                    component.message
                    for component in components.values()
                    if component.status == "degraded"
                ),
                _SUMMARY_STATUS_LABELS[status],
            ),
        )

    trace_id = diagnostics.get("trace_id") or snapshot.get("trace_id") or raw.get("trace_id")
    resolved_query_id = query_id or diagnostics.get("query_id") or snapshot.get("query_id") or raw.get("query_id")
    resolved_stock_code = (
        stock_code
        or diagnostics.get("stock_code")
        or snapshot.get("stock_code")
        or raw.get("code")
        or raw.get("stock_code")
    )

    return RunDiagnosticSummary(
        trace_id=trace_id,
        task_id=diagnostics.get("task_id"),
        query_id=resolved_query_id,
        stock_code=resolved_stock_code,
        trigger_source=diagnostics.get("trigger_source") or snapshot.get("trigger_source"),
        status=status,
        status_label=_SUMMARY_STATUS_LABELS[status],
        reason=reason,
        components=components,
    ).to_dict()


def format_copyable_diagnostics(summary: Dict[str, Any]) -> str:
    """Format a sanitized plain-text diagnostic payload for issue reports."""
    components = _as_dict(summary.get("components"))

    def _component_line(key: str) -> str:
        component = _as_dict(components.get(key))
        message = sanitize_diagnostic_text(component.get("message"), max_length=160) or "unknown"
        return f"{key}: {component.get('status', 'unknown')} - {message}"

    lines = [
        f"trace_id: {summary.get('trace_id') or 'unknown'}",
        f"query_id: {summary.get('query_id') or 'unknown'}",
        f"stock_code: {summary.get('stock_code') or 'unknown'}",
        f"trigger_source: {summary.get('trigger_source') or 'unknown'}",
        f"data_status: {summary.get('status', 'unknown')}",
        _component_line("realtime_quote"),
        _component_line("daily_data"),
        _component_line("news"),
        _component_line("llm"),
        _component_line("notification"),
        _component_line("history"),
        f"reason: {sanitize_diagnostic_text(summary.get('reason'), max_length=160) or 'unknown'}",
    ]
    return "\n".join(lines)
