# -*- coding: utf-8 -*-
"""
Shared data parsing and normalization helpers.
"""

import json
import math
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Tuple


_MODEL_PLACEHOLDER_VALUES = {"unknown", "error", "none", "null", "n/a"}
SIGNAL_ATTRIBUTION_WEIGHT_KEYS: Tuple[str, ...] = (
    "technical_indicators",
    "news_sentiment",
    "fundamentals",
    "market_conditions",
)
SIGNAL_ATTRIBUTION_SIGNAL_KEYS: Tuple[str, ...] = (
    "strongest_bullish_signal",
    "strongest_bearish_signal",
)


def normalize_model_used(value: Any) -> Optional[str]:
    """Normalize placeholder/empty model values to None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in _MODEL_PLACEHOLDER_VALUES:
        return None
    return text


def parse_json_field(value: Any) -> Any:
    """Best-effort JSON parse for string values; passthrough for others."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return value
    return value


def _non_empty_dict(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    return value if value else None


def _normalize_belong_boards(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if name is None:
            continue
        name_text = str(name).strip()
        if not name_text:
            continue
        board = {"name": name_text}
        if item.get("code") is not None:
            code_text = str(item.get("code")).strip()
            if code_text:
                board["code"] = code_text
        if item.get("type") is not None:
            type_text = str(item.get("type")).strip()
            if type_text:
                board["type"] = type_text
        normalized.append(board)
    return normalized


def _safe_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.endswith("%"):
                text = text[:-1].strip()
            numeric_value = float(text)
        else:
            numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    return numeric_value if math.isfinite(numeric_value) else None


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_sector_ranking_items(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if name is None:
            continue
        name_text = str(name).strip()
        if not name_text:
            continue
        ranking_item: Dict[str, Any] = {"name": name_text}
        for optional_field in ("code", "source", "updated_at"):
            if item.get(optional_field) is not None:
                optional_text = str(item.get(optional_field)).strip()
                if optional_text:
                    ranking_item[optional_field] = optional_text
        change_pct = _safe_float(item.get("change_pct"))
        if change_pct is not None:
            ranking_item["change_pct"] = change_pct
        rank = _safe_int(item.get("rank"))
        if rank is not None:
            ranking_item["rank"] = rank
        normalized.append(ranking_item)
    return normalized


def _normalize_sector_rankings(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None

    status = value.get("status")
    status_text = status.strip().lower() if isinstance(status, str) else None
    normalized: Dict[str, Any] = {
        "top": _normalize_sector_ranking_items(value.get("top")),
        "bottom": _normalize_sector_ranking_items(value.get("bottom")),
    }
    if status_text:
        normalized["status"] = status_text
    return normalized


def _extract_ranking_payload_from_block(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    if "top" in value or "bottom" in value:
        return value

    status = value.get("status")
    if not isinstance(status, str):
        status_is_valid = status is None
    else:
        status_is_valid = status.strip().lower() in {"ok", "partial"}
    if not status_is_valid:
        return None
    data = value.get("data")
    if isinstance(data, dict):
        payload = dict(data)
        if isinstance(status, str):
            payload["status"] = status.strip().lower()
        return payload
    return None


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    # NumPy scalars/arrays do not consistently implement Python's ``Real`` or
    # sequence protocols. Normalize them before deciding whether an overlay is
    # meaningful so an all-NaN ndarray cannot replace a finite fallback list.
    if type(value).__module__.startswith("numpy"):
        if hasattr(value, "tolist"):
            normalized = value.tolist()
        elif hasattr(value, "item"):
            normalized = value.item()
        else:
            normalized = value
        if normalized is not value:
            return _is_empty_value(normalized)
    if isinstance(value, Real):
        return not math.isfinite(float(value))
    if isinstance(value, str):
        stripped = value.strip()
        return not stripped or stripped.lower() in {
            "nan",
            "inf",
            "+inf",
            "-inf",
            "infinity",
            "+infinity",
            "-infinity",
        }
    if isinstance(value, (list, tuple)):
        return len(value) == 0 or all(_is_empty_value(item) for item in value)
    if isinstance(value, dict):
        return len(value) == 0
    return False


def _deep_merge_dicts(*values: Any) -> Optional[Dict[str, Any]]:
    merged: Dict[str, Any] = {}
    has_value = False
    for value in values:
        obj = parse_json_field(value)
        if not isinstance(obj, dict):
            continue
        has_value = True
        for key, item in obj.items():
            if _is_empty_value(item):
                continue
            existing = merged.get(key)
            if isinstance(existing, dict) and isinstance(item, dict):
                nested = _deep_merge_dicts(existing, item)
                if nested:
                    merged[key] = nested
            else:
                merged[key] = item
    return merged if has_value else None


def extract_fundamental_context(
    context_snapshot: Any,
    fallback_fundamental_payload: Any = None,
) -> Optional[Dict[str, Any]]:
    """
    Resolve fundamental_context from context snapshot, with optional fallback payload.
    """
    fallback_obj = parse_json_field(fallback_fundamental_payload)
    top_level_fundamental = None
    enhanced_fundamental = None
    snapshot_obj = parse_json_field(context_snapshot)
    if isinstance(snapshot_obj, dict):
        enhanced = snapshot_obj.get("enhanced_context")
        if isinstance(enhanced, dict):
            fundamental = enhanced.get("fundamental_context")
            if isinstance(fundamental, dict):
                enhanced_fundamental = fundamental
        raw_top_level = snapshot_obj.get("fundamental_context")
        if isinstance(raw_top_level, dict):
            top_level_fundamental = raw_top_level

    merged = _deep_merge_dicts(
        fallback_obj,
        top_level_fundamental,
        enhanced_fundamental,
    )
    normalized = normalize_json_safe_value(merged)
    return normalized if isinstance(normalized, dict) else None


_NULLISH_DISPLAY_TOKENS = {
    "none",
    "null",
    "nan",
    "inf",
    "+inf",
    "-inf",
    "infinity",
    "+infinity",
    "-infinity",
    "n/a",
    "na",
    "-",
    "--",
    "—",
}
_NONFINITE_JSON_STRING_TOKENS = {
    "nan",
    "inf",
    "+inf",
    "-inf",
    "infinity",
    "+infinity",
    "-infinity",
}


def normalize_json_safe_value(value: Any) -> Any:
    """Recursively convert report/API payload values to strict JSON types.

    Financial providers commonly return pandas/NumPy scalars and may carry
    NaN/Infinity sentinels. Python's default JSON encoder accepts those floats,
    but they are not valid JSON and can leak through API responses. This helper
    preserves finite values and booleans, converts tuples/arrays to lists, and
    maps non-finite numbers (including exact string sentinels) to ``None``.
    """
    if isinstance(value, dict):
        return {
            key: normalize_json_safe_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [normalize_json_safe_value(item) for item in value]
    if isinstance(value, bool):
        return value
    if type(value).__module__.startswith("numpy"):
        if hasattr(value, "tolist"):
            normalized = value.tolist()
        elif hasattr(value, "item"):
            normalized = value.item()
        else:
            normalized = value
        if normalized is not value:
            return normalize_json_safe_value(normalized)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric_value = float(value)
        return numeric_value if math.isfinite(numeric_value) else None
    if isinstance(value, str) and value.strip().lower() in _NONFINITE_JSON_STRING_TOKENS:
        return None
    return value


def display_value_or_na(value: Any, placeholder: str = "N/A") -> str:
    """Render a dashboard/report field, mapping missing values to a placeholder.

    ``dict.get(key, "N/A")`` only fires when the key is *absent*. The decision
    dashboard is produced by the model and routinely carries an explicit ``null``
    when a provider cannot supply a metric（例如部分美股或 ETF 缺少流通股本时无法
    计算换手率），所以默认值永远不会生效，报告里会直接渲染出字面量 ``None``。

    因此展示层统一走本函数：``None`` / 非有限浮点数 / 空串 / 以及模型可能吐出的
    ``"none"``、``"null"`` 等占位字符串都渲染为 ``placeholder``，其余原样输出。
    """
    if value is None:
        return placeholder
    if isinstance(value, Real) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return placeholder
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in _NULLISH_DISPLAY_TOKENS:
            return placeholder
        return text
    return str(value)


def display_numeric_with_suffix(
    value: Any,
    suffix: str,
    placeholder: str = "N/A",
) -> str:
    """Append a unit only when *value* is a real finite number.

    Model-produced dashboard fields occasionally contain explanatory text such as
    ``"数据缺失，无法判断"`` despite being nominally numeric. Unconditionally
    appending ``%`` or ``/100`` turns that into misleading output such as
    ``"数据缺失，无法判断%"``; missing values similarly became ``"N/A%"``.
    Numeric values and numeric strings keep their original display text and receive
    the suffix exactly once. Missing or non-numeric explanatory values are rendered
    without a unit.
    """
    rendered = display_value_or_na(value, placeholder)
    if rendered == placeholder or isinstance(value, bool):
        return rendered

    numeric_text = rendered
    if suffix and numeric_text.endswith(suffix):
        numeric_text = numeric_text[: -len(suffix)].rstrip()

    try:
        numeric_value = float(numeric_text.replace(",", ""))
    except (TypeError, ValueError):
        return rendered
    if not math.isfinite(numeric_value):
        return placeholder
    return f"{numeric_text}{suffix}"


def display_fraction_as_percent(
    value: Any,
    precision: int = 0,
    placeholder: str = "N/A",
) -> str:
    """Render a finite 0-1 fraction as a percentage for report boundaries.

    Numeric strings are accepted. A string already carrying ``%`` is treated as
    percentage points rather than multiplied again. Explanatory text is preserved
    without an appended unit; missing and non-finite values use ``placeholder``.
    """
    rendered = display_value_or_na(value, placeholder)
    if rendered == placeholder or isinstance(value, bool):
        return rendered

    has_percent_suffix = rendered.endswith("%")
    numeric_text = rendered[:-1].strip() if has_percent_suffix else rendered
    try:
        numeric_value = float(numeric_text.replace(",", ""))
    except (TypeError, ValueError):
        return rendered
    if not math.isfinite(numeric_value):
        return placeholder
    percentage = numeric_value if has_percent_suffix else numeric_value * 100
    return f"{percentage:.{max(0, int(precision))}f}%"


def _snapshot_official_daily_bar(snapshot_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the official (non-estimated) daily bar stored in a snapshot, if any."""
    enhanced = snapshot_obj.get("enhanced_context")
    if not isinstance(enhanced, dict):
        return None
    today = enhanced.get("today")
    if not isinstance(today, dict) or not today:
        return None
    # 实时估算 bar 不是官方日线，不能拿来纠正实时报价。
    if today.get("is_estimated"):
        return None
    return today


def _snapshot_is_confirmed_non_trading_day(snapshot_obj: Dict[str, Any]) -> bool:
    """Only an explicitly persisted ``is_trading_day is False`` counts (fail open)."""
    summary = snapshot_obj.get("market_phase_summary")
    return isinstance(summary, dict) and summary.get("is_trading_day") is False


def extract_realtime_detail_fields(context_snapshot: Any) -> Dict[str, Any]:
    """
    Extract stable realtime price/change fields from persisted context snapshots.

    Supports both the standard `enhanced_context.realtime` layout and the
    agent-mode top-level `realtime_quote` compatibility shape.

    非交易日校正：这两个字段是 API `meta.current_price` / `meta.change_pct` 的取数来源，
    也就是历史列表卡片和报告表头显示的数字。行情快照在收盘后仍会返回一份实时报价，
    其 `change_pct` 由 provider 自带的 `pre_close` 反推，而该 `pre_close` 可能落后一个
    交易日，于是同一份报告的表头和正文会给出两个涨跌幅
    （实例：MSFT 2026-08-28，provider `pre_close=503.09` → `+2.08%`，
    官方日线前收 505.06 → `+1.68%`）。因此当快照中的 `market_phase_summary`
    明确判定 `is_trading_day is False` 时，改用官方日线的 `close` / `pct_chg`，
    与 pipeline 侧 `resolve_result_price_change` 保持同一语义；phase 未知或
    官方日线缺字段时保持失败开放，沿用实时报价。
    """
    snapshot_obj = parse_json_field(context_snapshot)
    if not isinstance(snapshot_obj, dict):
        return {"current_price": None, "change_pct": None}

    current_price_candidates: List[Any] = []
    change_pct_candidates: List[Any] = []

    enhanced = snapshot_obj.get("enhanced_context")
    if isinstance(enhanced, dict):
        realtime = enhanced.get("realtime")
        if isinstance(realtime, dict):
            current_price_candidates.append(realtime.get("price"))
            change_pct_candidates.extend(
                (realtime.get("change_pct"), realtime.get("pct_chg"))
            )

    for field in ("realtime_quote_raw", "realtime_quote"):
        realtime_payload = snapshot_obj.get(field)
        if not isinstance(realtime_payload, dict):
            continue
        current_price_candidates.append(realtime_payload.get("price"))
        change_pct_candidates.extend(
            (realtime_payload.get("change_pct"), realtime_payload.get("pct_chg"))
        )

    current_price = next(
        (
            numeric_value
            for value in current_price_candidates
            if (numeric_value := _safe_float(value)) is not None
        ),
        None,
    )
    change_pct = next(
        (
            numeric_value
            for value in change_pct_candidates
            if (numeric_value := _safe_float(value)) is not None
        ),
        None,
    )

    if _snapshot_is_confirmed_non_trading_day(snapshot_obj):
        official = _snapshot_official_daily_bar(snapshot_obj)
        if official is not None:
            official_close = _safe_float(official.get("close"))
            official_change_pct = _safe_float(official.get("pct_chg"))
            # Invalid official fields must not erase an otherwise valid quote.
            if official_close is not None:
                current_price = official_close
            if official_change_pct is not None:
                change_pct = official_change_pct

    return {
        "current_price": current_price,
        "change_pct": change_pct,
    }


def extract_fundamental_detail_fields(
    context_snapshot: Any,
    fallback_fundamental_payload: Any = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Extract stable API-facing financial and dividend blocks from fundamental_context.
    """
    fundamental_ctx = extract_fundamental_context(
        context_snapshot=context_snapshot,
        fallback_fundamental_payload=fallback_fundamental_payload,
    )
    if not isinstance(fundamental_ctx, dict):
        return {"financial_report": None, "dividend_metrics": None}

    earnings_block = fundamental_ctx.get("earnings")
    earnings_data = earnings_block.get("data") if isinstance(earnings_block, dict) else None
    if not isinstance(earnings_data, dict):
        return {"financial_report": None, "dividend_metrics": None}

    financial_report = _non_empty_dict(earnings_data.get("financial_report"))
    dividend_metrics = _non_empty_dict(earnings_data.get("dividend"))
    return {
        "financial_report": financial_report,
        "dividend_metrics": dividend_metrics,
    }


def extract_board_detail_fields(
    context_snapshot: Any,
    fallback_fundamental_payload: Any = None,
) -> Dict[str, Any]:
    """
    Extract stable board detail fields from fundamental_context.
    """
    fundamental_ctx = extract_fundamental_context(
        context_snapshot=context_snapshot,
        fallback_fundamental_payload=fallback_fundamental_payload,
    )
    if not isinstance(fundamental_ctx, dict):
        return {"belong_boards": [], "sector_rankings": None, "concept_rankings": None}

    boards_block = fundamental_ctx.get("boards")
    sector_rankings = _extract_ranking_payload_from_block(boards_block)
    concept_rankings = (
        _extract_ranking_payload_from_block(fundamental_ctx.get("concept_boards"))
        or _extract_ranking_payload_from_block(fundamental_ctx.get("concepts"))
        or _extract_ranking_payload_from_block(fundamental_ctx.get("concept_rankings"))
    )
    if concept_rankings is None and isinstance(sector_rankings, dict):
        concept_rankings = _extract_ranking_payload_from_block(sector_rankings.get("concepts"))
    return {
        "belong_boards": _normalize_belong_boards(fundamental_ctx.get("belong_boards")),
        "sector_rankings": _normalize_sector_rankings(sector_rankings),
        "concept_rankings": _normalize_sector_rankings(concept_rankings),
    }


def extract_market_structure_detail_field(
    context_snapshot: Any,
    fallback_raw_result_payload: Any = None,
) -> Optional[Dict[str, Any]]:
    """Extract the stable market_structure detail payload from persisted payloads."""
    snapshot_obj = parse_json_field(context_snapshot)

    candidates = []
    if isinstance(snapshot_obj, dict):
        candidates.append(snapshot_obj.get("market_structure_context"))
        enhanced = snapshot_obj.get("enhanced_context")
        if isinstance(enhanced, dict):
            candidates.append(enhanced.get("market_structure_context"))

    raw_result_obj = parse_json_field(fallback_raw_result_payload)
    if isinstance(raw_result_obj, dict):
        candidates.append(raw_result_obj.get("market_structure_context"))

    for candidate in candidates:
        payload = parse_json_field(candidate)
        if not isinstance(payload, dict):
            continue
        if payload.get("schema_version") != "market-structure-v1":
            continue
        if not isinstance(payload.get("market_theme_context"), dict):
            continue
        if not isinstance(payload.get("stock_market_position"), dict):
            continue
        return payload
    return None


def normalize_signal_attribution_values(signal_attr: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Normalize signal_attribution values in-place.

    - Convert string percentages like "70%" to int 70
    - Convert "N/A", "N/A%", "" to None
    - Clamp negative numbers to 0
    - Normalize four non-zero contributions to sum = 100 (only if all four are valid numbers)
    - Preserve all-zero as "no effective signal"; do not fake 25/25/25/25
    """
    if not isinstance(signal_attr, dict):
        return None

    def _parse_contribution(raw: Any) -> Optional[float]:
        """
        Parse a single contribution value.

        Returns:
            - float in [0, 100] if valid
            - None if unparsable / N/A / empty
        """
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            val = float(raw)
            if not math.isfinite(val):
                return None
            # Clamp to [0, 100] — values outside this range are invalid
            return max(0.0, min(100.0, val))
        if isinstance(raw, str):
            text = raw.strip()
            if not text or text in ("N/A", "N/A%"):
                return None
            text = text.rstrip("%").strip()
            try:
                val = float(text)
                if not math.isfinite(val):
                    return None
                # Clamp to [0, 100]
                return max(0.0, min(100.0, val))
            except ValueError:
                return None
        return None

    parsed: Dict[str, Optional[float]] = {}
    for k in SIGNAL_ATTRIBUTION_WEIGHT_KEYS:
        parsed[k] = _parse_contribution(signal_attr.get(k))

    valid_values = [v for v in parsed.values() if v is not None]
    if len(valid_values) == 4:
        total = sum(valid_values)
        if total > 0:
            normalized = [(v / total) * 100.0 for v in valid_values]
            int_values = [round(v) for v in normalized]
            diff = 100 - sum(int_values)
            if diff != 0:
                max_idx = int_values.index(max(int_values))
                int_values[max_idx] += diff
            for i, k in enumerate(SIGNAL_ATTRIBUTION_WEIGHT_KEYS):
                parsed[k] = int_values[i]
        # else: total == 0 → all contributions are 0
        #   Keep as 0 (truthful "no contribution"), do NOT fake 25/25/25/25
        #   If LLM returned None for some fields, they stay None (meaning "unable to judge")

    for k in SIGNAL_ATTRIBUTION_WEIGHT_KEYS:
        signal_attr[k] = parsed[k]

    for k in SIGNAL_ATTRIBUTION_SIGNAL_KEYS:
        v = signal_attr.get(k)
        if v is not None and isinstance(v, str) and v.strip() == "":
            signal_attr[k] = None

    return signal_attr


def normalize_dashboard_signal_attribution(dashboard: Optional[Dict[str, Any]]) -> None:
    """Normalize signal_attribution in dashboard dict (in-place)."""
    if not isinstance(dashboard, dict):
        return
    signal_attr = dashboard.get("signal_attribution")
    if signal_attr is not None:
        if not isinstance(signal_attr, dict):
            dashboard.pop("signal_attribution", None)
            return
        normalize_signal_attribution_values(signal_attr)


def normalize_report_signal_attribution(payload: Optional[Dict[str, Any]]) -> None:
    """Normalize signal attribution in either a dashboard dict or full report dict."""
    if not isinstance(payload, dict):
        return
    normalize_dashboard_signal_attribution(payload)
    dashboard = payload.get("dashboard")
    if isinstance(dashboard, dict):
        normalize_dashboard_signal_attribution(dashboard)


def signal_attribution_weight_items(signal_attr: Any) -> List[Tuple[str, int]]:
    """Return displayable attribution weights as (key, integer percent) pairs."""
    if not isinstance(signal_attr, dict):
        return []
    items: List[Tuple[str, int]] = []
    for key in SIGNAL_ATTRIBUTION_WEIGHT_KEYS:
        value = signal_attr.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            number = float(value)
            if math.isfinite(number):
                items.append((key, int(round(number))))
    return items


def signal_attribution_has_content(signal_attr: Any) -> bool:
    """Whether a signal_attribution payload has anything meaningful to render."""
    if not isinstance(signal_attr, dict):
        return False
    if any(value != 0 for _, value in signal_attribution_weight_items(signal_attr)):
        return True
    return any(bool(signal_attr.get(key)) for key in SIGNAL_ATTRIBUTION_SIGNAL_KEYS)
