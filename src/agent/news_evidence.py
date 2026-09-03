# -*- coding: utf-8 -*-
"""Agent 运行期实际消费的新闻证据计数与可核验内容。

empty-news disclosure 必须反映「本次分析真正用到的新闻证据」。Agent 模式下情报
由 Agent 自己调用搜索工具取得，因此计数只能来自这些工具的真实返回，不能用分析
结束后补打的一次 `search_stock_news()` 代替：那次补查与 Agent 消费的证据无关，
两个方向都会失真——Agent 明明用了新闻却因补查失败而被标成「未纳入新闻面证据」，
或 Agent 没拿到新闻却因补查有结果而被错误地不提示。

用法：pipeline 在 `executor.run()` 前后开启并读取作用域；搜索工具在返回结果时
记录本次真正交给 Agent 的条数。工具在 ThreadPoolExecutor 中执行，
`src/agent/runner.py` 通过 `contextvars.copy_context()` 提交任务，因此 ContextVar
中的**可变**累加器在工作线程与父线程之间是同一个对象，工具线程里的累加对
pipeline 可见。请勿把它换成保存不可变值的 ContextVar，那样父线程读不到。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Mapping
from contextvars import ContextVar, Token
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)


class NewsEvidenceAccumulator:
    """线程安全地累计 Agent 实际消费的新闻条数和可核验内容。"""

    _MAX_TEXT_LENGTHS = {
        "title": 500,
        "snippet": 2000,
        "source": 200,
        "published_date": 100,
        "dimension": 100,
        "provider": 100,
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = 0
        self._attempted = False
        self._records: List[Dict[str, Optional[str]]] = []

    @classmethod
    def _safe_text(cls, value: Any, field: str) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return text[: cls._MAX_TEXT_LENGTHS.get(field, 500)]

    @staticmethod
    def _safe_url(value: Any) -> Optional[str]:
        """Keep a verifiable URL while dropping query/fragment credentials."""
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = urlsplit(text)
        except ValueError:
            return None
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))[:2000]

    @classmethod
    def _normalize_record(
        cls,
        item: Any,
        *,
        dimension: Optional[str],
        provider: Optional[str],
    ) -> Optional[Dict[str, Optional[str]]]:
        if isinstance(item, Mapping):
            getter = item.get
        else:
            getter = lambda key, default=None: getattr(item, key, default)
        title = cls._safe_text(getter("title"), "title")
        snippet = cls._safe_text(getter("snippet"), "snippet")
        source = cls._safe_text(getter("source"), "source")
        published_date = cls._safe_text(getter("published_date"), "published_date")
        url = cls._safe_url(getter("url"))
        if not any((title, snippet, source, published_date, url)):
            return None
        return {
            "dimension": cls._safe_text(dimension, "dimension"),
            "provider": cls._safe_text(provider, "provider"),
            "title": title,
            "snippet": snippet,
            "source": source,
            "published_date": published_date,
            "url": url,
        }

    def record(
        self,
        count: int,
        *,
        results: Optional[Iterable[Any]] = None,
        dimension: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        """记录一次工具调用；零命中也标记为已执行检索。"""
        try:
            value = int(count)
        except (TypeError, ValueError):
            value = 0
        if value < 0:
            value = 0
        normalized = [
            record
            for item in (results or [])
            if (record := self._normalize_record(
                item,
                dimension=dimension,
                provider=provider,
            )) is not None
        ]
        with self._lock:
            self._attempted = True
            self._total += value
            self._records.extend(normalized)

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    @property
    def attempted(self) -> bool:
        with self._lock:
            return self._attempted

    @property
    def records(self) -> List[Dict[str, Optional[str]]]:
        with self._lock:
            return [dict(item) for item in self._records]

    def render_context(self) -> Optional[str]:
        """Render only evidence actually returned to Agent tools."""
        records = self.records
        if not records:
            return None
        lines = ["## Agent 运行期新闻证据"]
        for index, item in enumerate(records, start=1):
            title = item.get("title") or "（无标题）"
            qualifiers = [
                value
                for value in (
                    item.get("dimension"),
                    item.get("provider"),
                    item.get("source"),
                    item.get("published_date"),
                )
                if value
            ]
            suffix = f" [{' | '.join(qualifiers)}]" if qualifiers else ""
            lines.append(f"{index}. {title}{suffix}")
            if item.get("snippet"):
                lines.append(f"   摘要：{item['snippet']}")
            if item.get("url"):
                lines.append(f"   URL：{item['url']}")
        return "\n".join(lines)

    def resolve(self, *, search_available: bool) -> Optional[int]:
        """收敛成 `news_result_count` 的三态语义。

        - 搜索渠道不可用：`None`，即未执行检索，披露「未配置搜索渠道」。
        - 渠道可用：从 `0` 起步，Agent 调用工具拿到多少算多少。

        渠道可用但 Agent 一次都没搜的情况刻意归入 `0` 而不是 `None`：此时报告
        确实没有新闻证据，但「未配置搜索渠道」是与事实相反的解释，而「未获取到
        可用的新闻面数据」在这两种子情形下都成立。
        """
        if not search_available:
            return None
        return self.total


_CURRENT_ACCUMULATOR: ContextVar[Optional[NewsEvidenceAccumulator]] = ContextVar(
    "news_evidence_accumulator",
    default=None,
)


def activate_news_evidence_scope() -> Token:
    """开启一次 Agent 运行的证据作用域，返回重置令牌。"""
    return _CURRENT_ACCUMULATOR.set(NewsEvidenceAccumulator())


def get_current_news_evidence() -> Optional[NewsEvidenceAccumulator]:
    return _CURRENT_ACCUMULATOR.get()


def reset_news_evidence_scope(token: Optional[Token]) -> None:
    if token is None:
        return
    try:
        _CURRENT_ACCUMULATOR.reset(token)
    except Exception as exc:  # pragma: no cover - 防御性 fail-open
        logger.warning("news evidence scope reset failed: %s", exc)


def record_news_evidence(
    count: int,
    *,
    results: Optional[Iterable[Any]] = None,
    dimension: Optional[str] = None,
    provider: Optional[str] = None,
) -> None:
    """供 Agent 搜索工具记录实际返回的条数与可核验内容。

    没有活动作用域时静默忽略：非 Agent 路径自己直接维护计数，工具也可能在
    Agent 分析之外被调用（例如报告页的后续资讯检索），那些都不应影响本次分析的
    披露判定。
    """
    accumulator = _CURRENT_ACCUMULATOR.get()
    if accumulator is None:
        return
    accumulator.record(
        count,
        results=results,
        dimension=dimension,
        provider=provider,
    )
