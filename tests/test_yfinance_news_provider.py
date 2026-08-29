# -*- coding: utf-8 -*-
"""Yahoo Finance news fallback provider."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from src.search_service import YFinanceNewsProvider  # noqa: E402


def _item(title, *, age_days=0, url="https://example.com/a", summary="s", source="Yahoo"):
    published = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat().replace(
        "+00:00", "Z"
    )
    return {
        "id": title,
        "content": {
            "title": title,
            "summary": summary,
            "description": f"<p>{summary}</p>",
            "pubDate": published,
            "provider": {"displayName": source},
            "canonicalUrl": {"url": url},
        },
    }


class _FakeYF:
    """Stands in for the yfinance module."""

    def __init__(self, news_by_ticker):
        self._news = news_by_ticker
        self.asked = []

    def Ticker(self, symbol):  # noqa: N802 - mirrors the yfinance API
        self.asked.append(symbol)
        news = self._news.get(symbol, [])
        if isinstance(news, Exception):
            raise news
        return SimpleNamespace(news=news)


@pytest.fixture
def fake_yf(monkeypatch):
    def _install(news_by_ticker):
        fake = _FakeYF(news_by_ticker)
        monkeypatch.setitem(sys.modules, "yfinance", fake)
        return fake

    return _install


# --- symbol resolution ------------------------------------------------------


def test_us_ticker_is_used_as_is(fake_yf):
    fake = fake_yf({"NVDA": [_item("N1")]})
    response = YFinanceNewsProvider().search("q", stock_code="NVDA")

    assert fake.asked == ["NVDA"]
    assert response.success is True
    assert [r.title for r in response.results] == ["N1"]


def test_market_sentinel_expands_to_region_indices(fake_yf):
    fake = fake_yf({"^GSPC": [_item("S1")], "^IXIC": [_item("S2", url="https://e.com/b")]})
    response = YFinanceNewsProvider().search("q", stock_code="market", region="us")

    # The market-review path has no symbol, so it must fan out to the indices
    # that actually carry market-wide headlines.
    assert fake.asked == ["^GSPC", "^IXIC"]
    assert {r.title for r in response.results} == {"S1", "S2"}


def test_missing_region_defaults_to_us(fake_yf):
    fake = fake_yf({})
    YFinanceNewsProvider().search("q", stock_code="market", region=None)
    assert fake.asked == ["^GSPC", "^IXIC"]


def test_unknown_region_serves_nothing_rather_than_us_news(fake_yf):
    fake = fake_yf({})
    response = YFinanceNewsProvider().search("q", stock_code="market", region="zz")

    # Silently serving US headlines for an unmapped market would be worse than
    # serving none: the review would read as if it covered that market.
    assert fake.asked == []
    assert response.success is False


def test_unresolvable_code_is_not_counted_as_a_provider_error():
    provider = YFinanceNewsProvider()
    response = provider.search("q", stock_code="")

    assert response.success is False
    assert "无法从" in (response.error_message or "")
    # A code this source cannot serve is "not applicable", not a fetch failure:
    # counting it would eventually trip the shared key error accounting.
    assert provider._key_errors == {"yfinance": 0}


# --- result shaping ---------------------------------------------------------


def test_items_outside_the_window_are_dropped(fake_yf):
    fake_yf({"NVDA": [_item("fresh", age_days=1), _item("stale", age_days=30, url="https://e.com/c")]})
    response = YFinanceNewsProvider().search("q", stock_code="NVDA", days=3)
    assert [r.title for r in response.results] == ["fresh"]


def test_duplicate_urls_across_indices_are_collapsed(fake_yf):
    shared = "https://example.com/same"
    fake_yf({"^GSPC": [_item("dup", url=shared)], "^IXIC": [_item("dup", url=shared)]})
    response = YFinanceNewsProvider().search("q", stock_code="market", region="us")
    assert len(response.results) == 1


def test_plain_summary_is_used_rather_than_html_description(fake_yf):
    fake_yf({"NVDA": [_item("N1", summary="plain text")]})
    response = YFinanceNewsProvider().search("q", stock_code="NVDA")
    # description carries HTML markup; the analysis prompt wants plain text.
    assert response.results[0].snippet == "plain text"
    assert "<p>" not in response.results[0].snippet


def test_publisher_becomes_the_source(fake_yf):
    fake_yf({"NVDA": [_item("N1", source="Reuters")]})
    response = YFinanceNewsProvider().search("q", stock_code="NVDA")
    assert response.results[0].source == "Reuters"


def test_max_results_is_respected(fake_yf):
    fake_yf({"NVDA": [_item(f"N{i}", url=f"https://e.com/{i}") for i in range(10)]})
    response = YFinanceNewsProvider().search("q", stock_code="NVDA", max_results=3)
    assert len(response.results) == 3


def test_malformed_items_are_skipped(fake_yf):
    fake_yf({"NVDA": [{"id": "x"}, {"content": "not-a-dict"}, _item("good")]})
    response = YFinanceNewsProvider().search("q", stock_code="NVDA")
    assert [r.title for r in response.results] == ["good"]


def test_one_failing_ticker_does_not_sink_the_others(fake_yf):
    fake_yf({"^GSPC": RuntimeError("boom"), "^IXIC": [_item("ok")]})
    response = YFinanceNewsProvider().search("q", stock_code="market", region="us")
    assert response.success is True
    assert [r.title for r in response.results] == ["ok"]


def test_provider_needs_no_credential():
    # It exists precisely for when the metered providers are exhausted.
    assert YFinanceNewsProvider().is_available is True


def test_provider_is_marked_symbol_scoped():
    """The open-ended intelligence paths must skip it rather than fail on it."""
    from src.search_service import BaseSearchProvider, TavilySearchProvider

    assert BaseSearchProvider.supports_open_ended_queries is True
    assert TavilySearchProvider.supports_open_ended_queries is True
    # It answers "news for this symbol", not "what happened in the sector".
    assert YFinanceNewsProvider.supports_open_ended_queries is False
