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


# --- multi-dimension intelligence path --------------------------------------


def test_symbol_scoped_provider_serves_only_the_news_dimension():
    """It can answer "news for NVDA" but not "研报 目标价 评级"."""
    from src.search_service import SearchService

    assert "latest_news" in SearchService.SYMBOL_SCOPED_INTEL_DIMENSIONS
    for open_ended in ("market_analysis", "risk_check", "industry_analysis", "announcements"):
        assert open_ended not in SearchService.SYMBOL_SCOPED_INTEL_DIMENSIONS


def test_comprehensive_intel_uses_the_fallback_for_latest_news(fake_yf, monkeypatch):
    from src.search_service import SearchService

    fake = fake_yf({"NVDA": [_item("Fresh NVDA headline")]})
    service = SearchService(
        searxng_public_instances_enabled=False,
        yfinance_news_enabled=True,
        news_max_age_days=3,
        news_strategy_profile="short",
    )
    # Only the keyless fallback is configured, mirroring the state where every
    # metered provider is out of quota.
    assert [p.name for p in service._providers] == ["YFinanceNews"]

    responses = service.search_comprehensive_intel("NVDA", "NVIDIA", max_searches=3)

    assert fake.asked == ["NVDA"]
    titles = [r.title for resp in responses.values() for r in resp.results]
    assert "Fresh NVDA headline" in titles


def test_news_dimension_falls_back_when_the_chosen_provider_returns_nothing(
    fake_yf, monkeypatch
):
    """One provider is tried per dimension; an exhausted one must not zero it out."""
    from src.search_service import SearchResponse, SearchService, TavilySearchProvider

    fake = fake_yf({"NVDA": [_item("Fallback headline")]})

    def _empty(self, query, max_results=5, days=7, **kwargs):
        return SearchResponse(
            query=query,
            results=[],
            provider=self.name,
            success=False,
            error_message="API 配额已用尽",
        )

    monkeypatch.setattr(TavilySearchProvider, "search", _empty)

    service = SearchService(
        tavily_keys=["dummy"],
        searxng_public_instances_enabled=False,
        yfinance_news_enabled=True,
        news_max_age_days=3,
        news_strategy_profile="short",
    )
    responses = service.search_comprehensive_intel("NVDA", "NVIDIA", max_searches=3)

    assert fake.asked == ["NVDA"]
    titles = [r.title for resp in responses.values() for r in resp.results]
    assert "Fallback headline" in titles


def test_open_ended_dimension_fails_over_to_another_capable_provider(monkeypatch):
    """Quota exhaustion on the rotated primary must not zero an analytical dimension."""
    from src.search_service import SearchResponse, SearchResult, SearchService

    today = datetime.now(timezone.utc).date().isoformat()

    class _Provider:
        supports_open_ended_queries = True
        is_available = True

        def __init__(self, name):
            self.name = name
            self.calls = []

        def search(self, query, max_results=5, days=7, **kwargs):
            self.calls.append(query)
            is_analysis = "analyst rating" in query
            if self.name == "Secondary" and is_analysis:
                return SearchResponse(
                    query=query,
                    results=[],
                    provider=self.name,
                    success=False,
                    error_message="quota exhausted",
                )
            title = (
                "NVIDIA analyst raises target price after earnings"
                if is_analysis
                else "NVIDIA announces a new AI platform"
            )
            return SearchResponse(
                query=query,
                results=[SearchResult(
                    title=title,
                    snippet="NVIDIA company update with material investor context.",
                    url=f"https://example.com/{self.name}/{len(self.calls)}",
                    source=self.name,
                    published_date=today,
                )],
                provider=self.name,
                success=True,
            )

    primary = _Provider("Primary")
    secondary = _Provider("Secondary")
    service = SearchService(searxng_public_instances_enabled=False)
    service._providers = [primary, secondary]
    monkeypatch.setattr("src.search_service.time.sleep", lambda _seconds: None)

    responses = service.search_comprehensive_intel("NVDA", "NVIDIA", max_searches=2)

    # latest_news starts at Primary; round-robin makes market_analysis start at
    # Secondary, whose exhausted response must fall back to Primary.
    assert responses["market_analysis"].provider == "Primary"
    assert responses["market_analysis"].results
    assert len(secondary.calls) == 1
    assert len(primary.calls) == 2


def test_open_ended_dimension_does_not_fall_back_to_the_symbol_source(
    fake_yf, monkeypatch
):
    from src.search_service import SearchResponse, SearchService, TavilySearchProvider

    fake = fake_yf({"NVDA": [_item("Should not appear")]})
    seen_dimensions = []

    def _empty(self, query, max_results=5, days=7, **kwargs):
        seen_dimensions.append(query)
        return SearchResponse(query=query, results=[], provider=self.name, success=False)

    monkeypatch.setattr(TavilySearchProvider, "search", _empty)

    service = SearchService(
        tavily_keys=["dummy"],
        searxng_public_instances_enabled=False,
        yfinance_news_enabled=True,
        news_max_age_days=3,
        news_strategy_profile="short",
    )
    # Force every dimension to be treated as open-ended.
    monkeypatch.setattr(SearchService, "SYMBOL_SCOPED_INTEL_DIMENSIONS", frozenset())
    service.search_comprehensive_intel("NVDA", "NVIDIA", max_searches=3)

    # "研报 目标价 评级" is not something a per-symbol news feed can answer.
    assert fake.asked == []
