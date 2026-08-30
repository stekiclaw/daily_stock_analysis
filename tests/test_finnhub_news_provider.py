# -*- coding: utf-8 -*-
"""Tests for the Finnhub company-news provider.

Added because the metered search providers (Brave/Tavily) hit their monthly
caps and the self-hosted SearXNG's upstream engines are CAPTCHA-walled from a
datacenter IP, which left US analyses running on 0-1 news items. Finnhub's
``/company-news`` is on the free tier and already keyed in this deployment.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.search_service import FinnhubNewsProvider


def _epoch(dt: datetime) -> int:
    return int(dt.timestamp())


def _article(headline: str, *, summary: str = "", when: datetime = None, url: str = None) -> dict:
    when = when or datetime.now(timezone.utc)
    return {
        "headline": headline,
        "summary": summary,
        "url": url if url is not None else f"https://example.com/{abs(hash(headline))}",
        "source": "Benzinga",
        "datetime": _epoch(when),
    }


def _response(payload, status: int = 200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.text = json.dumps(payload) if not isinstance(payload, str) else payload
    return resp


class TestSymbolGating:
    """The endpoint takes plain US tickers; anything else is declined up front."""

    def test_us_ticker_accepted(self):
        assert FinnhubNewsProvider._resolve_symbol("NVDA") == "NVDA"
        assert FinnhubNewsProvider._resolve_symbol("nvda") == "NVDA"

    def test_non_us_codes_declined(self):
        for code in ("600519", "HK00700", "00700.HK", "2330.TW", ""):
            assert FinnhubNewsProvider._resolve_symbol(code) == ""

    def test_declined_code_is_not_a_provider_failure(self):
        """It must not burn the key error budget, which tracks real fetch failures."""
        provider = FinnhubNewsProvider(["k"])
        with patch("src.search_service.requests.get") as get:
            result = provider.search("q", stock_code="600519")
        get.assert_not_called()
        assert result.success is False
        assert result.results == []
        assert provider._key_errors["k"] == 0


class TestRelevanceRanking:
    """``/company-news`` is a per-symbol feed in name only.

    Roughly a third of what it returns for a liquid ticker is same-day general
    market wire. Ranking on recency alone spends the whole max_results budget
    on that noise (measured 3/5 relevant for NVDA before this ranking).
    """

    def test_relevance_terms_drop_boilerplate_and_corporate_suffixes(self):
        terms = FinnhubNewsProvider._relevance_terms(
            "NVIDIA Corporation NVDA latest news events", "NVDA"
        )
        assert "NVDA" in terms
        assert "NVIDIA" in terms
        for noise in ("CORPORATION", "LATEST", "NEWS", "EVENTS"):
            assert noise not in terms

    def test_issuer_specific_articles_outrank_newer_generic_wire(self):
        now = datetime.now(timezone.utc)
        payload = [
            _article("Broadcom Faces Valuation Test", when=now),
            _article("Dell Stock Analysis Mixed Signals", when=now - timedelta(minutes=1)),
            _article("Nvidia: The Cracks Are Showing", when=now - timedelta(hours=6)),
        ]
        provider = FinnhubNewsProvider(["k"])
        with patch("src.search_service.requests.get", return_value=_response(payload)):
            result = provider.search(
                "NVIDIA Corporation NVDA latest news events",
                max_results=3,
                days=3,
                stock_code="NVDA",
            )
        assert result.success is True
        # The older-but-relevant item must come first despite two newer ones.
        assert result.results[0].title.startswith("Nvidia")

    def test_ticker_mentioned_only_in_summary_still_counts(self):
        now = datetime.now(timezone.utc)
        payload = [
            _article("Unrelated Market Roundup", when=now),
            _article(
                "Weekly Buzz",
                summary="Traders piled into NVDA this week.",
                when=now - timedelta(hours=3),
            ),
        ]
        provider = FinnhubNewsProvider(["k"])
        with patch("src.search_service.requests.get", return_value=_response(payload)):
            result = provider.search(
                "NVIDIA Corporation NVDA latest news events",
                max_results=2,
                stock_code="NVDA",
            )
        assert result.results[0].title == "Weekly Buzz"

    def test_substring_of_a_longer_word_is_not_a_match(self):
        """Word-boundary matching: 'MU' must not match 'MUSK' or 'AMUSEMENT'."""
        assert FinnhubNewsProvider._mentions("ELON MUSK COMMENTS", ["MU"]) is False
        assert FinnhubNewsProvider._mentions("MU BEATS ESTIMATES", ["MU"]) is True


class TestWindowAndDedup:
    def test_articles_older_than_the_window_are_dropped(self):
        now = datetime.now(timezone.utc)
        payload = [
            _article("Nvidia fresh", when=now - timedelta(days=1)),
            _article("Nvidia stale", when=now - timedelta(days=30)),
        ]
        provider = FinnhubNewsProvider(["k"])
        with patch("src.search_service.requests.get", return_value=_response(payload)):
            result = provider.search("NVIDIA NVDA", max_results=10, days=3, stock_code="NVDA")
        titles = [r.title for r in result.results]
        assert "Nvidia fresh" in titles
        assert "Nvidia stale" not in titles

    def test_duplicate_urls_are_collapsed(self):
        now = datetime.now(timezone.utc)
        payload = [
            _article("Nvidia A", when=now, url="https://example.com/same"),
            _article("Nvidia B", when=now, url="https://example.com/same"),
        ]
        provider = FinnhubNewsProvider(["k"])
        with patch("src.search_service.requests.get", return_value=_response(payload)):
            result = provider.search("NVIDIA NVDA", max_results=10, stock_code="NVDA")
        assert len(result.results) == 1

    def test_untitled_items_are_skipped(self):
        payload = [_article(""), _article("Nvidia real")]
        provider = FinnhubNewsProvider(["k"])
        with patch("src.search_service.requests.get", return_value=_response(payload)):
            result = provider.search("NVIDIA NVDA", max_results=10, stock_code="NVDA")
        assert [r.title for r in result.results] == ["Nvidia real"]


class TestErrorHandling:
    def test_non_200_is_reported_as_failure(self):
        provider = FinnhubNewsProvider(["k"])
        with patch("src.search_service.requests.get", return_value=_response("nope", status=429)):
            result = provider.search("NVIDIA NVDA", stock_code="NVDA")
        assert result.success is False
        assert "429" in result.error_message

    def test_object_payload_carrying_an_error_is_reported(self):
        """Quota/permission problems answer 200 with an object, not a list."""
        provider = FinnhubNewsProvider(["k"])
        payload = {"error": "You don't have access to this resource."}
        with patch("src.search_service.requests.get", return_value=_response(payload)):
            result = provider.search("NVIDIA NVDA", stock_code="NVDA")
        assert result.success is False
        assert "access" in result.error_message

    def test_open_ended_queries_are_not_supported(self):
        """The intelligence layer must skip this source for free-text lookups."""
        assert FinnhubNewsProvider.supports_open_ended_queries is False
