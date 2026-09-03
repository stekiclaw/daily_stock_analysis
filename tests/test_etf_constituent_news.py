# -*- coding: utf-8 -*-
"""Tests for the ETF constituent-news fallback.

A thinly-covered or leveraged ETF rarely generates news under its own ticker:
SOXS's most recent Yahoo headline was 29 days old against a 3-day analysis
window, so every fund-level source correctly returned nothing and the report
ran with no 舆情面 at all. The fund's price is driven by its holdings, so
their news is the news that explains the move.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.search_service import ETFConstituentNewsProvider, SearchService


def _holdings(rows):
    """Build a top_holdings frame shaped like yfinance's."""
    return pd.DataFrame(
        [{"Name": name} for _, name in rows],
        index=[sym for sym, _ in rows],
    )


def _news_item(title, *, when, summary="", url=None):
    return {
        "content": {
            "title": title,
            "summary": summary,
            "pubDate": when.isoformat().replace("+00:00", "Z"),
            "canonicalUrl": {"url": url or f"https://example.com/{abs(hash(title))}"},
            "provider": {"displayName": "Yahoo Finance"},
        }
    }


class TestCashHoldingFilter:
    """Leveraged/inverse funds hold swaps; their reported holdings are cash."""

    def test_money_market_tickers_are_excluded(self):
        for sym in ("DIRXX", "DGCXX", "FGXXX", "FRDXX"):
            assert ETFConstituentNewsProvider._is_cash_like(sym, "") is True

    def test_cash_like_names_are_excluded(self):
        cases = [
            ("ABC", "BNY Dreyfus Treasury Secs Csh Mgmt Inst"),
            ("DEF", "First American Government Obligs X"),
            ("GHI", "Some Money Market Fund"),
        ]
        for sym, name in cases:
            assert ETFConstituentNewsProvider._is_cash_like(sym, name) is True

    def test_real_equities_are_kept(self):
        for sym, name in (("NVDA", "NVIDIA Corp"), ("WDC", "Western Digital Corp"),
                          ("STX", "Seagate Technology Holdings PLC")):
            assert ETFConstituentNewsProvider._is_cash_like(sym, name) is False

    def test_four_letter_ticker_ending_in_x_is_not_treated_as_cash(self):
        """The XX rule is for 5-letter mutual-fund tickers, not equities."""
        assert ETFConstituentNewsProvider._is_cash_like("XRXX", "") is False  # 4 letters
        assert ETFConstituentNewsProvider._is_cash_like("LUMN", "Lumen") is False

    def test_t_bill_collateral_funds_are_excluded(self):
        """0DTE covered-call ETFs park collateral in a T-bill ETF.

        XDTE/QDTE/RDTE report exactly two holdings - "Roundhill Weekly T-Bill
        ETF" (5-7%) and a government money fund - and MSTU reports "The
        Laddered T-Bill ETF". WEEK/TLDR are four-letter tickers, so the
        five-letter XX rule cannot see them and only the name identifies them.
        """
        cases = [
            ("WEEK", "Roundhill Weekly T-Bill ETF"),
            ("TLDR", "The Laddered T-Bill ETF"),
            ("XBIL", "US Treasury 12 Month Bill ETF"),
        ]
        for sym, name in cases:
            assert ETFConstituentNewsProvider._is_cash_like(sym, name) is True

    def test_money_funds_without_the_word_market_are_excluded(self):
        """Real Yahoo names say "Money" without "Market", or abbreviate it."""
        cases = [
            ("IUGXY", "Invesco Premier US Government Money Inst"),
            ("MGMXY", "JPMorgan US Government MMkt IM"),
            ("ABCD", "Acme Govt Money Portfolio"),
        ]
        for sym, name in cases:
            # Deliberately non-XX tickers: the name alone must be enough.
            assert not (len(sym) == 5 and sym.endswith("XX"))
            assert ETFConstituentNewsProvider._is_cash_like(sym, name) is True

    def test_money_market_etf_share_classes_are_excluded(self):
        """Money-market *ETFs* have ordinary 4-letter tickers, not XX ones."""
        for sym, name in (("IQMM", "ProShares GENIUS Money Market ETF"),
                          ("SBIL", "Simplify Government Money Market ETF")):
            assert ETFConstituentNewsProvider._is_cash_like(sym, name) is True

    @pytest.mark.parametrize(
        "symbol,name",
        [
            # A false positive silently drops a real constituent's news, which
            # is worse than the wasted lookup a false negative costs. These are
            # real listed issuers whose names brush against cash vocabulary.
            ("ML", "MoneyLion Inc"),
            ("MNY", "MoneyHero Group Ltd"),
            ("3994.T", "Money Forward Inc"),
            ("MONY.L", "Moneysupermarket.com Group PLC"),
            ("BILL", "BILL Holdings Inc"),
            ("BILI", "Bilibili Inc ADR"),
            ("NTRS", "Northern Trust Corp"),
            ("TRST", "TrustCo Bancorp NY"),
            ("DLR", "Digital Realty Trust Inc"),
            ("BXMT", "Blackstone Mortgage Trust Inc Class A"),
            ("LADR", "Ladder Capital Corp Class A"),
            ("DEA", "Easterly Government Properties Inc"),
            ("BXSL", "Blackstone Secured Lending Fund Ordinary Shares"),
            ("TXN", "Texas Instruments Inc"),
            ("MKTX", "MarketAxess Holdings Inc"),
            ("GOVX", "GeoVax Labs Inc"),
            ("RGEN", "Repligen Corp"),
            ("STT", "State Street Corp"),
            ("XYZ", "Block Inc Class A"),
            ("TBBK", "The Bancorp Inc"),
            # Real issuer whose *ticker* is CASH; terms match names, not tickers.
            ("CASH", "Pathward Financial Inc"),
        ],
    )
    def test_operating_companies_are_never_treated_as_cash(self, symbol, name):
        assert ETFConstituentNewsProvider._is_cash_like(symbol, name) is False

    @pytest.mark.parametrize(
        "symbol,name",
        [
            # Live holdings that plain substring matching wrongly dropped:
            # "CASH" inside FirstCash, "DEPOSIT" inside Depository.
            ("FCFS", "FirstCash Holdings Inc"),
            ("LNW", "Light & Wonder Inc Chess Depository Interest"),
            # Yahoo spells some ADR holdings out in full; "Depositary" must not
            # match the DEPOSIT term either.
            ("TSM", "Taiwan Semiconductor Manufacturing Co Ltd "
                    "American Depositary Receipt"),
            ("BABA", "Alibaba Group Holding Ltd American Depositary Shares"),
            # "REPO" must not match Repligen/Repsol/Repossession-style names.
            ("RGEN", "Repligen Corp"),
            ("REPYY", "Repsol SA ADR"),
        ],
    )
    def test_cash_words_only_match_on_word_boundaries(self, symbol, name):
        """A false positive silently drops a real constituent's news.

        That is strictly worse than the wasted lookup a false negative costs,
        so the cash vocabulary is anchored to whole words.
        """
        assert ETFConstituentNewsProvider._is_cash_like(symbol, name) is False

    def test_word_boundaries_do_not_weaken_the_cash_terms(self):
        """The boundary rule must not let genuine cash positions through."""
        cases = [
            ("ABCD", "BNY Dreyfus Govt Cash Mgmt Instl"),
            ("EFGH", "BlackRock Cash Funds Treasury SL Agency"),
            ("IJKL", "Deposits with Broker for Short Positions"),
            ("MNOP", "Morgan Stanley Institutional Liquidity Treasury"),
            ("QRST", "First American Treasury Obligs X"),
        ]
        for sym, name in cases:
            assert ETFConstituentNewsProvider._is_cash_like(sym, name) is True


class TestConstituentResolution:
    def _provider_with_holdings(self, frame):
        provider = ETFConstituentNewsProvider()
        ticker = MagicMock()
        ticker.funds_data.top_holdings = frame
        return provider, ticker

    def test_equity_holdings_resolve_in_order(self):
        frame = _holdings([("NVDA", "NVIDIA Corp"), ("AVGO", "Broadcom Inc")])
        provider, ticker = self._provider_with_holdings(frame)
        with patch("yfinance.Ticker", return_value=ticker):
            assert provider._resolve_constituents("SOXQ") == [
                ("NVDA", "NVIDIA Corp"),
                ("AVGO", "Broadcom Inc"),
            ]

    def test_cash_only_fund_resolves_to_nothing(self):
        """SOXS reports only cash collateral - no constituent news exists."""
        frame = _holdings([("DIRXX", "BNY Dreyfus Treasury"), ("DGCXX", "Govt Cash Mgmt")])
        provider, ticker = self._provider_with_holdings(frame)
        with patch("yfinance.Ticker", return_value=ticker):
            assert provider._resolve_constituents("SOXS") == []

    def test_t_bill_collateral_only_fund_resolves_to_nothing(self):
        """XDTE/QDTE/RDTE hold only a T-bill ETF plus a government money fund.

        Before the T-BILL term, WEEK survived the filter and the provider spent
        a news lookup on a T-bill ETF's Yahoo feed, attributing whatever wire
        it carried to the fund as constituent driver news.
        """
        frame = _holdings([
            ("WEEK", "Roundhill Weekly T-Bill ETF"),
            ("FGXXX", "First American Government Obligs X"),
        ])
        provider, ticker = self._provider_with_holdings(frame)
        with patch("yfinance.Ticker", return_value=ticker):
            assert provider._resolve_constituents("XDTE") == []

    def test_resolution_is_capped(self):
        rows = [(f"T{i}", f"Company {i}") for i in range(20)]
        provider, ticker = self._provider_with_holdings(_holdings(rows))
        with patch("yfinance.Ticker", return_value=ticker):
            resolved = provider._resolve_constituents("BIG")
        assert len(resolved) == ETFConstituentNewsProvider.MAX_CONSTITUENTS

    def test_non_etf_declines_without_being_a_fetch_failure(self):
        provider = ETFConstituentNewsProvider()
        with patch("yfinance.Ticker", side_effect=Exception("no funds_data")):
            result = provider.search("q", stock_code="NVDA")
        assert result.success is False
        assert result.results == []
        assert provider._key_errors["etf-constituents"] == 0

    def test_empty_code_declines(self):
        assert ETFConstituentNewsProvider()._resolve_constituents("") == []


class TestConstituentNewsFetch:
    def _run(self, provider, holdings_frame, news_by_ticker, **kwargs):
        def fake_ticker(sym):
            t = MagicMock()
            t.funds_data.top_holdings = holdings_frame
            t.news = news_by_ticker.get(sym, [])
            return t

        with patch("yfinance.Ticker", side_effect=fake_ticker):
            return provider.search(
                kwargs.pop("query", "SOXQ latest news events"),
                stock_code=kwargs.pop("stock_code", "SOXQ"),
                **kwargs,
            )

    def test_results_are_attributed_to_the_holding(self):
        """The model must be able to tell this is not a fund announcement."""
        now = datetime.now(timezone.utc)
        frame = _holdings([("NVDA", "NVIDIA Corp")])
        news = {"NVDA": [_news_item("Nvidia beats estimates", when=now)]}
        result = self._run(ETFConstituentNewsProvider(), frame, news, max_results=5, days=3)

        assert result.success is True
        assert len(result.results) == 1
        assert "成分股" in result.results[0].source
        assert "NVDA" in result.results[0].source

    def test_articles_about_the_holding_outrank_newer_unrelated_wire(self):
        now = datetime.now(timezone.utc)
        frame = _holdings([("NVDA", "NVIDIA Corp")])
        news = {
            "NVDA": [
                _news_item("Wall Street Isn't Giving Up on CVS Health", when=now),
                _news_item("Nvidia's CFO explains the AI boom", when=now - timedelta(hours=5)),
            ]
        }
        result = self._run(ETFConstituentNewsProvider(), frame, news, max_results=2, days=3)
        assert result.results[0].title.startswith("Nvidia")

    def test_stale_articles_are_dropped(self):
        now = datetime.now(timezone.utc)
        frame = _holdings([("NVDA", "NVIDIA Corp")])
        news = {
            "NVDA": [
                _news_item("Nvidia fresh", when=now - timedelta(days=1)),
                _news_item("Nvidia stale", when=now - timedelta(days=40)),
            ]
        }
        result = self._run(ETFConstituentNewsProvider(), frame, news, max_results=5, days=3)
        titles = [r.title for r in result.results]
        assert titles == ["Nvidia fresh"]

    def test_same_article_across_two_holdings_is_deduped(self):
        """Index constituents frequently share wire coverage."""
        now = datetime.now(timezone.utc)
        frame = _holdings([("NVDA", "NVIDIA Corp"), ("AVGO", "Broadcom Inc")])
        shared = _news_item("Chip selloff deepens", when=now, url="https://example.com/one")
        news = {"NVDA": [shared], "AVGO": [dict(shared)]}
        result = self._run(ETFConstituentNewsProvider(), frame, news, max_results=5, days=3)
        assert len(result.results) == 1

    def test_one_failing_holding_does_not_sink_the_others(self):
        now = datetime.now(timezone.utc)
        frame = _holdings([("BAD", "Bad Corp"), ("NVDA", "NVIDIA Corp")])

        def fake_ticker(sym):
            t = MagicMock()
            t.funds_data.top_holdings = frame
            if sym == "BAD":
                type(t).news = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
            else:
                t.news = [_news_item("Nvidia news", when=now)]
            return t

        with patch("yfinance.Ticker", side_effect=fake_ticker):
            result = ETFConstituentNewsProvider().search(
                "SOXQ latest news", stock_code="SOXQ", max_results=5, days=3
            )
        assert [r.title for r in result.results] == ["Nvidia news"]


class TestRegistration:
    def test_opt_in_like_the_other_keyless_fallback(self):
        """Off by default: a fresh clone with no keys must still report having
        no news channel configured (#2225), which a silently-registered
        keyless provider would mask."""
        assert "ETFConstituentNews" not in [p.name for p in SearchService()._providers]
        assert SearchService().is_available is False

    def test_can_be_enabled(self):
        service = SearchService(etf_constituent_news_enabled=True)
        assert "ETFConstituentNews" in [p.name for p in service._providers]

    def test_survives_the_subprocess_kwargs_round_trip(self):
        service = SearchService(etf_constituent_news_enabled=True)
        rebuilt = SearchService(**service._constructor_kwargs)
        assert "ETFConstituentNews" in [p.name for p in rebuilt._providers]

    def test_ranks_after_fund_level_sources(self):
        """Real fund news must win; constituents are only a fallback."""
        service = SearchService(
            finnhub_news_keys=["k"],
            yfinance_news_enabled=True,
            etf_constituent_news_enabled=True,
        )
        names = [p.name for p in service._providers]
        assert names.index("ETFConstituentNews") > names.index("FinnhubNews")
        assert names.index("ETFConstituentNews") > names.index("YFinanceNews")

    def test_is_symbol_scoped(self):
        assert ETFConstituentNewsProvider.supports_open_ended_queries is False
