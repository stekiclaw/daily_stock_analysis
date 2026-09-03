# -*- coding: utf-8 -*-
"""Regression tests for YfinanceFetcher's derived 量比/换手率 (volume_ratio/turnover_rate).

Both fields used to be hardcoded to ``None`` for every quote YfinanceFetcher
returned, even though the ``ticker.info`` payload it already fetches (for
pe_ratio/pb_ratio) carries everything needed to derive them. That silently
starved every US/JP/KR/TW analysis of the volume-based signal the analysis
prompt explicitly asks the model to reason about (放量/缩量, 换手率).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from data_provider.realtime_types import RealtimeSource
from data_provider.yfinance_fetcher import (
    YfinanceFetcher,
    _compute_yfinance_volume_signals,
    _estimate_yfinance_amount,
    _resolve_yfinance_pb_ratio,
)


class TestComputeYfinanceVolumeSignals:
    """Unit coverage for the pure computation helper."""

    def test_derives_both_fields_from_a_full_info_payload(self) -> None:
        volume_ratio, turnover_rate = _compute_yfinance_volume_signals(
            volume=38_609_800,
            ticker_info={
                "averageVolume10days": 38_515_190,
                "averageVolume": 55_202_393,
                "floatShares": 14_569_223_952,
                "sharesOutstanding": 14_594_180_000,
            },
        )
        # today/avg10d, not the ~3-month average, when both are present.
        assert volume_ratio == round(38_609_800 / 38_515_190, 2)
        assert turnover_rate == round(38_609_800 / 14_569_223_952 * 100, 4)

    def test_falls_back_to_three_month_average_when_10day_missing(self) -> None:
        volume_ratio, _ = _compute_yfinance_volume_signals(
            volume=1_000_000,
            ticker_info={"averageVolume": 500_000},
        )
        assert volume_ratio == 2.0

    def test_falls_back_to_shares_outstanding_when_float_shares_missing(self) -> None:
        _, turnover_rate = _compute_yfinance_volume_signals(
            volume=1_000_000,
            ticker_info={"sharesOutstanding": 100_000_000},
        )
        assert turnover_rate == 1.0

    def test_missing_averages_yield_none_instead_of_a_fabricated_ratio(self) -> None:
        volume_ratio, turnover_rate = _compute_yfinance_volume_signals(
            volume=1_000_000,
            ticker_info={},
        )
        assert volume_ratio is None
        assert turnover_rate is None

    def test_zero_or_missing_today_volume_yields_none_for_both(self) -> None:
        info = {"averageVolume10days": 1_000, "floatShares": 1_000_000}
        assert _compute_yfinance_volume_signals(None, info) == (None, None)
        assert _compute_yfinance_volume_signals(0, info) == (None, None)

    def test_zero_denominator_in_info_does_not_raise_or_fabricate(self) -> None:
        volume_ratio, turnover_rate = _compute_yfinance_volume_signals(
            volume=1_000_000,
            ticker_info={"averageVolume10days": 0, "floatShares": 0},
        )
        assert volume_ratio is None
        assert turnover_rate is None


class TestResolveYfinancePbRatio:
    """Yahoo P/B must not mix quote and financial-statement currencies."""

    def test_cross_currency_adr_is_downgraded_without_threshold_guessing(self) -> None:
        assert (
            _resolve_yfinance_pb_ratio(
                {
                    "currency": "USD",
                    "financialCurrency": "TWD",
                    "priceToBook": 85.94378,
                }
            )
            is None
        )

    def test_same_currency_keeps_provider_value(self) -> None:
        assert (
            _resolve_yfinance_pb_ratio(
                {
                    "currency": "USD",
                    "financialCurrency": "USD",
                    "priceToBook": 12.34,
                }
            )
            == 12.34
        )

    def test_unknown_financial_currency_preserves_existing_behavior(self) -> None:
        assert _resolve_yfinance_pb_ratio({"currency": "USD", "priceToBook": 12.34}) == 12.34


class TestRealtimeQuoteCarriesVolumeSignals:
    """End-to-end through get_realtime_quote(), matching the app's real call path."""

    @patch("yfinance.Ticker")
    def test_us_stock_quote_carries_derived_signals(self, ticker_factory: MagicMock) -> None:
        ticker = ticker_factory.return_value
        ticker.fast_info = SimpleNamespace(
            lastPrice=209.18,
            previousClose=214.30,
            open=213.0,
            dayHigh=214.5,
            dayLow=208.0,
            lastVolume=13_088_800,
            marketCap=None,
        )
        ticker.info = {
            "shortName": "Nebius Group N.V.",
            "currency": "USD",
            "trailingPE": None,
            "priceToBook": None,
            "averageVolume10days": 23_000_000,
            "floatShares": 228_600_000,
        }

        quote = YfinanceFetcher().get_realtime_quote("NBIS")

        assert quote is not None
        assert quote.source is RealtimeSource.YFINANCE
        assert quote.volume_ratio == round(13_088_800 / 23_000_000, 2)
        assert quote.turnover_rate == round(13_088_800 / 228_600_000 * 100, 4)

    @patch("yfinance.Ticker")
    def test_missing_yahoo_shares_data_keeps_none_rather_than_guessing(
        self, ticker_factory: MagicMock
    ) -> None:
        ticker = ticker_factory.return_value
        ticker.fast_info = SimpleNamespace(
            lastPrice=100.0,
            previousClose=99.0,
            open=99.5,
            dayHigh=101.0,
            dayLow=98.5,
            lastVolume=500_000,
            marketCap=None,
        )
        ticker.info = {"shortName": "Test Co", "currency": "USD"}

        quote = YfinanceFetcher().get_realtime_quote("ZTST")

        assert quote is not None
        assert quote.volume_ratio is None
        assert quote.turnover_rate is None

    @patch("yfinance.Ticker")
    def test_cross_currency_adr_quote_drops_pb_and_marks_it_missing(
        self, ticker_factory: MagicMock
    ) -> None:
        ticker = ticker_factory.return_value
        ticker.fast_info = SimpleNamespace(
            lastPrice=417.52,
            previousClose=427.30,
            open=423.0,
            dayHigh=425.0,
            dayLow=415.0,
            lastVolume=12_000_000,
            marketCap=2_165_456_502_784,
        )
        ticker.info = {
            "shortName": "Taiwan Semiconductor Manufacturing Company Limited",
            "currency": "USD",
            "financialCurrency": "TWD",
            "trailingPE": 31.847445,
            "priceToBook": 85.94378,
            "averageVolume10days": 15_000_000,
            "floatShares": 5_186_474_013,
        }

        quote = YfinanceFetcher().get_realtime_quote("TSM")

        assert quote is not None
        assert quote.pb_ratio is None
        assert quote.missing_fields == ["pb_ratio"]
        assert quote.data_quality == "partial"


class TestEstimatedAmount:
    """成交额 was None on every realtime quote while the daily bars carried a
    value, so the prompt's 今日成交额 cell read N/A and could not be compared
    against the historical series it was shown alongside."""

    def test_uses_the_same_formula_as_the_daily_bar_column(self):
        # daily path: df['amount'] = df['volume'] * df['close']
        assert _estimate_yfinance_amount(1_000, 12.5) == 12_500.0

    def test_missing_or_nonpositive_inputs_yield_none_not_zero(self):
        for volume, price in ((None, 10.0), (0, 10.0), (100, None), (100, 0), (-5, 10.0)):
            assert _estimate_yfinance_amount(volume, price) is None

    @patch("yfinance.Ticker")
    def test_realtime_quote_carries_amount_and_clears_the_missing_flag(
        self, ticker_factory: MagicMock
    ) -> None:
        ticker = ticker_factory.return_value
        ticker.fast_info = SimpleNamespace(
            lastPrice=200.0,
            previousClose=198.0,
            open=199.0,
            dayHigh=201.0,
            dayLow=197.0,
            lastVolume=1_000_000,
            marketCap=None,
        )
        ticker.info = {
            "shortName": "Test Co",
            "currency": "USD",
            "trailingPE": 20.0,
            "priceToBook": 3.0,
            "averageVolume10days": 1_000_000,
            "floatShares": 100_000_000,
        }

        quote = YfinanceFetcher().get_realtime_quote("ZTST")

        assert quote is not None
        assert quote.amount == 200.0 * 1_000_000
        # amount no longer counts as missing, so the quote is no longer partial.
        assert quote.missing_fields is None
        assert quote.data_quality == "ok"
