# -*- coding: utf-8 -*-
"""
Issue #234 盘中实时技术指标的单元测试。

覆盖范围：
- _augment_historical_with_realtime：追加/更新逻辑和防护条件
- _compute_ma_status：均线排列文案
- _enhance_context：使用 realtime + trend_result 覆盖 today
"""

import os
import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_provider.realtime_types import UnifiedRealtimeQuote, RealtimeSource
from src.stock_analyzer import StockTrendAnalyzer, TrendAnalysisResult, TrendStatus
from src.core.pipeline import StockAnalysisPipeline


def _make_realtime_quote(
    price: float = 15.72,
    open_price: float = 15.62,
    high: float = 16.29,
    low: float = 15.55,
    volume: int = 13995600,
    amount: float = None,
    change_pct: float = 0.96,
    **overrides,
) -> UnifiedRealtimeQuote:
    return UnifiedRealtimeQuote(
        code="600519",
        name="贵州茅台",
        source=RealtimeSource.TENCENT,
        price=price,
        open_price=open_price,
        high=high,
        low=low,
        volume=volume,
        amount=amount,
        change_pct=change_pct,
        **overrides,
    )


def _make_historical_df(days: int = 25, last_date: date = None) -> pd.DataFrame:
    """构造历史 OHLCV DataFrame。"""
    if last_date is None:
        last_date = date.today() - timedelta(days=1)
    dates = [last_date - timedelta(days=i) for i in range(days - 1, -1, -1)]
    base = 100.0
    data = []
    for i, d in enumerate(dates):
        close = base + i * 0.5
        data.append({
            "code": "600519",
            "date": d,
            "open": close - 0.2,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": 1000000 + i * 10000,
            "amount": close * (1000000 + i * 10000),
            "pct_chg": 0.5,
            "ma5": close,
            "ma10": close - 0.1,
            "ma20": close - 0.2,
            "volume_ratio": 1.0,
        })
    return pd.DataFrame(data)


class TestAugmentHistoricalWithRealtime(unittest.TestCase):
    """_augment_historical_with_realtime 的测试。"""

    def setUp(self) -> None:
        self._db_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "test_issue234.db"
        )
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        with patch.dict(os.environ, {"DATABASE_PATH": self._db_path}):
            from src.config import Config
            Config._instance = None
            self.config = Config._load_from_env()
        self.pipeline = StockAnalysisPipeline(config=self.config)

    def test_returns_unchanged_when_realtime_none(self) -> None:
        df = _make_historical_df()
        result = self.pipeline._augment_historical_with_realtime(df, None, "600519")
        self.assertIs(result, df)
        self.assertEqual(len(result), len(df))

    def test_returns_unchanged_when_price_invalid(self) -> None:
        df = _make_historical_df()
        quote = _make_realtime_quote(price=0)
        result = self.pipeline._augment_historical_with_realtime(df, quote, "600519")
        self.assertEqual(len(result), len(df))
        quote2 = MagicMock()
        quote2.price = None
        result2 = self.pipeline._augment_historical_with_realtime(df, quote2, "600519")
        self.assertEqual(len(result2), len(df))

    def test_returns_unchanged_when_df_empty(self) -> None:
        df = pd.DataFrame()
        quote = _make_realtime_quote()
        result = self.pipeline._augment_historical_with_realtime(df, quote, "600519")
        self.assertTrue(result.empty)

    def test_returns_unchanged_when_df_missing_close(self) -> None:
        df = pd.DataFrame({"date": [date.today()], "open": [100]})
        quote = _make_realtime_quote()
        result = self.pipeline._augment_historical_with_realtime(df, quote, "600519")
        self.assertEqual(len(result), 1)
        self.assertNotIn("close", result.columns)

    @patch("src.core.pipeline.get_market_now")
    @patch("src.core.pipeline.is_market_open", return_value=True)
    @patch("src.core.pipeline.get_market_for_stock", return_value="cn")
    def test_appends_row_when_last_date_before_today(
        self, _mock_market, _mock_open, mock_now
    ) -> None:
        today = date.today()
        # 固定市场时钟为 UTC 当日，使 pipeline 的 market_today 等于 date.today()，
        # 不受 get_market_now 通常使用的市场时区影响（例如 CST=UTC+8）。
        mock_now.return_value = datetime(
            today.year, today.month, today.day, 10, 0, tzinfo=timezone.utc
        )
        df = _make_historical_df(last_date=today - timedelta(days=1))
        quote = _make_realtime_quote(price=15.72)
        result = self.pipeline._augment_historical_with_realtime(df, quote, "600519")
        self.assertEqual(len(result), len(df) + 1)
        last = result.iloc[-1]
        self.assertEqual(last["close"], 15.72)
        self.assertEqual(last["date"], today)

    @patch("src.core.pipeline.get_market_now")
    @patch("src.core.pipeline.is_market_open", return_value=True)
    @patch("src.core.pipeline.get_market_for_stock", return_value="cn")
    def test_updates_last_row_when_last_date_is_today(
        self, _mock_market, _mock_open, mock_now
    ) -> None:
        today = date.today()
        # 固定市场时钟为当日，使 last_date >= market_today，从而更新最后一行而不是追加。
        # 这可以避免 CI 在 CST 收盘后运行时出现日期边界偏移。
        mock_now.return_value = datetime(
            today.year, today.month, today.day, 10, 0, tzinfo=timezone.utc
        )
        df = _make_historical_df(last_date=today, days=25)
        df.loc[df.index[-1], "date"] = today
        quote = _make_realtime_quote(price=16.0)
        result = self.pipeline._augment_historical_with_realtime(df, quote, "600519")
        self.assertEqual(len(result), len(df))
        self.assertEqual(result.iloc[-1]["close"], 16.0)


class TestComputeMaStatus(unittest.TestCase):
    """_compute_ma_status 的测试。"""

    def test_bullish_alignment(self) -> None:
        status = StockAnalysisPipeline._compute_ma_status(11, 10, 9.5, 9)
        self.assertIn("多头", status)

    def test_bearish_alignment(self) -> None:
        status = StockAnalysisPipeline._compute_ma_status(8, 9, 9.5, 10)
        self.assertIn("空头", status)

    def test_consolidation(self) -> None:
        status = StockAnalysisPipeline._compute_ma_status(10, 10, 10, 10)
        self.assertIn("震荡", status)


class TestEnhanceContextRealtimeOverride(unittest.TestCase):
    """_enhance_context 使用实时行情和趋势结果覆盖 today 的测试。"""

    def setUp(self) -> None:
        self._db_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "test_issue234.db"
        )
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        with patch.dict(os.environ, {"DATABASE_PATH": self._db_path}):
            from src.config import Config
            Config._instance = None
            self.config = Config._load_from_env()
        self.pipeline = StockAnalysisPipeline(config=self.config)

    @patch("src.core.pipeline.get_market_now")
    @patch("src.core.pipeline.get_market_for_stock", return_value="cn")
    def test_today_overridden_when_realtime_and_trend_exist(
        self, _mock_market, mock_now
    ) -> None:
        today = date.today()
        # 固定市场时钟，使 _enhance_context 设置 enhanced['date'] == date.today().isoformat()，
        # 不受 get_market_now 通常使用的市场时区影响（例如 CST=UTC+8）。
        mock_now.return_value = datetime(
            today.year, today.month, today.day, 10, 0, tzinfo=timezone.utc
        )
        context = {
            "code": "600519",
            "date": (today - timedelta(days=1)).isoformat(),
            "today": {"close": 15.0, "ma5": 14.8, "ma10": 14.5},
            "yesterday": {"close": 14.5, "volume": 1000000},
        }
        quote = _make_realtime_quote(price=15.72, volume=2000000)
        trend = TrendAnalysisResult(
            code="600519",
            trend_status=TrendStatus.BULL,
            ma5=15.5,
            ma10=15.2,
            ma20=14.9,
        )
        enhanced = self.pipeline._enhance_context(
            context, quote, None, trend, "贵州茅台"
        )
        self.assertEqual(enhanced["today"]["close"], 15.72)
        self.assertEqual(enhanced["today"]["ma5"], 15.5)
        self.assertEqual(enhanced["today"]["ma10"], 15.2)
        self.assertEqual(enhanced["today"]["ma20"], 14.9)
        self.assertIn("多头", enhanced["ma_status"])
        self.assertEqual(enhanced["date"], today.isoformat())
        self.assertEqual(enhanced["today"]["date"], today.isoformat())
        self.assertEqual(enhanced["today"]["data_source"], "realtime:tencent")
        self.assertEqual(enhanced["today"]["realtime_source"], "tencent")
        self.assertIn("price_change_ratio", enhanced)
        self.assertIn("volume_change_ratio", enhanced)

    @patch("src.core.pipeline.get_market_now")
    @patch("src.core.pipeline.get_market_for_stock", return_value="cn")
    def test_tencent_688691_volume_change_ratio_uses_normalized_share_volume(
        self, _mock_market, mock_now
    ) -> None:
        today = date.today()
        mock_now.return_value = datetime(
            today.year, today.month, today.day, 10, 0, tzinfo=timezone.utc
        )
        context = {
            "code": "688691",
            "date": (today - timedelta(days=1)).isoformat(),
            "today": {
                "close": 128.46,
                "volume": 19512753,
                "amount": 2487341983,
                "date": (today - timedelta(days=1)).isoformat(),
                "dataSource": "AkshareFetcher",
            },
            "yesterday": {"close": 128.46, "volume": 19512753},
        }
        quote = UnifiedRealtimeQuote(
            code="688691",
            name="灿芯股份",
            source=RealtimeSource.TENCENT,
            price=122.70,
            open_price=120.09,
            high=125.96,
            low=116.20,
            volume=10931723,
            amount=1327404280,
            change_pct=3.40,
        )
        trend = TrendAnalysisResult(
            code="688691",
            trend_status=TrendStatus.BULL,
            ma5=120.014,
            ma10=119.425,
            ma20=115.8305,
        )

        enhanced = self.pipeline._enhance_context(
            context, quote, None, trend, "灿芯股份"
        )

        self.assertEqual(enhanced["today"]["volume"], 10931723)
        self.assertEqual(enhanced["today"]["amount"], 1327404280)
        self.assertEqual(enhanced["volume_change_ratio"], 0.56)
        self.assertEqual(enhanced["today"]["date"], today.isoformat())
        self.assertEqual(enhanced["today"]["data_source"], "realtime:tencent")
        self.assertEqual(enhanced["today"]["realtime_source"], "tencent")
        self.assertNotIn("dataSource", enhanced["today"])

    @patch("src.core.pipeline.get_market_now")
    @patch("src.core.pipeline.get_market_for_stock", return_value="cn")
    def test_realtime_metadata_and_partial_estimated_fields_are_propagated(
        self, _mock_market, mock_now
    ) -> None:
        today = date.today()
        mock_now.return_value = datetime(
            today.year, today.month, today.day, 10, 0, tzinfo=timezone.utc
        )
        context = {
            "code": "600519",
            "date": (today - timedelta(days=1)).isoformat(),
            "today": {
                "close": 15.0,
                "amount": 999999,
                "date": (today - timedelta(days=1)).isoformat(),
                "dataSource": "AkshareFetcher",
            },
            "yesterday": {"close": 14.5, "volume": 1000000},
        }
        quote = _make_realtime_quote(
            price=15.72,
            amount=None,
            fetched_at="2026-05-31T10:00:05+00:00",
            provider_timestamp="2026-05-31T10:00:00+00:00",
            is_stale=False,
            stale_seconds=5,
            fallback_from="efinance",
        )
        trend = TrendAnalysisResult(
            code="600519",
            trend_status=TrendStatus.BULL,
            ma5=15.5,
            ma10=15.2,
            ma20=14.9,
        )

        enhanced = self.pipeline._enhance_context(
            context,
            quote,
            None,
            trend,
            "贵州茅台",
            market_phase_context={"is_partial_bar": True},
        )

        self.assertEqual(enhanced["realtime"]["source"], "tencent")
        self.assertEqual(enhanced["realtime"]["fetched_at"], "2026-05-31T10:00:05+00:00")
        self.assertEqual(enhanced["realtime"]["provider_timestamp"], "2026-05-31T10:00:00+00:00")
        self.assertIs(enhanced["realtime"]["is_stale"], False)
        self.assertEqual(enhanced["realtime"]["stale_seconds"], 5)
        self.assertEqual(enhanced["realtime"]["fallback_from"], "efinance")
        self.assertTrue(enhanced["today"]["is_partial_bar"])
        self.assertTrue(enhanced["today"]["is_estimated"])
        self.assertEqual(
            enhanced["today"]["estimated_fields"],
            ["close", "open", "high", "low", "ma5", "ma10", "ma20", "volume", "pct_chg"],
        )
        self.assertEqual(enhanced["today"]["fetched_at"], "2026-05-31T10:00:05+00:00")
        self.assertEqual(enhanced["today"]["provider_timestamp"], "2026-05-31T10:00:00+00:00")
        self.assertEqual(enhanced["today"]["fallback_from"], "efinance")
        self.assertNotIn("amount", enhanced["today"])
        self.assertNotIn("dataSource", enhanced["today"])

    @patch("src.core.pipeline.get_market_now")
    @patch("src.core.pipeline.get_market_for_stock", return_value="cn")
    def test_realtime_today_does_not_backfill_historical_amount_or_source(
        self, _mock_market, mock_now
    ) -> None:
        today = date.today()
        mock_now.return_value = datetime(
            today.year, today.month, today.day, 10, 0, tzinfo=timezone.utc
        )
        context = {
            "code": "600519",
            "date": (today - timedelta(days=1)).isoformat(),
            "today": {
                "close": 15.0,
                "amount": 999999,
                "date": (today - timedelta(days=1)).isoformat(),
                "dataSource": "AkshareFetcher",
                "code": "600519",
            },
            "yesterday": {"close": 14.5, "volume": 1000000},
        }
        quote = _make_realtime_quote(price=15.72, amount=None)
        trend = TrendAnalysisResult(
            code="600519",
            trend_status=TrendStatus.BULL,
            ma5=15.5,
            ma10=15.2,
            ma20=14.9,
        )

        enhanced = self.pipeline._enhance_context(
            context, quote, None, trend, "贵州茅台"
        )

        self.assertNotIn("amount", enhanced["today"])
        self.assertNotIn("dataSource", enhanced["today"])
        self.assertEqual(enhanced["today"]["date"], today.isoformat())
        self.assertEqual(enhanced["today"]["data_source"], "realtime:tencent")
        self.assertEqual(enhanced["today"]["code"], "600519")

    def test_enhance_context_injects_runtime_news_window_days(self) -> None:
        context = {"code": "600519", "today": {"close": 15.0}}
        enhanced = self.pipeline._enhance_context(
            context, None, None, None, "贵州茅台"
        )
        self.assertEqual(
            enhanced["news_window_days"],
            self.pipeline.search_service.news_window_days,
        )

    def test_today_not_overridden_when_trend_missing(self) -> None:
        context = {"code": "600519", "today": {"close": 15.0}}
        quote = _make_realtime_quote(price=15.72)
        enhanced = self.pipeline._enhance_context(
            context, quote, None, None, "贵州茅台"
        )
        self.assertEqual(enhanced["today"]["close"], 15.0)

    def test_today_not_overridden_when_realtime_missing(self) -> None:
        context = {"code": "600519", "today": {"close": 15.0}}
        trend = TrendAnalysisResult(code="600519", ma5=15.0, ma10=14.8, ma20=14.5)
        enhanced = self.pipeline._enhance_context(
            context, None, None, trend, "贵州茅台"
        )
        self.assertEqual(enhanced["today"]["close"], 15.0)

    def test_today_not_overridden_when_trend_ma_zero(self) -> None:
        """StockTrendAnalyzer 因数据不足提前返回 ma5=0.0 时，不应覆盖 today。"""
        context = {"code": "600519", "today": {"close": 15.0, "ma5": 14.8}}
        quote = _make_realtime_quote(price=15.72)
        trend = TrendAnalysisResult(code="600519")  # 默认 ma5=ma10=ma20=0.0
        enhanced = self.pipeline._enhance_context(
            context, quote, None, trend, "贵州茅台"
        )
        self.assertEqual(enhanced["today"]["close"], 15.0)
        self.assertEqual(enhanced["today"]["ma5"], 14.8)

    def _non_trading_day_case(self):
        """构造一个「非交易日 + 有实时报价 + 有趋势结果」的输入。"""
        last_session = date.today() - timedelta(days=2)
        context = {
            "code": "600519",
            "date": last_session.isoformat(),
            "today": {
                "code": "600519",
                "date": last_session.isoformat(),
                "open": 14.9,
                "high": 15.2,
                "low": 14.7,
                "close": 15.0,
                "volume": 1200000,
                "amount": 18000000,
                "pct_chg": 3.45,
                "data_source": "AkshareFetcher",
            },
            "yesterday": {"close": 14.5, "volume": 1000000},
        }
        quote = _make_realtime_quote(price=15.72, volume=2000000, amount=None)
        trend = TrendAnalysisResult(
            code="600519",
            trend_status=TrendStatus.BULL,
            ma5=15.5,
            ma10=15.2,
            ma20=14.9,
        )
        return context, quote, trend

    @patch("src.core.pipeline.get_market_now")
    @patch("src.core.pipeline.get_market_for_stock", return_value="cn")
    def test_today_not_overridden_when_phase_is_non_trading_day(
        self, _mock_market, mock_now
    ) -> None:
        """非交易日不做实时覆盖：保留完整官方日线，不打 estimated 标记。"""
        today = date.today()
        mock_now.return_value = datetime(
            today.year, today.month, today.day, 10, 0, tzinfo=timezone.utc
        )
        context, quote, trend = self._non_trading_day_case()
        last_session_date = context["today"]["date"]

        enhanced = self.pipeline._enhance_context(
            context,
            quote,
            None,
            trend,
            "贵州茅台",
            market_phase_context={
                "market": "cn",
                "phase": "non_trading",
                "is_trading_day": False,
                "is_market_open_now": False,
                "is_partial_bar": False,
            },
        )

        self.assertEqual(enhanced["today"]["close"], 15.0)
        self.assertEqual(enhanced["today"]["open"], 14.9)
        self.assertEqual(enhanced["today"]["amount"], 18000000)
        self.assertEqual(enhanced["today"]["date"], last_session_date)
        self.assertEqual(enhanced["today"]["data_source"], "AkshareFetcher")
        self.assertEqual(enhanced["date"], last_session_date)
        self.assertNotIn("is_estimated", enhanced["today"])
        self.assertNotIn("estimated_fields", enhanced["today"])
        self.assertNotIn("is_partial_bar", enhanced["today"])
        self.assertNotIn("realtime_source", enhanced["today"])
        # 实时行情本身仍然透传，只是不改写日线 bar。
        self.assertEqual(enhanced["realtime"]["price"], 15.72)

    @patch("src.core.pipeline.get_market_now")
    @patch("src.core.pipeline.get_market_for_stock", return_value="cn")
    def test_non_trading_day_context_keeps_technical_block_available(
        self, _mock_market, mock_now
    ) -> None:
        """非交易日的 enhanced_context 不应把 technical 数据块降级为 partial。"""
        from src.schemas.analysis_context_pack import ContextFieldStatus
        from src.services.analysis_context_builder import (
            AnalysisContextBuilder,
            PipelineAnalysisArtifacts,
        )

        today = date.today()
        mock_now.return_value = datetime(
            today.year, today.month, today.day, 10, 0, tzinfo=timezone.utc
        )
        context, quote, trend = self._non_trading_day_case()
        phase = {
            "market": "cn",
            "phase": "non_trading",
            "is_trading_day": False,
            "is_market_open_now": False,
            "is_partial_bar": False,
        }

        enhanced = self.pipeline._enhance_context(
            context, quote, None, trend, "贵州茅台", market_phase_context=phase
        )
        pack = AnalysisContextBuilder.build(
            PipelineAnalysisArtifacts(
                code="600519",
                stock_name="贵州茅台",
                market="cn",
                phase=phase,
                base_context=context,
                enhanced_context=enhanced,
                realtime_quote=quote,
                trend_result=trend,
                chip_data=None,
                fundamental_context=None,
                news_context=None,
                news_result_count=None,
                metadata={},
            )
        )
        technical = pack.blocks["technical"]

        self.assertEqual(technical.status, ContextFieldStatus.AVAILABLE)
        self.assertEqual(technical.warnings, [])
        self.assertNotIn("intraday_overlay", technical.items)
        self.assertNotIn("intraday_realtime_overlay", pack.data_quality.warnings)
        self.assertEqual(pack.data_quality.block_scores["technical"], 100)

    @patch("src.core.pipeline.get_market_now")
    @patch("src.core.pipeline.get_market_for_stock", return_value="cn")
    def test_today_still_overridden_after_close_on_a_trading_day(
        self, _mock_market, mock_now
    ) -> None:
        """收盘后仍是交易日：保留覆盖，让当日盘后快照进入 today。"""
        today = date.today()
        mock_now.return_value = datetime(
            today.year, today.month, today.day, 10, 0, tzinfo=timezone.utc
        )
        context, quote, trend = self._non_trading_day_case()

        enhanced = self.pipeline._enhance_context(
            context,
            quote,
            None,
            trend,
            "贵州茅台",
            market_phase_context={
                "market": "cn",
                "phase": "postmarket",
                "is_trading_day": True,
                "is_market_open_now": False,
                "is_partial_bar": False,
            },
        )

        self.assertEqual(enhanced["today"]["close"], 15.72)
        self.assertEqual(enhanced["today"]["data_source"], "realtime:tencent")
        self.assertTrue(enhanced["today"]["is_estimated"])
        self.assertIs(enhanced["today"]["is_partial_bar"], False)

    @patch("src.core.pipeline.get_market_now")
    @patch("src.core.pipeline.get_market_for_stock", return_value="cn")
    def test_today_overridden_when_trading_day_flag_is_unknown(
        self, _mock_market, mock_now
    ) -> None:
        """phase 未知（is_trading_day=None）时保持失败开放，仍执行覆盖。"""
        today = date.today()
        mock_now.return_value = datetime(
            today.year, today.month, today.day, 10, 0, tzinfo=timezone.utc
        )
        context, quote, trend = self._non_trading_day_case()

        enhanced = self.pipeline._enhance_context(
            context,
            quote,
            None,
            trend,
            "贵州茅台",
            market_phase_context={
                "market": "cn",
                "phase": "unknown",
                "is_trading_day": None,
                "is_market_open_now": None,
                "is_partial_bar": None,
            },
        )

        self.assertEqual(enhanced["today"]["close"], 15.72)
        self.assertEqual(enhanced["today"]["data_source"], "realtime:tencent")


if __name__ == "__main__":
    unittest.main()
