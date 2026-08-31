# -*- coding: utf-8 -*-
"""非交易日「当前价 / 涨跌幅」取数护栏的回归测试。

背景：收盘后 provider 仍会返回一份实时报价，其 ``change_pct`` 由 provider 自带的
``pre_close`` 反推，而该 ``pre_close`` 可能落后一个交易日。真实案例 MSFT 2026-08-28：

- 官方日线：前收 505.06 → 收 513.53 → ``pct_chg = 1.68``
- provider 实时快照：``pre_close = 503.09`` → ``change_pct = 2.08``

于是报告表头 / 历史列表卡片显示 ``+2.08%``，正文却写 ``1.68%``，同一份报告自相矛盾；
``change_pct`` 还会流入 DecisionSignal 抽取与回测评估，因此并非纯展示问题。

覆盖两个方向：
- 非交易日必须改用官方日线的 close / pct_chg；
- 真实盘中（含盘后但仍是交易日）必须继续使用实时报价。

同时覆盖三个入口，避免只修一处：
- ``StockAnalysisPipeline.analyze_stock``（非 Agent 路径 Step 7.5）
- ``StockAnalysisPipeline._analyze_with_agent``（Agent 路径）
- ``extract_realtime_detail_fields``（API ``meta.current_price`` / ``meta.change_pct``，
  也就是历史列表卡片与报告表头的取数来源）
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from data_provider.realtime_types import RealtimeSource, UnifiedRealtimeQuote
from src.core.pipeline import (
    is_confirmed_non_trading_day,
    resolve_result_price_change,
)
from src.enums import ReportType
from src.utils.data_processing import extract_realtime_detail_fields
from tests.test_pipeline_market_phase_context import _make_pipeline


# 真实 MSFT 2026-08-28 数值，保证回归测试锚定线上复现过的缺陷。
OFFICIAL_CLOSE = 513.53
OFFICIAL_PCT = 1.68
QUOTE_PRICE = 513.530029296875
QUOTE_PCT = 2.08

OFFICIAL_TODAY = {
    "code": "MSFT",
    "date": "2026-08-28",
    "open": 505.33,
    "high": 517.78,
    "low": 504.8675,
    "close": OFFICIAL_CLOSE,
    "volume": 29207325.0,
    "pct_chg": OFFICIAL_PCT,
    "ma5": 498.8,
    "ma10": 490.47,
    "ma20": 493.35,
    "data_source": "AlphaVantageFetcher",
}
REALTIME_BLOCK = {"price": QUOTE_PRICE, "change_pct": QUOTE_PCT, "source": "fallback"}


def _phase(is_trading_day, phase="non_trading"):
    return {
        "market": "us",
        "phase": phase,
        "market_local_time": "2026-08-30T12:36:13-04:00",
        "session_date": "2026-08-30",
        "effective_daily_bar_date": "2026-08-28",
        "is_trading_day": is_trading_day,
        "is_market_open_now": False if is_trading_day is not True else True,
        "is_partial_bar": False,
        "minutes_to_open": None,
        "minutes_to_close": None,
        "trigger_source": "api",
        "analysis_intent": "auto",
        "warnings": [],
    }


class IsConfirmedNonTradingDayTestCase(unittest.TestCase):
    def test_only_explicit_false_counts(self):
        self.assertTrue(is_confirmed_non_trading_day({"is_trading_day": False}))
        self.assertFalse(is_confirmed_non_trading_day({"is_trading_day": True}))
        self.assertFalse(is_confirmed_non_trading_day({"is_trading_day": None}))
        self.assertFalse(is_confirmed_non_trading_day({}))
        self.assertFalse(is_confirmed_non_trading_day(None))
        # 不接受字符串等价物，避免 phase 语义漂移后静默改变行为
        self.assertFalse(is_confirmed_non_trading_day({"is_trading_day": "false"}))


class ResolveResultPriceChangeTestCase(unittest.TestCase):
    def test_non_trading_day_uses_official_daily_bar(self):
        price, change_pct = resolve_result_price_change(
            REALTIME_BLOCK, OFFICIAL_TODAY, _phase(False)
        )
        self.assertEqual(price, OFFICIAL_CLOSE)
        self.assertEqual(change_pct, OFFICIAL_PCT)

    def test_intraday_session_keeps_live_quote(self):
        price, change_pct = resolve_result_price_change(
            REALTIME_BLOCK, OFFICIAL_TODAY, _phase(True, phase="intraday")
        )
        self.assertEqual(price, QUOTE_PRICE)
        self.assertEqual(change_pct, QUOTE_PCT)

    def test_postmarket_on_a_trading_day_keeps_live_quote(self):
        price, change_pct = resolve_result_price_change(
            REALTIME_BLOCK, OFFICIAL_TODAY, _phase(True, phase="postmarket")
        )
        self.assertEqual(price, QUOTE_PRICE)
        self.assertEqual(change_pct, QUOTE_PCT)

    def test_unknown_phase_fails_open_to_live_quote(self):
        for phase_context in (None, {}, _phase(None), {"phase": "unknown"}):
            with self.subTest(phase_context=phase_context):
                price, change_pct = resolve_result_price_change(
                    REALTIME_BLOCK, OFFICIAL_TODAY, phase_context
                )
                self.assertEqual(price, QUOTE_PRICE)
                self.assertEqual(change_pct, QUOTE_PCT)

    def test_estimated_daily_bar_is_not_treated_as_official(self):
        estimated = {**OFFICIAL_TODAY, "is_estimated": True, "pct_chg": QUOTE_PCT}
        price, change_pct = resolve_result_price_change(
            REALTIME_BLOCK, estimated, _phase(False)
        )
        self.assertEqual(price, QUOTE_PRICE)
        self.assertEqual(change_pct, QUOTE_PCT)

    def test_missing_daily_bar_fails_open(self):
        for daily_bar in (None, {}, "not-a-dict"):
            with self.subTest(daily_bar=daily_bar):
                price, change_pct = resolve_result_price_change(
                    REALTIME_BLOCK, daily_bar, _phase(False)
                )
                self.assertEqual(price, QUOTE_PRICE)
                self.assertEqual(change_pct, QUOTE_PCT)

    def test_partial_daily_bar_falls_back_field_by_field(self):
        without_pct = {k: v for k, v in OFFICIAL_TODAY.items() if k != "pct_chg"}
        price, change_pct = resolve_result_price_change(
            REALTIME_BLOCK, without_pct, _phase(False)
        )
        self.assertEqual(price, OFFICIAL_CLOSE)
        self.assertEqual(change_pct, QUOTE_PCT)

    def test_nonfinite_official_fields_do_not_override_valid_quote(self):
        invalid_daily_bar = {
            **OFFICIAL_TODAY,
            "close": float("nan"),
            "pct_chg": "Infinity",
        }
        price, change_pct = resolve_result_price_change(
            REALTIME_BLOCK, invalid_daily_bar, _phase(False)
        )
        self.assertEqual(price, QUOTE_PRICE)
        self.assertEqual(change_pct, QUOTE_PCT)

    def test_official_fields_override_independently_and_preserve_zero(self):
        price, change_pct = resolve_result_price_change(
            REALTIME_BLOCK,
            {**OFFICIAL_TODAY, "close": 0, "pct_chg": "-Infinity"},
            _phase(False),
        )
        self.assertEqual(price, 0)
        self.assertEqual(change_pct, QUOTE_PCT)

        price, change_pct = resolve_result_price_change(
            REALTIME_BLOCK,
            {**OFFICIAL_TODAY, "close": float("inf"), "pct_chg": 0},
            _phase(False),
        )
        self.assertEqual(price, QUOTE_PRICE)
        self.assertEqual(change_pct, 0)

    def test_nonfinite_realtime_values_are_not_returned(self):
        price, change_pct = resolve_result_price_change(
            {"price": float("nan"), "change_pct": "Infinity"},
            OFFICIAL_TODAY,
            _phase(True, phase="intraday"),
        )
        self.assertIsNone(price)
        self.assertIsNone(change_pct)

    def test_missing_realtime_block_is_tolerated(self):
        price, change_pct = resolve_result_price_change(None, OFFICIAL_TODAY, _phase(False))
        self.assertEqual(price, OFFICIAL_CLOSE)
        self.assertEqual(change_pct, OFFICIAL_PCT)


class ExtractRealtimeDetailFieldsTestCase(unittest.TestCase):
    """API meta / 历史列表卡片取数：非交易日必须与报告正文一致。"""

    @staticmethod
    def _snapshot(is_trading_day, today=None, realtime=None, with_phase=True):
        snapshot = {
            "enhanced_context": {
                "code": "MSFT",
                "today": OFFICIAL_TODAY if today is None else today,
                "realtime": REALTIME_BLOCK if realtime is None else realtime,
            },
            "realtime_quote_raw": {"price": QUOTE_PRICE, "change_pct": QUOTE_PCT},
        }
        if with_phase:
            snapshot["market_phase_summary"] = _phase(is_trading_day)
        return snapshot

    def test_non_trading_day_snapshot_reports_official_change_pct(self):
        fields = extract_realtime_detail_fields(self._snapshot(False))
        self.assertEqual(fields["current_price"], OFFICIAL_CLOSE)
        self.assertEqual(fields["change_pct"], OFFICIAL_PCT)

    def test_trading_day_snapshot_keeps_live_quote(self):
        fields = extract_realtime_detail_fields(self._snapshot(True))
        self.assertEqual(fields["current_price"], QUOTE_PRICE)
        self.assertEqual(fields["change_pct"], QUOTE_PCT)

    def test_snapshot_without_phase_summary_fails_open(self):
        fields = extract_realtime_detail_fields(self._snapshot(False, with_phase=False))
        self.assertEqual(fields["current_price"], QUOTE_PRICE)
        self.assertEqual(fields["change_pct"], QUOTE_PCT)

    def test_legacy_snapshot_with_estimated_today_is_left_alone(self):
        """2328e8a3 之前的旧快照里 today 本身就是实时估算 bar，没有官方日线可用。"""
        estimated = {**OFFICIAL_TODAY, "is_estimated": True, "pct_chg": QUOTE_PCT}
        fields = extract_realtime_detail_fields(self._snapshot(False, today=estimated))
        self.assertEqual(fields["current_price"], QUOTE_PRICE)
        self.assertEqual(fields["change_pct"], QUOTE_PCT)

    def test_agent_shape_snapshot_is_corrected_too(self):
        """Agent 模式快照没有 enhanced_context.realtime，只有顶层 realtime_quote。"""
        snapshot = {
            "enhanced_context": {"code": "MSFT", "today": OFFICIAL_TODAY},
            "realtime_quote": {"price": QUOTE_PRICE, "change_pct": QUOTE_PCT},
            "market_phase_summary": _phase(False),
        }
        fields = extract_realtime_detail_fields(snapshot)
        self.assertEqual(fields["current_price"], OFFICIAL_CLOSE)
        self.assertEqual(fields["change_pct"], OFFICIAL_PCT)

    def test_json_string_snapshot_is_supported(self):
        import json

        fields = extract_realtime_detail_fields(json.dumps(self._snapshot(False)))
        self.assertEqual(fields["change_pct"], OFFICIAL_PCT)

    def test_nonfinite_preferred_quote_falls_back_to_later_finite_quote(self):
        snapshot = self._snapshot(
            True,
            realtime={"price": float("inf"), "change_pct": "Infinity"},
        )
        fields = extract_realtime_detail_fields(snapshot)
        self.assertEqual(fields["current_price"], QUOTE_PRICE)
        self.assertEqual(fields["change_pct"], QUOTE_PCT)

    def test_invalid_official_fields_do_not_erase_valid_quote(self):
        invalid_today = {
            **OFFICIAL_TODAY,
            "close": float("nan"),
            "pct_chg": "-Infinity",
        }
        fields = extract_realtime_detail_fields(
            self._snapshot(False, today=invalid_today)
        )
        self.assertEqual(fields["current_price"], QUOTE_PRICE)
        self.assertEqual(fields["change_pct"], QUOTE_PCT)

    def test_nonfinite_only_snapshot_returns_empty_fields(self):
        snapshot = self._snapshot(
            True,
            realtime={"price": float("inf"), "change_pct": "Infinity"},
        )
        snapshot["realtime_quote_raw"] = {
            "price": float("nan"),
            "change_pct": "-Infinity",
        }
        fields = extract_realtime_detail_fields(snapshot)
        self.assertEqual(
            fields,
            {"current_price": None, "change_pct": None},
        )

    def test_non_dict_snapshot_returns_empty_fields(self):
        self.assertEqual(
            extract_realtime_detail_fields(None),
            {"current_price": None, "change_pct": None},
        )


def _quote() -> UnifiedRealtimeQuote:
    return UnifiedRealtimeQuote(
        code="MSFT",
        name="Microsoft Corporation",
        source=RealtimeSource.FALLBACK,
        price=QUOTE_PRICE,
        change_pct=QUOTE_PCT,
        open_price=505.33,
        high=517.78,
        low=504.87,
        volume=29178300,
        pre_close=503.09,
    )


def _pipeline(agent_mode: bool):
    pipeline = _make_pipeline(agent_mode=agent_mode, save_context_snapshot=True)
    pipeline.config.enable_realtime_quote = True
    pipeline.fetcher_manager.get_stock_name.return_value = "Microsoft Corporation"
    pipeline.fetcher_manager.get_realtime_quote.return_value = _quote()
    pipeline.db.get_analysis_context.return_value = {
        "code": "MSFT",
        "stock_name": "Microsoft Corporation",
        "date": "2026-08-28",
        "today": dict(OFFICIAL_TODAY),
        "yesterday": {"close": 505.06, "volume": 28688756.0},
    }
    pipeline._ensure_agent_history = MagicMock()
    return pipeline


class LegacyPipelineResultFieldsTestCase(unittest.TestCase):
    """非 Agent 路径 Step 7.5：result.change_pct 会流入 DecisionSignal 与回测。"""

    def _run(self, phase_payload):
        pipeline = _pipeline(agent_mode=False)
        phase_context = SimpleNamespace(to_dict=MagicMock(return_value=phase_payload))
        with patch(
            "src.core.pipeline.build_market_phase_context", return_value=phase_context
        ):
            return pipeline.analyze_stock("MSFT", ReportType.SIMPLE, "q-legacy")

    def test_non_trading_day_result_uses_official_daily_figures(self):
        result = self._run(_phase(False))
        self.assertIsNotNone(result)
        self.assertEqual(result.change_pct, OFFICIAL_PCT)
        self.assertEqual(result.current_price, OFFICIAL_CLOSE)

    def test_intraday_session_result_keeps_live_quote(self):
        result = self._run(_phase(True, phase="intraday"))
        self.assertIsNotNone(result)
        self.assertEqual(result.change_pct, QUOTE_PCT)
        self.assertEqual(result.current_price, QUOTE_PRICE)


class AgentPipelineResultFieldsTestCase(unittest.TestCase):
    """Agent 路径必须与非 Agent 路径同护栏，避免只修一个入口。"""

    def _run_with_pipeline(self, phase_payload):
        from src.agent.executor import AgentResult

        pipeline = _pipeline(agent_mode=True)
        executor = MagicMock()
        executor.run.return_value = AgentResult(
            success=True,
            content="{}",
            dashboard={
                "stock_name": "Microsoft Corporation",
                "sentiment_score": 56,
                "trend_prediction": "震荡",
                "operation_advice": "观望",
                "decision_type": "hold",
            },
            provider="test",
        )
        with patch("src.agent.factory.build_agent_executor", return_value=executor):
            result = pipeline._analyze_with_agent(
                code="MSFT",
                report_type=ReportType.SIMPLE,
                query_id="q-agent",
                stock_name="Microsoft Corporation",
                realtime_quote=_quote(),
                chip_data=None,
                fundamental_context={"market": "us"},
                trend_result=None,
                market_phase_context=phase_payload,
                market_phase_summary=phase_payload,
            )
        return pipeline, result

    def _run(self, phase_payload):
        return self._run_with_pipeline(phase_payload)[1]

    def test_non_trading_day_result_uses_official_daily_figures(self):
        result = self._run(_phase(False))
        self.assertIsNotNone(result)
        self.assertEqual(result.change_pct, OFFICIAL_PCT)
        self.assertEqual(result.current_price, OFFICIAL_CLOSE)

    def test_intraday_session_result_keeps_live_quote(self):
        result = self._run(_phase(True, phase="intraday"))
        self.assertIsNotNone(result)
        self.assertEqual(result.change_pct, QUOTE_PCT)
        self.assertEqual(result.current_price, QUOTE_PRICE)

    def test_saved_agent_snapshot_persists_the_real_daily_bar_context(self):
        """Exercise the real Agent save path, not a hand-built ideal snapshot."""
        pipeline, result = self._run_with_pipeline(_phase(False))

        self.assertIsNotNone(result)
        pipeline.db.save_analysis_history.assert_called_once()
        snapshot = pipeline.db.save_analysis_history.call_args.kwargs["context_snapshot"]
        enhanced = snapshot["enhanced_context"]
        self.assertEqual(enhanced["date"], "2026-08-28")
        self.assertEqual(enhanced["today"], OFFICIAL_TODAY)
        self.assertEqual(enhanced["yesterday"], {"close": 505.06, "volume": 28688756.0})

        # The persisted shape must be sufficient for the same history/API reader
        # that serves meta.current_price and meta.change_pct after process restart.
        fields = extract_realtime_detail_fields(snapshot)
        self.assertEqual(fields["current_price"], OFFICIAL_CLOSE)
        self.assertEqual(fields["change_pct"], OFFICIAL_PCT)


if __name__ == "__main__":
    unittest.main()
