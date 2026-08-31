# -*- coding: utf-8 -*-
"""决策仪表盘空值渲染的回归测试。

背景：`dashboard.data_perspective` 由模型产出，数据管线拿不到的字段会以**显式
`null`** 出现（例如部分美股或 ETF 因股本字段缺失而无法计算换手率，以及部分
数据源没有量比时的 `volume_ratio`）。
展示层用的是 `dict.get(key, "N/A")` / Jinja `get(key, 'N/A')`，默认值只在 key
**缺失**时生效，key 存在但值为 `null` 时默认值不会触发，于是报告直接渲染出字面量
`None`。线上实测 42 份美股报告里有 30 份出现 `换手率 None%`、22 份出现 `量比 None`
（且同一行还给出了「缩量」这种量能判断，读起来像是有数据）。

三个渲染入口必须一致（避免只修一处）：
- `NotificationService.generate_dashboard_report` / `generate_single_stock_report`
- `HistoryService._generate_single_stock_markdown`（Web/API 报告正文）
- `templates/report_markdown.j2`（Jinja 模板渲染器）
"""

import json
import os
import re
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from src.analyzer import AnalysisResult
from src.services.report_renderer import render
from src.utils.data_processing import (
    display_fraction_as_percent,
    display_numeric_with_suffix,
    display_value_or_na,
    extract_fundamental_context,
    extract_fundamental_detail_fields,
)


_INVALID_NUMBER_TOKEN = re.compile(
    r"(?<![a-z])(nan|inf|infinity)(?![a-z])",
    re.IGNORECASE,
)
_LITERAL_NONE_TOKEN = re.compile(r"(?<![a-z])none(?![a-z])", re.IGNORECASE)


def _assert_no_invalid_number_rendering(test_case, output: str) -> None:
    test_case.assertIsNone(_INVALID_NUMBER_TOKEN.search(output))
    test_case.assertIsNone(_LITERAL_NONE_TOKEN.search(output))
    for invalid in ("None%", "N/A%", "N/A/100"):
        test_case.assertNotIn(invalid, output)


def _dashboard(*, null_fields: bool):
    """构造 data_perspective：null_fields=True 模拟美股缺换手率/量比的真实形态。"""
    return {
        "core_conclusion": {
            "one_sentence": "测试结论",
            "signal": "hold",
            "confidence": "中",
        },
        "data_perspective": {
            "trend_status": {
                "ma_alignment": None if null_fields else "空头排列",
                "is_bullish": False,
                "trend_score": None if null_fields else 45,
            },
            "price_position": {
                "current_price": None if null_fields else 90.28,
                "ma5": None if null_fields else 91.27,
                "ma10": None if null_fields else 93.53,
                "ma20": None if null_fields else 95.10,
                "bias_ma5": None if null_fields else -1.08,
                "bias_status": "安全",
                "support_level": None if null_fields else 90.04,
                "resistance_level": None if null_fields else 91.27,
            },
            "volume_analysis": {
                "volume_ratio": None if null_fields else 0.96,
                "volume_status": "缩量",
                # 部分美股或 ETF 因 floatShares/sharesOutstanding 缺失而无法计算。
                "turnover_rate": None,
                "volume_meaning": "成交量约为前一交易日的0.92倍。",
            },
        },
    }


def _result(dashboard) -> AnalysisResult:
    return AnalysisResult(
        code="SOXQ",
        name="Invesco PHLX Semiconductor ETF",
        sentiment_score=41,
        trend_prediction="看空",
        operation_advice="观望",
        decision_type="hold",
        confidence_level="中",
        dashboard=dashboard,
        analysis_summary="测试摘要",
        report_language="zh",
    )


class _Record:
    created_at = None


def _data_perspective_renderings(result: AnalysisResult, dashboard):
    """只包含真正渲染 data_perspective 数据透视块的三个入口。"""
    from src.notification import NotificationService
    from src.services.history_service import HistoryService

    notification = NotificationService()
    return {
        "notification_dashboard": notification.generate_dashboard_report([result]),
        "history_markdown": HistoryService.__new__(HistoryService)._generate_single_stock_markdown(
            result, _Record()
        ),
        "jinja_markdown": render(
            "markdown", [result], summary_only=False, extra_context={"report_language": "zh"}
        ),
    }


def _all_renderings(result: AnalysisResult, dashboard):
    """再加上不渲染数据透视块的简报入口，确保它同样不泄漏字面量 None。"""
    from src.notification import NotificationService

    renderings = _data_perspective_renderings(result, dashboard)
    renderings["notification_single"] = NotificationService().generate_single_stock_report(result)
    return renderings


class DisplayValueOrNaTestCase(unittest.TestCase):
    def test_missing_values_become_placeholder(self):
        for value in (
            None,
            float("nan"),
            float("inf"),
            "",
            "   ",
            "None",
            "null",
            "NaN",
            "inf",
            "-Infinity",
            "n/a",
            "-",
        ):
            with self.subTest(value=value):
                self.assertEqual(display_value_or_na(value), "N/A")

    def test_real_values_survive_including_zero_and_false(self):
        self.assertEqual(display_value_or_na(0), "0")
        self.assertEqual(display_value_or_na(0.0), "0.0")
        self.assertEqual(display_value_or_na(0.96), "0.96")
        self.assertEqual(display_value_or_na(-1.08), "-1.08")
        self.assertEqual(display_value_or_na(False), "False")
        self.assertEqual(display_value_or_na("空头排列"), "空头排列")

    def test_custom_placeholder(self):
        self.assertEqual(display_value_or_na(None, "—"), "—")

    def test_strings_are_stripped(self):
        self.assertEqual(display_value_or_na("  空头排列  "), "空头排列")

    def test_numeric_suffix_is_only_added_to_finite_numbers(self):
        for value in (None, float("nan"), "null", "数据缺失，无法判断"):
            with self.subTest(value=value):
                expected = "数据缺失，无法判断" if isinstance(value, str) and value.startswith("数据") else "N/A"
                self.assertEqual(display_numeric_with_suffix(value, "%"), expected)
        self.assertEqual(display_numeric_with_suffix(0, "%"), "0%")
        self.assertEqual(display_numeric_with_suffix(-1.08, "%"), "-1.08%")
        self.assertEqual(display_numeric_with_suffix("45", "/100"), "45/100")
        self.assertEqual(display_numeric_with_suffix("45/100", "/100"), "45/100")
        self.assertEqual(display_numeric_with_suffix("1.08%", "%"), "1.08%")

    def test_fraction_percent_rejects_nonfinite_and_preserves_text(self):
        for value in (None, float("nan"), float("inf"), "-Infinity"):
            with self.subTest(value=value):
                self.assertEqual(display_fraction_as_percent(value), "N/A")
        self.assertEqual(display_fraction_as_percent(0), "0%")
        self.assertEqual(display_fraction_as_percent(0.625), "62%")
        self.assertEqual(display_fraction_as_percent("12.5%", precision=1), "12.5%")
        self.assertEqual(
            display_fraction_as_percent("数据缺失，无法判断"),
            "数据缺失，无法判断",
        )


class FundamentalJsonSafetyTestCase(unittest.TestCase):
    def test_structured_fundamentals_are_recursively_strict_json_safe(self):
        import numpy as np

        fallback = {
            "earnings": {
                "data": {
                    "financial_report": {
                        "report_date": "2026-06-30",
                        "revenue": 123.0,
                        "all_nonfinite_overlay": [1.0, 2.0],
                    },
                    "dividend": {
                        "ttm_event_count": 4,
                    },
                }
            }
        }
        snapshot = {
            "enhanced_context": {
                "fundamental_context": {
                    "earnings": {
                        "data": {
                            "financial_report": {
                                # Missing provider sentinels must not replace the
                                # finite fallback value.
                                "revenue": float("nan"),
                                "roe": float("inf"),
                                "series": np.array([0.0, np.nan, np.inf]),
                                "all_nonfinite_overlay": np.array([np.nan, np.inf]),
                                "event_count": np.int64(0),
                                "audited": np.bool_(False),
                            },
                            "dividend": {
                                "yield": "-Infinity",
                                "ttm_event_count": np.float64(np.inf),
                                "events": (
                                    {"amount": np.float64(0.0)},
                                    {"amount": np.float64(np.nan)},
                                ),
                            },
                        }
                    }
                }
            }
        }

        context = extract_fundamental_context(snapshot, fallback)
        details = extract_fundamental_detail_fields(snapshot, fallback)

        # Exercise both the extraction result and the API schema boundary used
        # by analysis/history detail responses.
        from api.v1.schemas.history import ReportDetails

        api_details = ReportDetails(
            financial_report=details["financial_report"],
            dividend_metrics=details["dividend_metrics"],
        ).model_dump(mode="json")
        json.dumps(context, allow_nan=False)
        json.dumps(details, allow_nan=False)
        json.dumps(api_details, allow_nan=False)

        report = details["financial_report"]
        dividend = details["dividend_metrics"]
        self.assertEqual(report["revenue"], 123.0)
        self.assertIsNone(report.get("roe"))
        self.assertEqual(report["series"], [0.0, None, None])
        self.assertEqual(report["all_nonfinite_overlay"], [1.0, 2.0])
        self.assertEqual(report["event_count"], 0)
        self.assertIs(report["audited"], False)
        self.assertIsNone(dividend.get("yield"))
        self.assertEqual(dividend["ttm_event_count"], 4)
        self.assertEqual(dividend["events"], [{"amount": 0.0}, {"amount": None}])


class MarketSnapshotNullRenderingTestCase(unittest.TestCase):
    def test_nonfinite_snapshot_values_do_not_leak_or_receive_units(self):
        from src.services.history_service import HistoryService

        result = _result(_dashboard(null_fields=False))
        result.current_price = 513.53
        result.change_pct = 1.68
        result.market_snapshot = {
            "close": None,
            "prev_close": "NaN",
            "open": float("nan"),
            "high": float("inf"),
            "low": "-Infinity",
            "pct_chg": "NaN",
            "change_amount": None,
            "amplitude": float("nan"),
            "volume": None,
            "amount": float("inf"),
            "price": float("nan"),
            "volume_ratio": None,
            "turnover_rate": "inf",
            "source": None,
        }

        from src.notification import NotificationService

        outputs = {
            "notification": NotificationService().generate_single_stock_report(result),
            "history": HistoryService.__new__(
                HistoryService
            )._generate_single_stock_markdown(result, _Record()),
            "jinja": render(
                "markdown",
                [result],
                summary_only=False,
                extra_context={"report_language": "zh"},
            ),
        }

        for name, output in outputs.items():
            with self.subTest(renderer=name):
                self.assertNotIn("| None |", output)
                self.assertNotIn("N/A%", output)
                self.assertIsNone(
                    re.search(
                        r"(?<![a-z])(nan|inf|infinity)(?![a-z])",
                        output.lower(),
                    )
                )
        self.assertIn("**513.53** (+1.68%)", outputs["history"])

    def test_safe_number_formatter_rejects_nonfinite_values(self):
        from src.services.history_service import HistoryService

        for value in (float("nan"), float("inf"), "nan", "-Infinity"):
            with self.subTest(value=value):
                self.assertEqual(HistoryService._safe_format_number(value), "N/A")

    def test_history_market_fields_skip_nonfinite_values_and_use_valid_fallbacks(self):
        from src.services.history_service import HistoryService

        snapshot = {
            "enhanced_context": {
                "realtime": {
                    "price": float("inf"),
                    "change_pct": "Infinity",
                    "volume_ratio": float("nan"),
                    "turnover_rate": "-Infinity",
                },
            },
            "realtime_quote_raw": {
                "price": 513.53,
                "change_pct": 1.68,
                "volume_ratio": 0,
                "turnover_rate": 1.25,
            },
        }

        fields = HistoryService.__new__(HistoryService)._extract_history_market_fields(
            snapshot
        )

        self.assertEqual(
            fields,
            {
                "current_price": 513.53,
                "change_pct": 1.68,
                "volume_ratio": 0.0,
                "turnover_rate": 1.25,
            },
        )


class AnalyzerPromptFiniteNumberTestCase(unittest.TestCase):
    @staticmethod
    def _analyzer():
        from src.analyzer import GeminiAnalyzer

        analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
        analyzer._config_override = SimpleNamespace(
            news_max_age_days=3,
            news_strategy_profile="short",
        )
        analyzer._skill_instructions_override = ""
        analyzer._default_skill_policy_override = ""
        analyzer._use_legacy_default_prompt_override = True
        analyzer._resolved_prompt_state = None
        return analyzer

    def test_prompt_and_snapshot_reject_nonfinite_numeric_inputs(self):
        analyzer = self._analyzer()
        context = {
            "code": "MSFT",
            "date": "2026-08-28",
            "today": {
                "close": None,
                "open": float("nan"),
                "high": float("inf"),
                "low": "-Infinity",
                "pct_chg": float("nan"),
                "volume": float("nan"),
                "amount": float("inf"),
                "ma5": float("nan"),
                "ma10": "Infinity",
                "ma20": None,
            },
            "realtime": {
                "price": float("inf"),
                "volume_ratio": float("nan"),
                "turnover_rate": "-Infinity",
                "pe_ratio": float("nan"),
                "pb_ratio": float("inf"),
                "total_mv": float("nan"),
                "circ_mv": float("inf"),
                "change_60d": "Infinity",
            },
        }

        prompt = analyzer._format_prompt(context, "Microsoft", news_context=None)
        snapshot = analyzer._build_market_snapshot(context)
        combined = f"{prompt}\n{snapshot}"

        self.assertNotIn("None%", combined)
        self.assertNotIn("nan 元", combined.lower())
        self.assertNotIn("inf 元", combined.lower())
        self.assertNotIn("nan 股", combined.lower())
        self.assertNotIn("inf 亿元", combined.lower())
        self.assertIsNone(
            re.search(
                r"(?<![a-z])(nan|inf|infinity)(?![a-z])",
                combined.lower(),
            )
        )
        self.assertIn("| 收盘价 | N/A |", prompt)
        self.assertEqual(snapshot["volume"], "N/A")
        self.assertEqual(snapshot["amount"], "N/A")

    def test_prompt_and_snapshot_preserve_zero_and_finite_values(self):
        analyzer = self._analyzer()
        context = {
            "code": "MSFT",
            "date": "2026-08-28",
            "today": {
                "close": 0,
                "open": 1,
                "high": 2,
                "low": 0,
                "pct_chg": 0,
                "volume": 0,
                "amount": 0,
            },
            "realtime": {
                "price": 513.53,
                "volume_ratio": 0,
                "turnover_rate": 0,
                "total_mv": 1e8,
                "circ_mv": 5e7,
                "change_60d": 0,
            },
        }

        prompt = analyzer._format_prompt(context, "Microsoft", news_context=None)
        snapshot = analyzer._build_market_snapshot(context)

        self.assertIn("| 收盘价 | 0 元 |", prompt)
        self.assertIn("| 涨跌幅 | 0% |", prompt)
        self.assertIn("| **量比** | **0** |", prompt)
        self.assertIn("| **换手率** | **0%** |", prompt)
        self.assertEqual(snapshot["volume"], "0 股")
        self.assertEqual(snapshot["amount"], "0 元")

    def test_yesterday_chip_trend_and_fundamentals_reject_nonfinite_values(self):
        analyzer = self._analyzer()
        context = {
            "code": "MSFT",
            "date": "2026-08-28",
            "today": {"close": 1, "pct_chg": 0, "volume": 0, "amount": 0},
            "yesterday": {"close": 1, "volume": 1},
            "volume_change_ratio": float("nan"),
            "price_change_ratio": "Infinity",
            "chip": {
                "profit_ratio": None,
                "avg_cost": float("nan"),
                "concentration_90": float("inf"),
                "concentration_70": "-Infinity",
                "chip_status": float("nan"),
            },
            "trend_analysis": {
                "trend_status": "震荡",
                "ma_alignment": "缠绕",
                "trend_strength": float("inf"),
                "bias_ma5": None,
                "bias_ma10": float("-inf"),
                "volume_status": float("nan"),
                "volume_trend": "Infinity",
                "buy_signal": "观望",
                "signal_score": "Infinity",
            },
            "fundamental_context": {
                "earnings": {
                    "data": {
                        "financial_report": {
                            "report_date": "2026-06-30",
                            "revenue": float("nan"),
                            "net_profit_parent": float("inf"),
                            "operating_cash_flow": "-Infinity",
                            "roe": float("nan"),
                        },
                        "dividend": {
                            "ttm_cash_dividend_per_share": float("inf"),
                            "ttm_dividend_yield_pct": "Infinity",
                            "ttm_event_count": float("nan"),
                        },
                    }
                },
                "capital_flow": {
                    "data": {
                        "stock_flow": {
                            "main_net_inflow": float("nan"),
                            "inflow_5d": float("inf"),
                            "inflow_10d": "-Infinity",
                        }
                    }
                },
                "institution": {
                    "status": "ok",
                    "data": {
                        "foreign_net": float("nan"),
                        "trust_net": float("inf"),
                        "dealer_net": "-Infinity",
                        "total_net": None,
                    },
                },
            },
        }

        prompt = analyzer._format_prompt(context, "Microsoft", news_context=None)

        _assert_no_invalid_number_rendering(self, prompt)
        self.assertIn("成交量较昨日变化：N/A", prompt)
        self.assertIn("价格较昨日变化：N/A", prompt)
        self.assertIn("| **获利比例** | **N/A** |", prompt)
        self.assertIn("| 平均成本 | N/A |", prompt)
        self.assertIn("| 趋势强度 | N/A |", prompt)
        self.assertIn("| **乖离率(MA5)** | **N/A** |", prompt)
        self.assertNotIn("### 主力资金流向", prompt)
        self.assertNotIn("### 三大法人动向", prompt)

    def test_yesterday_chip_and_trend_preserve_zero_values(self):
        analyzer = self._analyzer()
        context = {
            "code": "MSFT",
            "date": "2026-08-28",
            "today": {"close": 0, "pct_chg": 0, "volume": 0, "amount": 0},
            "yesterday": {"close": 0, "volume": 0},
            "volume_change_ratio": 0,
            "price_change_ratio": 0,
            "chip": {
                "profit_ratio": 0,
                "avg_cost": 0,
                "concentration_90": 0,
                "concentration_70": 0,
                "chip_status": "中性",
            },
            "trend_analysis": {
                "trend_status": "震荡",
                "ma_alignment": "缠绕",
                "trend_strength": 0,
                "bias_ma5": 0,
                "bias_ma10": 0,
                "volume_status": "平量",
                "buy_signal": "观望",
                "signal_score": 0,
            },
        }

        prompt = analyzer._format_prompt(context, "Microsoft", news_context=None)

        self.assertIn("成交量较昨日变化：0倍", prompt)
        self.assertIn("价格较昨日变化：0%", prompt)
        self.assertIn("| **获利比例** | **0.0%** |", prompt)
        self.assertIn("| 平均成本 | 0 元 |", prompt)
        self.assertIn("| 90%筹码集中度 | 0.00% |", prompt)
        self.assertIn("| 趋势强度 | 0/100 |", prompt)
        self.assertIn("| **乖离率(MA5)** | **+0.00%** |", prompt)
        self.assertIn("| 系统评分 | 0/100 |", prompt)


class NullDashboardFieldRenderingTestCase(unittest.TestCase):
    """所有报告入口都不得把 null 渲染成字面量 None。"""

    def test_no_literal_none_leaks_into_any_report(self):
        dashboard = _dashboard(null_fields=True)
        for name, output in _all_renderings(_result(dashboard), dashboard).items():
            with self.subTest(renderer=name):
                self.assertIsNotNone(output, f"{name} returned None")
                self.assertNotIn("None%", output)
                self.assertNotIn("量比 None", output)
                self.assertNotIn("| None |", output)
                self.assertNotIn("None/100", output)

    def test_explicit_null_core_position_and_strategy_text_uses_safe_fallbacks(self):
        dashboard = _dashboard(null_fields=False)
        dashboard["core_conclusion"] = {
            "one_sentence": None,
            "time_sensitivity": None,
            "position_advice": {
                "no_position": None,
                "has_position": "None",
            },
        }
        dashboard["battle_plan"] = {
            "position_strategy": {
                "suggested_position": None,
                "entry_plan": "Infinity",
                "risk_control": float("nan"),
            }
        }
        dashboard["strategy_synthesis"] = {
            "final_signal": "hold",
            "consensus_level": "medium",
            "conflict_severity": "none",
            "conflict_count": None,
            "confidence": 0,
            "supporting_skills": [
                {"skill_id": "None", "signal": "NaN", "confidence": None},
            ],
            "opposing_skills": [
                {"skill_id": "Infinity", "signal": "Inf", "confidence": "None"},
            ],
            "conflicts": [
                {
                    "conflict_type": "NaN",
                    "severity": "Infinity",
                    "participants": ["None", "bull_trend"],
                },
            ],
        }
        dashboard["intelligence"] = {
            "risk_alerts": [None, "Infinity"],
            "positive_catalysts": [float("nan")],
        }

        outputs = _all_renderings(_result(dashboard), dashboard)
        for name, output in outputs.items():
            with self.subTest(renderer=name):
                _assert_no_invalid_number_rendering(self, output)
                self.assertIn("测试摘要", output)

        for name in ("notification_dashboard", "history_markdown", "jinja_markdown"):
            with self.subTest(renderer=name):
                output = outputs[name]
                self.assertIn("(N/A)", output)
                self.assertIn("仓位建议**: N/A", output)
                self.assertIn("建仓策略: N/A", output)
                self.assertIn("风控策略: N/A", output)

    def test_jinja_history_preserves_zero_sentiment_score(self):
        result = _result(_dashboard(null_fields=False))
        history_item = SimpleNamespace(
            created_at="2026-08-28 12:34:56",
            sentiment_score=0,
            action_label="观望",
            action="hold",
            operation_advice="观望",
            trend_prediction="震荡",
        )

        output = render(
            "markdown",
            [result],
            summary_only=False,
            extra_context={
                "report_language": "zh",
                "history_by_code": {result.code: [history_item]},
            },
        )

        self.assertIn("| 2026-08-28 12:34 | 0 | 观望 | 震荡 |", output)
        self.assertNotIn("| 2026-08-28 12:34 | N/A |", output)

    def test_missing_numeric_metrics_render_without_orphaned_units(self):
        dashboard = _dashboard(null_fields=True)
        for name, output in _data_perspective_renderings(_result(dashboard), dashboard).items():
            with self.subTest(renderer=name):
                self.assertIn("换手率 N/A", output)
                self.assertNotIn("换手率 N/A%", output)
                self.assertNotIn("N/A/100", output)

    def test_present_values_are_still_rendered_verbatim(self):
        """护栏不得吞掉真实数值。"""
        dashboard = _dashboard(null_fields=False)
        for name, output in _data_perspective_renderings(_result(dashboard), dashboard).items():
            with self.subTest(renderer=name):
                self.assertIn("0.96", output)
                self.assertIn("90.04", output)
                self.assertIn("-1.08", output)
                self.assertIn("空头排列", output)
                self.assertIn("45/100", output)

    def test_nonfinite_chip_sniper_strategy_and_fundamentals_are_sanitized(self):
        from src.notification import NotificationService
        from src.services.history_service import HistoryService

        dashboard = _dashboard(null_fields=False)
        dashboard["data_perspective"]["chip_structure"] = {
            "profit_ratio": float("nan"),
            "avg_cost": "Infinity",
            "concentration": float("-inf"),
            "chip_health": "一般",
        }
        dashboard["battle_plan"] = {
            "sniper_points": {
                "ideal_buy": float("inf"),
                "secondary_buy": float("nan"),
                "stop_loss": 0,
                "take_profit": "Target: Infinity",
            }
        }
        dashboard["strategy_synthesis"] = {
            "final_signal": "hold",
            "consensus_level": "medium",
            "conflict_severity": "none",
            "conflict_count": 0,
            "confidence": float("inf"),
            "supporting_skills": [
                {"skill_id": "bull_trend", "signal": "hold", "confidence": float("nan")}
            ],
            "opposing_skills": [],
            "conflicts": [],
        }
        result = _result(dashboard)
        result.fundamental_context = {
            "earnings": {
                "data": {
                    "financial_report": {
                        "report_date": "2026-06-30",
                        "revenue": 0,
                        "net_profit_parent": float("nan"),
                        "operating_cash_flow": float("inf"),
                        "roe": "Infinity",
                    },
                    "dividend": {
                        "ttm_cash_dividend_per_share": float("nan"),
                        "ttm_event_count": 0,
                        "ttm_dividend_yield_pct": float("inf"),
                        "events": [{"ex_dividend_date": "Infinity"}],
                    },
                }
            },
            "institution": {
                "status": "ok",
                "data": {
                    "foreign_net": float("nan"),
                    "trust_net": float("inf"),
                    "dealer_net": "-Infinity",
                    "total_net": 0,
                    "date": "2026-08-28",
                    "source": "test",
                },
            },
        }

        outputs = _all_renderings(result, dashboard)
        for name, output in outputs.items():
            with self.subTest(renderer=name):
                _assert_no_invalid_number_rendering(self, output)
                if name != "notification_single":
                    self.assertIn("| 🛑 止损位 | 0 |", output)
                    self.assertIn("(0)", output)

        self.assertEqual(NotificationService._format_percent(float("nan")), "N/A")
        self.assertEqual(NotificationService._format_percent("Infinity"), "N/A")
        self.assertEqual(NotificationService._format_percent(0), "0.00%")
        self.assertEqual(HistoryService._clean_sniper_value(float("inf")), "N/A")
        self.assertIsNone(
            HistoryService._normalize_display_sniper_value(float("nan"))
        )

    def test_explanatory_text_does_not_receive_numeric_units(self):
        dashboard = _dashboard(null_fields=False)
        perspective = dashboard["data_perspective"]
        perspective["trend_status"]["trend_score"] = "数据缺失，无法判断"
        perspective["price_position"]["bias_ma5"] = "数据缺失，无法判断"
        perspective["volume_analysis"]["turnover_rate"] = "数据缺失，无法判断"
        for name, output in _data_perspective_renderings(_result(dashboard), dashboard).items():
            with self.subTest(renderer=name):
                self.assertIn("数据缺失，无法判断", output)
                self.assertNotIn("数据缺失，无法判断%", output)
                self.assertNotIn("数据缺失，无法判断/100", output)

    def test_zero_valued_metrics_are_not_collapsed_to_placeholder(self):
        """0 是合法量比/乖离率，不能被当成缺失值。"""
        dashboard = _dashboard(null_fields=False)
        dashboard["data_perspective"]["volume_analysis"]["volume_ratio"] = 0
        dashboard["data_perspective"]["price_position"]["bias_ma5"] = 0
        for name, output in _data_perspective_renderings(_result(dashboard), dashboard).items():
            with self.subTest(renderer=name):
                self.assertIn("量比 0 ", output)
                self.assertIn("| 0% ", output)


if __name__ == "__main__":
    unittest.main()
