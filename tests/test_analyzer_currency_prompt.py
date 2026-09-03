# -*- coding: utf-8 -*-
"""Prices in the prompt must carry the instrument's own trading currency.

Rendering every quote as 元 tells the model a US or HK price is quoted in RMB,
which then flows into the battle-plan entry/stop/target levels.
"""

import unittest
from unittest.mock import patch

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    from tests.litellm_stub import ensure_litellm_stub

    ensure_litellm_stub()

from src.analyzer import (
    GeminiAnalyzer,
    currency_unit_label,
    resolve_market_currency,
)


def _analyzer(**kwargs):
    with patch.object(GeminiAnalyzer, "_init_litellm", return_value=None):
        return GeminiAnalyzer(**kwargs)



class ResolveMarketCurrencyTestCase(unittest.TestCase):
    def test_market_defaults(self):
        self.assertEqual(resolve_market_currency("600519"), "CNY")
        self.assertEqual(resolve_market_currency("00700.HK"), "HKD")
        self.assertEqual(resolve_market_currency("AAPL"), "USD")
        self.assertEqual(resolve_market_currency("2330.TW"), "TWD")
        self.assertEqual(resolve_market_currency("7203.T"), "JPY")
        self.assertEqual(resolve_market_currency("005930.KS"), "KRW")

    def test_explicit_currency_wins_over_market_guess(self):
        self.assertEqual(resolve_market_currency("AAPL", "hkd"), "HKD")

    def test_unknown_currency_keeps_yuan_default(self):
        self.assertEqual(currency_unit_label("ZZZ"), "元")
        self.assertEqual(currency_unit_label(None), "元")


class PromptCurrencyTestCase(unittest.TestCase):
    def _prompt(self, code, realtime=None):
        context = {
            "code": code,
            "stock_name": code,
            "date": "2026-08-29",
            "news_window_days": 3,
            "today": {"close": 231.4, "open": 230.0, "high": 232.0, "low": 229.1, "amount": 5.2e9},
            "chip": {
                "profit_ratio": 0.7,
                "avg_cost": 210.0,
                "concentration_90": 0.11,
                "concentration_70": 0.07,
            },
        }
        if realtime is not None:
            context["realtime"] = realtime
        return _analyzer()._format_prompt(context, code, news_context=None)

    def test_us_quote_uses_usd_label(self):
        prompt = self._prompt("AAPL", {"price": 231.4, "total_mv": 3.4e12})
        self.assertIn("231.4 美元", prompt)
        self.assertIn("210.0 美元", prompt)
        self.assertIn("亿美元", prompt)
        self.assertNotIn("231.4 元", prompt)

    def test_hk_quote_uses_hkd_label(self):
        prompt = self._prompt("00700.HK", {"price": 231.4})
        self.assertIn("231.4 港元", prompt)

    def test_source_reported_currency_overrides_market_guess(self):
        prompt = self._prompt("AAPL", {"price": 231.4, "currency": "HKD"})
        self.assertIn("231.4 港元", prompt)
        self.assertNotIn("231.4 美元", prompt)

    def test_a_share_prompt_keeps_yuan(self):
        prompt = self._prompt("600519", {"price": 231.4})
        self.assertIn("231.4 元", prompt)
        self.assertNotIn("美元", prompt)


class MarketSnapshotCurrencyTestCase(unittest.TestCase):
    def _snapshot(self, code, currency=None):
        realtime = {"price": 231.4}
        if currency:
            realtime["currency"] = currency
        context = {
            "code": code,
            "date": "2026-08-29",
            "today": {"close": 231.4, "open": 230.0, "high": 232.0, "low": 229.1, "amount": 5.2e9},
            "realtime": realtime,
        }
        return _analyzer()._build_market_snapshot(context)

    def test_turnover_follows_the_market_currency(self):
        self.assertIn("亿美元", self._snapshot("AAPL")["amount"])
        self.assertIn("亿港元", self._snapshot("00700.HK")["amount"])
        self.assertIn("亿元", self._snapshot("600519")["amount"])
        self.assertIn("亿新台币", self._snapshot("AAPL", currency="TWD")["amount"])


class SystemPromptCurrencyTestCase(unittest.TestCase):
    def test_battle_plan_examples_follow_the_market(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                analyzer = _analyzer(
                    skill_instructions="",
                    default_skill_policy="",
                    use_legacy_default_prompt=legacy,
                )
                us_prompt = analyzer._get_analysis_system_prompt("zh", stock_code="AAPL")
                cn_prompt = analyzer._get_analysis_system_prompt("zh", stock_code="600519")

                self.assertIn("XX美元", us_prompt)
                self.assertNotIn("XX元", us_prompt)
                self.assertNotIn("{currency_placeholder}", us_prompt)
                self.assertIn("XX元", cn_prompt)


if __name__ == "__main__":
    unittest.main()
