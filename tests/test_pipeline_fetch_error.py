# -*- coding: utf-8 -*-
"""Regression tests for pipeline data-fetch error handling."""

from datetime import date, datetime, timezone
import unittest
from unittest.mock import MagicMock, patch

from src.core.pipeline import StockAnalysisPipeline
from src.services.run_diagnostics import (
    activate_run_diagnostic_context,
    build_run_diagnostic_summary,
    current_diagnostic_snapshot,
    reset_run_diagnostic_context,
)


class PipelineFetchErrorTestCase(unittest.TestCase):
    """`fetch_and_save_stock_data` should preserve the original exception."""

    def test_fetch_and_save_handles_stock_name_lookup_failure(self):
        pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
        pipeline.fetcher_manager = MagicMock()
        pipeline.db = MagicMock()
        pipeline.fetcher_manager.get_stock_name.side_effect = RuntimeError("name lookup failed")

        success, error = StockAnalysisPipeline.fetch_and_save_stock_data(pipeline, "600519")

        self.assertFalse(success)
        self.assertIn("name lookup failed", error or "")

    @patch.object(
        StockAnalysisPipeline,
        "_resolve_resume_target_date",
        return_value=date(2026, 3, 27),
    )
    def test_fetch_and_save_uses_effective_trading_date_for_resume_check(self, _mock_target):
        pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
        pipeline.fetcher_manager = MagicMock()
        pipeline.db = MagicMock()
        pipeline.fetcher_manager.get_stock_name.return_value = "贵州茅台"
        pipeline.db.has_today_data.return_value = True
        current_time = datetime(2026, 3, 28, 1, 0, tzinfo=timezone.utc)

        success, error = StockAnalysisPipeline.fetch_and_save_stock_data(
            pipeline,
            "600519",
            current_time=current_time,
        )

        self.assertTrue(success)
        self.assertIsNone(error)
        _mock_target.assert_called_once_with(
            "600519", current_time=current_time, analysis_target=None
        )
        pipeline.db.has_today_data.assert_called_once_with("600519", date(2026, 3, 27))
        pipeline.fetcher_manager.get_daily_data.assert_not_called()

    def test_resolve_resume_target_date_normalizes_supported_a_share_formats(self):
        with patch("src.core.pipeline.get_market_for_stock", return_value="cn") as mock_market, patch(
            "src.core.pipeline.get_effective_trading_date",
            return_value=date(2026, 3, 27),
        ) as mock_target:
            for code in ("SH600519", "000001.SZ", "BJ920748"):
                result = StockAnalysisPipeline._resolve_resume_target_date(code)
                self.assertEqual(result, date(2026, 3, 27))

        self.assertEqual(
            [args.args[0] for args in mock_market.call_args_list],
            ["600519", "000001", "920748"],
        )
        self.assertEqual(mock_target.call_count, 3)


class PipelineResumeDiagnosticsTestCase(unittest.TestCase):
    """Checkpoint/resume hits must be recorded as a local-storage data source.

    Without a recorded run the daily component degrades to
    ``unknown - 日线数据未记录诊断信息`` even though the bars are present.
    """

    @patch.object(
        StockAnalysisPipeline,
        "_resolve_resume_target_date",
        return_value=date(2026, 8, 28),
    )
    def test_resume_cache_hit_records_local_storage_daily_run(self, _mock_target):
        pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
        pipeline.fetcher_manager = MagicMock()
        pipeline.db = MagicMock()
        pipeline.fetcher_manager.get_stock_name.return_value = "微软"
        pipeline.db.has_today_data.return_value = True

        token = activate_run_diagnostic_context(
            trace_id="trace-resume",
            query_id="query-resume",
            stock_code="MSFT",
            trigger_source="api",
        )
        try:
            success, error = StockAnalysisPipeline.fetch_and_save_stock_data(
                pipeline, "MSFT"
            )
            snapshot = current_diagnostic_snapshot()
        finally:
            reset_run_diagnostic_context(token)

        self.assertTrue(success)
        self.assertIsNone(error)
        pipeline.fetcher_manager.get_daily_data.assert_not_called()

        runs = [
            run for run in snapshot["provider_runs"]
            if run.get("data_type") == "daily_data"
        ]
        self.assertEqual(len(runs), 1)
        self.assertTrue(runs[0]["success"])
        self.assertTrue(runs[0]["cache_hit"])
        self.assertEqual(runs[0]["provider"], "LocalStorage")
        self.assertEqual(runs[0]["operation"], "resume_local_daily_data")
        self.assertEqual(runs[0]["data_date"], "2026-08-28")

        summary = build_run_diagnostic_summary(
            context_snapshot={"diagnostics": snapshot},
            raw_result={"success": True, "model_used": "gpt-5.6-sol"},
        )
        daily = summary["components"]["daily_data"]
        self.assertEqual(daily["status"], "ok")
        self.assertIn("本地存储缓存", daily["message"])
        self.assertIn("2026-08-28", daily["message"])

    @patch.object(
        StockAnalysisPipeline,
        "_resolve_resume_target_date",
        return_value=date(2026, 8, 28),
    )
    def test_force_refresh_does_not_record_cache_run(self, _mock_target):
        pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
        pipeline.fetcher_manager = MagicMock()
        pipeline.db = MagicMock()
        pipeline.fetcher_manager.get_stock_name.return_value = "微软"
        pipeline.db.has_today_data.return_value = True
        pipeline.fetcher_manager.get_daily_data.return_value = (None, "dummy")

        token = activate_run_diagnostic_context(
            trace_id="trace-force",
            query_id="query-force",
            stock_code="MSFT",
            trigger_source="api",
        )
        try:
            success, error = StockAnalysisPipeline.fetch_and_save_stock_data(
                pipeline, "MSFT", force_refresh=True
            )
            snapshot = current_diagnostic_snapshot()
        finally:
            reset_run_diagnostic_context(token)

        self.assertFalse(success)
        self.assertEqual(error, "获取数据为空")
        self.assertEqual(snapshot["provider_runs"], [])


if __name__ == "__main__":
    unittest.main()
