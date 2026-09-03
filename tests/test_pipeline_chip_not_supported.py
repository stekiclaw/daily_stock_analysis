# -*- coding: utf-8 -*-
"""Regression coverage: chip_not_supported reaches PipelineAnalysisArtifacts.metadata.

The read side (AnalysisContextBuilder scoring NOT_SUPPORTED over MISSING when
this flag is set) was already covered by
tests/test_analysis_context_builder.py::test_chip_missing_defaults_to_missing_and_explicit_not_supported.
Nothing ever wrote the flag, so it was dead — every non-A股 report's chip
block was scored as a data-quality failure (MISSING, 35/100) rather than
"this market doesn't have this data" (NOT_SUPPORTED, 70/100), even though
the underlying fetchers already knew and logged the distinction.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.core.pipeline import StockAnalysisPipeline
from src.schemas.analysis_context_pack import ContextFieldStatus
from src.services.analysis_context_builder import AnalysisContextBuilder


def _make_pipeline() -> StockAnalysisPipeline:
    pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
    pipeline.query_source = "api"
    return pipeline


def _build_artifacts(pipeline: StockAnalysisPipeline, *, chip_not_supported: bool):
    return pipeline._build_legacy_analysis_artifacts(
        code="NBIS",
        stock_name="Nebius Group N.V.",
        market="us",
        phase=None,
        context={"code": "NBIS", "today": {}, "yesterday": {}},
        enhanced_context={},
        realtime_quote=None,
        trend_result=None,
        chip_data=None,
        fundamental_context=None,
        news_context=None,
        news_result_count=None,
        query_id="q1",
        chip_not_supported=chip_not_supported,
    )


def test_chip_not_supported_flag_reaches_artifact_metadata():
    artifacts = _build_artifacts(_make_pipeline(), chip_not_supported=True)
    assert artifacts.metadata.get("chip_not_supported") is True


def test_chip_not_supported_defaults_to_absent_not_false():
    """Absent, not merely falsy: AnalysisContextBuilder only checks truthiness,
    but keeping the default case free of the key matches the metadata-filtering
    convention used everywhere else in this builder (e.g. _quote_metadata)."""
    artifacts = _build_artifacts(_make_pipeline(), chip_not_supported=False)
    assert "chip_not_supported" not in artifacts.metadata


def test_index_target_skip_sets_chip_not_supported():
    """The pipeline's index-skip branch (INDEX_SKIP_MODULES) is the same
    "not applicable" category as a market-unsupported stock, not a failure."""
    from src.core.pipeline import INDEX_SKIP_MODULES

    assert "chip_distribution" in INDEX_SKIP_MODULES


def test_unsupported_us_target_never_calls_chip_provider():
    pipeline = _make_pipeline()
    pipeline.fetcher_manager = SimpleNamespace(
        is_chip_distribution_unsupported_market=MagicMock(return_value=True),
        get_chip_distribution=MagicMock(),
    )

    chip_data, not_supported = pipeline._fetch_chip_distribution_for_target(
        code="MSFT",
        stock_name="Microsoft Corporation",
        is_index=False,
    )

    assert chip_data is None
    assert not_supported is True
    pipeline.fetcher_manager.get_chip_distribution.assert_not_called()


def test_supported_target_calls_chip_provider_and_is_not_structural_gap():
    chip = SimpleNamespace(profit_ratio=0.6, concentration_90=0.2)
    pipeline = _make_pipeline()
    pipeline.fetcher_manager = SimpleNamespace(
        is_chip_distribution_unsupported_market=MagicMock(return_value=False),
        get_chip_distribution=MagicMock(return_value=chip),
    )

    chip_data, not_supported = pipeline._fetch_chip_distribution_for_target(
        code="600519",
        stock_name="贵州茅台",
        is_index=False,
    )

    assert chip_data is chip
    assert not_supported is False
    pipeline.fetcher_manager.get_chip_distribution.assert_called_once_with("600519")


def test_agent_artifacts_receive_chip_not_supported_flag():
    artifacts = _make_pipeline()._build_agent_analysis_artifacts(
        code="MSFT",
        stock_name="Microsoft Corporation",
        market="us",
        phase=None,
        initial_context={},
        fundamental_context=None,
        query_id="agent-q1",
        base_context={"code": "MSFT", "today": {}, "yesterday": {}},
        chip_not_supported=True,
    )

    assert artifacts.metadata["chip_not_supported"] is True


def test_agent_final_artifacts_use_real_runtime_news_content_and_count():
    pipeline = _make_pipeline()
    news_content = pipeline._merge_agent_news_context(
        None,
        "## Agent 运行期新闻证据\n1. Microsoft update",
    )
    artifacts = pipeline._build_agent_analysis_artifacts(
        code="MSFT",
        stock_name="Microsoft Corporation",
        market="us",
        phase=None,
        initial_context={"news_context": news_content},
        fundamental_context=None,
        query_id="agent-q2",
        base_context={"code": "MSFT", "today": {}, "yesterday": {}},
        chip_not_supported=True,
        news_result_count=1,
    )

    pack = AnalysisContextBuilder.build(artifacts)
    assert pack.blocks["news"].status == ContextFieldStatus.AVAILABLE
    assert pack.blocks["news"].items["content"].value == news_content
    assert pack.blocks["chip"].status == ContextFieldStatus.NOT_SUPPORTED
    assert pack.metadata["news_result_count"] == 1


def test_agent_pretool_summary_warns_that_news_state_is_pending_runtime_tools():
    summary = _make_pipeline()._append_agent_runtime_news_note(
        "summary says news missing",
        report_language="zh",
        search_available=True,
    )

    assert "工具调用前" in summary
    assert "工具已返回真实证据时，不得再写新闻缺失" in summary
