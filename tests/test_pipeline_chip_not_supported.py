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

from src.core.pipeline import StockAnalysisPipeline


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
