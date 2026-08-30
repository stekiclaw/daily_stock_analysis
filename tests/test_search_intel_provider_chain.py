"""Provider-chain termination rules for the multi-dimension intelligence search.

`SearchService.search_comprehensive_intel` walks its providers in preference
order. It used to stop at the first provider that returned *any* result, which
in practice settled for a single weak item and never reached the next source
even though the code already computes a per-dimension target. These tests pin
the target-aware behaviour: keep going while below target, stop as soon as the
target is met, and dedupe whatever gets merged.
"""

from datetime import datetime, timezone

import pytest

from src.search_service import SearchResponse, SearchResult, SearchService


TODAY = datetime.now(timezone.utc).date().isoformat()


def _article(index: int, url: str | None = None) -> SearchResult:
    return SearchResult(
        title=f"NVIDIA announces a new AI platform ({index})",
        snippet="NVIDIA company update with material investor context.",
        url=url or f"https://example.com/nvda/{index}",
        source="example.com",
        published_date=TODAY,
    )


class _FakeProvider:
    """Minimal duck-typed provider, matching the shape the chain relies on."""

    supports_open_ended_queries = True
    is_available = True

    def __init__(self, name: str, results, *, success: bool = True, error: str | None = None):
        self.name = name
        self._results = results
        self._success = success
        self._error = error
        self.calls = []

    def search(self, query, max_results=5, days=7, **kwargs):
        self.calls.append(query)
        return SearchResponse(
            query=query,
            results=list(self._results),
            provider=self.name,
            success=self._success,
            error_message=self._error,
            search_time=1.5,
        )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("src.search_service.time.sleep", lambda _seconds: None)


def _service(providers):
    service = SearchService(searxng_public_instances_enabled=False)
    service._providers = list(providers)
    return service


def _latest_news(providers):
    # max_searches=1 keeps the assertions on the single `latest_news` dimension.
    responses = _service(providers).search_comprehensive_intel(
        "NVDA", "NVIDIA", max_searches=1
    )
    return responses["latest_news"]


def test_chain_continues_to_the_next_provider_while_below_target():
    """One weak hit must not stop the chain: the next source is still asked."""
    primary = _FakeProvider("Primary", [_article(1)])
    secondary = _FakeProvider("Secondary", [_article(2), _article(3), _article(4)])

    response = _latest_news([primary, secondary])

    assert len(primary.calls) == 1
    assert len(secondary.calls) == 1
    # Target is 3 per dimension; the merge fills it from both sources.
    assert len(response.results) == 3
    assert response.success is True
    assert response.provider == "Primary+Secondary"
    # Preference order is preserved: the primary's item stays first.
    assert response.results[0].url == "https://example.com/nvda/1"


def test_chain_stops_once_the_target_is_met():
    """Providers stay in preference order - a satisfied target must not query on."""
    primary = _FakeProvider("Primary", [_article(1), _article(2), _article(3)])
    secondary = _FakeProvider("Secondary", [_article(4)])

    response = _latest_news([primary, secondary])

    assert len(primary.calls) == 1
    assert secondary.calls == []
    assert len(response.results) == 3
    # A single contributor is reported verbatim, exactly as before.
    assert response.provider == "Primary"


def test_merged_results_are_deduplicated_by_url():
    """Providers do surface the same article; the merge must not double count."""
    shared = _article(1, url="https://example.com/nvda/shared-story")
    duplicate = _article(1, url="http://www.example.com/nvda/shared-story/#top")
    primary = _FakeProvider("Primary", [shared])
    secondary = _FakeProvider("Secondary", [duplicate, _article(2)])

    response = _latest_news([primary, secondary])

    assert len(secondary.calls) == 1
    urls = [item.url for item in response.results]
    assert urls == [
        "https://example.com/nvda/shared-story",
        "https://example.com/nvda/2",
    ]


def test_duplicate_only_provider_is_not_credited_in_the_merged_label():
    primary = _FakeProvider("Primary", [_article(1)])
    secondary = _FakeProvider("Secondary", [_article(1)])

    response = _latest_news([primary, secondary])

    assert len(secondary.calls) == 1
    assert len(response.results) == 1
    assert response.provider == "Primary"


def test_all_providers_empty_keeps_the_failure_accounting():
    """With nothing merged, the last attempt is returned verbatim as before."""
    primary = _FakeProvider("Primary", [], success=False, error="primary quota exhausted")
    secondary = _FakeProvider("Secondary", [], success=False, error="secondary quota exhausted")

    response = _latest_news([primary, secondary])

    assert len(primary.calls) == 1
    assert len(secondary.calls) == 1
    assert response.success is False
    assert response.results == []
    assert response.provider == "Secondary"
    assert response.error_message == "secondary quota exhausted"


def test_merged_search_time_sums_the_contributing_providers():
    primary = _FakeProvider("Primary", [_article(1)])
    secondary = _FakeProvider("Secondary", [_article(2)])

    response = _latest_news([primary, secondary])

    assert len(response.results) == 2
    assert response.search_time == pytest.approx(3.0)


def test_merged_provider_label_fits_the_persisted_column():
    """NewsIntel.provider is String(32); a long chain degrades instead of overflowing."""
    short = SearchService._merged_provider_label(["FinnhubNews", "ETFConstituentNews"])
    assert short == "FinnhubNews+ETFConstituentNews"
    assert len(short) <= SearchService.MERGED_PROVIDER_LABEL_MAX_LEN

    long_chain = SearchService._merged_provider_label(
        ["FinnhubNews", "ETFConstituentNews", "YFinanceNews", "Tavily"]
    )
    assert len(long_chain) <= SearchService.MERGED_PROVIDER_LABEL_MAX_LEN
    assert long_chain.startswith("FinnhubNews+")

    assert SearchService._merged_provider_label([]) == "None"
    # Repeat names collapse rather than inflating the label.
    assert SearchService._merged_provider_label(["Tavily", "Tavily"]) == "Tavily"
