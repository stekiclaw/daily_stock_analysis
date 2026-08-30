# -*- coding: utf-8 -*-
"""FINNHUB_PRIORITY must be honoured so a free-tier deployment can demote it.

Finnhub's ``/stock/candle`` is not on the free tier and answers 403, so on a
free key every US daily fetch spent a guaranteed-failing request and drove the
circuit breaker through open/half-open cycles. Paid keys are fine, so this is
configuration rather than a hardcoded demotion.
"""

import importlib
import os
from unittest.mock import patch

import pytest


def _reload_priority() -> int:
    import data_provider.finnhub_fetcher as mod

    importlib.reload(mod)
    return mod.FinnhubFetcher.priority


@pytest.mark.parametrize("value,expected", [("9", 9), ("0", 0), ("2", 2)])
def test_priority_reads_the_env_override(value, expected):
    with patch.dict(os.environ, {"FINNHUB_PRIORITY": value}):
        assert _reload_priority() == expected


def test_priority_defaults_to_two_when_unset():
    env = {k: v for k, v in os.environ.items() if k != "FINNHUB_PRIORITY"}
    with patch.dict(os.environ, env, clear=True):
        assert _reload_priority() == 2


def test_demoting_finnhub_moves_it_last_in_the_us_daily_chain():
    """The routing helper sorts on fetcher.priority, so the override must reach it."""
    from data_provider.base import DataFetcherManager

    class _Stub:
        def __init__(self, name, priority):
            self.name = name
            self.priority = priority

    manager = DataFetcherManager(fetchers=[
        _Stub("FinnhubFetcher", 9),
        _Stub("AlphaVantageFetcher", 3),
        _Stub("YfinanceFetcher", 4),
    ])
    ordered = manager._order_us_sources_by_priority(
        ["FinnhubFetcher", "AlphaVantageFetcher", "YfinanceFetcher"],
        pin_first=False,
    )
    assert ordered[-1] == "FinnhubFetcher"
    assert ordered[0] == "AlphaVantageFetcher"


def test_default_priority_keeps_finnhub_first_for_paid_keys():
    from data_provider.base import DataFetcherManager

    class _Stub:
        def __init__(self, name, priority):
            self.name = name
            self.priority = priority

    manager = DataFetcherManager(fetchers=[
        _Stub("FinnhubFetcher", 2),
        _Stub("AlphaVantageFetcher", 3),
        _Stub("YfinanceFetcher", 4),
    ])
    ordered = manager._order_us_sources_by_priority(
        ["FinnhubFetcher", "AlphaVantageFetcher", "YfinanceFetcher"],
        pin_first=False,
    )
    assert ordered[0] == "FinnhubFetcher"
