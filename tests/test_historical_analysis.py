"""As-of market data must never include bars after the requested session."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from data_provider.base import DataFetchError, DataFetcherManager
from src.market_analyzer import MarketAnalyzer
from src.services.history_loader import reset_historical_target_date, set_historical_target_date


def _bars(dates):
    return pd.DataFrame([
        {"date": day, "open": 100 + i, "high": 102 + i, "low": 99 + i,
         "close": 101 + i, "volume": 1000 + i, "amount": 10000 + i}
        for i, day in enumerate(dates)
    ])


def test_historical_daily_fetch_clips_provider_future_bars_and_passes_inclusive_end():
    seen = []

    def fetch(**kwargs):
        seen.append(kwargs)
        return _bars(["2026-10-22", "2026-10-23", "2026-10-26"])

    manager = DataFetcherManager(fetchers=[
        SimpleNamespace(name="YfinanceFetcher", priority=1, get_daily_data=fetch)
    ])
    token = set_historical_target_date(date(2026, 10, 23))
    try:
        frame, source = manager.get_daily_data("SPX", days=30)
    finally:
        reset_historical_target_date(token)

    assert source == "YfinanceFetcher"
    assert list(frame["date"]) == ["2026-10-22", "2026-10-23"]
    assert seen[0]["end_date"] == "2026-10-24"
    assert seen[0]["start_date"] <= "2026-10-22"


def test_historical_daily_fetch_rejects_missing_requested_session_bar():
    manager = DataFetcherManager(fetchers=[
        SimpleNamespace(name="YfinanceFetcher", priority=1,
                        get_daily_data=lambda **kwargs: _bars(["2026-10-22", "2026-10-26"]))
    ])
    token = set_historical_target_date(date(2026, 10, 23))
    try:
        with pytest.raises(DataFetchError):
            manager.get_daily_data("SPX", days=30)
    finally:
        reset_historical_target_date(token)


def test_historical_us_market_overview_uses_only_requested_session():
    analyzer = MarketAnalyzer.__new__(MarketAnalyzer)
    analyzer.region = "us"
    analyzer.analysis_date = date(2026, 10, 23)
    calls = []

    def fetch(code, **kwargs):
        calls.append((code, kwargs))
        return _bars(["2026-10-22", "2026-10-23", "2026-10-26"]), "test"

    analyzer.data_manager = SimpleNamespace(get_daily_data=fetch)
    overview = analyzer.get_market_overview()

    assert overview.date == "2026-10-23"
    assert len(overview.indices) == 4
    assert all(index.current == 102 for index in overview.indices)
    assert all(kwargs["end_date"] == "2026-10-24" for _, kwargs in calls)


def test_historical_trading_filter_checks_requested_friday_not_run_day():
    from main import _compute_trading_day_filter

    args = SimpleNamespace(analysis_date=date(2026, 10, 23), force_run=False,
                           no_market_review=True)
    config = SimpleNamespace(trading_day_check_enabled=True, market_review_enabled=False)

    def market_open(market, requested_date):
        assert requested_date == date(2026, 10, 23)
        return market == "us"

    with patch("src.core.trading_calendar.is_market_open", side_effect=market_open), \
         patch("src.core.trading_calendar.get_open_markets_today",
               side_effect=AssertionError("current-day calendar must not be queried")):
        codes, region, skip = _compute_trading_day_filter(config, args, ["AAPL", "MSFT"])

    assert codes == ["AAPL", "MSFT"]
    assert region is None
    assert not skip
