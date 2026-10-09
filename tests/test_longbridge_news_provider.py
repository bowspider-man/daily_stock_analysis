import sys
from types import ModuleType, SimpleNamespace

from src.search_service import LongbridgeNewsProvider


def test_longbridge_news_provider_maps_content_items(monkeypatch):
    provider = LongbridgeNewsProvider()
    provider._fetcher = SimpleNamespace(
        _config=object(),
        _get_ctx=lambda: object(),
        has_configured_credentials=lambda: True,
    )

    class FakeContentContext:
        def __init__(self, config):
            assert config is provider._fetcher._config

        def news(self, symbol):
            assert symbol == "700.HK"
            return [SimpleNamespace(
                title="Tencent news",
                description="Original Longbridge summary",
                url="https://longbridge.com/news/1",
                published_at=1773805586,
            )]

    fake_openapi = ModuleType("longbridge.openapi")
    fake_openapi.ContentContext = FakeContentContext
    fake_longbridge = ModuleType("longbridge")
    fake_longbridge.__path__ = []
    fake_longbridge.openapi = fake_openapi
    monkeypatch.setitem(sys.modules, "longbridge", fake_longbridge)
    monkeypatch.setitem(sys.modules, "longbridge.openapi", fake_openapi)

    response = provider.search_stock("700.HK", "腾讯控股", 5)

    assert response.success is True
    assert response.provider == "Longbridge"
    assert len(response.results) == 1
    result = response.results[0]
    assert result.title == "Tencent news"
    assert result.snippet == "Original Longbridge summary"
    assert result.url == "https://longbridge.com/news/1"
    assert result.published_date == "1773805586"


def test_longbridge_news_provider_reports_unsupported_symbol():
    provider = LongbridgeNewsProvider()

    response = provider.search_stock("600519", "贵州茅台", 5)

    assert response.success is True
    assert response.results == []
    assert "does not support" in (response.error_message or "")
