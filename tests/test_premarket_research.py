from dataclasses import dataclass
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.services.premarket_evidence import audit_financial_context, audit_news_result
from src.services.premarket_price_plan import _scenario, build_premarket_price_plan
from src.services.premarket_research import apply_us_premarket_research


def test_legacy_premarket_prompt_uses_frozen_usd_units():
    from src.analyzer import GeminiAnalyzer

    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    context = {
        "code": "SPCX",
        "date": "2026-10-20",
        "premarket_signal_date": "2026-10-20",
        "today": {"close": 148.51, "amount": 150000000, "volume": 1000000},
    }
    with patch.object(analyzer, "_get_skill_prompt_sections", return_value=("", "", False)):
        prompt = analyzer._format_prompt(context, "SPCX", report_language="zh")
    assert "Frozen US premarket signal bar: 2026-10-20" in prompt
    assert "148.51 美元" in prompt
    assert "1.50 亿美元" in prompt
    assert "148.51 元" not in prompt
    assert analyzer._format_amount(150000000) == "1.50 亿元"


def test_agent_comprehensive_search_excludes_post_cutoff_news():
    from src.agent.news_evidence import (
        activate_news_evidence_scope, get_current_news_evidence, reset_news_evidence_scope,
    )
    from src.agent.tools.search_tools import _handle_search_comprehensive_intel

    @dataclass
    class Item:
        title: str
        published_date: str
        url: str = "https://www.sec.gov/Archives/example"
        source: str = "SEC"
        snippet: str = "test"

    @dataclass
    class Response:
        results: list[Item]
        success: bool = True
        query: str = "test"
        provider: str = "stub"

    class Service:
        is_available = True

        def search_comprehensive_intel(self, **_kwargs):
            return {"latest_news": Response([
                Item("before", "2026-10-21T11:00:00Z"),
                Item("after", "2026-10-21T13:00:00Z"),
                Item("unknown", ""),
            ])}

        def format_intel_report(self, results, _name):
            return str(len(results["latest_news"].results))

    token = activate_news_evidence_scope(datetime(2026, 10, 21, 12, tzinfo=timezone.utc))
    try:
        with patch("src.agent.tools.search_tools._get_search_service", return_value=Service()), \
             patch("src.agent.tools.search_tools._persist_news_response"), \
             patch("src.services.premarket_evidence._source_publication", return_value=None):
            output = _handle_search_comprehensive_intel("SPCX", "SPCX")
        scope = get_current_news_evidence()
        assert output["report"] == "1"
        assert output["dimensions"]["latest_news"]["results_count"] == 1
        assert scope.total == 1
        assert len(scope.publication_audit) == 3
    finally:
        reset_news_evidence_scope(token)


def _bars(count=65):
    dates = pd.bdate_range("2026-07-20", periods=count)
    closes = [100 + index * 0.35 for index in range(count)]
    return pd.DataFrame({
        "date": dates.date,
        "open": [value - 0.3 for value in closes],
        "high": [value + 1.0 for value in closes],
        "low": [value - 1.0 for value in closes],
        "close": closes,
        "volume": [1000 + index for index in range(count)],
        "amount": [100000 + index * 1000 for index in range(count)],
    })


def test_premarket_plan_ignores_execution_day_bar():
    bars = _bars()
    signal_date = bars.iloc[-1]["date"]
    kwargs = {"signal_date": signal_date, "ma5": 121.7, "ma10": 120.8, "ma20": 119.2}
    original = build_premarket_price_plan(bars, **kwargs)
    future = bars.iloc[-1].copy()
    future["date"] = signal_date + pd.Timedelta(days=1)
    future["close"] = 500.0
    future["high"] = 510.0
    future["low"] = 490.0
    extended = build_premarket_price_plan(pd.concat([bars, pd.DataFrame([future])]), **kwargs)
    assert extended == original
    assert original["plan_policy"] == "dsa_structural_v1"
    assert original["signal_date"] == signal_date.isoformat()
    assert original["atr14"] is not None


def test_small_positive_structural_reward_is_not_rejected_by_ratio():
    plan = _scenario(
        name="理想买入参考", kind="pullback", anchor=100.0, atr=4.0,
        resistances=[102.0, 103.0], supports=[98.0], base_low=95.0,
    )
    assert plan["entry_low"] == 99.0
    assert plan["entry_high"] == 101.0
    assert plan["stop"] == 97.0
    assert plan["target1"] == 101.6
    assert plan["target2"] == 102.6
    assert plan["status"] == "待开盘复核"
    assert (plan["target1"] - plan["entry_high"]) / (plan["entry_high"] - plan["stop"]) < 1.5


def test_pressure_inside_entry_interval_remains_observation():
    plan = _scenario(
        name="理想买入参考", kind="pullback", anchor=148.51, atr=1.0,
        resistances=[148.66, 155.33, 160.0], supports=[143.27], base_low=143.27,
    )
    assert plan["status"] == "观察"
    assert "压力" in plan["reason"]


def test_news_cutoff_requires_source_and_time():
    cutoff = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)
    base = dict(title="Article", url="https://example.org/article", source="Example", retrieved_at=None)
    same_day = SimpleNamespace(**base, published_date="2026-10-21")
    earlier_date = SimpleNamespace(**base, published_date="2026-10-20")
    exact = SimpleNamespace(**base, published_date="2026-10-21T11:30:00Z")
    future = SimpleNamespace(**base, published_date="2026-10-21T13:00:00Z")
    assert audit_news_result(same_day, cutoff)["status"] == "unverified"
    assert audit_news_result(earlier_date, cutoff)["status"] == "partial"
    assert audit_news_result(exact, cutoff)["status"] == "metadata_verified"
    assert audit_news_result(future, cutoff)["status"] == "unverified"


def test_source_page_time_corrects_same_day_search_date():
    cutoff = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)
    item = SimpleNamespace(
        title="Article", url="https://www.sec.gov/Archives/example", source="SEC",
        published_date="2026-10-21", retrieved_at=None,
    )
    with patch("src.services.premarket_evidence._source_publication", return_value="2026-10-21T11:00:00Z"):
        audit = audit_news_result(item, cutoff, verify_source=True)
    assert audit["status"] == "source_verified"
    assert audit["published_at_utc"] == "2026-10-21T11:00:00+00:00"
    assert audit["timestamp_corrected"] is True
    with patch("src.services.premarket_evidence._source_publication", return_value="2026-10-21T13:00:00Z"):
        audit = audit_news_result(item, cutoff, verify_source=True)
    assert audit["status"] == "unverified"


def test_article_meta_parser_reads_source_publication_tag():
    from src.services.premarket_evidence import _PublicationMetaParser

    parser = _PublicationMetaParser()
    parser.feed('<html><head><meta content="2026-10-20T16:00:51-04:00" '
                'property="article:published_time"></head></html>')
    assert parser.published == "2026-10-20T16:00:51-04:00"


def test_financial_period_alone_cannot_verify_publication():
    audit = audit_financial_context(
        {"financial_report": {"report_date": "2026-09-30", "revenue": 100}},
        datetime(2026, 10, 21, 12, tzinfo=timezone.utc),
    )
    assert audit["status"] == "unverified"
    assert "披露" in audit["reason"]


def test_premarket_result_keeps_existing_technical_score_and_sanitizes_model_plan():
    bars = _bars()
    components = {"trend": 8, "bias_ma5": 20, "volume_5d": 10, "ma_support": 0, "macd": 8, "rsi": 5}
    trend = SimpleNamespace(
        ma5=121.7, ma10=125.0, ma20=124.0,
        current_price=float(bars.iloc[-1]["close"]), bias_ma5=-0.5,
        trend_status=SimpleNamespace(value="弱势空头"), ma_alignment="MA5<MA10≥MA20",
        trend_strength=40, signal_score=51, score_components=components,
        volume_ratio_5d=1.2, volume_status=SimpleNamespace(value="量能正常"),
        support_levels=[110.0], resistance_levels=[125.0],
        macd_status=SimpleNamespace(value="多头"), rsi_status=SimpleNamespace(value="中性"),
    )
    result = SimpleNamespace(
        dashboard={"intelligence": {"positive_catalysts": ["模型自称重大利好"]},
                   "battle_plan": {"sniper_points": {"ideal_buy": 148.51, "secondary_buy": 155.33,
                                                      "stop_loss": 143.27, "take_profit": 155.33}}},
        sentiment_score=88,
    )
    plan = apply_us_premarket_research(
        result, bars=bars, trend=trend, signal_date=bars.iloc[-1]["date"],
        data_quality={"overall_score": 97},
        news_audit=[], financial_audit={"status": "unverified"},
    )
    assert result.sentiment_score == 51
    assert result.dashboard["raw_model_score"] == 88
    assert result.dashboard["data_block_availability"] == 97
    assert result.dashboard["battle_plan"]["price_plan"] == plan
    assert result.dashboard["battle_plan"]["sniper_points"]["take_profit"] == "按情景分别复核"
    assert result.dashboard["intelligence"]["positive_catalysts"] == []
    assert result.market_snapshot["currency"] == "USD"
    assert "乖离" in result.technical_analysis


def test_dashboard_report_uses_signal_close_usd_and_separate_plans():
    from src.analyzer import AnalysisResult
    from src.notification import NotificationService
    from src.stock_analyzer import StockTrendAnalyzer

    bars = _bars()
    signal_date = bars.iloc[-1]["date"]
    trend = StockTrendAnalyzer().analyze(bars, "SPCX")
    assert sum(trend.score_components.values()) == trend.signal_score
    result = AnalysisResult(
        code="SPCX", name="测试标的", sentiment_score=99,
        trend_prediction="模型称强势多头", operation_advice="买入",
        dashboard={"intelligence": {"positive_catalysts": ["模型自称重大利好"]}},
        report_language="zh", success=True,
    )
    apply_us_premarket_research(
        result, bars=bars, trend=trend, signal_date=signal_date,
        data_quality={"overall_score": 97},
        financial_audit={"status": "unverified", "reason": "缺披露时间"},
    )
    service = NotificationService()
    report = service.generate_dashboard_report([result], report_date="2026-10-21")
    assert f"{signal_date.isoformat()} 收盘行情（USD）" in report
    assert "数据块可用性**: 97/100" in report
    assert "技术评分" in report
    assert "昨量比" in report
    assert "| 情景 | 参考位 | 入场区间 | 止损 | 目标一 | 目标二 | 状态 |" in report
    assert "模型自称重大利好" not in report
    assert "新增计划股数0" in report
    assert "现在买入" not in report
    assert "未读取账户，不推断实际持仓" in report


def test_premarket_report_summarizes_verified_financials_without_raw_table():
    from src.analyzer import AnalysisResult
    from src.notification import NotificationService
    from src.stock_analyzer import StockTrendAnalyzer

    bars = _bars()
    signal_date = bars.iloc[-1]["date"]
    result = AnalysisResult(
        code="SPCX", name="测试标的", sentiment_score=50,
        trend_prediction="震荡", operation_advice="观望",
        dashboard={}, report_language="zh", success=True,
    )
    result.fundamental_context = {
        "financial_report": {"report_date": "2026-09-30", "revenue": 123456789},
    }
    apply_us_premarket_research(
        result, bars=bars, trend=StockTrendAnalyzer().analyze(bars, "SPCX"),
        signal_date=signal_date, financial_audit={"status": "source_verified"},
    )
    assert result.fundamental_context is None
    report = NotificationService().generate_dashboard_report([result], report_date="2026-10-21")
    assert "财务披露时刻与出处已核验" in report
    assert "财务摘要" not in report
    assert "123456789" not in report


def test_agent_premarket_prompt_excludes_live_and_undated_tools():
    from src.agent.executor import AgentExecutor

    tool_names = [
        "get_realtime_quote", "get_daily_history", "analyze_trend",
        "get_stock_info", "get_capital_flow", "get_chip_distribution", "search_stock_news",
    ]
    registry = SimpleNamespace(to_openai_tools=lambda: [
        {"type": "function", "function": {"name": name}} for name in tool_names
    ])
    executor = AgentExecutor(registry, None)
    captured = {}

    def capture(messages, tool_decls, parse_dashboard):
        captured["messages"] = messages
        captured["tools"] = [item["function"]["name"] for item in tool_decls]
        return SimpleNamespace(success=True)

    with patch.object(executor, "_run_loop", side_effect=capture):
        executor.run("分析SPCX", context={
            "stock_code": "SPCX", "report_language": "zh",
            "premarket_signal_date": "2026-10-20",
            "trend_result": {"signal_score": 51},
        })
    assert captured["tools"] == ["get_daily_history", "analyze_trend", "search_stock_news"]
    assert "指定信号日" in captured["messages"][0]["content"]
    assert "2026-10-20" in captured["messages"][1]["content"]
    assert "51" in captured["messages"][1]["content"]
