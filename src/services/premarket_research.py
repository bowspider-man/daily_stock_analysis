"""Apply deterministic signal-close facts to a premarket model draft."""

from __future__ import annotations

from typing import Any
import math

import pandas as pd

from src.services.premarket_price_plan import build_premarket_price_plan
from src.services.premarket_evidence import summarize_evidence


def apply_us_premarket_research(
    result: Any,
    *,
    bars: pd.DataFrame,
    trend: Any,
    signal_date: Any,
    data_quality: dict[str, Any] | None = None,
    news_audit: list[dict[str, Any]] | None = None,
    financial_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Override model-made levels and technical prose, retaining its raw score in audit."""
    plan = build_premarket_price_plan(
        bars,
        signal_date=signal_date,
        ma5=trend.ma5,
        ma10=trend.ma10,
        ma20=trend.ma20,
    )
    dashboard = result.dashboard if isinstance(result.dashboard, dict) else {}
    result.dashboard = dashboard
    dashboard["raw_model_score"] = result.sentiment_score
    result.sentiment_score = trend.signal_score
    result.trend_prediction = trend.trend_status.value
    result.operation_advice = "观望"
    result.decision_type = "hold"
    result.action = "watch"
    result.action_label = "待开盘复核" if plan["status"] == "待开盘复核" else "观察"
    dashboard["technical_score_components"] = dict(trend.score_components)
    dashboard["technical_score"] = trend.signal_score
    dashboard["premarket_signal_date"] = signal_date.isoformat()
    audited_news = list(news_audit or [])
    audited_financial = dict(financial_audit or {"status": "unavailable"})
    dashboard["premarket_news_audit"] = audited_news
    dashboard["premarket_financial_audit"] = audited_financial
    dashboard["premarket_evidence_status"] = summarize_evidence(audited_news, audited_financial)
    accepted_titles = [
        item["title"] for item in audited_news
        if item.get("status") in {"source_verified", "metadata_verified", "partial"} and item.get("title")
    ]
    intelligence = dashboard.get("intelligence")
    if not isinstance(intelligence, dict):
        intelligence = {}
        dashboard["intelligence"] = intelligence
    intelligence.update({
        "latest_news": "；".join(accepted_titles[:2]) if accepted_titles else "无可核实的盘前资讯",
        "positive_catalysts": accepted_titles[:3],
        "risk_alerts": [],
        "sentiment_summary": "仅列出通过发布时间和来源元数据核验的条目；事实内容仍待原文复核",
        "earnings_outlook": (
            "财务披露时刻与出处已核验，具体数据仍待原文复核"
            if audited_financial.get("status") == "source_verified"
            else "财务披露时刻或原始出处未核，不作具体业绩断言"
        ),
    })
    result.news_summary = intelligence["latest_news"]
    result.fundamental_analysis = (
        "财务披露时刻与来源已核；具体业绩以原文复核为准。"
        if audited_financial.get("status") == "source_verified"
        else "财务披露时刻或原始出处未核，不作具体业绩断言。"
    )
    result.company_highlights = ""
    result.sector_position = ""
    result.market_sentiment = ""
    result.hot_topics = ""
    result.key_points = ""
    result.fundamental_context = None
    result.market_structure_context = None
    result.risk_warning = "盘前价格计划不是买入指令；开盘跳空和止损成交存在偏差风险。"
    if isinstance(data_quality, dict):
        score = data_quality.get("overall_score")
        if isinstance(score, int) and not isinstance(score, bool) and 0 <= score <= 100:
            dashboard["data_block_availability"] = score

    perspective = dashboard.get("data_perspective")
    if not isinstance(perspective, dict):
        perspective = {}
        dashboard["data_perspective"] = perspective
    perspective["trend_status"] = {
        "ma_alignment": trend.ma_alignment,
        "is_bullish": trend.ma5 > trend.ma10 > trend.ma20,
        "trend_score": trend.trend_strength,
    }
    perspective["price_position"] = {
        "current_price": round(trend.current_price, 2),
        "ma5": round(trend.ma5, 2),
        "ma10": round(trend.ma10, 2),
        "ma20": round(trend.ma20, 2),
        "bias_ma5": round(trend.bias_ma5, 2),
        "bias_status": "接近均线，未确认支撑" if trend.bias_ma5 < 0 else "均线上方，仍须复核",
        "support_level": min(trend.support_levels) if trend.support_levels else None,
        "resistance_level": min(trend.resistance_levels) if trend.resistance_levels else None,
    }
    frame = bars.sort_values("date")
    previous_volume_ratio = None
    if len(frame) >= 2 and "volume" in frame.columns:
        latest_volume = _number(frame.iloc[-1].get("volume"))
        prior_volume = _number(frame.iloc[-2].get("volume"))
        if latest_volume is not None and prior_volume is not None and prior_volume > 0:
            previous_volume_ratio = round(latest_volume / prior_volume, 2)
    perspective["volume_analysis"] = {
        "volume_ratio": round(trend.volume_ratio_5d, 2),
        "volume_status": trend.volume_status.value,
        "previous_day_volume_ratio": previous_volume_ratio,
        "volume_meaning": "5日量能状态用于评分；昨量比仅描述相邻两日成交量。",
    }
    perspective["chip_structure"] = {}
    components = trend.score_components
    bias_interpretation = "低于MA5，尚未确认支撑" if trend.bias_ma5 < 0 else "位于MA5上方，仍待开盘确认"
    result.technical_analysis = (
        f"技术评分{trend.signal_score}/100：趋势{trend.trend_status.value}"
        f"({components.get('trend', 0)}/30)，MA5乖离{trend.bias_ma5:.2f}%"
        f"({components.get('bias_ma5', 0)}/20；{bias_interpretation})，5日量能{trend.volume_status.value}"
        f"({components.get('volume_5d', 0)}/15)，均线支撑"
        f"({components.get('ma_support', 0)}/10)，MACD {trend.macd_status.value}"
        f"({components.get('macd', 0)}/15)，RSI {trend.rsi_status.value}"
        f"({components.get('rsi', 0)}/10)。"
    )
    result.ma_analysis = (
        f"{trend.ma_alignment}；MA5={trend.ma5:.2f}，MA10={trend.ma10:.2f}，"
        f"MA20={trend.ma20:.2f}。接近MA5不等于已确认支撑。"
    )
    result.trend_analysis = result.ma_analysis
    result.short_term_outlook = "开盘后的15/30分钟确认由独立开盘复核完成。"
    result.medium_term_outlook = "盘前不推断未完成交易日的走势。"
    result.pattern_analysis = ""
    result.volume_analysis = (
        f"5日量能状态：{trend.volume_status.value}（当日量/5日均量="
        f"{trend.volume_ratio_5d:.2f}）；昨量比："
        f"{previous_volume_ratio:.2f}" if previous_volume_ratio is not None else
        f"5日量能状态：{trend.volume_status.value}；昨量比：数据缺失"
    )
    battle = dashboard.get("battle_plan")
    if not isinstance(battle, dict):
        battle = {}
        dashboard["battle_plan"] = battle
    battle["price_plan"] = plan
    battle["sniper_points"] = {
        "ideal_buy": plan["ideal"]["reference"] if plan["ideal"] else "数据不足",
        "secondary_buy": plan["secondary"]["reference"] if plan["secondary"] else "数据不足",
        "stop_loss": "按情景分别复核",
        "take_profit": "按情景分别复核",
    }
    battle["position_strategy"] = {
        "suggested_position": "新增计划股数0",
        "entry_plan": "仅待开盘复核或观察；盘前不触发交易",
        "risk_control": "开盘跳空超出入场上限则原计划失效；不读取账户",
    }
    battle["action_checklist"] = []
    if "signal_attribution" in dashboard:
        dashboard["raw_model_signal_attribution"] = dashboard.pop("signal_attribution")
    if "strategy_synthesis" in dashboard:
        dashboard["raw_model_strategy_synthesis"] = dashboard.pop("strategy_synthesis")
    dashboard["phase_decision"] = {
        "action_window": "盘前计划",
        "immediate_action": "等待开盘后复核",
        "watch_conditions": ["核对入场区间、结构压力和15/30分钟确认"],
        "next_check_time": "美股开盘后由独立开盘分析复核",
        "confidence_reason": "技术指标冻结于信号日收盘；资讯另行核验",
        "data_limitations": [],
    }
    core = dashboard.get("core_conclusion")
    if not isinstance(core, dict):
        core = {}
        dashboard["core_conclusion"] = core
    core["signal_type"] = "🟡待开盘复核" if plan["status"] == "待开盘复核" else "⚪观察"
    core["one_sentence"] = "技术结构和价格计划仅为信号日盘前研究，待开盘另行复核"
    core["time_sensitivity"] = "待开盘复核"
    core["position_advice"] = {
        "no_position": "观察；不开盘前买入",
        "has_position": "未读取账户，不推断实际持仓",
    }
    result.analysis_summary = core["one_sentence"]
    result.buy_reason = "盘前无买入结论；仅提供两套条件参考。"
    result.market_snapshot = _market_snapshot(frame, signal_date)
    result.current_price = float(frame.iloc[-1]["close"])
    return plan


def _market_snapshot(frame: pd.DataFrame, signal_date: Any) -> dict[str, Any]:
    last = frame.iloc[-1]
    previous = frame.iloc[-2] if len(frame) >= 2 else None
    close = float(last["close"])
    previous_close = float(previous["close"]) if previous is not None else None
    change = close - previous_close if previous_close else None
    open_price = _number(last.get("open"))
    volume = _number(last.get("volume"))
    amount = _number(last.get("amount"))
    return {
        "date": signal_date.isoformat(),
        "currency": "USD",
        "close": f"{close:.2f}",
        "prev_close": f"{previous_close:.2f}" if previous_close else "N/A",
        "open": f"{open_price:.2f}" if open_price is not None else "N/A",
        "high": f"{float(last['high']):.2f}",
        "low": f"{float(last['low']):.2f}",
        "pct_chg": f"{change / previous_close * 100:.2f}%" if change is not None else "N/A",
        "change_amount": f"{change:.2f}" if change is not None else "N/A",
        "amplitude": f"{(float(last['high']) - float(last['low'])) / previous_close * 100:.2f}%" if previous_close else "N/A",
        "volume": f"{volume / 1e8:.2f} 亿股" if volume is not None else "N/A",
        "amount": f"{amount / 1e8:.2f} 亿美元" if amount is not None else "N/A",
    }


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
