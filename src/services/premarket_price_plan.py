"""Deterministic, signal-day-only reference plans for US premarket research.

These are conditional research levels, never orders.  Opening confirmation and
the final breakout-retest stop belong to the separate opening review.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import pandas as pd


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _price(value: float | None) -> float | None:
    return round(value + 1e-9, 2) if value is not None else None


def _atr14(bars: pd.DataFrame) -> float | None:
    if len(bars) < 15:
        return None
    highs = pd.to_numeric(bars["high"], errors="coerce").tolist()
    lows = pd.to_numeric(bars["low"], errors="coerce").tolist()
    closes = pd.to_numeric(bars["close"], errors="coerce").tolist()
    true_ranges: list[float] = []
    for index in range(1, len(bars)):
        high, low, previous_close = highs[index], lows[index], closes[index - 1]
        if not all(math.isfinite(item) for item in (high, low, previous_close)) or high < low:
            return None
        true_ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
    atr = sum(true_ranges[:14]) / 14
    for true_range in true_ranges[14:]:
        atr = (atr * 13 + true_range) / 14
    return atr if atr > 0 else None


def _levels(bars: pd.DataFrame, ma10: float, ma20: float) -> tuple[list[float], list[float]]:
    """Only pivots with two completed right-hand bars count as confirmed."""
    highs = bars["high"].astype(float).tolist()
    lows = bars["low"].astype(float).tolist()
    resistance = {highs[-1], max(highs[-20:])}
    support = {min(lows[-20:])}
    for index in range(2, len(bars) - 2):
        if highs[index] > max(highs[index - 2:index]) and highs[index] >= max(highs[index + 1:index + 3]):
            resistance.add(highs[index])
        if lows[index] < min(lows[index - 2:index]) and lows[index] <= min(lows[index + 1:index + 3]):
            support.add(lows[index])
    last_close = float(bars.iloc[-1]["close"])
    resistance.update(level for level in (ma10, ma20) if level > last_close)
    support.update(level for level in (ma10, ma20) if level < last_close)
    return sorted(resistance), sorted(support)


def _scenario(
    *,
    name: str,
    kind: str,
    anchor: float,
    atr: float,
    resistances: list[float],
    supports: list[float],
    base_low: float,
) -> dict[str, Any]:
    if kind == "breakout_retest":
        low, high = _price(anchor + 0.1 * atr), _price(anchor + 0.5 * atr)
        stop = _price(anchor - 0.5 * atr)
        stop_source = anchor
        trigger = "开盘后突破压力、回踩守住，再由15/30分钟量价复核；跳空超出上限则失效"
    else:
        low, high = _price(anchor - 0.25 * atr), _price(anchor + 0.25 * atr)
        lower_support = [level for level in supports if level < low]
        stop_source = max(lower_support) if lower_support else None
        stop = _price(stop_source - 0.25 * atr) if stop_source is not None else None
        trigger = "开盘后触及参考区间并重新站回均线，由15/30分钟走势复核"
    scenario: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "reference": _price(anchor),
        "entry_low": low,
        "entry_high": high,
        "stop": stop,
        "stop_source_level": _price(stop_source),
        "target1": None,
        "target2": None,
        "status": "观察",
        "reason": "",
        "opening_trigger": trigger,
        "target_evidence": [],
    }
    if low is None or high is None or stop is None or not stop < low < high:
        scenario["reason"] = "入场区间下方缺少有效止损依据"
        return scenario
    # A pressure inside the proposed interval makes a single pullback plan
    # internally contradictory.  It is not an R-multiple threshold.
    if kind != "breakout_retest" and any(low < level <= high for level in resistances):
        scenario["reason"] = "入场区间穿越未突破的结构压力"
        return scenario
    upper = [level for level in resistances if level > high]
    target_sources = {}
    for level in upper:
        target = _price(level - 0.1 * atr)
        if target is not None and target > high:
            target_sources.setdefault(target, _price(level))
    targets = sorted(target_sources)
    if targets:
        scenario["target1"] = targets[0]
        scenario["target1_source_level"] = target_sources[targets[0]]
        scenario["target_evidence"].append("上方已确认压力减0.1×ATR14")
    if len(targets) >= 2:
        scenario["target2"] = targets[1]
        scenario["target2_source_level"] = target_sources[targets[1]]
        scenario["target_evidence"].append("次级已确认压力减0.1×ATR14")
    elif targets and kind == "breakout_retest" and base_low < anchor:
        measured = _price(anchor + (anchor - base_low))
        if measured is not None and measured > targets[0]:
            scenario["target2"] = measured
            scenario["target2_source_level"] = _price(base_low)
            scenario["target_evidence"].append("20日整理区间测量目标，非价格预测")
    if scenario["target1"] is None or scenario["target2"] is None:
        scenario["reason"] = "上方结构目标不足；不强填目标"
        return scenario
    scenario["status"] = "待开盘复核"
    scenario["reason"] = "盘前数字关系成立；开盘触发及成交价格尚未确认"
    return scenario


def build_premarket_price_plan(
    bars: pd.DataFrame,
    *,
    signal_date: date,
    ma5: float,
    ma10: float,
    ma20: float,
) -> dict[str, Any]:
    """Return two separate conditional plans without a minimum R-multiple gate."""
    plan: dict[str, Any] = {
        "plan_policy": "dsa_structural_v1",
        "signal_date": signal_date.isoformat(),
        "currency": "USD",
        "status": "观察",
        "atr14": None,
        "ideal": None,
        "secondary": None,
        "planned_new_shares": 0,
        "minimum_reward_risk_required": False,
        "round_trip_cost_per_share": None,
    }
    needed = {"date", "high", "low", "close"}
    if bars is None or not needed.issubset(bars.columns):
        plan["reason"] = "缺少完整日线列"
        return plan
    frame = bars.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame = frame.loc[frame["date"].notna() & (frame["date"] <= signal_date)]
    frame = frame.drop_duplicates(subset="date", keep="last").sort_values("date")
    if len(frame) < 20 or frame.iloc[-1]["date"] != signal_date:
        plan["reason"] = "信号日日线缺失或历史长度不足"
        return plan
    averages = [_finite(value) for value in (ma5, ma10, ma20)]
    atr = _atr14(frame)
    if any(value is None or value <= 0 for value in averages) or atr is None:
        plan["reason"] = "均线或ATR14不可验证"
        return plan
    ma5, ma10, ma20 = averages
    if min(ma5, ma10, ma20) < 1:
        plan["reason"] = "低价证券最小报价单位未核实"
        return plan
    try:
        resistances, supports = _levels(frame.tail(60), ma10, ma20)
    except (TypeError, ValueError):
        plan["reason"] = "结构高低点不可验证"
        return plan
    plan["atr14"] = round(atr, 4)
    bull = ma5 > ma10 > ma20
    plan["regime"] = "多头回踩" if bull else "非多头：收复／突破观察"
    base_low = float(frame.tail(20)["low"].min())
    plan["ideal"] = _scenario(
        name="理想买入参考", kind="pullback" if bull else "reclaim_watch",
        anchor=ma5, atr=atr, resistances=resistances,
        supports=supports, base_low=base_low,
    )
    breakout_anchor = max(ma10, ma20, float(frame.tail(20)["high"].max()))
    plan["secondary"] = _scenario(
        name="次优买入参考", kind="deep_pullback" if bull else "breakout_retest",
        anchor=ma10 if bull else breakout_anchor,
        atr=atr, resistances=resistances, supports=supports, base_low=base_low,
    )
    # In a weak/bearish structure, the MA5 reclaim is a watch level, not an
    # immediately qualified pullback setup.
    if not bull:
        plan["ideal"]["status"] = "观察"
        plan["ideal"]["reason"] = "信号日非多头，MA5只是收复观察位"
    plan["status"] = (
        "待开盘复核"
        if any(item["status"] == "待开盘复核" for item in (plan["ideal"], plan["secondary"]))
        else "观察"
    )
    return plan
