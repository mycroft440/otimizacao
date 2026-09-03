from __future__ import annotations

import math

import numpy as np
import pandas as pd

import optimize_b3_pine as opt


def _rolling_sum_prefix_reference(values: np.ndarray, window: int) -> np.ndarray:
    """Independent scalar prefix implementation of the float64 rolling contract.

    The production engine uses a float64 prefix/cumsum and strict comparisons in
    the subsequent direction state. Re-summing each window independently is
    mathematically equivalent but not bit-equivalent: different rounding can turn
    a flat SMA into a tiny positive/negative move and change a discrete signal.
    This reference implementation deliberately rebuilds the same *numeric
    contract* with an explicit scalar prefix loop without calling production
    rolling helpers.
    """
    data = np.asarray(values, dtype=np.float64)
    out = np.full(data.shape, np.nan, dtype=np.float64)
    if window <= 0 or len(data) < window:
        return out
    prefix = np.empty(len(data) + 1, dtype=np.float64)
    prefix[0] = np.float64(0.0)
    for i, value in enumerate(data):
        prefix[i + 1] = np.float64(prefix[i] + value)
    for i in range(window - 1, len(data)):
        out[i] = np.float64(prefix[i + 1] - prefix[i + 1 - window])
    return out


def _rolling_mean_contiguous_reference(
    values: np.ndarray, window: int, valid_from: int
) -> np.ndarray:
    """Independent float64 prefix SMA for the contiguous valid ratio segment."""
    data = np.asarray(values, dtype=np.float64)
    out = np.full(data.shape, np.nan, dtype=np.float64)
    if window <= 0 or valid_from >= len(data):
        return out
    segment = data[valid_from:]
    if len(segment) < window:
        return out
    prefix = np.empty(len(segment) + 1, dtype=np.float64)
    prefix[0] = np.float64(0.0)
    for i, value in enumerate(segment):
        prefix[i + 1] = np.float64(prefix[i] + value)
    for j in range(window - 1, len(segment)):
        total = np.float64(prefix[j + 1] - prefix[j + 1 - window])
        out[valid_from + j] = np.float64(total / window)
    return out


def _indicator_for_ticker(frame, gap_period: int, signal_period: int, momentum_period: int, vol_period: int):
    opens = frame["open"].to_numpy(dtype=np.float64)
    closes = frame["close"].to_numpy(dtype=np.float64)
    n = len(closes)

    gaps = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        gaps[i] = np.float64(opens[i] - closes[i - 1])

    positive = np.maximum(gaps, np.float64(0.0))
    negative = np.maximum(-gaps, np.float64(0.0))
    pos_sum = _rolling_sum_prefix_reference(positive, gap_period)
    neg_sum = _rolling_sum_prefix_reference(negative, gap_period)
    ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(gap_period - 1, n):
        pos = float(pos_sum[i])
        neg = float(neg_sum[i])
        if not math.isfinite(pos) or not math.isfinite(neg):
            continue
        ratio[i] = np.float64(1.0 if neg == 0.0 else 100.0 * pos / neg)

    signal = _rolling_mean_contiguous_reference(ratio, signal_period, gap_period - 1)

    state = np.zeros(n, dtype=bool)
    valid = np.flatnonzero(np.isfinite(signal))
    if len(valid) >= 2:
        current = False
        start = int(valid[0])
        for i in range(start + 1, n):
            if not math.isfinite(float(signal[i])) or not math.isfinite(float(signal[i - 1])):
                state[i] = current
                continue
            if signal[i] > signal[i - 1]:
                current = True
            elif signal[i] < signal[i - 1]:
                current = False
            state[i] = current

    momentum = np.full(n, np.nan, dtype=float)
    for i in range(momentum_period, n):
        prev = float(closes[i - momentum_period])
        cur = float(closes[i])
        if prev > 0.0 and cur > 0.0:
            momentum[i] = cur / prev - 1.0

    returns = np.zeros(n, dtype=float)
    for i in range(1, n):
        prev = float(closes[i - 1])
        returns[i] = closes[i] / prev - 1.0 if prev > 0.0 else 0.0
    vol_valid = np.zeros(n, dtype=bool)
    for i in range(vol_period - 1, n):
        values = returns[i - vol_period + 1 : i + 1]
        if len(values) > 1:
            variance = float(np.var(values, ddof=1))
            vol_valid[i] = math.isfinite(variance) and variance > 0.0

    return state, momentum, vol_valid


def weekly_schedule_reference(master_dates: pd.DatetimeIndex, start: str):
    """Rebuild weekly decision/execution dates without primary-engine mappings."""
    master = pd.DatetimeIndex(master_dates)
    if len(master) < 2:
        raise RuntimeError("reference engine: calendario insuficiente")
    iso = master.isocalendar()
    keys = list(zip(iso["year"].astype(int), iso["week"].astype(int)))
    first_positions = []
    previous = None
    for i, key in enumerate(keys):
        if key != previous:
            first_positions.append(i)
            previous = key
    start_ts = pd.Timestamp(start)
    exec_positions = [i for i in first_positions if master[i] >= start_ts and i > 0]
    execution_dates = pd.DatetimeIndex([master[i] for i in exec_positions])
    decision_dates = pd.DatetimeIndex([master[i - 1] for i in exec_positions])
    return execution_dates, decision_dates


def build_weekly_inputs(
    market: opt.MarketData,
    gap_period: int,
    signal_period: int,
    momentum_period: int,
    vol_period: int,
):
    execution_dates, decision_dates = weekly_schedule_reference(market.master_dates, market.start)
    weeks = len(execution_dates)
    assets = len(market.tickers)
    gap_state = np.zeros((weeks, assets), dtype=bool)
    momentum = np.full((weeks, assets), np.nan, dtype=float)
    vol_valid = np.zeros((weeks, assets), dtype=bool)

    for ti, ticker in enumerate(market.tickers):
        frame = market.frames[ticker]
        state, mom, vol = _indicator_for_ticker(
            frame, gap_period, signal_period, momentum_period, vol_period
        )
        didx = pd.Index(frame["date"]).get_indexer(decision_dates)
        for w, idx in enumerate(didx):
            if idx < 0:
                continue
            gap_state[w, ti] = bool(state[idx])
            momentum[w, ti] = float(mom[idx])
            vol_valid[w, ti] = bool(vol[idx])

    targets = np.full(weeks, -1, dtype=np.int16)
    for w in range(weeks):
        best = -1
        best_score = -math.inf
        for ti in range(assets):
            score = float(momentum[w, ti])
            eligible = gap_state[w, ti] and vol_valid[w, ti] and math.isfinite(score) and score > 0.0
            if eligible and score > best_score:
                best = ti
                best_score = score
        targets[w] = best
    return gap_state, momentum, vol_valid, targets, execution_dates, decision_dates


def _buy_cost(raw: float, qty: int, fee: float, base: float, extra: float) -> float:
    odd = qty % 100
    gross = raw * ((1.0 + base) * qty + extra * odd)
    return gross * (1.0 + fee)


def affordable_qty_reference(cash: float, raw: float, fee: float, base: float, extra: float) -> int:
    if not math.isfinite(cash) or not math.isfinite(raw) or cash <= 0.0 or raw <= 0.0:
        return 0
    upper = max(0, int(cash / raw) + 1)
    q = upper
    while q > 0 and _buy_cost(raw, q, fee, base, extra) > cash + 1e-9:
        q -= 1
    while _buy_cost(raw, q + 1, fee, base, extra) <= cash + 1e-9:
        q += 1
    return q


def _open_on(frame: pd.DataFrame, day: pd.Timestamp) -> float:
    idx = pd.Index(frame["date"]).get_indexer([pd.Timestamp(day)])[0]
    if idx < 0:
        return float("nan")
    return float(frame.iloc[idx]["open"])


def _final_close_reference(frame: pd.DataFrame, end: str) -> float:
    prior = frame[frame["date"] <= pd.Timestamp(end)]
    if prior.empty:
        return float("nan")
    return float(prior.iloc[-1]["close"])


def simulate_reference(
    market: opt.MarketData,
    gap_period: int,
    signal_period: int,
    momentum_period: int,
    vol_period: int,
    *,
    initial_cash: float,
    fee_bps: float,
    slippage_bps: float,
    odd_lot_extra_bps: float,
):
    gap_state, momentum, vol_valid, base_targets, execution_dates, decision_dates = build_weekly_inputs(
        market, gap_period, signal_period, momentum_period, vol_period
    )
    fee = fee_bps / 10000.0
    base = slippage_bps / 10000.0
    extra = odd_lot_extra_bps / 10000.0
    cash = float(initial_cash)
    shares = 0
    holding = -1
    trades = 0
    skipped = 0
    fees_paid = 0.0
    slippage_paid = 0.0
    actual_targets = []

    for w, proposed in enumerate(base_targets):
        target = int(proposed)
        execution_day = execution_dates[w]
        if holding >= 0 and target >= 0 and holding != target:
            incumbent_score = float(momentum[w, holding])
            target_score = float(momentum[w, target])
            incumbent_ok = (
                gap_state[w, holding]
                and vol_valid[w, holding]
                and math.isfinite(incumbent_score)
                and incumbent_score > 0.0
            )
            if incumbent_ok and math.isfinite(target_score) and incumbent_score == target_score:
                target = holding
        actual_targets.append(target)
        if target == holding:
            continue

        projected = cash
        valid = True
        if holding >= 0:
            raw_sell = _open_on(market.frames[market.tickers[holding]], execution_day)
            if not math.isfinite(raw_sell) or raw_sell <= 0.0:
                valid = False
            else:
                odd = shares % 100
                slip = raw_sell * (base * shares + extra * odd)
                gross = raw_sell * shares - slip
                projected += gross - gross * fee
        if target >= 0:
            raw_buy = _open_on(market.frames[market.tickers[target]], execution_day)
            if not math.isfinite(raw_buy) or raw_buy <= 0.0:
                valid = False
            elif affordable_qty_reference(projected, raw_buy, fee, base, extra) <= 0:
                valid = False
        if not valid:
            skipped += 1
            continue

        if holding >= 0:
            raw_sell = _open_on(market.frames[market.tickers[holding]], execution_day)
            odd = shares % 100
            slip = raw_sell * (base * shares + extra * odd)
            gross = raw_sell * shares - slip
            charge = gross * fee
            cash += gross - charge
            fees_paid += charge
            slippage_paid += slip
            shares = 0
            holding = -1
            trades += 1

        if target >= 0:
            raw_buy = _open_on(market.frames[market.tickers[target]], execution_day)
            qty = affordable_qty_reference(cash, raw_buy, fee, base, extra)
            odd = qty % 100
            gross = raw_buy * ((1.0 + base) * qty + extra * odd)
            charge = gross * fee
            slip = raw_buy * (base * qty + extra * odd)
            cash -= gross + charge
            if cash < -1e-7:
                raise RuntimeError("reference engine gerou caixa negativo")
            cash = max(cash, 0.0)
            shares = qty
            holding = target
            fees_paid += charge
            slippage_paid += slip
            trades += int(qty > 0)

    final_equity = cash
    if holding >= 0:
        price = _final_close_reference(market.frames[market.tickers[holding]], market.end)
        if not math.isfinite(price) or price <= 0.0:
            raise RuntimeError("reference engine sem preco final valido")
        final_equity += shares * price

    return {
        "final_equity": final_equity,
        "total_return": final_equity / initial_cash - 1.0,
        "cash": cash,
        "shares": shares,
        "holding": holding,
        "trades": trades,
        "skipped": skipped,
        "fees": fees_paid,
        "slippage": slippage_paid,
        "base_targets": base_targets,
        "actual_targets": np.asarray(actual_targets, dtype=np.int16),
        "gap_state": gap_state,
        "momentum": momentum,
        "vol_valid": vol_valid,
        "execution_dates": execution_dates,
        "decision_dates": decision_dates,
    }
