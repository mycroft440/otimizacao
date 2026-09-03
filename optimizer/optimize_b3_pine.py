#!/usr/bin/env python3
"""Otimizador exaustivo da estratégia Pine B3 Gap Momentum + Top1 semanal.

O código usa somente candles oficiais congelados pelo workflow.
A estratégia reproduz a semântica do Pine V17 em modo 1x para a otimização dos
períodos do indicador:
- Gap Ratio por janela;
- SMA do Gap Ratio e estado persistente pela direção da SMA;
- Momentum positivo como score;
- volatilidade amostral de 21 retornos apenas como gate > 0;
- decisão no último pregão da semana e execução na primeira abertura seguinte;
- Top1; empate exato preserva incumbente elegível;
- se o mesmo Top1 continua, mantém exatamente a posição: sem venda, recompra ou
  reinvestimento semanal de caixa residual;
- troca de ativo atômica: valida venda e nova compra antes de alterar a carteira;
- custos/slippage adversos por lado e penalidade fracionária ponderada;
- lote mínimo de 1 ação; sem dividendos/JCP, IR ou alavancagem nesta etapa de
  otimização dos parâmetros.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config


@dataclass
class MarketData:
    tickers: list[str]
    frames: dict[str, pd.DataFrame]
    master_dates: pd.DatetimeIndex
    execution_dates: pd.DatetimeIndex
    decision_dates: pd.DatetimeIndex
    exec_open: np.ndarray  # [week, ticker]
    decision_index: np.ndarray  # [ticker, week], -1 quando não há candle na data mestre
    final_close: np.ndarray  # [ticker]
    start: str
    end: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--start", default=config.DEFAULT_START)
    p.add_argument("--end", default=config.DEFAULT_END)
    p.add_argument("--gap-min", type=int, default=config.DEFAULT_GAP_MIN)
    p.add_argument("--gap-max", type=int, default=config.DEFAULT_GAP_MAX)
    p.add_argument("--signal-min", type=int, default=config.DEFAULT_SIGNAL_MIN)
    p.add_argument("--signal-max", type=int, default=config.DEFAULT_SIGNAL_MAX)
    p.add_argument("--momentum-min", type=int, default=config.DEFAULT_MOMENTUM_MIN)
    p.add_argument("--momentum-max", type=int, default=config.DEFAULT_MOMENTUM_MAX)
    p.add_argument("--vol-period", type=int, default=config.DEFAULT_VOL_PERIOD)
    p.add_argument("--initial-cash", type=float, default=config.DEFAULT_INITIAL_CASH)
    p.add_argument("--fee-bps", type=float, default=config.DEFAULT_FEE_BPS)
    p.add_argument("--slippage-bps", type=float, default=config.DEFAULT_SLIPPAGE_BPS)
    p.add_argument("--odd-lot-extra-bps", type=float, default=config.DEFAULT_ODD_LOT_EXTRA_BPS)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if window <= 0 or len(values) < window:
        return out
    c = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    out[window - 1 :] = c[window:] - c[:-window]
    return out


def rolling_mean_contiguous(values: np.ndarray, window: int, valid_from: int) -> np.ndarray:
    """SMA quando os valores passam a ser contínuos a partir de valid_from."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if window <= 0 or valid_from >= len(values):
        return out
    segment = values[valid_from:]
    if len(segment) < window:
        return out
    c = np.concatenate(([0.0], np.cumsum(segment, dtype=np.float64)))
    means = (c[window:] - c[:-window]) / window
    out[valid_from + window - 1 :] = means
    return out


def persistent_direction_state(signal_line: np.ndarray) -> np.ndarray:
    """Equivale ao ta.valuewhen(changed, changedState, 0) do Pine."""
    state = np.zeros(len(signal_line), dtype=np.bool_)
    valid = np.flatnonzero(np.isfinite(signal_line))
    if len(valid) < 2:
        return state
    start = int(valid[0])
    seq = signal_line[start:]
    diff = np.diff(seq)
    event = np.where(diff > 0.0, 1, np.where(diff < 0.0, 0, -1)).astype(np.int8)
    event_idx = np.where(event >= 0, np.arange(len(event), dtype=np.int32), -1)
    last_idx = np.maximum.accumulate(event_idx)
    after = np.zeros(len(event), dtype=np.bool_)
    mask = last_idx >= 0
    if np.any(mask):
        after[mask] = event[last_idx[mask]] == 1
    state[start + 1 :] = after
    return state


def sample_vol_positive(closes: np.ndarray, period: int) -> np.ndarray:
    returns = np.zeros(len(closes), dtype=np.float64)
    if len(closes) > 1:
        prev = closes[:-1]
        returns[1:] = np.where(prev > 0.0, closes[1:] / prev - 1.0, 0.0)
    s1 = rolling_sum(returns, period)
    s2 = rolling_sum(returns * returns, period)
    var = np.full(len(closes), np.nan, dtype=np.float64)
    ok = np.isfinite(s1) & np.isfinite(s2)
    if period > 1:
        var[ok] = np.maximum(0.0, (s2[ok] - s1[ok] * s1[ok] / period) / (period - 1))
    return np.isfinite(var) & (var > 0.0)


def _load_master_calendar(data_root: Path, observed_dates: set[pd.Timestamp]) -> pd.DatetimeIndex:
    """Load the independent B3 session calendar when present.

    Hardened workflows build ``data/calendars/b3_sessions.csv`` directly from
    annual B3 COTAHIST archives. Synthetic/unit-test datasets may omit it and use
    the legacy observed-date union as an explicit compatibility fallback.
    """
    calendar_path = data_root / "data" / "calendars" / "b3_sessions.csv"
    if not calendar_path.exists():
        return pd.DatetimeIndex(sorted(observed_dates))
    calendar = pd.read_csv(calendar_path, usecols=["date"])
    dates = pd.DatetimeIndex(
        pd.to_datetime(calendar["date"].astype(str).str.slice(0, 10), errors="raise")
    )
    if dates.empty or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise RuntimeError("Calendário B3 oficial vazio, duplicado ou fora de ordem.")
    official = set(dates.tolist())
    outside = sorted(day for day in observed_dates if day not in official)
    if outside:
        raise RuntimeError(
            "Candles contêm datas ausentes do calendário COTAHIST oficial: "
            + ", ".join(str(day.date()) for day in outside[:10])
        )
    return dates


def load_market(data_root: Path, start: str, end: str) -> MarketData:
    universe_path = data_root / "data" / "universes" / "fixed_40_2018.json"
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    tickers = [str(x).upper() for x in universe["tickers"]]
    frames: dict[str, pd.DataFrame] = {}
    all_dates: set[pd.Timestamp] = set()

    for ticker in tickers:
        path = data_root / "data" / "candles" / f"{ticker.lower()}_1d.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, usecols=["date", "open", "close"])
        df["date"] = pd.to_datetime(df["date"].astype(str).str.slice(0, 10), errors="raise")
        df = df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df[
            np.isfinite(df["open"])
            & np.isfinite(df["close"])
            & (df["open"] > 0)
            & (df["close"] > 0)
        ].copy()
        frames[ticker] = df
        all_dates.update(df["date"].tolist())

    master = _load_master_calendar(data_root, all_dates)
    start_ts = pd.Timestamp(start).normalize()
    requested_end_ts = pd.Timestamp(end).normalize() if end else master.max()
    master = master[master <= requested_end_ts]
    if len(master) < 2:
        raise RuntimeError("Calendário mestre insuficiente.")
    effective_end_ts = pd.Timestamp(master.max())
    if effective_end_ts < start_ts:
        raise RuntimeError("Nenhum pregão B3 dentro do período solicitado.")

    iso = master.isocalendar()
    week_key = (iso["year"].astype(str) + "-" + iso["week"].astype(str)).to_numpy()
    first_of_week = np.ones(len(master), dtype=bool)
    first_of_week[1:] = week_key[1:] != week_key[:-1]
    exec_positions = np.flatnonzero(first_of_week & (master >= start_ts))
    exec_positions = exec_positions[exec_positions > 0]
    execution_dates = master[exec_positions]
    decision_dates = master[exec_positions - 1]

    n_tickers = len(tickers)
    n_weeks = len(execution_dates)
    exec_open = np.full((n_weeks, n_tickers), np.nan, dtype=np.float64)
    decision_index = np.full((n_tickers, n_weeks), -1, dtype=np.int32)
    final_close = np.full(n_tickers, np.nan, dtype=np.float64)

    for ti, ticker in enumerate(tickers):
        df = frames[ticker]
        idx = pd.Index(df["date"])
        eidx = idx.get_indexer(execution_dates)
        didx = idx.get_indexer(decision_dates)
        decision_index[ti, :] = didx.astype(np.int32)
        ok = eidx >= 0
        if np.any(ok):
            exec_open[ok, ti] = df["open"].to_numpy(dtype=np.float64)[eidx[ok]]
        prior = df[df["date"] <= effective_end_ts]
        if not prior.empty:
            final_close[ti] = float(prior.iloc[-1]["close"])

    return MarketData(
        tickers=tickers,
        frames=frames,
        master_dates=master,
        execution_dates=execution_dates,
        decision_dates=decision_dates,
        exec_open=exec_open,
        decision_index=decision_index,
        final_close=final_close,
        start=start,
        end=effective_end_ts.date().isoformat(),
    )


def precompute_shard(
    market: MarketData,
    gap_values: list[int],
    signal_values: list[int],
    momentum_values: list[int],
    vol_period: int,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
    pairs = [(g, s) for g in gap_values for s in signal_values]
    p_index = {pair: i for i, pair in enumerate(pairs)}
    P, W, N = len(pairs), len(market.execution_dates), len(market.tickers)
    M = len(momentum_values)
    gap_state = np.zeros((P, W, N), dtype=np.bool_)
    momentum = np.full((M, W, N), np.nan, dtype=np.float64)
    vol_valid = np.zeros((W, N), dtype=np.bool_)

    for ti, ticker in enumerate(market.tickers):
        df = market.frames[ticker]
        opens = df["open"].to_numpy(dtype=np.float64)
        closes = df["close"].to_numpy(dtype=np.float64)
        didx = market.decision_index[ti]
        decision_ok = didx >= 0

        vol = sample_vol_positive(closes, vol_period)
        if np.any(decision_ok):
            vol_valid[decision_ok, ti] = vol[didx[decision_ok]]

        for mi, m in enumerate(momentum_values):
            valid_w = decision_ok & (didx >= m)
            if not np.any(valid_w):
                continue
            ii = didx[valid_w]
            prev = closes[ii - m]
            cur = closes[ii]
            score = np.where((prev > 0.0) & (cur > 0.0), cur / prev - 1.0, np.nan)
            momentum[mi, valid_w, ti] = score

        gaps = np.zeros(len(closes), dtype=np.float64)
        if len(closes) > 1:
            gaps[1:] = opens[1:] - closes[:-1]
        positive = np.maximum(gaps, 0.0)
        negative = np.maximum(-gaps, 0.0)

        for g in gap_values:
            pos_sum = rolling_sum(positive, g)
            neg_sum = rolling_sum(negative, g)
            ratio = np.full(len(closes), np.nan, dtype=np.float64)
            valid_ratio = np.isfinite(pos_sum) & np.isfinite(neg_sum)
            zero_neg = valid_ratio & (neg_sum == 0.0)
            ratio[zero_neg] = 1.0
            nz = valid_ratio & (neg_sum != 0.0)
            ratio[nz] = 100.0 * pos_sum[nz] / neg_sum[nz]
            valid_from = g - 1

            for s in signal_values:
                line = rolling_mean_contiguous(ratio, s, valid_from)
                state = persistent_direction_state(line)
                pi = p_index[(g, s)]
                if np.any(decision_ok):
                    gap_state[pi, decision_ok, ti] = state[didx[decision_ok]]

    return pairs, gap_state, momentum, vol_valid


def first_ranked_targets(gap_state: np.ndarray, mom: np.ndarray, vol_valid: np.ndarray) -> np.ndarray:
    """Top1 por semana e por par Gap/Signal, preservando ordem do universo em empate."""
    P, W, _N = gap_state.shape
    base = np.isfinite(mom) & (mom > 0.0) & vol_valid
    scores = np.where(base, mom, -np.inf)
    order = np.argsort(-scores, axis=1, kind="stable")
    targets = np.full((P, W), -1, dtype=np.int16)
    unresolved = np.ones((P, W), dtype=np.bool_)
    widx = np.arange(W)
    for rank in range(order.shape[1]):
        asset = order[:, rank]
        base_ok = base[widx, asset]
        gap_ok = gap_state[:, widx, asset]
        take = unresolved & gap_ok & base_ok[None, :]
        if np.any(take):
            targets = np.where(take, asset[None, :], targets)
            unresolved &= ~take
        if not np.any(unresolved):
            break
    return targets


def slip_amount(raw: np.ndarray, qty: np.ndarray, base: float, extra: float) -> np.ndarray:
    odd = qty % 100
    return raw * (base * qty + extra * odd)


def affordable_qty(cash: np.ndarray, raw: np.ndarray, fee: float, base: float, extra: float) -> np.ndarray:
    """Forma fechada equivalente ao scan de até 100 ações do Pine."""
    q = np.zeros(len(cash), dtype=np.int64)
    ok = np.isfinite(raw) & (raw > 0.0) & np.isfinite(cash) & (cash > 0.0)
    if not np.any(ok):
        return q
    rraw = raw[ok]
    cc = cash[ok]
    A = rraw * (1.0 + fee) * (1.0 + base)
    B = rraw * (1.0 + fee) * extra
    lots = np.floor(cc / (A * 100.0)).astype(np.int64)
    remainder = cc - lots * A * 100.0
    odd = np.floor(np.maximum(0.0, remainder) / (A + B)).astype(np.int64)
    odd = np.clip(odd, 0, 99)
    qq = lots * 100 + odd
    rr = qq % 100
    total = rraw * ((1.0 + base) * qq + extra * rr) * (1.0 + fee)
    too_much = total > cc + 1e-9
    qq[too_much] = np.maximum(0, qq[too_much] - 1)
    q[ok] = qq
    return q


def simulate_pairs(
    targets: np.ndarray,
    gap_state: np.ndarray,
    mom: np.ndarray,
    vol_valid: np.ndarray,
    market: MarketData,
    *,
    initial_cash: float,
    fee_bps: float,
    slippage_bps: float,
    odd_lot_extra_bps: float,
) -> dict[str, np.ndarray]:
    """Gerencia cada combinação como uma carteira Top1 independente.

    Política idêntica ao Pine V17 em 1x:
    - mesmo alvo: nenhuma ordem e nenhuma alteração de quantidade;
    - alvo diferente: preflight completo, depois venda e compra atômicas;
    - alvo CASH: vende a posição se houver abertura válida;
    - falha de execução: mantém a carteira exatamente como estava.
    """
    P, W = targets.shape
    fee = fee_bps / 10000.0
    base = slippage_bps / 10000.0
    extra = odd_lot_extra_bps / 10000.0
    cash = np.full(P, float(initial_cash), dtype=np.float64)
    shares = np.zeros(P, dtype=np.int64)
    holding = np.full(P, -1, dtype=np.int16)
    trades = np.zeros(P, dtype=np.int32)
    skipped = np.zeros(P, dtype=np.int32)
    fees_paid = np.zeros(P, dtype=np.float64)
    slip_paid = np.zeros(P, dtype=np.float64)

    for w in range(W):
        target = targets[:, w].copy()
        hmask = (holding >= 0) & (target >= 0) & (holding != target)
        if np.any(hmask):
            rows = np.flatnonzero(hmask)
            h = holding[rows].astype(np.int64)
            t = target[rows].astype(np.int64)
            inc_m = mom[w, h]
            top_m = mom[w, t]
            inc_ok = (
                gap_state[rows, w, h]
                & vol_valid[w, h]
                & np.isfinite(inc_m)
                & (inc_m > 0.0)
            )
            tie = inc_ok & np.isfinite(top_m) & (inc_m == top_m)
            if np.any(tie):
                target[rows[tie]] = holding[rows[tie]]

        changed = target != holding
        rows = np.flatnonzero(changed)
        if len(rows):
            projected = cash[rows].copy()
            valid = np.ones(len(rows), dtype=np.bool_)
            old = holding[rows].astype(np.int64)
            new = target[rows].astype(np.int64)

            sell_mask = old >= 0
            if np.any(sell_mask):
                rr = np.flatnonzero(sell_mask)
                raw = market.exec_open[w, old[rr]]
                ok = np.isfinite(raw) & (raw > 0.0)
                valid[rr] &= ok
                if np.any(ok):
                    q = shares[rows[rr]][ok]
                    raw_ok = raw[ok]
                    slip = slip_amount(raw_ok, q, base, extra)
                    gross = raw_ok * q - slip
                    sale_fee = gross * fee
                    projected[rr[ok]] += gross - sale_fee

            buy_mask = new >= 0
            if np.any(buy_mask):
                rr = np.flatnonzero(buy_mask)
                raw = market.exec_open[w, new[rr]]
                ok = np.isfinite(raw) & (raw > 0.0)
                valid[rr] &= ok
                if np.any(ok):
                    qq = affordable_qty(projected[rr[ok]], raw[ok], fee, base, extra)
                    valid[rr[ok]] &= qq > 0

            good_rows = rows[valid]
            bad_rows = rows[~valid]
            if len(bad_rows):
                skipped[bad_rows] += 1

            if len(good_rows):
                oldg = holding[good_rows].astype(np.int64)
                newg = target[good_rows].astype(np.int64)

                sell = oldg >= 0
                if np.any(sell):
                    gr = good_rows[sell]
                    old_asset = oldg[sell]
                    raw = market.exec_open[w, old_asset]
                    q = shares[gr]
                    slip = slip_amount(raw, q, base, extra)
                    gross = raw * q - slip
                    ff = gross * fee
                    cash[gr] += gross - ff
                    fees_paid[gr] += ff
                    slip_paid[gr] += slip
                    shares[gr] = 0
                    holding[gr] = -1
                    trades[gr] += 1

                buy = newg >= 0
                if np.any(buy):
                    gr = good_rows[buy]
                    new_asset = newg[buy]
                    raw = market.exec_open[w, new_asset]
                    q = affordable_qty(cash[gr], raw, fee, base, extra)
                    odd = q % 100
                    gross = raw * ((1.0 + base) * q + extra * odd)
                    ff = gross * fee
                    slip = raw * (base * q + extra * odd)
                    new_cash = cash[gr] - gross - ff
                    if np.any(new_cash < -1e-7):
                        raise RuntimeError("invariante violada: compra gerou caixa negativo")
                    cash[gr] = np.maximum(new_cash, 0.0)
                    shares[gr] = q
                    holding[gr] = new_asset.astype(np.int16)
                    fees_paid[gr] += ff
                    slip_paid[gr] += slip
                    trades[gr] += (q > 0).astype(np.int32)

        if np.any(cash < -1e-7):
            raise RuntimeError("invariante violada: caixa negativo")
        if np.any(shares < 0):
            raise RuntimeError("invariante violada: posição vendida/quantidade negativa")
        if np.any((holding < 0) != (shares == 0)):
            raise RuntimeError("invariante violada: holding e quantidade divergentes")

    equity = cash.copy()
    invested = holding >= 0
    if np.any(invested):
        rows = np.flatnonzero(invested)
        px = market.final_close[holding[rows].astype(np.int64)]
        if np.any(~np.isfinite(px)) or np.any(px <= 0.0):
            raise RuntimeError("preço final inválido para posição aberta")
        equity[rows] += shares[rows] * px

    return {
        "final_equity": equity,
        "total_return": equity / float(initial_cash) - 1.0,
        "cash": cash,
        "shares": shares,
        "holding": holding,
        "trades": trades,
        "skipped": skipped,
        "fees": fees_paid,
        "slippage": slip_paid,
    }


def main() -> None:
    args = parse_args()
    try:
        config.validate_run_config(
            start=args.start,
            end=args.end,
            gap_min=args.gap_min,
            gap_max=args.gap_max,
            signal_min=args.signal_min,
            signal_max=args.signal_max,
            momentum_min=args.momentum_min,
            momentum_max=args.momentum_max,
            vol_period=args.vol_period,
            initial_cash=args.initial_cash,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            odd_lot_extra_bps=args.odd_lot_extra_bps,
            shard_id=args.shard_id,
            shards=args.shards,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    all_gaps = list(range(args.gap_min, args.gap_max + 1))
    gap_values = all_gaps[args.shard_id :: args.shards]
    signal_values = list(range(args.signal_min, args.signal_max + 1))
    momentum_values = list(range(args.momentum_min, args.momentum_max + 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not gap_values:
        pd.DataFrame(
            columns=[
                "gap_period",
                "signal_period",
                "momentum_period",
                "vol_period",
                "final_equity",
                "total_return",
            ]
        ).to_csv(args.output, index=False)
        return

    market = load_market(args.data_root, args.start, args.end)
    pairs, gap_state, momentum, vol_valid = precompute_shard(
        market, gap_values, signal_values, momentum_values, args.vol_period
    )

    rows: list[dict[str, object]] = []
    for mi, m in enumerate(momentum_values):
        mom = momentum[mi]
        targets = first_ranked_targets(gap_state, mom, vol_valid)
        sim = simulate_pairs(
            targets,
            gap_state,
            mom,
            vol_valid,
            market,
            initial_cash=args.initial_cash,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            odd_lot_extra_bps=args.odd_lot_extra_bps,
        )
        for pi, (g, s) in enumerate(pairs):
            h = int(sim["holding"][pi])
            rows.append(
                {
                    "gap_period": g,
                    "signal_period": s,
                    "momentum_period": m,
                    "vol_period": args.vol_period,
                    "final_equity": float(sim["final_equity"][pi]),
                    "total_return": float(sim["total_return"][pi]),
                    "trades": int(sim["trades"][pi]),
                    "skipped_executions": int(sim["skipped"][pi]),
                    "fees_paid": float(sim["fees"][pi]),
                    "slippage_impact": float(sim["slippage"][pi]),
                    "final_holding": market.tickers[h] if h >= 0 else "CASH",
                    "start": market.start,
                    "end": market.end,
                    "shard": args.shard_id,
                }
            )

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["final_equity", "gap_period", "signal_period", "momentum_period"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    result.to_csv(args.output, index=False, float_format="%.12f")

    snapshot_meta_path = args.data_root / "SNAPSHOT_META.json"
    snapshot_meta = (
        json.loads(snapshot_meta_path.read_text(encoding="utf-8"))
        if snapshot_meta_path.exists()
        else {}
    )
    meta = {
        "schema_version": 2,
        "shard": args.shard_id,
        "shards": args.shards,
        "gap_values": gap_values,
        "signal_min": args.signal_min,
        "signal_max": args.signal_max,
        "momentum_min": args.momentum_min,
        "momentum_max": args.momentum_max,
        "vol_period": args.vol_period,
        "rows": int(len(result)),
        "start": market.start,
        "end": market.end,
        "initial_cash": args.initial_cash,
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
        "odd_lot_extra_bps": args.odd_lot_extra_bps,
        "portfolio_policy": "pine_v17_hold_same_target_no_residual_reinvestment",
        "momentum_dtype": "float64",
        "csv_sha256": sha256_file(args.output),
        "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "optimizer_sha": os.environ.get("GITHUB_SHA", ""),
        "github_repository": os.environ.get("GITHUB_REPOSITORY", ""),
        "snapshot_upstream_sha": snapshot_meta.get("upstream_sha", ""),
        "snapshot_universe_sha256": snapshot_meta.get("universe_sha256", ""),
        "snapshot_requested_end": snapshot_meta.get("requested_end", ""),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if len(result):
        best = result.iloc[0]
        print(
            f"shard={args.shard_id} tested={len(result)} best="
            f"G{int(best.gap_period)}/S{int(best.signal_period)}/M{int(best.momentum_period)} "
            f"equity={best.final_equity:.2f} return={best.total_return:.2%}"
        )


if __name__ == "__main__":
    main()
