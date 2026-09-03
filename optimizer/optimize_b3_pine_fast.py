#!/usr/bin/env python3
"""Acelerador exato do otimizador B3 Pine com CLI endurecida."""
from __future__ import annotations

import sys

import numpy as np

import config
import optimize_b3_pine as opt


def fast_precompute_shard(
    market: opt.MarketData,
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
    m_values = np.asarray(momentum_values, dtype=np.int64)[:, None]

    for ti, ticker in enumerate(market.tickers):
        df = market.frames[ticker]
        opens = df["open"].to_numpy(dtype=np.float64)
        closes = df["close"].to_numpy(dtype=np.float64)
        didx = market.decision_index[ti].astype(np.int64, copy=False)
        decision_ok = didx >= 0

        vol = opt.sample_vol_positive(closes, vol_period)
        if np.any(decision_ok):
            vol_valid[decision_ok, ti] = vol[didx[decision_ok]]

        if M and W and len(closes):
            prev_idx = didx[None, :] - m_values
            valid = decision_ok[None, :] & (prev_idx >= 0)
            safe_cur = np.clip(didx, 0, len(closes) - 1)
            safe_prev = np.clip(prev_idx, 0, len(closes) - 1)
            cur = closes[safe_cur][None, :]
            prev = closes[safe_prev]
            with np.errstate(divide="ignore", invalid="ignore"):
                scores = cur / prev - 1.0
            scores = np.where(valid & (prev > 0.0) & (cur > 0.0), scores, np.nan)
            momentum[:, :, ti] = scores

        gaps = np.zeros(len(closes), dtype=np.float64)
        if len(closes) > 1:
            gaps[1:] = opens[1:] - closes[:-1]
        positive = np.maximum(gaps, 0.0)
        negative = np.maximum(-gaps, 0.0)

        for g in gap_values:
            pos_sum = opt.rolling_sum(positive, g)
            neg_sum = opt.rolling_sum(negative, g)
            ratio = np.full(len(closes), np.nan, dtype=np.float64)
            valid_ratio = np.isfinite(pos_sum) & np.isfinite(neg_sum)
            zero_neg = valid_ratio & (neg_sum == 0.0)
            ratio[zero_neg] = 1.0
            nz = valid_ratio & (neg_sum != 0.0)
            ratio[nz] = 100.0 * pos_sum[nz] / neg_sum[nz]
            valid_from = g - 1
            segment = ratio[valid_from:]
            csum = np.concatenate(([0.0], np.cumsum(segment, dtype=np.float64)))
            for s in signal_values:
                line = np.full(len(closes), np.nan, dtype=np.float64)
                if len(segment) >= s:
                    means = (csum[s:] - csum[:-s]) / s
                    line[valid_from + s - 1 :] = means
                state = opt.persistent_direction_state(line)
                pi = p_index[(g, s)]
                if np.any(decision_ok):
                    gap_state[pi, decision_ok, ti] = state[didx[decision_ok]]

    return pairs, gap_state, momentum, vol_valid


def _hardened_parse_args():
    args = _ORIGINAL_PARSE_ARGS()
    provided = set(sys.argv[1:])
    if "--fee-bps" not in provided:
        args.fee_bps = config.DEFAULT_FEE_BPS
    if "--slippage-bps" not in provided:
        args.slippage_bps = config.DEFAULT_SLIPPAGE_BPS
    if "--odd-lot-extra-bps" not in provided:
        args.odd_lot_extra_bps = config.DEFAULT_ODD_LOT_EXTRA_BPS
    if "--initial-cash" not in provided:
        args.initial_cash = config.DEFAULT_INITIAL_CASH
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
    return args


_ORIGINAL_PARSE_ARGS = opt.parse_args


def main() -> None:
    opt.parse_args = _hardened_parse_args
    opt.precompute_shard = fast_precompute_shard
    opt.main()


if __name__ == "__main__":
    main()
