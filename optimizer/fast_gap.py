from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import optimize_b3_pine as opt


def _ticker_arrays(market, ti: int):
    if hasattr(market, "opens") and hasattr(market, "lengths"):
        n = int(market.lengths[ti])
        return market.opens[ti, :n], market.closes[ti, :n]
    ticker = market.tickers[ti]
    frame = market.frames[ticker]
    return (
        frame["open"].to_numpy(dtype=np.float64),
        frame["close"].to_numpy(dtype=np.float64),
    )


def gap_state_for_ticker(market, ti: int, gap_values: list[int], signal_values: list[int]):
    opens, closes = _ticker_arrays(market, ti)
    didx = market.decision_index[ti].astype(np.int64, copy=False)
    decision_ok = didx >= 0
    W = len(didx)
    P = len(gap_values) * len(signal_values)
    out = np.zeros((P, W), dtype=np.bool_)

    gaps = np.zeros(len(closes), dtype=np.float64)
    if len(closes) > 1:
        gaps[1:] = opens[1:] - closes[:-1]
    positive = np.maximum(gaps, 0.0)
    negative = np.maximum(-gaps, 0.0)

    pi = 0
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
            if np.any(decision_ok):
                out[pi, decision_ok] = state[didx[decision_ok]]
            pi += 1
    return ti, out


def compute_gap_state(market, gap_values: list[int], signal_values: list[int]) -> np.ndarray:
    P = len(gap_values) * len(signal_values)
    W = len(market.execution_dates)
    N = len(market.tickers)
    result = np.zeros((P, W, N), dtype=np.bool_)
    if N == 0:
        return result

    default_workers = min(4, os.cpu_count() or 1, N)
    workers = max(1, min(N, int(os.environ.get("B3_PRECOMPUTE_WORKERS", default_workers))))
    if workers == 1:
        for ti in range(N):
            _, state = gap_state_for_ticker(market, ti, gap_values, signal_values)
            result[:, :, ti] = state
        return result

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="b3-gap") as pool:
        futures = [
            pool.submit(gap_state_for_ticker, market, ti, gap_values, signal_values)
            for ti in range(N)
        ]
        for future in futures:
            ti, state = future.result()
            result[:, :, ti] = state
    return result
