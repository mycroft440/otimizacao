#!/usr/bin/env python3
"""Acelerador exato do otimizador B3 Pine com cache, threads e batching."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import config
import fast_batch
import fast_gap
import fast_shared
import optimize_b3_pine as opt


def fast_precompute_shard(
    market,
    gap_values: list[int],
    signal_values: list[int],
    momentum_values: list[int],
    vol_period: int,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
    pairs = [(g, s) for g in gap_values for s in signal_values]
    gap_state = fast_gap.compute_gap_state(market, gap_values, signal_values)
    if isinstance(market, fast_shared.FastMarketData):
        if int(market.vol_period) != int(vol_period):
            raise RuntimeError("Fast cache VOL_PERIOD incompatível.")
        if not np.array_equal(market.momentum_values, np.asarray(momentum_values, dtype=np.int32)):
            raise RuntimeError("Fast cache Momentum incompatível.")
        momentum = market.momentum
        vol_valid = market.vol_valid
    else:
        momentum, vol_valid = fast_shared.compute_shared_features(
            market, momentum_values, vol_period
        )
    return pairs, gap_state, momentum, vol_valid


def _hardened_parse_args():
    args = opt.parse_args()
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


def _snapshot_meta(args):
    path = args.data_root / "SNAPSHOT_META.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_meta(args, market, gap_values, rows: int, cache_used: bool, batch_size: int) -> None:
    snapshot_meta = _snapshot_meta(args)
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
        "rows": int(rows),
        "start": market.start,
        "end": market.end,
        "initial_cash": args.initial_cash,
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
        "odd_lot_extra_bps": args.odd_lot_extra_bps,
        "portfolio_policy": "pine_v17_hold_same_target_no_residual_reinvestment",
        "momentum_dtype": "float64",
        "performance_engine": "exact_batch_v1",
        "fast_cache_used": bool(cache_used),
        "momentum_batch_size": int(batch_size),
        "precompute_workers": int(os.environ.get("B3_PRECOMPUTE_WORKERS", min(4, os.cpu_count() or 1))),
        "csv_sha256": opt.sha256_file(args.output),
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


def _write_empty_shard(args) -> None:
    columns = [
        "gap_period", "signal_period", "momentum_period", "vol_period",
        "final_equity", "total_return", "trades", "skipped_executions",
        "fees_paid", "slippage_impact", "final_holding", "start", "end", "shard",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(args.output, index=False)
    snapshot = _snapshot_meta(args)
    class EmptyMarket:
        start = args.start
        end = str(snapshot.get("actual_master_end") or snapshot.get("requested_end") or args.end or "")
    _write_meta(args, EmptyMarket(), [], 0, False, 0)


def _result_frame(args, market, pairs, momentum_values, results):
    M, P = len(momentum_values), len(pairs)
    gap_col = np.tile(np.asarray([g for g, _s in pairs], dtype=np.int32), M)
    signal_col = np.tile(np.asarray([s for _g, s in pairs], dtype=np.int32), M)
    momentum_col = np.repeat(np.asarray(momentum_values, dtype=np.int32), P)
    holding_idx = results["holding"].reshape(-1)
    holding = np.full(M * P, "CASH", dtype=object)
    invested = holding_idx >= 0
    if np.any(invested):
        tickers = np.asarray(market.tickers, dtype=object)
        holding[invested] = tickers[holding_idx[invested].astype(np.int64)]
    frame = pd.DataFrame({
        "gap_period": gap_col,
        "signal_period": signal_col,
        "momentum_period": momentum_col,
        "vol_period": np.full(M * P, args.vol_period, dtype=np.int32),
        "final_equity": results["final_equity"].reshape(-1),
        "total_return": results["total_return"].reshape(-1),
        "trades": results["trades"].reshape(-1),
        "skipped_executions": results["skipped"].reshape(-1),
        "fees_paid": results["fees"].reshape(-1),
        "slippage_impact": results["slippage"].reshape(-1),
        "final_holding": holding,
        "start": market.start,
        "end": market.end,
        "shard": args.shard_id,
    })
    return frame.sort_values(
        ["final_equity", "gap_period", "signal_period", "momentum_period"],
        ascending=[False, True, True, True], kind="stable",
    )


def main() -> None:
    args = _hardened_parse_args()
    gap_values = list(range(args.gap_min, args.gap_max + 1))[args.shard_id :: args.shards]
    if not gap_values:
        _write_empty_shard(args)
        print(f"shard={args.shard_id} tested=0 empty_gap_assignment=true")
        return

    signal_values = list(range(args.signal_min, args.signal_max + 1))
    momentum_values = list(range(args.momentum_min, args.momentum_max + 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    market = fast_shared.load_fast_cache(
        args.data_root, args.start, args.end, momentum_values, args.vol_period
    )
    cache_used = market is not None
    if market is None:
        market = opt.load_market(args.data_root, args.start, args.end)

    pairs, gap_state, momentum, vol_valid = fast_precompute_shard(
        market, gap_values, signal_values, momentum_values, args.vol_period
    )
    M, P = len(momentum_values), len(pairs)
    batch_size = max(1, int(os.environ.get("B3_MOMENTUM_BATCH", "32")))
    outputs = {
        "final_equity": np.empty((M, P), dtype=np.float64),
        "total_return": np.empty((M, P), dtype=np.float64),
        "cash": np.empty((M, P), dtype=np.float64),
        "shares": np.empty((M, P), dtype=np.int64),
        "holding": np.empty((M, P), dtype=np.int16),
        "trades": np.empty((M, P), dtype=np.int32),
        "skipped": np.empty((M, P), dtype=np.int32),
        "fees": np.empty((M, P), dtype=np.float64),
        "slippage": np.empty((M, P), dtype=np.float64),
    }

    for lo in range(0, M, batch_size):
        hi = min(M, lo + batch_size)
        targets = np.stack([
            opt.first_ranked_targets(gap_state, momentum[mi], vol_valid)
            for mi in range(lo, hi)
        ], axis=0)
        simulated = fast_batch.simulate_momentum_batch(
            targets, gap_state, momentum[lo:hi], vol_valid, market,
            initial_cash=args.initial_cash,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            odd_lot_extra_bps=args.odd_lot_extra_bps,
        )
        for key in outputs:
            outputs[key][lo:hi] = simulated[key]

    result = _result_frame(args, market, pairs, momentum_values, outputs)
    result.to_csv(args.output, index=False, float_format="%.12f")
    _write_meta(args, market, gap_values, len(result), cache_used, batch_size)
    best = result.iloc[0]
    print(
        f"shard={args.shard_id} tested={len(result)} cache={cache_used} batch={batch_size} best="
        f"G{int(best.gap_period)}/S{int(best.signal_period)}/M{int(best.momentum_period)} "
        f"equity={best.final_equity:.2f} return={best.total_return:.2%}"
    )


if __name__ == "__main__":
    main()
