#!/usr/bin/env python3
"""Busca exaustiva B3 incluindo VOL_PERIOD sem materializar dezenas de milhões de linhas.

Cada shard cobre uma partição determinística de GAP_PERIOD e testa integralmente
Signal x Momentum x Vol. O cálculo completo acontece em memória, mas apenas os
melhores resultados locais são persistidos. Isso preserva a vencedora global e o
Top-K global enquanto evita gerar gigabytes de CSV para a grade 4D.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import config
import fast_batch
import fast_gap
import fast_shared
import optimize_b3_pine as opt
import optimize_b3_pine_fast as fast


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
    p.add_argument("--vol-min", type=int, default=config.DEFAULT_VOL_MIN)
    p.add_argument("--vol-max", type=int, default=config.DEFAULT_VOL_MAX)
    p.add_argument("--initial-cash", type=float, default=config.DEFAULT_INITIAL_CASH)
    p.add_argument("--fee-bps", type=float, default=config.DEFAULT_FEE_BPS)
    p.add_argument("--slippage-bps", type=float, default=config.DEFAULT_SLIPPAGE_BPS)
    p.add_argument("--odd-lot-extra-bps", type=float, default=config.DEFAULT_ODD_LOT_EXTRA_BPS)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--shards", type=int, default=config.DEFAULT_SHARDS)
    p.add_argument("--top-k", type=int, default=100)
    return p.parse_args()


def _validate(args: argparse.Namespace) -> None:
    if args.vol_max < args.vol_min:
        raise ValueError("faixa de VOL_PERIOD invertida")
    if args.top_k <= 0:
        raise ValueError("top-k precisa ser > 0")
    for vol in (args.vol_min, args.vol_max):
        config.validate_run_config(
            start=args.start,
            end=args.end,
            gap_min=args.gap_min,
            gap_max=args.gap_max,
            signal_min=args.signal_min,
            signal_max=args.signal_max,
            momentum_min=args.momentum_min,
            momentum_max=args.momentum_max,
            vol_period=vol,
            initial_cash=args.initial_cash,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            odd_lot_extra_bps=args.odd_lot_extra_bps,
            shard_id=args.shard_id,
            shards=args.shards,
        )


def _ticker_closes(market, ti: int) -> np.ndarray:
    if isinstance(market, fast_shared.FastMarketData):
        n = int(market.lengths[ti])
        return market.closes[ti, :n]
    ticker = market.tickers[ti]
    return market.frames[ticker]["close"].to_numpy(dtype=np.float64)


def compute_vol_valid(market, period: int) -> np.ndarray:
    """Calcula somente o gate de volatilidade para um período, sem refazer Momentum."""
    W = len(market.execution_dates)
    N = len(market.tickers)
    out = np.zeros((W, N), dtype=np.bool_)
    for ti in range(N):
        closes = _ticker_closes(market, ti)
        didx = market.decision_index[ti].astype(np.int64, copy=False)
        ok = didx >= 0
        if not np.any(ok):
            continue
        vol = opt.sample_vol_positive(closes, period)
        out[ok, ti] = vol[didx[ok]]
    return out


def _simulate_range(
    lo: int,
    hi: int,
    *,
    gap_state: np.ndarray,
    momentum: np.ndarray,
    vol_valid: np.ndarray,
    market,
    args,
):
    targets = np.stack(
        [opt.first_ranked_targets(gap_state, momentum[mi], vol_valid) for mi in range(lo, hi)],
        axis=0,
    )
    simulated = fast_batch.simulate_momentum_batch(
        targets,
        gap_state,
        momentum[lo:hi],
        vol_valid,
        market,
        initial_cash=args.initial_cash,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        odd_lot_extra_bps=args.odd_lot_extra_bps,
    )
    return lo, hi, simulated


def simulate_all_momentum(
    *,
    gap_state: np.ndarray,
    momentum: np.ndarray,
    vol_valid: np.ndarray,
    market,
    args,
) -> tuple[dict[str, np.ndarray], int, int]:
    M, P = momentum.shape[0], gap_state.shape[0]
    batch_size = max(1, int(os.environ.get("B3_MOMENTUM_BATCH", "32")))
    ranges = [(lo, min(M, lo + batch_size)) for lo in range(0, M, batch_size)]
    default_workers = min(4, os.cpu_count() or 1, max(1, len(ranges)))
    workers = max(1, min(len(ranges) or 1, int(os.environ.get("B3_BATCH_WORKERS", default_workers))))

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

    def store(lo: int, hi: int, simulated) -> None:
        for key in outputs:
            outputs[key][lo:hi] = simulated[key]

    if workers == 1 or len(ranges) <= 1:
        for lo, hi in ranges:
            rlo, rhi, simulated = _simulate_range(
                lo,
                hi,
                gap_state=gap_state,
                momentum=momentum,
                vol_valid=vol_valid,
                market=market,
                args=args,
            )
            store(rlo, rhi, simulated)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="b3-volgrid") as pool:
            futures = [
                pool.submit(
                    _simulate_range,
                    lo,
                    hi,
                    gap_state=gap_state,
                    momentum=momentum,
                    vol_valid=vol_valid,
                    market=market,
                    args=args,
                )
                for lo, hi in ranges
            ]
            for future in as_completed(futures):
                rlo, rhi, simulated = future.result()
                store(rlo, rhi, simulated)

    return outputs, batch_size, workers


def _validate_frame(frame: pd.DataFrame, *, expected_rows: int, vol_period: int) -> None:
    if len(frame) != expected_rows:
        raise RuntimeError(f"cardinalidade local incorreta: {len(frame)} != {expected_rows}")
    if set(frame["vol_period"].astype(int)) != {int(vol_period)}:
        raise RuntimeError("VOL_PERIOD divergente no frame local")
    finite_cols = ["final_equity", "total_return", "fees_paid", "slippage_impact"]
    for col in finite_cols:
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"resultado não finito em {col}")
    if (frame["final_equity"] < 0).any():
        raise RuntimeError("final_equity negativo")
    for col in ("trades", "skipped_executions", "fees_paid", "slippage_impact"):
        if (frame[col] < 0).any():
            raise RuntimeError(f"valor negativo em {col}")


def main() -> None:
    args = parse_args()
    try:
        _validate(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    gap_values = list(range(args.gap_min, args.gap_max + 1))[args.shard_id :: args.shards]
    signal_values = list(range(args.signal_min, args.signal_max + 1))
    momentum_values = list(range(args.momentum_min, args.momentum_max + 1))
    vol_values = list(range(args.vol_min, args.vol_max + 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not gap_values:
        raise SystemExit("shard sem GAP_PERIOD; reduza SHARDS ou amplie a faixa")

    # O preflight hardened já materializa o cache com VOL=21. Momentum e preços não
    # dependem do VOL_PERIOD, então reutilizamos essas matrizes e recalculamos somente
    # o pequeno gate booleano de volatilidade para 1..60.
    cache_period = config.DEFAULT_VOL_PERIOD
    market = fast_shared.load_fast_cache(
        args.data_root, args.start, args.end, momentum_values, cache_period
    )
    cache_used = market is not None
    if market is None:
        market = opt.load_market(args.data_root, args.start, args.end)
        momentum, _ = fast_shared.compute_shared_features(market, momentum_values, cache_period)
    else:
        momentum = market.momentum

    pairs = [(g, s) for g in gap_values for s in signal_values]
    gap_state = fast_gap.compute_gap_state(market, gap_values, signal_values)
    expected_per_vol = len(pairs) * len(momentum_values)
    expected_total = expected_per_vol * len(vol_values)

    local_best_parts: list[pd.DataFrame] = []
    tested_rows = 0
    batch_size = 0
    batch_workers = 0

    frame_args = SimpleNamespace(**vars(args))
    for vol in vol_values:
        vol_valid = compute_vol_valid(market, vol)
        simulated, batch_size, batch_workers = simulate_all_momentum(
            gap_state=gap_state,
            momentum=momentum,
            vol_valid=vol_valid,
            market=market,
            args=args,
        )
        frame_args.vol_period = vol
        frame = fast._result_frame(frame_args, market, pairs, momentum_values, simulated)
        _validate_frame(frame, expected_rows=expected_per_vol, vol_period=vol)
        tested_rows += len(frame)
        local_best_parts.append(frame.head(min(args.top_k, len(frame))).copy())
        best = frame.iloc[0]
        print(
            f"shard={args.shard_id} vol={vol} tested={len(frame)} best="
            f"G{int(best.gap_period)}/S{int(best.signal_period)}/M{int(best.momentum_period)} "
            f"equity={best.final_equity:.2f}",
            flush=True,
        )

    if tested_rows != expected_total:
        raise RuntimeError(f"total local incorreto: {tested_rows} != {expected_total}")

    leaders = pd.concat(local_best_parts, ignore_index=True)
    leaders = leaders.sort_values(
        ["final_equity", "gap_period", "signal_period", "momentum_period", "vol_period"],
        ascending=[False, True, True, True, True],
        kind="stable",
    ).head(args.top_k).reset_index(drop=True)
    leaders.to_csv(args.output, index=False, float_format="%.12f")

    snapshot_meta_path = args.data_root / "SNAPSHOT_META.json"
    snapshot_meta = (
        json.loads(snapshot_meta_path.read_text(encoding="utf-8"))
        if snapshot_meta_path.exists()
        else {}
    )
    meta = {
        "schema_version": 4,
        "mode": "exhaustive_4d_streaming_topk",
        "shard": args.shard_id,
        "shards": args.shards,
        "gap_values": gap_values,
        "signal_min": args.signal_min,
        "signal_max": args.signal_max,
        "momentum_min": args.momentum_min,
        "momentum_max": args.momentum_max,
        "vol_min": args.vol_min,
        "vol_max": args.vol_max,
        "vol_values": vol_values,
        "tested_rows": int(tested_rows),
        "output_rows": int(len(leaders)),
        "top_k": int(args.top_k),
        "start": market.start,
        "end": market.end,
        "initial_cash": args.initial_cash,
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
        "odd_lot_extra_bps": args.odd_lot_extra_bps,
        "portfolio_policy": "pine_v17_hold_same_target_no_residual_reinvestment",
        "momentum_dtype": "float64",
        "performance_engine": "exact_batch_v2_parallel_vol_grid_streaming",
        "fast_cache_used": bool(cache_used),
        "momentum_batch_size": int(batch_size),
        "momentum_batch_workers": int(batch_workers),
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
    best = leaders.iloc[0]
    print(
        f"shard={args.shard_id} exhaustive_tested={tested_rows} retained={len(leaders)} "
        f"vol={args.vol_min}..{args.vol_max} best="
        f"G{int(best.gap_period)}/S{int(best.signal_period)}/M{int(best.momentum_period)}/V{int(best.vol_period)} "
        f"equity={best.final_equity:.2f} return={best.total_return:.2%}"
    )


if __name__ == "__main__":
    main()
