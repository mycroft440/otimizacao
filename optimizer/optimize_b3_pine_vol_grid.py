#!/usr/bin/env python3
"""Busca exaustiva B3 4D com acelerações exatas e sem mudar a estratégia.

A grade lógica continua sendo GAP x Signal x Momentum x Vol. O motor evita
trabalho redundante somente quando consegue provar equivalência exata:
- Momentum é calculado uma vez e seu ranking estável é reutilizado entre VOLs;
- todos os gates VOL são derivados dos mesmos prefix sums, preservando float64;
- gates VOL byte-a-byte idênticos compartilham uma única simulação;
- apenas o Top-K local é materializado em pandas, mantendo a ordenação canônica.

Nenhuma heurística elimina combinações. ``tested_rows`` continua registrando toda
a cardinalidade lógica da grade, mesmo quando uma simulação física é reutilizada.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import config
import fast_batch
import fast_gap
import fast_shared
import optimize_b3_pine as opt


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
    """Compatibilidade: gate de um único VOL com a semântica canônica."""
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


def compute_all_vol_valid(market, periods: list[int]) -> np.ndarray:
    """Calcula todos os gates VOL reutilizando os mesmos prefix sums float64.

    A expressão é a mesma de ``sample_vol_positive``; a única diferença é que
    materializamos apenas os pontos de decisão usados pelo backtest.
    """
    V = len(periods)
    W = len(market.execution_dates)
    N = len(market.tickers)
    out = np.zeros((V, W, N), dtype=np.bool_)
    if V == 0 or W == 0 or N == 0:
        return out

    for ti in range(N):
        closes = _ticker_closes(market, ti)
        didx = market.decision_index[ti].astype(np.int64, copy=False)
        if len(closes) == 0 or not np.any(didx >= 0):
            continue

        returns = np.zeros(len(closes), dtype=np.float64)
        if len(closes) > 1:
            prev = closes[:-1]
            returns[1:] = np.where(prev > 0.0, closes[1:] / prev - 1.0, 0.0)
        c1 = np.concatenate(([0.0], np.cumsum(returns, dtype=np.float64)))
        squared = returns * returns
        c2 = np.concatenate(([0.0], np.cumsum(squared, dtype=np.float64)))

        for vi, period in enumerate(periods):
            if period <= 1:
                continue
            valid = (didx >= 0) & (didx >= period - 1)
            if not np.any(valid):
                continue
            idx = didx[valid]
            s1 = c1[idx + 1] - c1[idx + 1 - period]
            s2 = c2[idx + 1] - c2[idx + 1 - period]
            var = np.maximum(
                0.0,
                (s2 - s1 * s1 / period) / (period - 1),
            )
            out[vi, valid, ti] = np.isfinite(var) & (var > 0.0)
    return out


def group_identical_vol_gates(
    vol_values: list[int],
    gates: np.ndarray,
) -> list[tuple[list[int], np.ndarray]]:
    """Agrupa somente gates comprovadamente byte-a-byte idênticos."""
    if gates.shape[0] != len(vol_values):
        raise ValueError("quantidade de gates VOL incompatível")
    groups: list[tuple[list[int], np.ndarray]] = []
    by_bytes: dict[bytes, int] = {}
    for vi, vol in enumerate(vol_values):
        gate = np.ascontiguousarray(gates[vi], dtype=np.bool_)
        key = gate.tobytes(order="C")
        gi = by_bytes.get(key)
        if gi is None:
            by_bytes[key] = len(groups)
            groups.append(([int(vol)], gate))
            continue
        prior_vols, prior_gate = groups[gi]
        if not np.array_equal(prior_gate, gate):
            raise RuntimeError("colisão impossível na chave exata do gate VOL")
        prior_vols.append(int(vol))
    return groups


def build_momentum_rank_cache(momentum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ranking estável de Momentum independente de GAP e VOL."""
    positive = np.isfinite(momentum) & (momentum > 0.0)
    scores = np.where(positive, momentum, -np.inf)
    order = np.argsort(-scores, axis=2, kind="stable")
    max_index = int(order.max()) if order.size else 0
    if max_index > np.iinfo(np.int16).max:
        raise RuntimeError("universo excede capacidade do cache int16")
    return order.astype(np.int16, copy=False), positive


def first_ranked_targets_from_cache(
    gap_state: np.ndarray,
    momentum_positive: np.ndarray,
    vol_valid: np.ndarray,
    order: np.ndarray,
) -> np.ndarray:
    """Equivalente a opt.first_ranked_targets, sem repetir argsort por VOL."""
    P, W, N = gap_state.shape
    if momentum_positive.shape != (W, N) or vol_valid.shape != (W, N):
        raise ValueError("shape de elegibilidade incompatível")
    if order.shape != (W, N):
        raise ValueError("shape do ranking Momentum incompatível")

    eligible = momentum_positive & vol_valid
    targets = np.full((P, W), -1, dtype=np.int16)
    unresolved = np.ones((P, W), dtype=np.bool_)
    widx = np.arange(W)

    for rank in range(N):
        asset = order[:, rank].astype(np.intp, copy=False)
        base_ok = eligible[widx, asset]
        gap_ok = gap_state[:, widx, asset]
        take = unresolved & gap_ok & base_ok[None, :]
        if np.any(take):
            targets = np.where(take, asset[None, :], targets)
            unresolved &= ~take
        if not np.any(unresolved):
            break
    return targets


def _simulate_range(
    lo: int,
    hi: int,
    *,
    gap_state: np.ndarray,
    momentum: np.ndarray,
    momentum_order: np.ndarray,
    momentum_positive: np.ndarray,
    vol_valid: np.ndarray,
    market,
    args,
):
    targets = np.stack(
        [
            first_ranked_targets_from_cache(
                gap_state,
                momentum_positive[mi],
                vol_valid,
                momentum_order[mi],
            )
            for mi in range(lo, hi)
        ],
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
    momentum_order: np.ndarray,
    momentum_positive: np.ndarray,
    vol_valid: np.ndarray,
    market,
    args,
) -> tuple[dict[str, np.ndarray], int, int]:
    M, P = momentum.shape[0], gap_state.shape[0]
    batch_size = max(1, int(os.environ.get("B3_MOMENTUM_BATCH", "32")))
    ranges = [(lo, min(M, lo + batch_size)) for lo in range(0, M, batch_size)]
    default_workers = min(4, os.cpu_count() or 1, max(1, len(ranges)))
    workers = max(
        1,
        min(
            len(ranges) or 1,
            int(os.environ.get("B3_BATCH_WORKERS", default_workers)),
        ),
    )

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
                momentum_order=momentum_order,
                momentum_positive=momentum_positive,
                vol_valid=vol_valid,
                market=market,
                args=args,
            )
            store(rlo, rhi, simulated)
    else:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="b3-volgrid",
        ) as pool:
            futures = [
                pool.submit(
                    _simulate_range,
                    lo,
                    hi,
                    gap_state=gap_state,
                    momentum=momentum,
                    momentum_order=momentum_order,
                    momentum_positive=momentum_positive,
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


def _validate_simulated(
    simulated: dict[str, np.ndarray],
    *,
    expected_shape: tuple[int, int],
    ticker_count: int,
) -> None:
    for key in (
        "final_equity",
        "total_return",
        "cash",
        "shares",
        "holding",
        "trades",
        "skipped",
        "fees",
        "slippage",
    ):
        if simulated[key].shape != expected_shape:
            raise RuntimeError(f"shape incorreto em {key}: {simulated[key].shape}")

    for key in ("final_equity", "total_return", "cash", "fees", "slippage"):
        if not np.all(np.isfinite(simulated[key])):
            raise RuntimeError(f"resultado não finito em {key}")

    if np.any(simulated["final_equity"] < 0.0):
        raise RuntimeError("final_equity negativo")
    if np.any(simulated["cash"] < -1e-7):
        raise RuntimeError("cash negativo")
    for key in ("shares", "trades", "skipped"):
        if np.any(simulated[key] < 0):
            raise RuntimeError(f"valor negativo em {key}")
    for key in ("fees", "slippage"):
        if np.any(simulated[key] < 0.0):
            raise RuntimeError(f"valor negativo em {key}")
    if np.any(simulated["holding"] < -1) or np.any(simulated["holding"] >= ticker_count):
        raise RuntimeError("holding fora do universo")


def exact_top_k_indices(
    final_equity: np.ndarray,
    pairs: list[tuple[int, int]],
    momentum_values: list[int],
    top_k: int,
) -> np.ndarray:
    """Top-K exato com o mesmo desempate do sort canônico, sem DataFrame completo."""
    equity = np.asarray(final_equity, dtype=np.float64).reshape(-1)
    P = len(pairs)
    M = len(momentum_values)
    if len(equity) != M * P:
        raise ValueError("cardinalidade do equity incompatível")
    k = min(int(top_k), len(equity))
    if k <= 0:
        return np.empty(0, dtype=np.int64)

    if k == len(equity):
        candidates = np.arange(len(equity), dtype=np.int64)
    else:
        threshold = np.partition(equity, len(equity) - k)[len(equity) - k]
        candidates = np.flatnonzero(equity >= threshold)

    pair_idx = candidates % P
    momentum_idx = candidates // P
    gap_values = np.asarray([g for g, _s in pairs], dtype=np.int32)[pair_idx]
    signal_values = np.asarray([s for _g, s in pairs], dtype=np.int32)[pair_idx]
    momentum_col = np.asarray(momentum_values, dtype=np.int32)[momentum_idx]

    order = np.lexsort(
        (
            momentum_col,
            signal_values,
            gap_values,
            -equity[candidates],
        )
    )
    return candidates[order[:k]]


def _top_k_frame(
    args,
    market,
    pairs: list[tuple[int, int]],
    momentum_values: list[int],
    simulated: dict[str, np.ndarray],
    *,
    vol_period: int,
) -> pd.DataFrame:
    M, P = len(momentum_values), len(pairs)
    _validate_simulated(
        simulated,
        expected_shape=(M, P),
        ticker_count=len(market.tickers),
    )
    selected = exact_top_k_indices(
        simulated["final_equity"],
        pairs,
        momentum_values,
        args.top_k,
    )
    pair_idx = selected % P
    momentum_idx = selected // P
    gap_col = np.asarray([g for g, _s in pairs], dtype=np.int32)[pair_idx]
    signal_col = np.asarray([s for _g, s in pairs], dtype=np.int32)[pair_idx]
    momentum_col = np.asarray(momentum_values, dtype=np.int32)[momentum_idx]

    holding_idx = simulated["holding"].reshape(-1)[selected]
    holding = np.full(len(selected), "CASH", dtype=object)
    invested = holding_idx >= 0
    if np.any(invested):
        tickers = np.asarray(market.tickers, dtype=object)
        holding[invested] = tickers[holding_idx[invested].astype(np.int64)]

    def picked(key: str) -> np.ndarray:
        return simulated[key].reshape(-1)[selected]

    return pd.DataFrame(
        {
            "gap_period": gap_col,
            "signal_period": signal_col,
            "momentum_period": momentum_col,
            "vol_period": np.full(len(selected), vol_period, dtype=np.int32),
            "final_equity": picked("final_equity"),
            "total_return": picked("total_return"),
            "trades": picked("trades"),
            "skipped_executions": picked("skipped"),
            "fees_paid": picked("fees"),
            "slippage_impact": picked("slippage"),
            "final_holding": holding,
            "start": market.start,
            "end": market.end,
            "shard": args.shard_id,
        }
    )


def _vol_label(vols: list[int]) -> str:
    if not vols:
        return ""
    if len(vols) == 1:
        return str(vols[0])
    contiguous = vols == list(range(vols[0], vols[-1] + 1))
    return f"{vols[0]}..{vols[-1]}" if contiguous else ",".join(map(str, vols))


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

    cache_period = config.DEFAULT_VOL_PERIOD
    market = fast_shared.load_fast_cache(
        args.data_root,
        args.start,
        args.end,
        momentum_values,
        cache_period,
    )
    cache_used = market is not None
    if market is None:
        market = opt.load_market(args.data_root, args.start, args.end)
        momentum, _ = fast_shared.compute_shared_features(
            market,
            momentum_values,
            cache_period,
        )
    else:
        momentum = market.momentum

    pairs = [(g, s) for g in gap_values for s in signal_values]
    gap_state = fast_gap.compute_gap_state(market, gap_values, signal_values)

    # Ranking de Momentum é invariável para GAP e VOL.
    momentum_order, momentum_positive = build_momentum_rank_cache(momentum)

    # Todos os VOLs usam os mesmos retornos/cumsums. Depois agrupamos apenas
    # matrizes booleanas comprovadamente idênticas.
    all_vol_gates = compute_all_vol_valid(market, vol_values)
    vol_groups = group_identical_vol_gates(vol_values, all_vol_gates)

    expected_per_vol = len(pairs) * len(momentum_values)
    expected_total = expected_per_vol * len(vol_values)
    physical_total = expected_per_vol * len(vol_groups)

    local_best_parts: list[pd.DataFrame] = []
    batch_size = 0
    batch_workers = 0

    for vols, vol_valid in vol_groups:
        representative = vols[0]
        simulated, batch_size, batch_workers = simulate_all_momentum(
            gap_state=gap_state,
            momentum=momentum,
            momentum_order=momentum_order,
            momentum_positive=momentum_positive,
            vol_valid=vol_valid,
            market=market,
            args=args,
        )
        top = _top_k_frame(
            args,
            market,
            pairs,
            momentum_values,
            simulated,
            vol_period=representative,
        )
        if len(top) != min(args.top_k, expected_per_vol):
            raise RuntimeError("Top-K local com cardinalidade incorreta")

        # Um gate idêntico implica alvos e carteira idênticos. Replicamos apenas
        # as linhas Top-K para preservar a grade lógica e o desempate por VOL.
        for vol in vols:
            part = top.copy()
            part["vol_period"] = int(vol)
            local_best_parts.append(part)

        best = top.iloc[0]
        print(
            f"shard={args.shard_id} vols={_vol_label(vols)} "
            f"logical_per_vol={expected_per_vol} physical_once={expected_per_vol} best="
            f"G{int(best.gap_period)}/S{int(best.signal_period)}/"
            f"M{int(best.momentum_period)} equity={best.final_equity:.2f}",
            flush=True,
        )

    leaders = pd.concat(local_best_parts, ignore_index=True)
    leaders = leaders.sort_values(
        [
            "final_equity",
            "gap_period",
            "signal_period",
            "momentum_period",
            "vol_period",
        ],
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
    group_meta = [
        {
            "representative": int(vols[0]),
            "vol_values": [int(v) for v in vols],
        }
        for vols, _gate in vol_groups
    ]
    reduction = (
        float(len(vol_values)) / float(len(vol_groups))
        if vol_groups
        else 1.0
    )
    meta = {
        "schema_version": 4,
        "mode": "exhaustive_4d_exact_vol_gate_dedup_topk",
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
        "vol_gate_groups": group_meta,
        "unique_vol_gates": int(len(vol_groups)),
        "tested_rows": int(expected_total),
        "physical_simulated_rows": int(physical_total),
        "simulation_reduction_factor": reduction,
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
        "performance_engine": "exact_batch_v3_vol_gate_dedup_rank_cache_numpy_topk",
        "fast_cache_used": bool(cache_used),
        "momentum_batch_size": int(batch_size),
        "momentum_batch_workers": int(batch_workers),
        "momentum_rank_cache_bytes": int(momentum_order.nbytes + momentum_positive.nbytes),
        "vol_gate_cache_bytes": int(all_vol_gates.nbytes),
        "csv_sha256": opt.sha256_file(args.output),
        "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "optimizer_sha": os.environ.get("GITHUB_SHA", ""),
        "github_repository": os.environ.get("GITHUB_REPOSITORY", ""),
        "snapshot_upstream_sha": snapshot_meta.get("upstream_sha", ""),
        "snapshot_universe_sha256": snapshot_meta.get("universe_sha256", ""),
        "snapshot_requested_end": snapshot_meta.get("requested_end", ""),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    best = leaders.iloc[0]
    print(
        f"shard={args.shard_id} exhaustive_tested={expected_total} "
        f"physical_simulated={physical_total} unique_vol_gates={len(vol_groups)} "
        f"reduction={reduction:.2f}x retained={len(leaders)} "
        f"vol={args.vol_min}..{args.vol_max} best="
        f"G{int(best.gap_period)}/S{int(best.signal_period)}/"
        f"M{int(best.momentum_period)}/V{int(best.vol_period)} "
        f"equity={best.final_equity:.2f} return={best.total_return:.2%}"
    )


if __name__ == "__main__":
    main()
