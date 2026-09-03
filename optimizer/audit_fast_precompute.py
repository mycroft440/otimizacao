#!/usr/bin/env python3
"""Gate de equivalencia entre o motor original e todos os caminhos acelerados."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config
import fast_batch
import fast_shared
import optimize_b3_pine as opt
import optimize_b3_pine_fast as fast

RANDOM_SEED = 20260903


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--start", default=config.DEFAULT_START)
    p.add_argument("--end", default=config.DEFAULT_END)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def same(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if np.issubdtype(a.dtype, np.floating):
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def representative_values() -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(RANDOM_SEED)
    gaps = {5, 40, 41, 80}
    signals = {2, 8, 15, 20, 60}
    momentums = {5, 35, 49, 63, 123, 252}
    gaps.update(int(x) for x in rng.integers(5, 81, size=6))
    signals.update(int(x) for x in rng.integers(2, 61, size=6))
    momentums.update(int(x) for x in rng.integers(5, 253, size=8))
    return sorted(gaps), sorted(signals), sorted(momentums)


def _compare_precompute(label, original, candidate):
    checks = {
        "pairs": original[0] == candidate[0],
        "gap_state_bit_exact": same(original[1], candidate[1]),
        "momentum_bit_exact": same(original[2], candidate[2]),
        "vol_valid_bit_exact": same(original[3], candidate[3]),
    }
    if not all(checks.values()):
        raise SystemExit(f"{label} PRECOMPUTE EQUIVALENCE FAIL: {checks}")
    return checks


def main() -> None:
    a = args()
    market = opt.load_market(a.data_root, a.start, a.end)
    gaps, signals, momentums = representative_values()
    original = opt.precompute_shard(market, gaps, signals, momentums, config.DEFAULT_VOL_PERIOD)
    accelerated = fast.fast_precompute_shard(
        market, gaps, signals, momentums, config.DEFAULT_VOL_PERIOD
    )
    direct_checks = _compare_precompute("FAST DIRECT", original, accelerated)

    # O próprio gate materializa o cache antes de o workflow publicar o snapshot.
    fast_shared.build_fast_cache(
        a.data_root,
        a.start,
        a.end,
        config.DEFAULT_MOMENTUM_MIN,
        config.DEFAULT_MOMENTUM_MAX,
        config.DEFAULT_VOL_PERIOD,
    )
    cached_market = fast_shared.load_fast_cache(
        a.data_root, a.start, a.end, momentums, config.DEFAULT_VOL_PERIOD
    )
    if cached_market is None:
        raise SystemExit("FAST CACHE BUILD FAIL: cache não foi materializado")
    cached = fast.fast_precompute_shard(
        cached_market, gaps, signals, momentums, config.DEFAULT_VOL_PERIOD
    )
    cache_checks = _compare_precompute("FAST CACHE", original, cached)

    pairs, gap_state, momentum, vol_valid = original
    scalar_results = []
    targets_all = []
    portfolio_checks = []
    for mi, m in enumerate(momentums):
        targets = opt.first_ranked_targets(gap_state, momentum[mi], vol_valid)
        targets_all.append(targets)
        scalar = opt.simulate_pairs(
            targets,
            gap_state,
            momentum[mi],
            vol_valid,
            market,
            initial_cash=config.DEFAULT_INITIAL_CASH,
            fee_bps=config.DEFAULT_FEE_BPS,
            slippage_bps=config.DEFAULT_SLIPPAGE_BPS,
            odd_lot_extra_bps=config.DEFAULT_ODD_LOT_EXTRA_BPS,
        )
        scalar_results.append(scalar)

        cache_targets = opt.first_ranked_targets(cached[1], cached[2][mi], cached[3])
        if not np.array_equal(targets, cache_targets):
            raise SystemExit(f"FAST CACHE TARGET FAIL momentum={m}")
        cache_scalar = opt.simulate_pairs(
            cache_targets,
            cached[1],
            cached[2][mi],
            cached[3],
            cached_market,
            initial_cash=config.DEFAULT_INITIAL_CASH,
            fee_bps=config.DEFAULT_FEE_BPS,
            slippage_bps=config.DEFAULT_SLIPPAGE_BPS,
            odd_lot_extra_bps=config.DEFAULT_ODD_LOT_EXTRA_BPS,
        )
        keys = ["final_equity", "cash", "shares", "holding", "trades", "skipped", "fees", "slippage"]
        one = {key: same(scalar[key], cache_scalar[key]) for key in keys}
        if not all(one.values()):
            raise SystemExit(f"FAST CACHE PORTFOLIO FAIL momentum={m} checks={one}")
        portfolio_checks.append({"momentum": m, "checks": one})

    targets_batch = np.stack(targets_all, axis=0)
    batch = fast_batch.simulate_momentum_batch(
        targets_batch,
        gap_state,
        momentum,
        vol_valid,
        market,
        initial_cash=config.DEFAULT_INITIAL_CASH,
        fee_bps=config.DEFAULT_FEE_BPS,
        slippage_bps=config.DEFAULT_SLIPPAGE_BPS,
        odd_lot_extra_bps=config.DEFAULT_ODD_LOT_EXTRA_BPS,
    )
    batch_checks = []
    keys = ["final_equity", "cash", "shares", "holding", "trades", "skipped", "fees", "slippage"]
    for mi, m in enumerate(momentums):
        one = {key: same(scalar_results[mi][key], batch[key][mi]) for key in keys}
        if not all(one.values()):
            raise SystemExit(f"FAST BATCH PORTFOLIO FAIL momentum={m} checks={one}")
        batch_checks.append({"momentum": m, "checks": one})

    payload = {
        "status": "PASS",
        "schema_version": 3,
        "mode": "bit_exact_cache_threads_batch_and_portfolio",
        "random_seed": RANDOM_SEED,
        "grid": {
            "gap": gaps,
            "signal": signals,
            "momentum": momentums,
            "vol": config.DEFAULT_VOL_PERIOD,
            "gap_signal_pairs": len(pairs),
            "parameter_combinations_checked": len(pairs) * len(momentums),
        },
        "direct_precompute_checks": direct_checks,
        "cache_precompute_checks": cache_checks,
        "cache_portfolio_checks": portfolio_checks,
        "batch_portfolio_checks": batch_checks,
        "cache_sha256": fast_shared.sha256_file(a.data_root / fast_shared.CACHE_FILE),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
