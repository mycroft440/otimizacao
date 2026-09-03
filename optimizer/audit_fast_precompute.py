#!/usr/bin/env python3
"""Gate de equivalencia entre o precompute original e o acelerado."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config
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


def main() -> None:
    a = args()
    market = opt.load_market(a.data_root, a.start, a.end)
    gaps, signals, momentums = representative_values()
    original = opt.precompute_shard(market, gaps, signals, momentums, config.DEFAULT_VOL_PERIOD)
    accelerated = fast.fast_precompute_shard(
        market, gaps, signals, momentums, config.DEFAULT_VOL_PERIOD
    )

    checks = {
        "pairs": original[0] == accelerated[0],
        "gap_state_bit_exact": same(original[1], accelerated[1]),
        "momentum_bit_exact": same(original[2], accelerated[2]),
        "vol_valid_bit_exact": same(original[3], accelerated[3]),
    }
    if not all(checks.values()):
        raise SystemExit(f"FAST PRECOMPUTE EQUIVALENCE FAIL: {checks}")

    # Confere também carteira em todos os momentums da amostra ampliada.
    pairs, gap_state, momentum, vol_valid = original
    _, gap_fast, momentum_fast, vol_fast = accelerated
    portfolio_checks = []
    for mi in range(len(momentums)):
        mom = momentum[mi]
        mom_fast = momentum_fast[mi]
        t1 = opt.first_ranked_targets(gap_state, mom, vol_valid)
        t2 = opt.first_ranked_targets(gap_fast, mom_fast, vol_fast)
        if not np.array_equal(t1, t2):
            raise SystemExit(f"FAST TARGET EQUIVALENCE FAIL momentum={momentums[mi]}")
        s1 = opt.simulate_pairs(
            t1,
            gap_state,
            mom,
            vol_valid,
            market,
            initial_cash=config.DEFAULT_INITIAL_CASH,
            fee_bps=config.DEFAULT_FEE_BPS,
            slippage_bps=config.DEFAULT_SLIPPAGE_BPS,
            odd_lot_extra_bps=config.DEFAULT_ODD_LOT_EXTRA_BPS,
        )
        s2 = opt.simulate_pairs(
            t2,
            gap_fast,
            mom_fast,
            vol_fast,
            market,
            initial_cash=config.DEFAULT_INITIAL_CASH,
            fee_bps=config.DEFAULT_FEE_BPS,
            slippage_bps=config.DEFAULT_SLIPPAGE_BPS,
            odd_lot_extra_bps=config.DEFAULT_ODD_LOT_EXTRA_BPS,
        )
        keys = ["final_equity", "cash", "shares", "holding", "trades", "skipped", "fees", "slippage"]
        one = {key: same(s1[key], s2[key]) for key in keys}
        if not all(one.values()):
            raise SystemExit(
                f"FAST PORTFOLIO EQUIVALENCE FAIL momentum={momentums[mi]} checks={one}"
            )
        portfolio_checks.append({"momentum": momentums[mi], "checks": one})

    payload = {
        "status": "PASS",
        "schema_version": 2,
        "mode": "bit_exact_precompute_and_portfolio_deterministic_broad_sample",
        "random_seed": RANDOM_SEED,
        "grid": {
            "gap": gaps,
            "signal": signals,
            "momentum": momentums,
            "vol": config.DEFAULT_VOL_PERIOD,
            "gap_signal_pairs": len(pairs),
            "parameter_combinations_checked": len(pairs) * len(momentums),
        },
        "checks": checks,
        "portfolio_checks": portfolio_checks,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
