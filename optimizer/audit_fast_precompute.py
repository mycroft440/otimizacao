#!/usr/bin/env python3
"""Gate de equivalencia entre o precompute original e o acelerado."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import optimize_b3_pine as opt
import optimize_b3_pine_fast as fast


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--start", default="2018-01-02")
    p.add_argument("--end", default="")
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def same(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if np.issubdtype(a.dtype, np.floating):
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def main() -> None:
    a = args()
    market = opt.load_market(a.data_root, a.start, a.end)
    gaps = [5, 40, 80]
    signals = [2, 9, 20, 60]
    momentums = [5, 63, 123, 252]
    original = opt.precompute_shard(market, gaps, signals, momentums, 21)
    accelerated = fast.fast_precompute_shard(market, gaps, signals, momentums, 21)

    checks = {
        "pairs": original[0] == accelerated[0],
        "gap_state_bit_exact": same(original[1], accelerated[1]),
        "momentum_bit_exact": same(original[2], accelerated[2]),
        "vol_valid_bit_exact": same(original[3], accelerated[3]),
    }
    if not all(checks.values()):
        raise SystemExit(f"FAST PRECOMPUTE EQUIVALENCE FAIL: {checks}")

    # Confere tambem o resultado de carteira para a grade pequena representativa.
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
            t1, gap_state, mom, vol_valid, market,
            initial_cash=1000.0, fee_bps=3.25,
            slippage_bps=10.0, odd_lot_extra_bps=5.0,
        )
        s2 = opt.simulate_pairs(
            t2, gap_fast, mom_fast, vol_fast, market,
            initial_cash=1000.0, fee_bps=3.25,
            slippage_bps=10.0, odd_lot_extra_bps=5.0,
        )
        keys = ["final_equity", "cash", "shares", "holding", "trades", "skipped", "fees", "slippage"]
        one = {key: same(s1[key], s2[key]) for key in keys}
        if not all(one.values()):
            raise SystemExit(f"FAST PORTFOLIO EQUIVALENCE FAIL momentum={momentums[mi]} checks={one}")
        portfolio_checks.append({"momentum": momentums[mi], "checks": one})

    payload = {
        "status": "PASS",
        "mode": "bit_exact_precompute_and_portfolio_small_grid",
        "grid": {"gap": gaps, "signal": signals, "momentum": momentums, "vol": 21},
        "checks": checks,
        "portfolio_checks": portfolio_checks,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
