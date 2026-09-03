#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import optimize_b3_pine as opt
import reference_engine as ref


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--start", required=True)
    p.add_argument("--end", default="")
    p.add_argument("--initial-cash", required=True, type=float)
    p.add_argument("--fee-bps", required=True, type=float)
    p.add_argument("--slippage-bps", required=True, type=float)
    p.add_argument("--odd-lot-extra-bps", required=True, type=float)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-7)


def audit_cases() -> list[tuple[int, int, int, int]]:
    fixed = [(5, 2, 5, 21), (40, 20, 63, 21), (41, 15, 49, 21), (80, 60, 252, 21)]
    rng = np.random.default_rng(20260903)
    cases = list(fixed)
    while len(cases) < 8:
        case = (
            int(rng.integers(5, 81)),
            int(rng.integers(2, 61)),
            int(rng.integers(5, 253)),
            21,
        )
        if case not in cases:
            cases.append(case)
    return cases


def main():
    args = parse_args()
    market = opt.load_market(args.data_root, args.start, args.end)
    cases = audit_cases()
    reports = []
    failures = []

    for g, s, m, v in cases:
        pairs, gap, momentum, vol = opt.precompute_shard(market, [g], [s], [m], v)
        targets = opt.first_ranked_targets(gap, momentum[0], vol)
        fast = opt.simulate_pairs(
            targets, gap, momentum[0], vol, market,
            initial_cash=args.initial_cash,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            odd_lot_extra_bps=args.odd_lot_extra_bps,
        )
        slow = ref.simulate_reference(
            market, g, s, m, v,
            initial_cash=args.initial_cash,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            odd_lot_extra_bps=args.odd_lot_extra_bps,
        )
        checks = {
            "pairs": pairs == [(g, s)],
            "execution_dates": bool(np.array_equal(market.execution_dates.values, slow["execution_dates"].values)),
            "decision_dates": bool(np.array_equal(market.decision_dates.values, slow["decision_dates"].values)),
            "gap_state": bool(np.array_equal(gap[0], slow["gap_state"])),
            "momentum": bool(np.allclose(momentum[0], slow["momentum"], rtol=1e-12, atol=1e-12, equal_nan=True)),
            "vol_valid": bool(np.array_equal(vol, slow["vol_valid"])),
            "base_targets": bool(np.array_equal(targets[0], slow["base_targets"])),
            "final_equity": close(fast["final_equity"][0], slow["final_equity"]),
            "cash": close(fast["cash"][0], slow["cash"]),
            "shares": int(fast["shares"][0]) == int(slow["shares"]),
            "holding": int(fast["holding"][0]) == int(slow["holding"]),
            "trades": int(fast["trades"][0]) == int(slow["trades"]),
            "skipped": int(fast["skipped"][0]) == int(slow["skipped"]),
            "fees": close(fast["fees"][0], slow["fees"]),
            "slippage": close(fast["slippage"][0], slow["slippage"]),
        }
        if not all(checks.values()):
            failures.append({"case": [g, s, m, v], "checks": checks})
        reports.append({"case": [g, s, m, v], "checks": checks})

    payload = {
        "status": "PASS" if not failures else "FAIL",
        "schema_version": 2,
        "mode": "independent_slow_reference_vs_primary_engine",
        "schedule_mapping_independent": True,
        "execution_open_lookup_independent": True,
        "random_seed": 20260903,
        "case_count": len(cases),
        "cases": reports,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "case_count": len(cases),
        "failures": failures,
    }, ensure_ascii=False))
    if failures:
        raise SystemExit("REFERENCE ENGINE AUDIT FAIL")


if __name__ == "__main__":
    main()
