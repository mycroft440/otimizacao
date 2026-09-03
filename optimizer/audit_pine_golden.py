#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optimize_b3_pine as opt
import reduce_results as red


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--fixture", required=True, type=Path)
    p.add_argument("--meta", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def close(a, b, tol=1e-7):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=tol)


def main():
    args = parse_args()
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    if meta.get("source") != "TradingView Pine external export":
        raise SystemExit("golden meta precisa declarar fonte externa do TradingView Pine")
    expected_hash = str(meta.get("fixture_sha256") or "")
    actual_hash = sha256_file(args.fixture)
    if not expected_hash or expected_hash != actual_hash:
        raise SystemExit("hash da fixture Pine diverge do meta")
    for field in ["pine_script_sha256", "exported_at", "start", "end", "gap_period", "signal_period", "momentum_period", "vol_period"]:
        if field not in meta:
            raise SystemExit(f"golden meta sem {field}")

    fixture = pd.read_csv(args.fixture)
    required = {"date", "holding", "shares", "cash", "equity", "weekly_target"}
    missing = required - set(fixture.columns)
    if missing:
        raise SystemExit(f"fixture Pine sem colunas de carteira: {sorted(missing)}")
    fixture["date"] = pd.to_datetime(fixture["date"], errors="raise")

    market = opt.load_market(args.data_root, str(meta["start"]), str(meta["end"]))
    summary, curve = red.detailed_backtest(
        market,
        int(meta["gap_period"]), int(meta["signal_period"]), int(meta["momentum_period"]), int(meta["vol_period"]),
        initial_cash=float(meta.get("initial_cash", 1000.0)),
        fee_bps=float(meta.get("fee_bps", 3.25)),
        slippage_bps=float(meta.get("slippage_bps", 10.0)),
        odd_lot_extra_bps=float(meta.get("odd_lot_extra_bps", 5.0)),
    )
    curve["date"] = pd.to_datetime(curve["date"])
    merged = fixture.merge(curve, on="date", how="left", suffixes=("_pine", "_python"), validate="one_to_one")
    if merged[["equity_python", "cash_python"]].isna().any().any():
        raise SystemExit("fixture contem data que nao existe na curva Python")

    failures = []
    for row in merged.itertuples(index=False):
        checks = {
            "holding": str(row.holding_pine) == str(row.holding_python),
            "weekly_target": str(row.weekly_target_pine) == str(row.weekly_target_python),
            "shares": int(row.shares_pine) == int(row.shares_python),
            "cash": close(row.cash_pine, row.cash_python),
            "equity": close(row.equity_pine, row.equity_python),
        }
        if not all(checks.values()):
            failures.append({"date": row.date.date().isoformat(), "checks": checks})

    indicator_checks = None
    indicator_columns = {"decision_date", "ticker", "gap_state", "momentum", "vol_valid", "selected_top1"}
    if indicator_columns.issubset(fixture.columns):
        pairs, gap, momentum, vol = opt.precompute_shard(
            market,
            [int(meta["gap_period"])], [int(meta["signal_period"])],
            [int(meta["momentum_period"])], int(meta["vol_period"]),
        )
        targets = opt.first_ranked_targets(gap, momentum[0], vol)[0]
        ticker_index = {ticker: i for i, ticker in enumerate(market.tickers)}
        decision_lookup = {pd.Timestamp(day): w for w, day in enumerate(market.decision_dates)}
        indicator_failures = []
        for row in fixture.drop_duplicates(["decision_date", "ticker"]).itertuples(index=False):
            day = pd.Timestamp(row.decision_date)
            ticker = str(row.ticker).upper()
            if day not in decision_lookup or ticker not in ticker_index:
                indicator_failures.append({"decision_date": str(day.date()), "ticker": ticker, "reason": "not_in_python_context"})
                continue
            w = decision_lookup[day]
            ti = ticker_index[ticker]
            expected_top = market.tickers[int(targets[w])] if int(targets[w]) >= 0 else "CASH"
            checks = {
                "gap_state": bool(row.gap_state) == bool(gap[0, w, ti]),
                "momentum": close(row.momentum, momentum[0, w, ti], tol=1e-10) if np.isfinite(momentum[0, w, ti]) else pd.isna(row.momentum),
                "vol_valid": bool(row.vol_valid) == bool(vol[w, ti]),
                "selected_top1": str(row.selected_top1) == expected_top,
            }
            if not all(checks.values()):
                indicator_failures.append({"decision_date": str(day.date()), "ticker": ticker, "checks": checks})
        indicator_checks = {"status": "PASS" if not indicator_failures else "FAIL", "failures": indicator_failures[:100]}
        if indicator_failures:
            failures.extend([{"indicator": item} for item in indicator_failures[:100]])

    payload = {
        "status": "PASS" if not failures else "FAIL",
        "schema_version": 1,
        "fixture_sha256": actual_hash,
        "pine_script_sha256": meta["pine_script_sha256"],
        "source": meta["source"],
        "portfolio_rows_checked": int(len(merged)),
        "portfolio_failures": [item for item in failures if "date" in item][:100],
        "indicator_checks": indicator_checks,
        "python_summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "portfolio_rows_checked": len(merged)}, ensure_ascii=False))
    if failures:
        raise SystemExit("PINE GOLDEN AUDIT FAIL")


if __name__ == "__main__":
    main()
