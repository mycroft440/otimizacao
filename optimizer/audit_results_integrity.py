#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--initial-cash", required=True, type=float)
    p.add_argument("--universe", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    if not math.isfinite(args.initial_cash) or args.initial_cash <= 0:
        raise SystemExit("initial_cash invalido")
    files = sorted(args.results_dir.rglob("shard_*.csv"))
    if not files:
        raise SystemExit("nenhum shard encontrado")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    required = {
        "gap_period", "signal_period", "momentum_period", "vol_period",
        "final_equity", "total_return", "trades", "skipped_executions",
        "fees_paid", "slippage_impact", "final_holding", "shard",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"colunas ausentes: {missing}")

    failures = []
    numeric_nonnegative = ["final_equity", "trades", "skipped_executions", "fees_paid", "slippage_impact"]
    for name in ["final_equity", "total_return", "fees_paid", "slippage_impact"]:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            failures.append(f"{name}: contem NaN/inf")
    for name in numeric_nonnegative:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        if np.any(values < 0):
            failures.append(f"{name}: contem valor negativo")

    for name in ["trades", "skipped_executions", "shard", "gap_period", "signal_period", "momentum_period", "vol_period"]:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(values == np.rint(values)):
            failures.append(f"{name}: precisa conter somente inteiros finitos")

    equity = pd.to_numeric(frame["final_equity"], errors="coerce").to_numpy(dtype=float)
    reported = pd.to_numeric(frame["total_return"], errors="coerce").to_numpy(dtype=float)
    expected = equity / args.initial_cash - 1.0
    if not np.allclose(reported, expected, rtol=1e-10, atol=1e-10, equal_nan=False):
        bad = np.flatnonzero(~np.isclose(reported, expected, rtol=1e-10, atol=1e-10))[:10]
        failures.append(f"total_return inconsistente em {len(bad)} exemplos: {bad.tolist()}")

    universe = json.loads(args.universe.read_text(encoding="utf-8"))
    allowed = {str(x).upper() for x in universe["tickers"]} | {"CASH"}
    holdings = frame["final_holding"].astype(str).str.upper()
    invalid_holdings = sorted(set(holdings[~holdings.isin(allowed)].tolist()))
    if invalid_holdings:
        failures.append(f"final_holding invalido: {invalid_holdings[:20]}")

    payload = {
        "status": "PASS" if not failures else "FAIL",
        "rows": int(len(frame)),
        "initial_cash": args.initial_cash,
        "checks": {
            "required_schema": not missing,
            "finite_financial_fields": not any("NaN/inf" in x for x in failures),
            "nonnegative_accounting_fields": not any("negativo" in x for x in failures),
            "integer_count_fields": not any("inteiros finitos" in x for x in failures),
            "total_return_reconciles": not any("total_return inconsistente" in x for x in failures),
            "holding_in_universe_or_cash": not invalid_holdings,
        },
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if failures:
        raise SystemExit("RESULT INTEGRITY AUDIT FAIL")


if __name__ == "__main__":
    main()
