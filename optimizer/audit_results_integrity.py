#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

import audit_shard_set


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--initial-cash", required=True, type=float)
    p.add_argument("--universe", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--expected-shards", type=int, default=None)
    p.add_argument("--fee-bps", type=float, default=None)
    p.add_argument("--slippage-bps", type=float, default=None)
    p.add_argument("--odd-lot-extra-bps", type=float, default=None)
    return p.parse_args()


def _env_or(value: float | int | None, env_name: str, cast):
    if value is not None:
        return value
    raw = os.environ.get(env_name)
    return cast(raw) if raw not in (None, "") else None


def _first_meta(results_dir: Path) -> dict[str, object]:
    metas = sorted(results_dir.rglob("shard_*.json"))
    if not metas:
        raise SystemExit("metadata shard_*.json ausente")
    return json.loads(metas[0].read_text(encoding="utf-8"))


def _snapshot_meta_from_universe(universe_path: Path) -> dict[str, object]:
    try:
        root = universe_path.parents[2]
    except IndexError:
        return {}
    path = root / "SNAPSHOT_META.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    if not math.isfinite(args.initial_cash) or args.initial_cash <= 0:
        raise SystemExit("initial_cash invalido")

    first_meta = _first_meta(args.results_dir)
    expected_shards = _env_or(args.expected_shards, "SHARDS", int)
    if expected_shards is None:
        expected_shards = int(first_meta.get("shards", 0))
    fee_bps = _env_or(args.fee_bps, "FEE_BPS", float)
    if fee_bps is None:
        fee_bps = float(first_meta.get("fee_bps"))
    slippage_bps = _env_or(args.slippage_bps, "SLIPPAGE_BPS", float)
    if slippage_bps is None:
        slippage_bps = float(first_meta.get("slippage_bps"))
    odd_lot_extra_bps = _env_or(args.odd_lot_extra_bps, "ODD_LOT_EXTRA_BPS", float)
    if odd_lot_extra_bps is None:
        odd_lot_extra_bps = float(first_meta.get("odd_lot_extra_bps"))

    shard_set = audit_shard_set.audit_shard_set(
        args.results_dir,
        expected_shards=int(expected_shards),
        initial_cash=args.initial_cash,
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
        odd_lot_extra_bps=float(odd_lot_extra_bps),
    )
    if shard_set["status"] != "PASS":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "FAIL",
            "schema_version": 3,
            "rows": 0,
            "initial_cash": args.initial_cash,
            "shard_set": shard_set,
            "checks": {"shard_metadata_reconciles": False},
            "failures": list(shard_set["failures"]),
        }
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise SystemExit("RESULT INTEGRITY AUDIT FAIL: SHARD SET")

    provenance_failures: list[str] = []
    contract = dict(shard_set.get("canonical_contract") or {})
    current_run_id = os.environ.get("GITHUB_RUN_ID", "")
    current_sha = os.environ.get("GITHUB_SHA", "")
    current_repo = os.environ.get("GITHUB_REPOSITORY", "")
    if current_run_id and str(contract.get("github_run_id", "")) != current_run_id:
        provenance_failures.append(
            f"github_run_id dos shards={contract.get('github_run_id')} atual={current_run_id}"
        )
    if current_sha and str(contract.get("optimizer_sha", "")) != current_sha:
        provenance_failures.append(
            f"optimizer_sha dos shards={contract.get('optimizer_sha')} atual={current_sha}"
        )
    if current_repo and str(contract.get("github_repository", "")) != current_repo:
        provenance_failures.append(
            f"github_repository dos shards={contract.get('github_repository')} atual={current_repo}"
        )

    snapshot_meta = _snapshot_meta_from_universe(args.universe)
    if snapshot_meta:
        if str(contract.get("snapshot_upstream_sha", "")) != str(snapshot_meta.get("upstream_sha", "")):
            provenance_failures.append("snapshot_upstream_sha dos shards diverge do snapshot congelado")
        if str(contract.get("snapshot_universe_sha256", "")) != str(snapshot_meta.get("universe_sha256", "")):
            provenance_failures.append("snapshot_universe_sha256 dos shards diverge do snapshot congelado")
        if str(contract.get("snapshot_requested_end", "")) != str(snapshot_meta.get("requested_end", "")):
            provenance_failures.append("snapshot_requested_end dos shards diverge do snapshot congelado")

    files = sorted(args.results_dir.rglob("shard_*.csv"))
    if not files:
        raise SystemExit("nenhum shard encontrado")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    required = {
        "gap_period",
        "signal_period",
        "momentum_period",
        "vol_period",
        "final_equity",
        "total_return",
        "trades",
        "skipped_executions",
        "fees_paid",
        "slippage_impact",
        "final_holding",
        "shard",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"colunas ausentes: {missing}")

    failures = list(provenance_failures)
    numeric_nonnegative = [
        "final_equity",
        "trades",
        "skipped_executions",
        "fees_paid",
        "slippage_impact",
    ]
    for name in ["final_equity", "total_return", "fees_paid", "slippage_impact"]:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            failures.append(f"{name}: contem NaN/inf")
    for name in numeric_nonnegative:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        if np.any(values < 0):
            failures.append(f"{name}: contem valor negativo")

    for name in [
        "trades",
        "skipped_executions",
        "shard",
        "gap_period",
        "signal_period",
        "momentum_period",
        "vol_period",
    ]:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(values == np.rint(values)):
            failures.append(f"{name}: precisa conter somente inteiros finitos")

    equity = pd.to_numeric(frame["final_equity"], errors="coerce").to_numpy(dtype=float)
    reported = pd.to_numeric(frame["total_return"], errors="coerce").to_numpy(dtype=float)
    expected = equity / args.initial_cash - 1.0
    if not np.allclose(reported, expected, rtol=1e-10, atol=1e-10, equal_nan=False):
        bad = np.flatnonzero(~np.isclose(reported, expected, rtol=1e-10, atol=1e-10))[:10]
        failures.append(f"total_return inconsistente; exemplos de indices: {bad.tolist()}")

    universe = json.loads(args.universe.read_text(encoding="utf-8"))
    allowed = {str(x).upper() for x in universe["tickers"]} | {"CASH"}
    holdings = frame["final_holding"].astype(str).str.upper()
    invalid_holdings = sorted(set(holdings[~holdings.isin(allowed)].tolist()))
    if invalid_holdings:
        failures.append(f"final_holding invalido: {invalid_holdings[:20]}")

    payload = {
        "status": "PASS" if not failures else "FAIL",
        "schema_version": 3,
        "rows": int(len(frame)),
        "initial_cash": args.initial_cash,
        "expected_costs": {
            "fee_bps": float(fee_bps),
            "slippage_bps": float(slippage_bps),
            "odd_lot_extra_bps": float(odd_lot_extra_bps),
        },
        "current_workflow_identity": {
            "github_run_id": current_run_id or None,
            "optimizer_sha": current_sha or None,
            "github_repository": current_repo or None,
        },
        "shard_set": shard_set,
        "checks": {
            "shard_metadata_reconciles": shard_set["status"] == "PASS",
            "shards_belong_to_current_run": not provenance_failures,
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
