#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd

SHARD_RE = re.compile(r"^shard_(\d+)$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--expected-shards", required=True, type=int)
    p.add_argument("--initial-cash", required=True, type=float)
    p.add_argument("--fee-bps", required=True, type=float)
    p.add_argument("--slippage-bps", required=True, type=float)
    p.add_argument("--odd-lot-extra-bps", required=True, type=float)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def close(a: object, b: float) -> bool:
    try:
        value = float(a)
    except Exception:
        return False
    return math.isfinite(value) and math.isclose(value, float(b), rel_tol=0.0, abs_tol=1e-12)


def audit_shard_set(
    results_dir: Path,
    *,
    expected_shards: int,
    initial_cash: float,
    fee_bps: float,
    slippage_bps: float,
    odd_lot_extra_bps: float,
) -> dict[str, object]:
    if expected_shards <= 0:
        raise ValueError("expected_shards precisa ser > 0")

    csvs = sorted(results_dir.rglob("shard_*.csv"))
    jsons = sorted(results_dir.rglob("shard_*.json"))
    failures: list[str] = []
    if len(csvs) != expected_shards:
        failures.append(f"CSV count {len(csvs)} != expected {expected_shards}")
    if len(jsons) != expected_shards:
        failures.append(f"JSON count {len(jsons)} != expected {expected_shards}")

    seen_ids: set[int] = set()
    canonical_contract = None
    file_reports = []

    for csv_path in csvs:
        match = SHARD_RE.match(csv_path.stem)
        if not match:
            failures.append(f"nome de shard invalido: {csv_path}")
            continue
        shard_id = int(match.group(1))
        if shard_id in seen_ids:
            failures.append(f"shard id duplicado: {shard_id}")
        seen_ids.add(shard_id)
        meta_path = csv_path.with_suffix(".json")
        if not meta_path.exists():
            failures.append(f"metadata ausente para {csv_path.name}")
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            frame = pd.read_csv(csv_path)
        except Exception as exc:
            failures.append(f"falha ao ler shard {shard_id}: {exc}")
            continue

        if int(meta.get("schema_version", 0)) != 2:
            failures.append(f"shard {shard_id}: schema_version precisa ser 2")
        if int(meta.get("shard", -1)) != shard_id:
            failures.append(f"shard {shard_id}: metadata shard={meta.get('shard')}")
        if int(meta.get("shards", -1)) != expected_shards:
            failures.append(f"shard {shard_id}: metadata shards={meta.get('shards')}")
        if int(meta.get("rows", -1)) != len(frame):
            failures.append(f"shard {shard_id}: rows metadata={meta.get('rows')} csv={len(frame)}")
        if "shard" not in frame.columns or not (
            pd.to_numeric(frame["shard"], errors="coerce") == shard_id
        ).all():
            failures.append(f"shard {shard_id}: coluna shard do CSV diverge")

        actual_csv_hash = sha256(csv_path)
        if str(meta.get("csv_sha256", "")) != actual_csv_hash:
            failures.append(f"shard {shard_id}: csv_sha256 nao confere")

        if "gap_period" in frame.columns:
            actual_gaps = sorted(
                set(pd.to_numeric(frame["gap_period"], errors="raise").astype(int).tolist())
            )
            expected_gaps = sorted(int(x) for x in meta.get("gap_values", []))
            if actual_gaps != expected_gaps:
                failures.append(
                    f"shard {shard_id}: gap_values divergem metadata={expected_gaps} csv={actual_gaps}"
                )
        else:
            failures.append(f"shard {shard_id}: gap_period ausente")

        for key, expected in (
            ("initial_cash", initial_cash),
            ("fee_bps", fee_bps),
            ("slippage_bps", slippage_bps),
            ("odd_lot_extra_bps", odd_lot_extra_bps),
        ):
            if not close(meta.get(key), expected):
                failures.append(f"shard {shard_id}: {key}={meta.get(key)} esperado={expected}")

        contract = {
            "shards": meta.get("shards"),
            "signal_min": meta.get("signal_min"),
            "signal_max": meta.get("signal_max"),
            "momentum_min": meta.get("momentum_min"),
            "momentum_max": meta.get("momentum_max"),
            "vol_period": meta.get("vol_period"),
            "start": meta.get("start"),
            "end": meta.get("end"),
            "initial_cash": meta.get("initial_cash"),
            "fee_bps": meta.get("fee_bps"),
            "slippage_bps": meta.get("slippage_bps"),
            "odd_lot_extra_bps": meta.get("odd_lot_extra_bps"),
            "portfolio_policy": meta.get("portfolio_policy"),
            "momentum_dtype": meta.get("momentum_dtype"),
            "github_run_id": meta.get("github_run_id"),
            "optimizer_sha": meta.get("optimizer_sha"),
            "github_repository": meta.get("github_repository"),
            "snapshot_upstream_sha": meta.get("snapshot_upstream_sha"),
            "snapshot_universe_sha256": meta.get("snapshot_universe_sha256"),
            "snapshot_requested_end": meta.get("snapshot_requested_end"),
        }
        if canonical_contract is None:
            canonical_contract = contract
        elif contract != canonical_contract:
            failures.append(f"shard {shard_id}: contrato/proveniencia diverge dos demais shards")

        file_reports.append(
            {
                "shard": shard_id,
                "csv": str(csv_path),
                "json": str(meta_path),
                "rows": int(len(frame)),
                "csv_sha256": actual_csv_hash,
                "json_sha256": sha256(meta_path),
            }
        )

    expected_ids = set(range(expected_shards))
    if seen_ids != expected_ids:
        failures.append(
            f"IDs de shard divergentes; missing={sorted(expected_ids-seen_ids)} "
            f"extra={sorted(seen_ids-expected_ids)}"
        )

    return {
        "status": "PASS" if not failures else "FAIL",
        "schema_version": 2,
        "expected_shards": expected_shards,
        "observed_shard_ids": sorted(seen_ids),
        "canonical_contract": canonical_contract,
        "files": file_reports,
        "failures": failures,
    }


def main():
    args = parse_args()
    payload = audit_shard_set(
        args.results_dir,
        expected_shards=args.expected_shards,
        initial_cash=args.initial_cash,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        odd_lot_extra_bps=args.odd_lot_extra_bps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "shards": payload["observed_shard_ids"],
                "failures": payload["failures"],
            },
            ensure_ascii=False,
        )
    )
    if payload["status"] != "PASS":
        raise SystemExit("SHARD SET AUDIT FAIL")


if __name__ == "__main__":
    main()
