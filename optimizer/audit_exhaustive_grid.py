#!/usr/bin/env python3
"""Audita que os shards cobrem exatamente toda a grade cartesiana, passo 1.

A checagem anterior por cardinalidade + duplicatas já era forte, mas em teoria uma
linha fora da faixa poderia substituir uma combinação ausente mantendo a mesma
cardinalidade. Este auditor fecha essa brecha: transforma cada tripla
(GAP, SIGNAL, MOMENTUM) em um ID canônico da grade e exige frequência exatamente 1
para todos os IDs esperados, além de VOL_PERIOD constante.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--gap-min", type=int, required=True)
    p.add_argument("--gap-max", type=int, required=True)
    p.add_argument("--signal-min", type=int, required=True)
    p.add_argument("--signal-max", type=int, required=True)
    p.add_argument("--momentum-min", type=int, required=True)
    p.add_argument("--momentum-max", type=int, required=True)
    p.add_argument("--vol-period", type=int, required=True)
    return p.parse_args()


def _integer_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise SystemExit(f"AUDIT FAIL: {name} contém valor não numérico/não finito")
    rounded = np.rint(values)
    if not np.all(values == rounded):
        bad = values[values != rounded][:10].tolist()
        raise SystemExit(f"AUDIT FAIL: {name} contém valor não inteiro: {bad}")
    return rounded.astype(np.int64)


def main() -> None:
    args = parse_args()
    csvs = sorted(args.results_dir.rglob("shard_*.csv"))
    if not csvs:
        raise SystemExit(f"AUDIT FAIL: nenhum shard_*.csv em {args.results_dir}")

    frames = [pd.read_csv(path) for path in csvs]
    results = pd.concat(frames, ignore_index=True)
    required = {"gap_period", "signal_period", "momentum_period", "vol_period"}
    missing_columns = sorted(required - set(results.columns))
    if missing_columns:
        raise SystemExit(f"AUDIT FAIL: colunas ausentes: {missing_columns}")

    if args.gap_max < args.gap_min or args.signal_max < args.signal_min or args.momentum_max < args.momentum_min:
        raise SystemExit("AUDIT FAIL: faixa invertida")

    gap_count = args.gap_max - args.gap_min + 1
    signal_count = args.signal_max - args.signal_min + 1
    momentum_count = args.momentum_max - args.momentum_min + 1
    expected = gap_count * signal_count * momentum_count

    gap = _integer_column(results, "gap_period")
    signal = _integer_column(results, "signal_period")
    momentum = _integer_column(results, "momentum_period")
    vol = _integer_column(results, "vol_period")

    in_bounds = (
        (gap >= args.gap_min)
        & (gap <= args.gap_max)
        & (signal >= args.signal_min)
        & (signal <= args.signal_max)
        & (momentum >= args.momentum_min)
        & (momentum <= args.momentum_max)
    )
    if not np.all(in_bounds):
        idx = np.flatnonzero(~in_bounds)[:10]
        examples = [
            {
                "gap": int(gap[i]),
                "signal": int(signal[i]),
                "momentum": int(momentum[i]),
                "vol": int(vol[i]),
            }
            for i in idx
        ]
        raise SystemExit(f"AUDIT FAIL: combinação fora da grade: {examples}")

    if not np.all(vol == args.vol_period):
        bad = sorted(set(int(x) for x in vol[vol != args.vol_period][:20]))
        raise SystemExit(
            f"AUDIT FAIL: VOL_PERIOD divergente; esperado={args.vol_period}, encontrados={bad}"
        )

    # ID bijetivo para cada ponto da grade cartesiana.
    ids = (
        ((gap - args.gap_min) * signal_count + (signal - args.signal_min))
        * momentum_count
        + (momentum - args.momentum_min)
    )
    if np.any(ids < 0) or np.any(ids >= expected):
        raise SystemExit("AUDIT FAIL: ID canônico fora dos limites")

    frequencies = np.bincount(ids, minlength=expected)
    missing_ids = np.flatnonzero(frequencies == 0)
    duplicate_ids = np.flatnonzero(frequencies > 1)

    if len(results) != expected or len(missing_ids) or len(duplicate_ids):
        def decode(value: int) -> dict[str, int]:
            g_offset, rem = divmod(int(value), signal_count * momentum_count)
            s_offset, m_offset = divmod(rem, momentum_count)
            return {
                "gap": args.gap_min + g_offset,
                "signal": args.signal_min + s_offset,
                "momentum": args.momentum_min + m_offset,
                "vol": args.vol_period,
            }

        report = {
            "status": "FAIL",
            "expected_combinations": expected,
            "rows": int(len(results)),
            "missing_count": int(len(missing_ids)),
            "duplicate_count": int(len(duplicate_ids)),
            "missing_examples": [decode(x) for x in missing_ids[:10]],
            "duplicate_examples": [decode(x) for x in duplicate_ids[:10]],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise SystemExit("AUDIT FAIL: a grade cartesiana não foi coberta exatamente uma vez")

    report = {
        "status": "PASS",
        "step": 1,
        "gap_period": [args.gap_min, args.gap_max, 1],
        "signal_period": [args.signal_min, args.signal_max, 1],
        "momentum_period": [args.momentum_min, args.momentum_max, 1],
        "vol_period": args.vol_period,
        "expected_combinations": expected,
        "rows": int(len(results)),
        "unique_canonical_ids": int(np.count_nonzero(frequencies)),
        "missing_count": 0,
        "duplicate_count": 0,
        "exact_cartesian_grid_once": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
