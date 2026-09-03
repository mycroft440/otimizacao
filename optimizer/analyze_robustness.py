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
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--json", required=True, type=Path)
    p.add_argument("--md", required=True, type=Path)
    p.add_argument("--radius", type=int, default=2)
    return p.parse_args()


def f(value):
    return float(value) if math.isfinite(float(value)) else None


def main():
    args = parse_args()
    frame = pd.read_csv(args.results)
    required = {"gap_period", "signal_period", "momentum_period", "final_equity", "total_return"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"resultados sem colunas: {sorted(missing)}")
    if frame.empty:
        raise SystemExit("resultados vazios")
    frame = frame.sort_values(
        ["final_equity", "gap_period", "signal_period", "momentum_period"],
        ascending=[False, True, True, True], kind="stable"
    ).reset_index(drop=True)
    best = frame.iloc[0]
    returns = pd.to_numeric(frame["total_return"], errors="raise").to_numpy(dtype=float)
    if not np.all(np.isfinite(returns)):
        raise SystemExit("retornos nao finitos")

    quantiles = {str(q): f(np.quantile(returns, q)) for q in [0.0, 0.5, 0.9, 0.99, 0.999, 1.0]}
    r = int(args.radius)
    neighbor_mask = (
        (frame["gap_period"].sub(int(best.gap_period)).abs() <= r)
        & (frame["signal_period"].sub(int(best.signal_period)).abs() <= r)
        & (frame["momentum_period"].sub(int(best.momentum_period)).abs() <= r)
    )
    neighborhood = frame[neighbor_mask].copy()
    neighbor_returns = neighborhood["total_return"].to_numpy(dtype=float)
    local = {
        "radius_each_dimension": r,
        "count": int(len(neighborhood)),
        "median_return": f(np.median(neighbor_returns)),
        "mean_return": f(np.mean(neighbor_returns)),
        "min_return": f(np.min(neighbor_returns)),
        "max_return": f(np.max(neighbor_returns)),
        "std_return": f(np.std(neighbor_returns, ddof=1)) if len(neighbor_returns) > 1 else None,
    }
    local_median = float(np.median(neighbor_returns))
    best_return = float(best.total_return)
    local["winner_vs_local_median_multiple"] = (
        f((1.0 + best_return) / (1.0 + local_median)) if local_median > -1.0 else None
    )

    top_count = max(1, int(math.ceil(len(frame) * 0.001)))
    top = frame.head(top_count)
    concentration = {}
    for column in ["gap_period", "signal_period", "momentum_period"]:
        counts = top[column].value_counts(normalize=True).head(10)
        concentration[column] = [
            {"value": int(index), "share_of_top_0_1pct": float(value)}
            for index, value in counts.items()
        ]

    payload = {
        "status": "PASS",
        "schema_version": 1,
        "rows": int(len(frame)),
        "warning": "in-sample robustness analysis; does not replace OOS/walk-forward validation",
        "winner": {
            "gap_period": int(best.gap_period),
            "signal_period": int(best.signal_period),
            "momentum_period": int(best.momentum_period),
            "final_equity": float(best.final_equity),
            "total_return": best_return,
        },
        "return_quantiles": quantiles,
        "top_0_1pct_count": top_count,
        "local_neighborhood": local,
        "top_0_1pct_parameter_concentration": concentration,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Robustez in-sample do landscape de parametros",
        "",
        "Esta analise mede estabilidade local e distribuicao da grade; **nao substitui OOS/walk-forward**.",
        "",
        f"- Vencedora: G{int(best.gap_period)}/S{int(best.signal_period)}/M{int(best.momentum_period)}",
        f"- Retorno da vencedora: {best_return * 100:.2f}%",
        f"- Mediana de toda a grade: {float(np.median(returns)) * 100:.2f}%",
        f"- Percentil 99: {float(np.quantile(returns, 0.99)) * 100:.2f}%",
        f"- Percentil 99,9: {float(np.quantile(returns, 0.999)) * 100:.2f}%",
        "",
        f"## Vizinhanca ±{r} em Gap/Signal/Momentum",
        "",
        f"- Combinacoes na vizinhanca: {len(neighborhood)}",
        f"- Retorno mediano local: {local_median * 100:.2f}%",
        f"- Retorno medio local: {float(np.mean(neighbor_returns)) * 100:.2f}%",
        f"- Pior retorno local: {float(np.min(neighbor_returns)) * 100:.2f}%",
        f"- Melhor retorno local: {float(np.max(neighbor_returns)) * 100:.2f}%",
    ]
    args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
