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


def _integer_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise SystemExit(f"{column}: contem valor nao finito")
    if not (values == values.round()).all():
        raise SystemExit(f"{column}: precisa ser inteiro")
    return values.astype(int)


def main():
    args = parse_args()
    if args.radius < 0:
        raise SystemExit("--radius precisa ser >= 0")

    frame = pd.read_csv(args.results)
    required = {"gap_period", "signal_period", "momentum_period", "final_equity", "total_return"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"resultados sem colunas: {sorted(missing)}")
    if frame.empty:
        raise SystemExit("resultados vazios")

    for column in ["gap_period", "signal_period", "momentum_period"]:
        frame[column] = _integer_series(frame, column)
    for column in ["final_equity", "total_return"]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise SystemExit(f"{column}: contem valor nao finito")
        frame[column] = values.astype(float)

    key = ["gap_period", "signal_period", "momentum_period"]
    if frame.duplicated(key).any():
        raise SystemExit("landscape contem combinacoes duplicadas")

    frame = frame.sort_values(
        ["final_equity", "gap_period", "signal_period", "momentum_period"],
        ascending=[False, True, True, True], kind="stable"
    ).reset_index(drop=True)
    best = frame.iloc[0]
    returns = frame["total_return"].to_numpy(dtype=float)

    quantile_points = [0.0, 0.5, 0.9, 0.99, 0.999, 1.0]
    quantiles = {str(q): f(np.quantile(returns, q)) for q in quantile_points}
    q99 = float(np.quantile(returns, 0.99))
    r = int(args.radius)
    neighbor_mask = (
        (frame["gap_period"].sub(int(best.gap_period)).abs() <= r)
        & (frame["signal_period"].sub(int(best.signal_period)).abs() <= r)
        & (frame["momentum_period"].sub(int(best.momentum_period)).abs() <= r)
    )
    neighborhood = frame[neighbor_mask].copy()
    neighbor_returns = neighborhood["total_return"].to_numpy(dtype=float)
    if len(neighbor_returns) == 0:
        raise SystemExit("vizinhanca da vencedora ficou vazia")

    local_median = float(np.median(neighbor_returns))
    best_return = float(best.total_return)
    local = {
        "radius_each_dimension": r,
        "count": int(len(neighborhood)),
        "median_return": f(local_median),
        "mean_return": f(np.mean(neighbor_returns)),
        "min_return": f(np.min(neighbor_returns)),
        "max_return": f(np.max(neighbor_returns)),
        "std_return": f(np.std(neighbor_returns, ddof=1)) if len(neighbor_returns) > 1 else None,
        "share_positive_return": float(np.mean(neighbor_returns > 0.0)),
        "share_at_or_above_global_99pct": float(np.mean(neighbor_returns >= q99)),
        "winner_vs_local_median_wealth_multiple": (
            f((1.0 + best_return) / (1.0 + local_median)) if local_median > -1.0 else None
        ),
    }

    # One-step axis neighbors are especially useful to detect a single-point spike.
    axis_neighbors = []
    best_tuple = (
        int(best.gap_period), int(best.signal_period), int(best.momentum_period)
    )
    by_key = frame.set_index(key)
    for label, offset in [
        ("gap-1", (-1, 0, 0)), ("gap+1", (1, 0, 0)),
        ("signal-1", (0, -1, 0)), ("signal+1", (0, 1, 0)),
        ("momentum-1", (0, 0, -1)), ("momentum+1", (0, 0, 1)),
    ]:
        candidate = tuple(best_tuple[i] + offset[i] for i in range(3))
        if candidate not in by_key.index:
            continue
        row = by_key.loc[candidate]
        if isinstance(row, pd.DataFrame):
            raise SystemExit(f"chave duplicada inesperada na vizinhanca: {candidate}")
        axis_neighbors.append({
            "direction": label,
            "gap_period": candidate[0],
            "signal_period": candidate[1],
            "momentum_period": candidate[2],
            "total_return": float(row["total_return"]),
            "final_equity": float(row["final_equity"]),
            "winner_wealth_multiple_vs_neighbor": (
                float(best.final_equity) / float(row["final_equity"])
                if float(row["final_equity"]) > 0 else None
            ),
        })

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
        "status": "ANALYSIS_COMPLETE",
        "schema_version": 2,
        "rows": int(len(frame)),
        "warning": "in-sample diagnostic only; does not establish strategy robustness and does not replace OOS/walk-forward",
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
        "one_step_axis_neighbors": axis_neighbors,
        "top_0_1pct_parameter_concentration": concentration,
        "strategy_robustness_verdict": "NOT_ESTABLISHED_BY_IN_SAMPLE_ANALYSIS",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Robustez in-sample do landscape de parametros",
        "",
        "Esta analise mede estabilidade local e distribuicao da grade; **nao comprova robustez da estrategia e nao substitui OOS/walk-forward**.",
        "",
        f"- Vencedora: G{int(best.gap_period)}/S{int(best.signal_period)}/M{int(best.momentum_period)}",
        f"- Retorno da vencedora: {best_return * 100:.2f}%",
        f"- Mediana de toda a grade: {float(np.median(returns)) * 100:.2f}%",
        f"- Percentil 99: {q99 * 100:.2f}%",
        f"- Percentil 99,9: {float(np.quantile(returns, 0.999)) * 100:.2f}%",
        "",
        f"## Vizinhanca ±{r} em Gap/Signal/Momentum",
        "",
        f"- Combinacoes na vizinhanca: {len(neighborhood)}",
        f"- Retorno mediano local: {local_median * 100:.2f}%",
        f"- Retorno medio local: {float(np.mean(neighbor_returns)) * 100:.2f}%",
        f"- Pior retorno local: {float(np.min(neighbor_returns)) * 100:.2f}%",
        f"- Melhor retorno local: {float(np.max(neighbor_returns)) * 100:.2f}%",
        f"- Parcela local acima do percentil 99 global: {local['share_at_or_above_global_99pct'] * 100:.2f}%",
        "",
        "## Vizinhos imediatos por eixo",
        "",
    ]
    if axis_neighbors:
        for item in axis_neighbors:
            lines.append(
                f"- {item['direction']}: G{item['gap_period']}/S{item['signal_period']}/M{item['momentum_period']} "
                f"→ {item['total_return'] * 100:.2f}%"
            )
    else:
        lines.append("- Nenhum vizinho de um passo existe dentro da grade configurada.")
    args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
