#!/usr/bin/env python3
"""Gera retornos por ano e separa anos completos de ano terminal parcial."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--curve", required=True, type=Path)
    p.add_argument("--initial-cash", required=True, type=float)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--json", required=True, type=Path)
    p.add_argument("--md", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.curve)
    if not {"date", "equity"}.issubset(df.columns):
        raise SystemExit("curve precisa conter date,equity")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["equity"] = pd.to_numeric(df["equity"], errors="raise")
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    if df.empty or args.initial_cash <= 0:
        raise SystemExit("curva vazia ou capital inicial invalido")

    rows = []
    prior_equity = float(args.initial_cash)
    years = sorted(df["date"].dt.year.unique())
    terminal_date = df["date"].iloc[-1].date()
    for year in years:
        yd = df[df["date"].dt.year == year]
        start_equity = prior_equity
        end_equity = float(yd.iloc[-1]["equity"])
        ret = end_equity / start_equity - 1.0 if start_equity > 0 else float("nan")
        last_date = yd.iloc[-1]["date"].date()
        # O ultimo ano do dataset so e completo se chegar a 31/dez. Para anos anteriores,
        # o fechamento anual observado na ultima sessao B3 e suficiente.
        is_terminal = year == terminal_date.year
        complete = (not is_terminal) or (last_date.month == 12 and last_date.day >= 28)
        rows.append(
            {
                "year": int(year),
                "start_equity": start_equity,
                "end_equity": end_equity,
                "return": ret,
                "return_pct": ret * 100.0,
                "last_observation": last_date.isoformat(),
                "complete_year": bool(complete),
            }
        )
        prior_equity = end_equity

    out = pd.DataFrame(rows)
    complete_returns = out.loc[out["complete_year"], "return"].to_numpy(dtype=float)
    avg_complete = float(np.mean(complete_returns)) if len(complete_returns) else float("nan")
    geometric_complete = (
        float(np.prod(1.0 + complete_returns) ** (1.0 / len(complete_returns)) - 1.0)
        if len(complete_returns) and np.all(1.0 + complete_returns > 0)
        else float("nan")
    )
    total_return = float(df.iloc[-1]["equity"] / args.initial_cash - 1.0)

    payload = {
        "status": "PASS",
        "initial_cash": args.initial_cash,
        "final_equity": float(df.iloc[-1]["equity"]),
        "total_return": total_return,
        "total_return_pct": total_return * 100.0,
        "complete_years": int(out["complete_year"].sum()),
        "partial_terminal_year": bool(not out.iloc[-1]["complete_year"]),
        "average_complete_year_return": avg_complete,
        "average_complete_year_return_pct": avg_complete * 100.0,
        "geometric_mean_complete_year_return": geometric_complete,
        "geometric_mean_complete_year_return_pct": geometric_complete * 100.0,
        "years": rows,
    }

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.csv, index=False, float_format="%.12f")
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Retorno por ano — melhor combinacao",
        "",
        "| Ano | Retorno | Patrimonio final | Status |",
        "|---:|---:|---:|---|",
    ]
    for row in rows:
        status = "completo" if row["complete_year"] else "parcial"
        lines.append(
            f"| {row['year']} | {row['return_pct']:.2f}% | R$ {row['end_equity']:.2f} | {status} |"
        )
    lines += [
        "",
        f"**Retorno total:** {total_return * 100:.2f}%",
        f"**Media aritmetica somente dos anos completos:** {avg_complete * 100:.2f}%" if np.isfinite(avg_complete) else "**Media dos anos completos:** N/D",
        f"**Media geometrica somente dos anos completos:** {geometric_complete * 100:.2f}%" if np.isfinite(geometric_complete) else "**Media geometrica dos anos completos:** N/D",
        "",
        "O ano terminal parcial e exibido, mas nao e contado como um ano completo na media.",
    ]
    args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
