#!/usr/bin/env python3
"""Valida no holdout combinações escolhidas exclusivamente no período de treino."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optimize_b3_pine as opt  # noqa: E402
import reduce_results as red  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--training-top", required=True, type=Path)
    p.add_argument("--start", required=True)
    p.add_argument("--end", default="")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--initial-cash", type=float, default=1000.0)
    p.add_argument("--fee-bps", type=float, default=3.25)
    p.add_argument("--slippage-bps", type=float, default=10.0)
    p.add_argument("--odd-lot-extra-bps", type=float, default=5.0)
    return p.parse_args()


def pct(value: float) -> str:
    return "N/D" if not math.isfinite(value) else f"{value * 100:.2f}%"


def main() -> None:
    args = parse_args()
    if args.top <= 0:
        raise SystemExit("--top precisa ser positivo")
    source = pd.read_csv(args.training_top).head(args.top).copy()
    required = {"gap_period", "signal_period", "momentum_period", "vol_period"}
    if len(source) < args.top or not required.issubset(source.columns):
        raise SystemExit("training-top insuficiente ou schema invalido")

    market = opt.load_market(args.data_root, args.start, args.end)
    results = []
    curves = []
    for rank, row in enumerate(source.itertuples(index=False), start=1):
        g = int(row.gap_period)
        s = int(row.signal_period)
        m = int(row.momentum_period)
        v = int(row.vol_period)
        summary, curve = red.detailed_backtest(
            market, g, s, m, v,
            initial_cash=args.initial_cash,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            odd_lot_extra_bps=args.odd_lot_extra_bps,
        )
        summary = dict(summary)
        summary["training_rank"] = rank
        summary["selection_source"] = "training_only"
        summary["oos_start"] = args.start
        summary["oos_end"] = market.end
        results.append(summary)
        curve = curve.copy()
        curve.insert(0, "training_rank", rank)
        curve.insert(1, "gap_period", g)
        curve.insert(2, "signal_period", s)
        curve.insert(3, "momentum_period", m)
        curves.append(curve)

    frame = pd.DataFrame(results).sort_values("training_rank")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "OOS_TOP3.csv", index=False, float_format="%.12f")
    pd.concat(curves, ignore_index=True).to_csv(
        args.output_dir / "OOS_TOP3_EQUITY_DAILY.csv", index=False, float_format="%.12f"
    )

    payload = {
        "status": "PASS",
        "selection": "parameters selected only on training period; holdout never used for ranking",
        "oos_start": args.start,
        "oos_end": market.end,
        "top": args.top,
        "results": results,
    }
    (args.output_dir / "OOS_TOP3.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Validacao fora da amostra — Top 3 escolhidos no treino",
        "",
        f"Treino usado para escolher parametros: arquivo `{args.training_top.name}`.",
        f"Holdout: **{args.start} ate {market.end}**. O holdout nao participa do ranking de treino.",
        "",
        "| Rank treino | Gap | Signal | Momentum | Retorno OOS | CAGR OOS | Max DD OOS | Capital final |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['training_rank']} | {item['gap_period']} | {item['signal_period']} | {item['momentum_period']} | "
            f"{pct(float(item['total_return']))} | {pct(float(item['cagr']))} | {pct(float(item['max_drawdown']))} | "
            f"R$ {float(item['final_equity']):.2f} |"
        )
    lines += [
        "",
        "Estes resultados sao OOS em relacao a escolha dos parametros. O universo fixo do Pine continua sendo uma limitacao metodologica separada.",
    ]
    (args.output_dir / "OOS_TOP3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
