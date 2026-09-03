#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

import metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--best", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--curve", required=True, type=Path)
    p.add_argument("--summary", required=True, type=Path)
    p.add_argument("--fee-bps", required=True, type=float)
    p.add_argument("--slippage-bps", required=True, type=float)
    p.add_argument("--odd-lot-extra-bps", required=True, type=float)
    return p.parse_args()


def main():
    args = parse_args()
    best = json.loads(args.best.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    curve = pd.read_csv(args.curve)
    if curve.empty:
        raise SystemExit("curve vazia")
    initial = float(best["initial_cash"])
    annual = metrics.annual_metrics(curve[["date", "equity"]], initial)
    risk = metrics.risk_metrics(curve[["date", "equity"]], initial)

    last = curve.iloc[-1]
    mtm = float(last["equity"])
    cash = float(last["cash"])
    shares = int(last["shares"])
    holding = str(last["holding"])
    liquidation = mtm
    liquidation_cost = 0.0
    inferred_close = None
    if holding != "CASH" and shares > 0:
        inferred_close = (mtm - cash) / shares
        fee = args.fee_bps / 10000.0
        base = args.slippage_bps / 10000.0
        extra = args.odd_lot_extra_bps / 10000.0
        odd = shares % 100
        slip = inferred_close * (base * shares + extra * odd)
        gross_after_slip = inferred_close * shares - slip
        sale_fee = gross_after_slip * fee
        liquidation_cost = slip + sale_fee
        liquidation = cash + gross_after_slip - sale_fee

    best.update(
        {
            "schema_version": 2,
            "mark_to_market_equity": mtm,
            "liquidation_equity_estimate": liquidation,
            "liquidation_cost_estimate": liquidation_cost,
            "liquidation_return_estimate": liquidation / initial - 1.0,
            "terminal_mark_price_inferred": inferred_close,
            "average_complete_year_return": annual["average_complete_year_return"],
            "geometric_mean_complete_year_return": annual["geometric_mean_complete_year_return"],
            "annual_returns": {str(row["year"]): row["return"] for row in annual["years"]},
            "annual_coverage": annual["years"],
            "cagr": risk["cagr"],
            "max_drawdown": risk["max_drawdown_close_to_close"],
            "annual_volatility": risk["annual_volatility_close_to_close"],
            "sharpe": risk["sharpe_rf0_close_to_close"],
            "calmar": risk["calmar_close_to_close"],
            "risk_metric_conventions": {
                "drawdown": "daily close-to-close equity",
                "volatility": "sample std of daily close-to-close returns annualized by sqrt(252)",
                "sharpe": "daily close-to-close arithmetic mean / sample std, risk-free=0, annualized sqrt(252)",
                "calmar": "CAGR / abs(close-to-close max drawdown)",
            },
        }
    )
    execution = manifest.setdefault("execution", {})
    execution["terminal_valuation"] = {
        "primary": "mark_to_market_at_last_available_close",
        "liquidation_estimate_also_reported": True,
        "liquidation_estimate_uses_terminal_close_with_configured_fee_and_slippage": True,
    }
    manifest["metric_conventions"] = best["risk_metric_conventions"]

    args.best.write_text(json.dumps(best, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    original = args.summary.read_text(encoding="utf-8").rstrip()
    extra = [
        "",
        "## Auditoria endurecida de metricas e valor terminal",
        "",
        f"- Mark-to-market final: **R$ {mtm:.2f}**",
        f"- Liquidation equity estimada: **R$ {liquidation:.2f}**",
        f"- Custo estimado para liquidar a posicao terminal: **R$ {liquidation_cost:.2f}**",
        f"- Max drawdown reportado: **{float(risk['max_drawdown_close_to_close']) * 100:.2f}%**, medido close-to-close diario.",
        f"- Sharpe: **{float(risk['sharpe_rf0_close_to_close']):.4f}**, risk-free=0 e anualizacao sqrt(252)." if math.isfinite(float(risk['sharpe_rf0_close_to_close'])) else "- Sharpe: **N/D**.",
        "- Anos iniciais e terminais parciais nao entram na media de anos completos.",
    ]
    args.summary.write_text(original + "\n" + "\n".join(extra) + "\n", encoding="utf-8")
    print(json.dumps({
        "mark_to_market_equity": mtm,
        "liquidation_equity_estimate": liquidation,
        "liquidation_cost_estimate": liquidation_cost,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
