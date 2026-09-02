#!/usr/bin/env python3
"""Testes determinísticos do gerenciamento de carteira do otimizador Pine V17."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import optimize_b3_pine as opt


def market(tickers: list[str], opens: list[list[float]], finals: list[float]) -> opt.MarketData:
    weeks = len(opens)
    execution_dates = pd.DatetimeIndex(pd.date_range("2020-01-06", periods=weeks, freq="7D"))
    decision_dates = execution_dates - pd.Timedelta(days=3)
    frames = {}
    for i, ticker in enumerate(tickers):
        dates = sorted(set(execution_dates.tolist() + decision_dates.tolist()))
        frames[ticker] = pd.DataFrame(
            {
                "date": dates,
                "open": [float(finals[i])] * len(dates),
                "close": [float(finals[i])] * len(dates),
            }
        )
    return opt.MarketData(
        tickers=tickers,
        frames=frames,
        master_dates=pd.DatetimeIndex(sorted(set(execution_dates.tolist() + decision_dates.tolist()))),
        execution_dates=execution_dates,
        decision_dates=decision_dates,
        exec_open=np.asarray(opens, dtype=np.float64),
        decision_index=np.zeros((len(tickers), weeks), dtype=np.int32),
        final_close=np.asarray(finals, dtype=np.float64),
        start="2020-01-06",
        end=execution_dates[-1].date().isoformat(),
    )


def simulate(
    targets: list[int],
    opens: list[list[float]],
    finals: list[float],
    *,
    mom: list[list[float]] | None = None,
    initial_cash: float = 100.0,
) -> dict[str, np.ndarray]:
    n_assets = len(finals)
    w = len(targets)
    mkt = market([f"T{i}" for i in range(n_assets)], opens, finals)
    target_array = np.asarray([targets], dtype=np.int16)
    gap_state = np.ones((1, w, n_assets), dtype=np.bool_)
    vol_valid = np.ones((w, n_assets), dtype=np.bool_)
    if mom is None:
        mom_array = np.tile(np.arange(1, n_assets + 1, dtype=np.float64), (w, 1)) / 10.0
    else:
        mom_array = np.asarray(mom, dtype=np.float64)
    return opt.simulate_pairs(
        target_array,
        gap_state,
        mom_array,
        vol_valid,
        mkt,
        initial_cash=initial_cash,
        fee_bps=0.0,
        slippage_bps=0.0,
        odd_lot_extra_bps=0.0,
    )


def scalar(result: dict[str, np.ndarray], key: str):
    return result[key][0].item()


def main() -> None:
    checks: dict[str, bool] = {}

    # 1) Mesmo Top1: preço cai e o caixa residual passa a comprar 1 ação, mas V17
    # exige manutenção integral. Quantidade deve continuar 1 e caixa R$40.
    r = simulate([0, 0], [[60.0], [40.0]], [40.0])
    checks["same_target_no_residual_reinvestment"] = (
        scalar(r, "shares") == 1
        and abs(float(scalar(r, "cash")) - 40.0) < 1e-12
        and scalar(r, "trades") == 1
        and scalar(r, "holding") == 0
    )

    # 2) Troca normal: compra A, vende A na próxima semana e compra B.
    r = simulate([0, 1], [[40.0, 30.0], [50.0, 30.0]], [50.0, 30.0])
    checks["atomic_switch_sell_then_buy"] = (
        scalar(r, "holding") == 1
        and scalar(r, "shares") == 4
        and abs(float(scalar(r, "cash"))) < 1e-12
        and scalar(r, "trades") == 3
        and scalar(r, "skipped") == 0
    )

    # 3) Nova compra impossível: a troca inteira falha e a posição antiga permanece.
    r = simulate([0, 1], [[40.0, 30.0], [50.0, 1000.0]], [50.0, 1000.0])
    checks["failed_switch_keeps_previous_portfolio"] = (
        scalar(r, "holding") == 0
        and scalar(r, "shares") == 2
        and abs(float(scalar(r, "cash")) - 20.0) < 1e-12
        and scalar(r, "trades") == 1
        and scalar(r, "skipped") == 1
    )

    # 4) Alvo vira caixa: vende o Top1 e não abre nova posição.
    r = simulate([0, -1], [[40.0], [50.0]], [50.0])
    checks["cash_target_liquidates_position"] = (
        scalar(r, "holding") == -1
        and scalar(r, "shares") == 0
        and abs(float(scalar(r, "cash")) - 120.0) < 1e-12
        and scalar(r, "trades") == 2
    )

    # 5) Empate exato: líder base seria T0, mas incumbente T1 tem mesmo Momentum e
    # continua elegível. Deve manter T1 sem giro.
    r = simulate(
        [1, 0],
        [[1000.0, 50.0], [50.0, 50.0]],
        [50.0, 50.0],
        mom=[[0.10, 0.20], [0.20, 0.20]],
    )
    checks["exact_tie_preserves_eligible_incumbent"] = (
        scalar(r, "holding") == 1
        and scalar(r, "shares") == 2
        and scalar(r, "trades") == 1
        and scalar(r, "skipped") == 0
    )

    # 6) Precisão: o array de Momentum deve ser float64 para não alterar Top1 por
    # quantização float32. Esta checagem usa o contrato explícito do código.
    tiny = np.array([[[1.00000001, 1.00000002]]], dtype=np.float64)
    g = np.ones((1, 1, 2), dtype=np.bool_)
    v = np.ones((1, 2), dtype=np.bool_)
    target = opt.first_ranked_targets(g, tiny[0], v)
    checks["float64_ranking_distinguishes_close_scores"] = int(target[0, 0]) == 1

    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise SystemExit("PORTFOLIO AUDIT FAIL: " + ", ".join(failed))

    payload = {
        "status": "PASS",
        "policy": "pine_v17_hold_same_target_no_residual_reinvestment",
        "checks": checks,
    }
    out = Path("portfolio_management_audit.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
