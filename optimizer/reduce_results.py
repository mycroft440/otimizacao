#!/usr/bin/env python3
"""Une shards, audita cardinalidade e recalcula a vencedora em replay diário."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import metrics  # noqa: E402
import optimize_b3_pine as opt  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--start", default=config.DEFAULT_START)
    p.add_argument("--end", default=config.DEFAULT_END)
    p.add_argument("--gap-min", type=int, default=config.DEFAULT_GAP_MIN)
    p.add_argument("--gap-max", type=int, default=config.DEFAULT_GAP_MAX)
    p.add_argument("--signal-min", type=int, default=config.DEFAULT_SIGNAL_MIN)
    p.add_argument("--signal-max", type=int, default=config.DEFAULT_SIGNAL_MAX)
    p.add_argument("--momentum-min", type=int, default=config.DEFAULT_MOMENTUM_MIN)
    p.add_argument("--momentum-max", type=int, default=config.DEFAULT_MOMENTUM_MAX)
    p.add_argument("--vol-period", type=int, default=config.DEFAULT_VOL_PERIOD)
    p.add_argument("--initial-cash", type=float, default=config.DEFAULT_INITIAL_CASH)
    p.add_argument("--fee-bps", type=float, default=config.DEFAULT_FEE_BPS)
    p.add_argument("--slippage-bps", type=float, default=config.DEFAULT_SLIPPAGE_BPS)
    p.add_argument("--odd-lot-extra-bps", type=float, default=config.DEFAULT_ODD_LOT_EXTRA_BPS)
    p.add_argument("--upstream-sha", default="")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scalar_affordable(cash: float, raw: float, fee: float, base: float, extra: float) -> int:
    if not math.isfinite(raw) or raw <= 0 or cash <= 0:
        return 0
    return int(
        opt.affordable_qty(
            np.array([cash], dtype=np.float64),
            np.array([raw], dtype=np.float64),
            fee,
            base,
            extra,
        )[0]
    )


def build_close_matrix(market: opt.MarketData) -> np.ndarray:
    D, N = len(market.master_dates), len(market.tickers)
    matrix = np.full((D, N), np.nan, dtype=np.float64)
    master_index = pd.Index(market.master_dates)
    for ti, ticker in enumerate(market.tickers):
        df = market.frames[ticker]
        rows = df[df["date"].isin(market.master_dates)]
        positions = master_index.get_indexer(rows["date"])
        ok = positions >= 0
        matrix[positions[ok], ti] = rows["close"].to_numpy(dtype=np.float64)[ok]
    return matrix


def detailed_backtest(
    market: opt.MarketData,
    gap_period: int,
    signal_period: int,
    momentum_period: int,
    vol_period: int,
    *,
    initial_cash: float,
    fee_bps: float,
    slippage_bps: float,
    odd_lot_extra_bps: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    config.validate_periods(gap_period, signal_period, momentum_period, vol_period)
    if not math.isfinite(float(initial_cash)) or initial_cash <= 0:
        raise ValueError("initial_cash precisa ser finito e > 0")
    for name, value in (
        ("fee_bps", fee_bps),
        ("slippage_bps", slippage_bps),
        ("odd_lot_extra_bps", odd_lot_extra_bps),
    ):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} precisa ser finito e >= 0")

    pairs, gap_state, momentum, vol_valid = opt.precompute_shard(
        market, [gap_period], [signal_period], [momentum_period], vol_period
    )
    if pairs != [(gap_period, signal_period)]:
        raise RuntimeError("precompute retornou par inesperado")
    mom = momentum[0]
    base_targets = opt.first_ranked_targets(gap_state, mom, vol_valid)[0]

    fee = fee_bps / 10000.0
    base = slippage_bps / 10000.0
    extra = odd_lot_extra_bps / 10000.0
    cash = float(initial_cash)
    shares = 0
    holding = -1
    trades = 0
    skipped = 0
    fees_paid = 0.0
    slippage_paid = 0.0

    close_matrix = build_close_matrix(market)
    last_close = np.full(len(market.tickers), np.nan, dtype=np.float64)
    exec_lookup = {pd.Timestamp(d): i for i, d in enumerate(market.execution_dates)}
    start_ts = pd.Timestamp(market.start)
    rows: list[dict[str, object]] = []

    for di, day in enumerate(market.master_dates):
        if day < start_ts:
            values = close_matrix[di]
            ok = np.isfinite(values) & (values > 0)
            last_close[ok] = values[ok]
            continue

        target_for_day = holding
        if day in exec_lookup:
            w = exec_lookup[day]
            target = int(base_targets[w])

            if holding >= 0 and target >= 0 and holding != target:
                inc_m = float(mom[w, holding])
                top_m = float(mom[w, target])
                inc_ok = bool(
                    gap_state[0, w, holding]
                    and vol_valid[w, holding]
                    and math.isfinite(inc_m)
                    and inc_m > 0.0
                )
                if inc_ok and math.isfinite(top_m) and inc_m == top_m:
                    target = holding

            target_for_day = target

            if target != holding:
                projected = cash
                valid = True

                if holding >= 0:
                    raw_sell = float(market.exec_open[w, holding])
                    if not math.isfinite(raw_sell) or raw_sell <= 0.0:
                        valid = False
                    else:
                        odd = shares % 100
                        slip = raw_sell * (base * shares + extra * odd)
                        gross = raw_sell * shares - slip
                        projected += gross - gross * fee

                if target >= 0:
                    raw_buy = float(market.exec_open[w, target])
                    if not math.isfinite(raw_buy) or raw_buy <= 0.0:
                        valid = False
                    elif scalar_affordable(projected, raw_buy, fee, base, extra) <= 0:
                        valid = False

                if valid:
                    if holding >= 0:
                        raw_sell = float(market.exec_open[w, holding])
                        odd = shares % 100
                        slip = raw_sell * (base * shares + extra * odd)
                        gross = raw_sell * shares - slip
                        f = gross * fee
                        cash += gross - f
                        fees_paid += f
                        slippage_paid += slip
                        shares = 0
                        holding = -1
                        trades += 1

                    if target >= 0:
                        raw_buy = float(market.exec_open[w, target])
                        q = scalar_affordable(cash, raw_buy, fee, base, extra)
                        odd = q % 100
                        gross = raw_buy * ((1.0 + base) * q + extra * odd)
                        f = gross * fee
                        slip = raw_buy * (base * q + extra * odd)
                        new_cash = cash - gross - f
                        if new_cash < -1e-7:
                            raise RuntimeError("replay detalhado gerou caixa negativo")
                        cash = max(0.0, new_cash)
                        shares = q
                        holding = target
                        fees_paid += f
                        slippage_paid += slip
                        trades += int(q > 0)
                else:
                    skipped += 1

        if cash < -1e-7 or shares < 0 or ((holding < 0) != (shares == 0)):
            raise RuntimeError("invariante de carteira violada no replay detalhado")

        values = close_matrix[di]
        ok = np.isfinite(values) & (values > 0)
        last_close[ok] = values[ok]
        equity = cash
        if holding >= 0:
            px = float(last_close[holding])
            if not math.isfinite(px) or px <= 0.0:
                raise RuntimeError("posição aberta sem preço de fechamento válido")
            equity += shares * px
        rows.append(
            {
                "date": day.date().isoformat(),
                "equity": equity,
                "cash": cash,
                "holding": market.tickers[holding] if holding >= 0 else "CASH",
                "shares": shares,
                "weekly_target": market.tickers[target_for_day] if target_for_day >= 0 else "CASH",
            }
        )

    curve = pd.DataFrame(rows)
    if curve.empty:
        raise RuntimeError("Curva detalhada vazia.")

    risk = metrics.risk_metrics(curve[["date", "equity"]], initial_cash)
    annual = metrics.annual_metrics(curve[["date", "equity"]], initial_cash)
    final = float(curve.iloc[-1]["equity"])
    annual_returns = {
        str(int(item["year"])): float(item["return"])
        for item in annual["years"]
    }

    summary: dict[str, object] = {
        "gap_period": gap_period,
        "signal_period": signal_period,
        "momentum_period": momentum_period,
        "vol_period": vol_period,
        "start": str(curve.iloc[0]["date"]),
        "end": str(curve.iloc[-1]["date"]),
        "initial_cash": initial_cash,
        "final_equity": final,
        "profit": final - initial_cash,
        "total_return": float(risk["total_return"]),
        "cagr": float(risk["cagr"]),
        "average_complete_year_return": float(annual["average_complete_year_return"]),
        "geometric_mean_complete_year_return": float(annual["geometric_mean_complete_year_return"]),
        "annual_returns": annual_returns,
        "annual_years": annual["years"],
        "max_drawdown": float(risk["max_drawdown_close_to_close"]),
        "max_drawdown_close_to_close": float(risk["max_drawdown_close_to_close"]),
        "annual_volatility": float(risk["annual_volatility_close_to_close"]),
        "annual_volatility_close_to_close": float(risk["annual_volatility_close_to_close"]),
        "sharpe": float(risk["sharpe_rf0_close_to_close"]),
        "sharpe_rf0_close_to_close": float(risk["sharpe_rf0_close_to_close"]),
        "calmar_close_to_close": float(risk["calmar_close_to_close"]),
        "trades": trades,
        "skipped_executions": skipped,
        "fees_paid": fees_paid,
        "slippage_impact": slippage_paid,
        "final_holding": market.tickers[holding] if holding >= 0 else "CASH",
        "portfolio_policy": "pine_v17_hold_same_target_no_residual_reinvestment",
        "metric_convention": {
            "risk_free_rate": 0.0,
            "annualization_sessions": 252,
            "drawdown": "daily close-to-close marked equity",
            "volatility": "sample standard deviation of daily close-to-close returns",
        },
    }
    return summary, curve


def _close_enough(a: float, b: float, *, rel: float = 1e-10, abs_: float = 1e-7) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


def reconcile_fast_vs_detailed(best_row: pd.Series, best: dict[str, object]) -> dict[str, object]:
    checks = {
        "final_equity": _close_enough(best_row.final_equity, float(best["final_equity"])),
        "fees_paid": _close_enough(best_row.fees_paid, float(best["fees_paid"])),
        "slippage_impact": _close_enough(best_row.slippage_impact, float(best["slippage_impact"])),
        "trades": int(best_row.trades) == int(best["trades"]),
        "skipped_executions": int(best_row.skipped_executions) == int(best["skipped_executions"]),
        "final_holding": str(best_row.final_holding) == str(best["final_holding"]),
    }
    if not all(checks.values()):
        raise SystemExit(f"PORTFOLIO RECONCILIATION FAIL: {checks}")
    return {"status": "PASS", "checks": checks}


def fmt_pct(v: float) -> str:
    return "N/D" if not math.isfinite(v) else f"{v * 100:.2f}%"


def main() -> None:
    args = parse_args()
    try:
        config.validate_run_config(
            start=args.start,
            end=args.end,
            gap_min=args.gap_min,
            gap_max=args.gap_max,
            signal_min=args.signal_min,
            signal_max=args.signal_max,
            momentum_min=args.momentum_min,
            momentum_max=args.momentum_max,
            vol_period=args.vol_period,
            initial_cash=args.initial_cash,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            odd_lot_extra_bps=args.odd_lot_extra_bps,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csvs = sorted(args.results_dir.rglob("shard_*.csv"))
    if not csvs:
        raise SystemExit(f"Nenhum shard CSV em {args.results_dir}")
    results = pd.concat([pd.read_csv(path) for path in csvs], ignore_index=True)

    key = ["gap_period", "signal_period", "momentum_period", "vol_period"]
    duplicates = int(results.duplicated(key).sum())
    expected = (
        (args.gap_max - args.gap_min + 1)
        * (args.signal_max - args.signal_min + 1)
        * (args.momentum_max - args.momentum_min + 1)
    )
    unique = int(len(results.drop_duplicates(key)))
    if duplicates or unique != expected:
        raise SystemExit(
            f"AUDIT FAIL: expected={expected} unique={unique} duplicates={duplicates} rows={len(results)}"
        )

    results = results.sort_values(
        ["final_equity", "gap_period", "signal_period", "momentum_period"],
        ascending=[False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    results.to_csv(
        args.output_dir / "all_results.csv.gz",
        index=False,
        compression="gzip",
        float_format="%.12f",
    )
    results.head(100).to_csv(
        args.output_dir / "top_100.csv", index=False, float_format="%.12f"
    )

    best_row = results.iloc[0]
    market = opt.load_market(args.data_root, args.start, args.end)
    best, best_curve = detailed_backtest(
        market,
        int(best_row.gap_period),
        int(best_row.signal_period),
        int(best_row.momentum_period),
        args.vol_period,
        initial_cash=args.initial_cash,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        odd_lot_extra_bps=args.odd_lot_extra_bps,
    )
    portfolio_reconciliation = reconcile_fast_vs_detailed(best_row, best)
    best_curve.to_csv(
        args.output_dir / "BEST_EQUITY_DAILY.csv", index=False, float_format="%.12f"
    )
    (args.output_dir / "PORTFOLIO_RECONCILIATION.json").write_text(
        json.dumps(portfolio_reconciliation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    canonical = None
    if (
        args.gap_min <= 40 <= args.gap_max
        and args.signal_min <= 20 <= args.signal_max
        and args.momentum_min <= 63 <= args.momentum_max
    ):
        canonical, _ = detailed_backtest(
            market,
            40,
            20,
            63,
            args.vol_period,
            initial_cash=args.initial_cash,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            odd_lot_extra_bps=args.odd_lot_extra_bps,
        )

    input_hashes: dict[str, str] = {}
    universe_path = args.data_root / "data" / "universes" / "fixed_40_2018.json"
    input_hashes[str(universe_path.relative_to(args.data_root))] = sha256_file(universe_path)
    for ticker in market.tickers:
        path = args.data_root / "data" / "candles" / f"{ticker.lower()}_1d.csv"
        input_hashes[str(path.relative_to(args.data_root))] = sha256_file(path)

    manifest = {
        "status": "IN_SAMPLE_EXHAUSTIVE_SUCCESS",
        "strategy": "B3 Pine Gap Momentum + positive momentum + Top1 weekly",
        "upstream_repository": "mycroft440/b3-strategy-lab",
        "upstream_sha": args.upstream_sha,
        "universe_count": len(market.tickers),
        "universe": market.tickers,
        "search_space": {
            "gap_period": [args.gap_min, args.gap_max, 1],
            "signal_period": [args.signal_min, args.signal_max, 1],
            "momentum_period": [args.momentum_min, args.momentum_max, 1],
            "vol_period": args.vol_period,
            "expected_combinations": expected,
            "tested_unique_combinations": unique,
        },
        "execution": {
            "start": market.start,
            "end": market.end,
            "initial_cash": args.initial_cash,
            "fee_bps_per_side": args.fee_bps,
            "slippage_bps_per_side": args.slippage_bps,
            "odd_lot_extra_bps_weighted": args.odd_lot_extra_bps,
            "top_n": 1,
            "rebalance": "weekly",
            "decision": "last B3 session close of previous week",
            "execution_price": "first B3 session open of next week",
            "same_target_policy": "hold_exactly_no_order_no_residual_reinvestment",
            "switch_policy": "atomic_preflight_sell_then_buy_or_keep_previous_portfolio",
            "momentum_precision": "float64",
            "dividends_jcp": "excluded",
            "income_tax": "excluded",
            "leverage": 1.0,
        },
        "metric_convention": best["metric_convention"],
        "portfolio_reconciliation": portfolio_reconciliation,
        "best": best,
        "canonical_40_20_63": canonical,
        "input_sha256": input_hashes,
        "warning": (
            "A vencedora foi escolhida no mesmo período medido; resultado in-sample e sujeito "
            "a overfitting. O universo fixo Pine também não é survivorship-safe."
        ),
    }
    (args.output_dir / "BEST.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    improvement = None
    if canonical:
        improvement = float(best["final_equity"]) / float(canonical["final_equity"]) - 1.0

    lines = [
        "# Otimização exaustiva — B3 Estratégia Live",
        "",
        "**Status:** `IN_SAMPLE_EXHAUSTIVE_SUCCESS`",
        f"**Combinações testadas:** {unique:,}".replace(",", "."),
        f"**Período solicitado:** {market.start} até {market.end}",
        f"**Curva observada:** {best['start']} até {best['end']}",
        f"**Universo:** {len(market.tickers)} ações",
        "**Carteira:** Top1 semanal; mesmo Top1 = manutenção integral, sem nova ordem.",
        "**Reconciliação motor rápido × replay detalhado:** PASS.",
        "",
        "## Melhor combinação por patrimônio final",
        "",
        f"- GAP_PERIOD: **{best['gap_period']}**",
        f"- SIGNAL_PERIOD: **{best['signal_period']}**",
        f"- MOMENTUM_PERIOD: **{best['momentum_period']}**",
        f"- VOL_PERIOD: **{best['vol_period']}** (fixo)",
        f"- Capital inicial: **R$ {float(best['initial_cash']):.2f}**",
        f"- Patrimônio mark-to-market final: **R$ {float(best['final_equity']):.2f}**",
        f"- Retorno total: **{fmt_pct(float(best['total_return']))}**",
        f"- CAGR: **{fmt_pct(float(best['cagr']))}**",
        f"- Média dos anos completos: **{fmt_pct(float(best['average_complete_year_return']))}**",
        f"- Max drawdown close-to-close: **{fmt_pct(float(best['max_drawdown_close_to_close']))}**",
        f"- Sharpe rf=0 close-to-close: **{float(best['sharpe_rf0_close_to_close']):.4f}**"
        if math.isfinite(float(best["sharpe_rf0_close_to_close"]))
        else "- Sharpe rf=0 close-to-close: **N/D**",
        f"- Trades: **{best['trades']}**",
        f"- Execuções puladas: **{best['skipped_executions']}**",
        f"- Taxas: **R$ {float(best['fees_paid']):.2f}**",
        f"- Slippage: **R$ {float(best['slippage_impact']):.2f}**",
        "",
        "### Retorno por ano",
        "",
    ]
    for item in best["annual_years"]:
        marker = "" if item["complete_year"] else " (parcial)"
        lines.append(f"- {item['year']}{marker}: **{fmt_pct(float(item['return']))}**")

    if canonical:
        lines += [
            "",
            "## Pine original 40/20/63",
            "",
            f"- Patrimônio final: **R$ {float(canonical['final_equity']):.2f}**",
            f"- Retorno total: **{fmt_pct(float(canonical['total_return']))}**",
            f"- CAGR: **{fmt_pct(float(canonical['cagr']))}**",
            f"- Vantagem patrimonial da vencedora: **{fmt_pct(float(improvement))}**",
        ]

    lines += [
        "",
        "## Metodologia",
        "",
        f"Busca exaustiva: GAP {args.gap_min}–{args.gap_max}, Signal {args.signal_min}–{args.signal_max} e Momentum {args.momentum_min}–{args.momentum_max}, passo 1; Vol{args.vol_period} fixo.",
        "",
        "O sinal usa o fechamento da última sessão da semana anterior e a execução usa a abertura da primeira sessão B3 seguinte. Quando o Top1 não muda, nenhuma ordem é criada. Quando muda, a operação é atômica: se venda ou nova compra não puder ser executada, a carteira anterior permanece intacta.",
        "",
        "Métricas de risco usam a curva diária marcada a fechamento; Sharpe usa taxa livre de risco zero e anualização por sqrt(252).",
        "",
        "**Atenção:** a vencedora continua sendo in-sample; a validação OOS/walk-forward é separada.",
    ]
    (args.output_dir / "OPTIMIZATION_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
