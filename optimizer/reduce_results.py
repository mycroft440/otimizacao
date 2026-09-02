#!/usr/bin/env python3
"""Une shards, audita cardinalidade e recalcula a melhor configuração diariamente."""
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
import optimize_b3_pine as opt  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--start", default="2018-01-02")
    p.add_argument("--end", default="")
    p.add_argument("--gap-min", type=int, default=5)
    p.add_argument("--gap-max", type=int, default=80)
    p.add_argument("--signal-min", type=int, default=2)
    p.add_argument("--signal-max", type=int, default=60)
    p.add_argument("--momentum-min", type=int, default=5)
    p.add_argument("--momentum-max", type=int, default=252)
    p.add_argument("--vol-period", type=int, default=21)
    p.add_argument("--initial-cash", type=float, default=1000.0)
    p.add_argument("--fee-bps", type=float, default=3.0)
    p.add_argument("--slippage-bps", type=float, default=10.0)
    p.add_argument("--odd-lot-extra-bps", type=float, default=5.0)
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
    return int(opt.affordable_qty(np.array([cash]), np.array([raw]), fee, base, extra)[0])


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
    pairs, gap_state, momentum, vol_valid = opt.precompute_shard(
        market, [gap_period], [signal_period], [momentum_period], vol_period
    )
    mom = momentum[0].astype(np.float64)
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

            # Preserva incumbente em empate exato quando continua elegível.
            if holding >= 0 and target >= 0 and holding != target:
                inc_m = float(mom[w, holding])
                top_m = float(mom[w, target])
                inc_ok = bool(gap_state[0, w, holding] and vol_valid[w, holding] and math.isfinite(inc_m) and inc_m > 0)
                if inc_ok and math.isfinite(top_m) and inc_m == top_m:
                    target = holding

            target_for_day = target
            if target != holding:
                projected = cash
                valid = True
                if holding >= 0:
                    raw_sell = float(market.exec_open[w, holding])
                    if not math.isfinite(raw_sell) or raw_sell <= 0:
                        valid = False
                    else:
                        odd = shares % 100
                        slip = raw_sell * (base * shares + extra * odd)
                        gross = raw_sell * shares - slip
                        projected += gross - gross * fee
                if target >= 0:
                    raw_buy = float(market.exec_open[w, target])
                    if not math.isfinite(raw_buy) or raw_buy <= 0:
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
                        gross = raw_buy * ((1 + base) * q + extra * odd)
                        f = gross * fee
                        slip = raw_buy * (base * q + extra * odd)
                        cash -= gross + f
                        cash = max(0.0, cash)
                        shares = q
                        holding = target
                        fees_paid += f
                        slippage_paid += slip
                        trades += int(q > 0)
                else:
                    skipped += 1
            elif holding >= 0:
                # Compra complementar de caixa residual, sem vender/recomprar a posição.
                raw_buy = float(market.exec_open[w, holding])
                if math.isfinite(raw_buy) and raw_buy > 0:
                    q = scalar_affordable(cash, raw_buy, fee, base, extra)
                    if q > 0:
                        odd = q % 100
                        gross = raw_buy * ((1 + base) * q + extra * odd)
                        f = gross * fee
                        slip = raw_buy * (base * q + extra * odd)
                        cash -= gross + f
                        cash = max(0.0, cash)
                        shares += q
                        fees_paid += f
                        slippage_paid += slip
                        trades += 1

        values = close_matrix[di]
        ok = np.isfinite(values) & (values > 0)
        last_close[ok] = values[ok]
        equity = cash
        if holding >= 0 and math.isfinite(last_close[holding]):
            equity += shares * float(last_close[holding])
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
    eq = curve["equity"].to_numpy(dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    daily = pd.Series(eq).pct_change().dropna().to_numpy(dtype=float)
    elapsed_years = (pd.Timestamp(curve.iloc[-1]["date"]) - pd.Timestamp(curve.iloc[0]["date"])).days / 365.2425
    final = float(eq[-1])
    total_return = final / initial_cash - 1.0
    cagr = (final / initial_cash) ** (1.0 / elapsed_years) - 1.0 if elapsed_years > 0 and final > 0 else float("nan")
    vol = float(np.std(daily, ddof=1) * math.sqrt(252.0)) if len(daily) > 1 else float("nan")
    sharpe = float(np.mean(daily) / np.std(daily, ddof=1) * math.sqrt(252.0)) if len(daily) > 1 and np.std(daily, ddof=1) > 0 else float("nan")

    # Retornos por ano usando o último patrimônio observado de cada ano.
    year_end = curve.assign(year=pd.to_datetime(curve["date"]).dt.year).groupby("year", sort=True)["equity"].last()
    annual_returns = year_end.pct_change()
    if len(year_end):
        first_year = int(year_end.index[0])
        first_base = initial_cash
        annual_returns.loc[first_year] = year_end.loc[first_year] / first_base - 1.0
    avg_annual = float(annual_returns.mean()) if len(annual_returns.dropna()) else float("nan")

    summary = {
        "gap_period": gap_period,
        "signal_period": signal_period,
        "momentum_period": momentum_period,
        "vol_period": vol_period,
        "start": str(curve.iloc[0]["date"]),
        "end": str(curve.iloc[-1]["date"]),
        "initial_cash": initial_cash,
        "final_equity": final,
        "profit": final - initial_cash,
        "total_return": total_return,
        "cagr": cagr,
        "average_annual_return": avg_annual,
        "max_drawdown": float(np.min(dd)),
        "annual_volatility": vol,
        "sharpe": sharpe,
        "trades": trades,
        "skipped_executions": skipped,
        "fees_paid": fees_paid,
        "slippage_impact": slippage_paid,
        "final_holding": market.tickers[holding] if holding >= 0 else "CASH",
    }
    return summary, curve


def fmt_pct(v: float) -> str:
    return "N/D" if not math.isfinite(v) else f"{v * 100:.2f}%"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csvs = sorted(args.results_dir.rglob("shard_*.csv"))
    if not csvs:
        raise SystemExit(f"Nenhum shard CSV em {args.results_dir}")
    frames = [pd.read_csv(path) for path in csvs]
    results = pd.concat(frames, ignore_index=True)

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
    results.to_csv(args.output_dir / "all_results.csv.gz", index=False, compression="gzip", float_format="%.12f")
    top100 = results.head(100).copy()
    top100.to_csv(args.output_dir / "top_100.csv", index=False, float_format="%.12f")

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
    best_curve.to_csv(args.output_dir / "BEST_EQUITY_DAILY.csv", index=False, float_format="%.12f")

    canonical = None
    if args.gap_min <= 40 <= args.gap_max and args.signal_min <= 20 <= args.signal_max and args.momentum_min <= 63 <= args.momentum_max:
        canonical, _ = detailed_backtest(
            market, 40, 20, 63, args.vol_period,
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
            "dividends_jcp": "excluded",
            "income_tax": "excluded",
            "leverage": 1.0,
        },
        "best": best,
        "canonical_40_20_63": canonical,
        "input_sha256": input_hashes,
        "warning": "A vencedora foi escolhida no mesmo período medido; resultado in-sample e sujeito a overfitting, viés de seleção e sobrevivência já declarados pelo universo upstream.",
    }
    (args.output_dir / "BEST.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    improvement = None
    if canonical:
        improvement = best["final_equity"] / canonical["final_equity"] - 1.0

    lines = [
        "# Otimização exaustiva — B3 Estratégia Live",
        "",
        f"**Status:** `IN_SAMPLE_EXHAUSTIVE_SUCCESS`",
        f"**Combinações testadas:** {unique:,}".replace(",", "."),
        f"**Período:** {market.start} até {market.end}",
        f"**Universo:** {len(market.tickers)} ações do `fixed_40_2018.json`",
        "",
        "## Melhor combinação por patrimônio final",
        "",
        f"- GAP_PERIOD: **{best['gap_period']}**",
        f"- SIGNAL_PERIOD: **{best['signal_period']}**",
        f"- MOMENTUM_PERIOD: **{best['momentum_period']}**",
        f"- VOL_PERIOD: **{best['vol_period']}** (fixo)",
        f"- Capital inicial: **R$ {best['initial_cash']:.2f}**",
        f"- Capital final: **R$ {best['final_equity']:.2f}**",
        f"- Lucro: **R$ {best['profit']:.2f}**",
        f"- Retorno total: **{fmt_pct(float(best['total_return']))}**",
        f"- CAGR: **{fmt_pct(float(best['cagr']))}**",
        f"- Retorno anual médio: **{fmt_pct(float(best['average_annual_return']))}**",
        f"- Max drawdown: **{fmt_pct(float(best['max_drawdown']))}**",
        f"- Volatilidade anual: **{fmt_pct(float(best['annual_volatility']))}**",
        f"- Sharpe (rf=0): **{best['sharpe']:.3f}**" if math.isfinite(float(best['sharpe'])) else "- Sharpe: **N/D**",
        f"- Trades: **{best['trades']}**",
        f"- Execuções puladas: **{best['skipped_executions']}**",
        f"- Custos operacionais pagos: **R$ {best['fees_paid']:.2f}**",
        f"- Impacto de slippage: **R$ {best['slippage_impact']:.2f}**",
        "",
    ]
    if canonical:
        lines += [
            "## Pine original 40/20/63",
            "",
            f"- Capital final: **R$ {canonical['final_equity']:.2f}**",
            f"- Retorno total: **{fmt_pct(float(canonical['total_return']))}**",
            f"- CAGR: **{fmt_pct(float(canonical['cagr']))}**",
            f"- Max drawdown: **{fmt_pct(float(canonical['max_drawdown']))}**",
            f"- Vantagem patrimonial da vencedora sobre 40/20/63: **{fmt_pct(float(improvement))}**",
            "",
        ]
    lines += [
        "## Metodologia",
        "",
        f"A busca foi exaustiva dentro de GAP {args.gap_min}–{args.gap_max}, Signal {args.signal_min}–{args.signal_max} e Momentum {args.momentum_min}–{args.momentum_max}, passo 1. O Vol21 e o Top1 semanal foram preservados para manter a família da estratégia Pine.",
        "",
        "A decisão usa apenas o fechamento da última sessão B3 da semana anterior e a execução usa a abertura da primeira sessão da semana seguinte. Não há look-ahead intencional.",
        "",
        "**Atenção:** a melhor combinação é in-sample. Ela é a mais lucrativa no histórico testado, não uma promessa de desempenho futuro.",
    ]
    (args.output_dir / "OPTIMIZATION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
