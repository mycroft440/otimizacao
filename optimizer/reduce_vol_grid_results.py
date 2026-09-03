#!/usr/bin/env python3
"""Reduz shards da grade GAP x Signal x Momentum x Vol sem carregar 66M linhas.

Cada shard já provou e contou sua partição completa e reteve Top-K local. Como o
Top-K global nunca pode conter mais de K elementos de um único shard, unir os
Top-K locais preserva exatamente o Top-K global e, em particular, a vencedora.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config
import optimize_b3_pine as opt
import reduce_results as base

SHARD_RE = re.compile(r"^shard_(\d+)\.csv$")


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
    p.add_argument("--vol-min", type=int, default=config.DEFAULT_VOL_MIN)
    p.add_argument("--vol-max", type=int, default=config.DEFAULT_VOL_MAX)
    p.add_argument("--expected-shards", type=int, default=config.DEFAULT_SHARDS)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--initial-cash", type=float, default=config.DEFAULT_INITIAL_CASH)
    p.add_argument("--fee-bps", type=float, default=config.DEFAULT_FEE_BPS)
    p.add_argument("--slippage-bps", type=float, default=config.DEFAULT_SLIPPAGE_BPS)
    p.add_argument("--odd-lot-extra-bps", type=float, default=config.DEFAULT_ODD_LOT_EXTRA_BPS)
    p.add_argument("--upstream-sha", default="")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _close(a: object, b: float) -> bool:
    try:
        value = float(a)
    except Exception:
        return False
    return math.isfinite(value) and math.isclose(value, float(b), rel_tol=0.0, abs_tol=1e-12)


def _audit_shards(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    csvs = sorted(args.results_dir.rglob("shard_*.csv"))
    if len(csvs) != args.expected_shards:
        raise SystemExit(f"SHARD AUDIT FAIL: {len(csvs)} CSVs != {args.expected_shards}")

    signal_count = args.signal_max - args.signal_min + 1
    momentum_count = args.momentum_max - args.momentum_min + 1
    vol_count = args.vol_max - args.vol_min + 1
    expected_total = (
        (args.gap_max - args.gap_min + 1) * signal_count * momentum_count * vol_count
    )

    gap_frequency = {g: 0 for g in range(args.gap_min, args.gap_max + 1)}
    seen_shards: set[int] = set()
    candidates: list[pd.DataFrame] = []
    reports: list[dict[str, object]] = []
    total_tested = 0
    provenance: dict[str, str] | None = None

    for csv_path in csvs:
        match = SHARD_RE.match(csv_path.name)
        if not match:
            raise SystemExit(f"SHARD AUDIT FAIL: nome inválido {csv_path.name}")
        shard = int(match.group(1))
        if shard in seen_shards:
            raise SystemExit(f"SHARD AUDIT FAIL: shard duplicado {shard}")
        seen_shards.add(shard)

        meta_path = csv_path.with_suffix(".json")
        if not meta_path.exists():
            raise SystemExit(f"SHARD AUDIT FAIL: metadata ausente para {csv_path.name}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(csv_path)

        if int(meta.get("schema_version", 0)) != 4:
            raise SystemExit(f"SHARD AUDIT FAIL: schema inesperado no shard {shard}")
        if int(meta.get("shard", -1)) != shard or int(meta.get("shards", -1)) != args.expected_shards:
            raise SystemExit(f"SHARD AUDIT FAIL: identidade do shard {shard} divergente")
        if int(meta.get("vol_min", -1)) != args.vol_min or int(meta.get("vol_max", -1)) != args.vol_max:
            raise SystemExit(f"SHARD AUDIT FAIL: faixa VOL divergente no shard {shard}")
        if int(meta.get("signal_min", -1)) != args.signal_min or int(meta.get("signal_max", -1)) != args.signal_max:
            raise SystemExit(f"SHARD AUDIT FAIL: faixa Signal divergente no shard {shard}")
        if int(meta.get("momentum_min", -1)) != args.momentum_min or int(meta.get("momentum_max", -1)) != args.momentum_max:
            raise SystemExit(f"SHARD AUDIT FAIL: faixa Momentum divergente no shard {shard}")
        if int(meta.get("top_k", -1)) < args.top_k:
            raise SystemExit(f"SHARD AUDIT FAIL: Top-K local insuficiente no shard {shard}")
        if int(meta.get("output_rows", -1)) != len(frame):
            raise SystemExit(f"SHARD AUDIT FAIL: output_rows divergente no shard {shard}")
        if str(meta.get("csv_sha256", "")) != sha256(csv_path):
            raise SystemExit(f"SHARD AUDIT FAIL: hash CSV divergente no shard {shard}")

        gaps = [int(x) for x in meta.get("gap_values", [])]
        expected_local = len(gaps) * signal_count * momentum_count * vol_count
        tested_local = int(meta.get("tested_rows", -1))
        if tested_local != expected_local:
            raise SystemExit(
                f"SHARD AUDIT FAIL: tested_rows={tested_local} expected={expected_local} shard={shard}"
            )
        total_tested += tested_local
        for gap in gaps:
            if gap not in gap_frequency:
                raise SystemExit(f"SHARD AUDIT FAIL: GAP fora da grade {gap}")
            gap_frequency[gap] += 1

        if frame.empty:
            raise SystemExit(f"SHARD AUDIT FAIL: Top-K vazio no shard {shard}")
        required = {
            "gap_period", "signal_period", "momentum_period", "vol_period", "final_equity",
            "total_return", "trades", "skipped_executions", "fees_paid", "slippage_impact",
            "final_holding", "start", "end", "shard",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise SystemExit(f"SHARD AUDIT FAIL: colunas ausentes {missing} no shard {shard}")
        if set(pd.to_numeric(frame["shard"], errors="raise").astype(int)) != {shard}:
            raise SystemExit(f"SHARD AUDIT FAIL: coluna shard divergente em {shard}")
        if not set(pd.to_numeric(frame["gap_period"], errors="raise").astype(int)).issubset(set(gaps)):
            raise SystemExit(f"SHARD AUDIT FAIL: GAP candidato fora da partição {shard}")
        vol = pd.to_numeric(frame["vol_period"], errors="raise").astype(int)
        if ((vol < args.vol_min) | (vol > args.vol_max)).any():
            raise SystemExit(f"SHARD AUDIT FAIL: VOL candidato fora da faixa {shard}")
        finite = frame[["final_equity", "total_return", "fees_paid", "slippage_impact"]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(finite)):
            raise SystemExit(f"SHARD AUDIT FAIL: resultado não finito no shard {shard}")
        if (finite[:, 0] < 0).any() or (finite[:, 2:] < 0).any():
            raise SystemExit(f"SHARD AUDIT FAIL: resultado financeiro negativo inválido no shard {shard}")

        for key, expected in (
            ("initial_cash", args.initial_cash),
            ("fee_bps", args.fee_bps),
            ("slippage_bps", args.slippage_bps),
            ("odd_lot_extra_bps", args.odd_lot_extra_bps),
        ):
            if not _close(meta.get(key), expected):
                raise SystemExit(f"SHARD AUDIT FAIL: {key} divergente no shard {shard}")

        current_provenance = {
            "github_run_id": str(meta.get("github_run_id") or ""),
            "optimizer_sha": str(meta.get("optimizer_sha") or ""),
            "github_repository": str(meta.get("github_repository") or ""),
            "snapshot_upstream_sha": str(meta.get("snapshot_upstream_sha") or ""),
            "snapshot_universe_sha256": str(meta.get("snapshot_universe_sha256") or ""),
            "snapshot_requested_end": str(meta.get("snapshot_requested_end") or ""),
        }
        if provenance is None:
            provenance = current_provenance
        elif provenance != current_provenance:
            raise SystemExit(f"SHARD AUDIT FAIL: proveniência divergente no shard {shard}")

        candidates.append(frame)
        reports.append({"shard": shard, "gap_values": gaps, "tested_rows": tested_local, "retained": len(frame)})

    if seen_shards != set(range(args.expected_shards)):
        raise SystemExit("SHARD AUDIT FAIL: conjunto de IDs de shard incompleto")
    bad_gaps = {g: count for g, count in gap_frequency.items() if count != 1}
    if bad_gaps:
        raise SystemExit(f"SHARD AUDIT FAIL: cobertura GAP não bijetiva: {bad_gaps}")
    if total_tested != expected_total:
        raise SystemExit(f"GRID AUDIT FAIL: tested={total_tested} expected={expected_total}")

    merged = pd.concat(candidates, ignore_index=True)
    merged = merged.sort_values(
        ["final_equity", "gap_period", "signal_period", "momentum_period", "vol_period"],
        ascending=[False, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    top = merged.head(args.top_k).copy()
    audit = {
        "status": "PASS",
        "schema_version": 1,
        "mode": "exhaustive_4d_streaming_topk",
        "gap_period": [args.gap_min, args.gap_max, 1],
        "signal_period": [args.signal_min, args.signal_max, 1],
        "momentum_period": [args.momentum_min, args.momentum_max, 1],
        "vol_period": [args.vol_min, args.vol_max, 1],
        "expected_combinations": expected_total,
        "tested_combinations": total_tested,
        "exact_partition_coverage": True,
        "global_top_k_exact": True,
        "top_k": args.top_k,
        "shards": reports,
        "provenance": provenance,
    }
    return top, audit


def main() -> None:
    args = parse_args()
    if args.gap_max < args.gap_min or args.signal_max < args.signal_min or args.momentum_max < args.momentum_min or args.vol_max < args.vol_min:
        raise SystemExit("faixa invertida")
    if args.expected_shards <= 0 or args.top_k <= 0:
        raise SystemExit("expected-shards e top-k precisam ser > 0")
    config.validate_periods(args.gap_min, args.signal_min, args.momentum_min, args.vol_min)
    config.validate_periods(args.gap_max, args.signal_max, args.momentum_max, args.vol_max)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    top, grid_audit = _audit_shards(args)
    top.to_csv(args.output_dir / "top_100.csv", index=False, float_format="%.12f")
    (args.output_dir / "EXHAUSTIVE_GRID_AUDIT.json").write_text(
        json.dumps(grid_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    best_row = top.iloc[0]
    market = opt.load_market(args.data_root, args.start, args.end)
    best, best_curve = base.detailed_backtest(
        market,
        int(best_row.gap_period),
        int(best_row.signal_period),
        int(best_row.momentum_period),
        int(best_row.vol_period),
        initial_cash=args.initial_cash,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        odd_lot_extra_bps=args.odd_lot_extra_bps,
    )
    reconciliation = base.reconcile_fast_vs_detailed(best_row, best)
    best_curve.to_csv(args.output_dir / "BEST_EQUITY_DAILY.csv", index=False, float_format="%.12f")
    (args.output_dir / "PORTFOLIO_RECONCILIATION.json").write_text(
        json.dumps(reconciliation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    canonical = None
    if (
        args.gap_min <= 40 <= args.gap_max
        and args.signal_min <= 20 <= args.signal_max
        and args.momentum_min <= 63 <= args.momentum_max
        and args.vol_min <= 21 <= args.vol_max
    ):
        canonical, _ = base.detailed_backtest(
            market,
            40,
            20,
            63,
            21,
            initial_cash=args.initial_cash,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            odd_lot_extra_bps=args.odd_lot_extra_bps,
        )

    input_hashes: dict[str, str] = {}
    universe_path = args.data_root / "data" / "universes" / "fixed_40_2018.json"
    input_hashes[str(universe_path.relative_to(args.data_root))] = base.sha256_file(universe_path)
    for ticker in market.tickers:
        path = args.data_root / "data" / "candles" / f"{ticker.lower()}_1d.csv"
        input_hashes[str(path.relative_to(args.data_root))] = base.sha256_file(path)

    manifest = {
        "status": "IN_SAMPLE_EXHAUSTIVE_SUCCESS",
        "strategy": "B3 Pine Gap Momentum + positive momentum + volatility gate + Top1 weekly",
        "upstream_repository": "mycroft440/b3-strategy-lab",
        "upstream_sha": args.upstream_sha,
        "universe_count": len(market.tickers),
        "universe": market.tickers,
        "search_space": {
            "gap_period": [args.gap_min, args.gap_max, 1],
            "signal_period": [args.signal_min, args.signal_max, 1],
            "momentum_period": [args.momentum_min, args.momentum_max, 1],
            "vol_period": [args.vol_min, args.vol_max, 1],
            "expected_combinations": grid_audit["expected_combinations"],
            "tested_unique_combinations": grid_audit["tested_combinations"],
            "storage_mode": "streaming_exact_top_k",
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
        "portfolio_reconciliation": reconciliation,
        "grid_audit": grid_audit,
        "best": best,
        "canonical_40_20_63_21": canonical,
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
        "# Otimização exaustiva 4D — B3 Estratégia Live",
        "",
        "**Status:** `IN_SAMPLE_EXHAUSTIVE_SUCCESS`",
        f"**Combinações testadas:** {int(grid_audit['tested_combinations']):,}".replace(",", "."),
        f"**Período:** {market.start} até {market.end}",
        f"**Grade:** GAP {args.gap_min}–{args.gap_max}; Signal {args.signal_min}–{args.signal_max}; Momentum {args.momentum_min}–{args.momentum_max}; Vol {args.vol_min}–{args.vol_max}.",
        "**Cobertura da grade:** PASS — partições determinísticas, sem sobreposição de GAP e contagem exata por shard.",
        "**Top-100 global:** exato a partir dos Top-100 locais de cada shard.",
        "**Reconciliação motor rápido × replay detalhado:** PASS.",
        "",
        "## Melhor combinação",
        "",
        f"- GAP_PERIOD: **{best['gap_period']}**",
        f"- SIGNAL_PERIOD: **{best['signal_period']}**",
        f"- MOMENTUM_PERIOD: **{best['momentum_period']}**",
        f"- VOL_PERIOD: **{best['vol_period']}**",
        f"- Capital inicial: **R$ {float(best['initial_cash']):.2f}**",
        f"- Patrimônio final: **R$ {float(best['final_equity']):.2f}**",
        f"- Retorno total: **{base.fmt_pct(float(best['total_return']))}**",
        f"- CAGR: **{base.fmt_pct(float(best['cagr']))}**",
        f"- Max drawdown: **{base.fmt_pct(float(best['max_drawdown_close_to_close']))}**",
        f"- Sharpe rf=0: **{float(best['sharpe_rf0_close_to_close']):.4f}**" if math.isfinite(float(best["sharpe_rf0_close_to_close"])) else "- Sharpe rf=0: **N/D**",
        f"- Trades: **{best['trades']}**",
        f"- Execuções puladas: **{best['skipped_executions']}**",
    ]
    if improvement is not None:
        lines += [
            "",
            "## Referência 40/20/63/21",
            "",
            f"- Patrimônio final: **R$ {float(canonical['final_equity']):.2f}**",
            f"- Vantagem patrimonial da vencedora: **{base.fmt_pct(float(improvement))}**",
        ]
    lines += [
        "",
        "## Observação",
        "",
        "Para evitar materializar mais de 66 milhões de linhas, cada shard testa sua partição inteira e persiste somente seu Top-K. A vencedora e o Top-K global permanecem exatos; o arquivo gigante de todos os resultados deixa de ser necessário.",
        "",
        "**Atenção:** a validação estatística continua in-sample; OOS/walk-forward é separada.",
    ]
    (args.output_dir / "OPTIMIZATION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
