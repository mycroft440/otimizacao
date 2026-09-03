#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import optimize_b3_pine as opt  # noqa: E402
import reduce_results as red  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--training-top", required=True, type=Path)
    p.add_argument("--training-manifest", required=True, type=Path)
    p.add_argument("--start", required=True)
    p.add_argument("--end", default="")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--initial-cash", type=float, default=config.DEFAULT_INITIAL_CASH)
    p.add_argument("--fee-bps", type=float, default=config.DEFAULT_FEE_BPS)
    p.add_argument("--slippage-bps", type=float, default=config.DEFAULT_SLIPPAGE_BPS)
    p.add_argument("--odd-lot-extra-bps", type=float, default=config.DEFAULT_ODD_LOT_EXTRA_BPS)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pct(value: float) -> str:
    return "N/D" if not math.isfinite(value) else f"{value * 100:.2f}%"


def verify_training_source(top_path: Path, manifest_path: Path, oos_start: str) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    execution = manifest.get("execution") or {}
    training = manifest.get("training") or {}
    training_start = str(training.get("start") or execution.get("start") or "")
    training_end = str(training.get("end") or execution.get("end") or "")
    if not training_start or not training_end:
        raise SystemExit("training manifest sem start/end verificaveis")
    if pd.Timestamp(training_end) >= pd.Timestamp(oos_start):
        raise SystemExit(
            f"OOS LEAKAGE: training_end={training_end} precisa ser anterior a oos_start={oos_start}"
        )

    selection = manifest.get("selection_provenance") or {}
    if selection.get("mode") != "training_only":
        raise SystemExit("training manifest nao prova selection_provenance.mode=training_only")
    expected_hash = str(selection.get("top_100_sha256") or "")
    if not expected_hash:
        raise SystemExit("training manifest sem hash do top_100 usado para selecionar parametros")
    actual_hash = sha256_file(top_path)
    if actual_hash != expected_hash:
        raise SystemExit(
            f"training-top hash divergente: esperado={expected_hash} atual={actual_hash}"
        )
    return {
        "training_start": training_start,
        "training_end": training_end,
        "training_top_sha256": actual_hash,
        "training_optimizer_sha": manifest.get("optimizer_sha", ""),
        "training_upstream_sha": manifest.get("upstream_sha", ""),
    }


def main():
    args = parse_args()
    if args.top <= 0:
        raise SystemExit("--top precisa ser positivo")
    provenance = verify_training_source(args.training_top, args.training_manifest, args.start)

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
        config.validate_run_config(
            start=args.start, end=market.end,
            gap_min=g, gap_max=g, signal_min=s, signal_max=s,
            momentum_min=m, momentum_max=m, vol_period=v,
            initial_cash=args.initial_cash, fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            odd_lot_extra_bps=args.odd_lot_extra_bps,
        )
        summary, curve = red.detailed_backtest(
            market, g, s, m, v,
            initial_cash=args.initial_cash,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            odd_lot_extra_bps=args.odd_lot_extra_bps,
        )
        summary = dict(summary)
        summary["training_rank"] = rank
        summary["selection_source"] = "cryptographically_verified_training_only"
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
        "schema_version": 2,
        "selection": "parameters selected only on cryptographically verified training artifact",
        **provenance,
        "oos_start": args.start,
        "oos_end": market.end,
        "top": args.top,
        "results": results,
    }
    (args.output_dir / "OOS_TOP3.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Validacao fora da amostra — Top escolhidos somente no treino",
        "",
        f"Treino: **{provenance['training_start']} ate {provenance['training_end']}**.",
        f"Holdout: **{args.start} ate {market.end}**.",
        "O arquivo de ranking de treino foi conferido por SHA-256 e o manifest prova selecao training-only.",
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
        "O universo Pine fixo continua sendo uma limitacao metodologica separada e nao e survivorship-safe.",
    ]
    (args.output_dir / "OOS_TOP3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
