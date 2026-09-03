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
import metrics
import optimize_b3_pine as opt
import reduce_results as red


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--windows", required=True, type=Path)
    p.add_argument("--training-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--snapshot-end", required=True)
    p.add_argument("--initial-cash", type=float, default=config.DEFAULT_INITIAL_CASH)
    p.add_argument("--fee-bps", type=float, default=config.DEFAULT_FEE_BPS)
    p.add_argument("--slippage-bps", type=float, default=config.DEFAULT_SLIPPAGE_BPS)
    p.add_argument("--odd-lot-extra-bps", type=float, default=config.DEFAULT_OD_LOT_EXTRA_BPS if hasattr(config, 'DEFAULT_OD_LOT_EXTRA_BPS') else config.DEFAULT_ODD_LOT_EXTRA_BPS)
    return p.parse_args()


def verify_training(window: dict[str, str], directory: Path) -> tuple[pd.Series, dict[str, object]]:
    manifest_path = directory / "MANIFEST.json"
    top_path = directory / "top_100.csv"
    if not manifest_path.exists() or not top_path.exists():
        raise SystemExit(f"{window['id']}: artefato de treino incompleto")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    execution = manifest.get("execution") or {}
    provenance = manifest.get("selection_provenance") or {}
    if provenance.get("mode") != "training_only":
        raise SystemExit(f"{window['id']}: manifest nao e training_only")
    if str(execution.get("start")) != window["train_start"] or str(execution.get("end")) != window["train_end"]:
        raise SystemExit(
            f"{window['id']}: periodo do manifest diverge da janela: "
            f"{execution.get('start')}..{execution.get('end')}"
        )
    if pd.Timestamp(window["train_end"]) >= pd.Timestamp(window["oos_start"]):
        raise SystemExit(f"{window['id']}: train_end precisa ser anterior a oos_start")
    expected_hash = str(provenance.get("top_100_sha256") or "")
    actual_hash = sha256_file(top_path)
    if not expected_hash or expected_hash != actual_hash:
        raise SystemExit(f"{window['id']}: hash do top_100 nao confere")
    top = pd.read_csv(top_path)
    if top.empty:
        raise SystemExit(f"{window['id']}: top_100 vazio")
    return top.iloc[0], manifest


def main():
    args = parse_args()
    windows_payload = json.loads(args.windows.read_text(encoding="utf-8"))
    windows = list(windows_payload["windows"])
    capital = float(args.initial_cash)
    all_curves = []
    results = []
    snapshot_end = pd.Timestamp(args.snapshot_end)

    for window in windows:
        train_dir = args.training_root / window["id"]
        winner, manifest = verify_training(window, train_dir)
        oos_start = pd.Timestamp(window["oos_start"])
        configured_end = pd.Timestamp(window["oos_end"]) if window.get("oos_end") else snapshot_end
        oos_end = min(configured_end, snapshot_end)
        if oos_start > snapshot_end:
            continue
        if oos_end < oos_start:
            raise SystemExit(f"{window['id']}: OOS vazio")

        g, s, m, v = (
            int(winner.gap_period), int(winner.signal_period),
            int(winner.momentum_period), int(winner.vol_period),
        )
        market = opt.load_market(args.data_root, oos_start.date().isoformat(), oos_end.date().isoformat())
        summary, curve = red.detailed_backtest(
            market, g, s, m, v,
            initial_cash=capital,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            odd_lot_extra_bps=args.odd_lot_extra_bps,
        )
        start_capital = capital
        capital = float(summary["final_equity"])
        window_result = {
            "window_id": window["id"],
            "train_start": window["train_start"],
            "train_end": window["train_end"],
            "oos_start": str(summary["start"]),
            "oos_end": str(summary["end"]),
            "gap_period": g,
            "signal_period": s,
            "momentum_period": m,
            "vol_period": v,
            "start_equity": start_capital,
            "end_equity": capital,
            "oos_return": capital / start_capital - 1.0,
            "oos_cagr": float(summary["cagr"]),
            "oos_max_drawdown": float(summary["max_drawdown"]),
            "trades": int(summary["trades"]),
            "training_optimizer_sha": manifest.get("optimizer_sha", ""),
            "training_upstream_sha": manifest.get("upstream_sha", ""),
        }
        results.append(window_result)
        curve = curve.copy()
        curve.insert(0, "window_id", window["id"])
        curve.insert(1, "gap_period", g)
        curve.insert(2, "signal_period", s)
        curve.insert(3, "momentum_period", m)
        all_curves.append(curve)

    if not results:
        raise SystemExit("nenhuma janela OOS disponivel")
    combined = pd.concat(all_curves, ignore_index=True)
    combined = combined.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    risk = metrics.risk_metrics(combined[["date", "equity"]], args.initial_cash)
    annual = metrics.annual_metrics(combined[["date", "equity"]], args.initial_cash)
    parameter_changes = 0
    for previous, current in zip(results, results[1:]):
        a = (previous["gap_period"], previous["signal_period"], previous["momentum_period"])
        b = (current["gap_period"], current["signal_period"], current["momentum_period"])
        parameter_changes += int(a != b)

    payload = {
        "status": "PASS",
        "schema_version": 1,
        "method": "rolling_3y_train_then_next_year_oos_rank1_with_capital_carry",
        "selection_is_strictly_prior_to_each_holdout": True,
        "initial_cash": args.initial_cash,
        "final_equity": capital,
        "total_oos_return": capital / args.initial_cash - 1.0,
        "risk": risk,
        "annual": annual,
        "parameter_changes": parameter_changes,
        "windows": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "WALK_FORWARD.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(results).to_csv(args.output_dir / "WALK_FORWARD_WINDOWS.csv", index=False, float_format="%.12f")
    combined.to_csv(args.output_dir / "WALK_FORWARD_EQUITY_DAILY.csv", index=False, float_format="%.12f")

    lines = [
        "# Walk-forward — validacao sequencial fora da amostra",
        "",
        "Cada janela escolhe parametros apenas no treino anterior; o capital final OOS e carregado para a janela seguinte.",
        "",
        "| Janela | Treino | OOS | Parametros | Retorno OOS | Capital final |",
        "|---|---|---|---|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['window_id']} | {item['train_start']}..{item['train_end']} | "
            f"{item['oos_start']}..{item['oos_end']} | "
            f"{item['gap_period']}/{item['signal_period']}/{item['momentum_period']} | "
            f"{item['oos_return'] * 100:.2f}% | R$ {item['end_equity']:.2f} |"
        )
    lines += [
        "",
        f"**Retorno OOS agregado:** {(capital / args.initial_cash - 1.0) * 100:.2f}%",
        f"**Capital final OOS:** R$ {capital:.2f}",
        f"**CAGR OOS agregado:** {float(risk['cagr']) * 100:.2f}%" if math.isfinite(float(risk['cagr'])) else "**CAGR OOS agregado:** N/D",
        f"**Max DD close-to-close OOS:** {float(risk['max_drawdown_close_to_close']) * 100:.2f}%",
        f"**Mudancas de parametros entre janelas:** {parameter_changes}",
    ]
    (args.output_dir / "WALK_FORWARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
