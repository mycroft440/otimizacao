#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
import validate_top_oos as oos_guard


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
    p.add_argument("--odd-lot-extra-bps", type=float, default=config.DEFAULT_ODD_LOT_EXTRA_BPS)
    return p.parse_args()


def _parse_required_date(value: object, label: str, window_id: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(str(value))
    except Exception as exc:
        raise SystemExit(f"{window_id}: {label} invalido: {value!r}") from exc
    if pd.isna(result):
        raise SystemExit(f"{window_id}: {label} invalido: {value!r}")
    return result.normalize()


def verify_training(
    window: dict[str, str],
    directory: Path,
    *,
    expected_config: dict[str, float] | None = None,
) -> tuple[pd.Series, dict[str, object]]:
    window_id = window["id"]
    manifest_path = directory / "MANIFEST.json"
    top_path = directory / "top_100.csv"
    if not manifest_path.exists() or not top_path.exists():
        raise SystemExit(f"{window_id}: artefato de treino incompleto")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    execution = manifest.get("execution") or {}
    provenance = manifest.get("selection_provenance") or {}
    if provenance.get("mode") != "training_only":
        raise SystemExit(f"{window_id}: manifest nao e training_only")

    requested_start = _parse_required_date(window["train_start"], "train_start solicitado", window_id)
    requested_end = _parse_required_date(window["train_end"], "train_end solicitado", window_id)
    oos_start = _parse_required_date(window["oos_start"], "oos_start", window_id)
    execution_start = _parse_required_date(execution.get("start"), "execution.start", window_id)
    execution_end = _parse_required_date(execution.get("end"), "execution.end", window_id)
    provenance_start = _parse_required_date(
        provenance.get("training_start"), "selection_provenance.training_start", window_id
    )
    provenance_end = _parse_required_date(
        provenance.get("training_end"), "selection_provenance.training_end", window_id
    )

    if execution_start != requested_start or provenance_start != execution_start:
        raise SystemExit(
            f"{window_id}: inicio do treino divergente: solicitado={requested_start.date()} "
            f"execution={execution_start.date()} provenance={provenance_start.date()}"
        )

    # The manifest may record the effective last B3 session just before a
    # calendar cutoff such as 31/Dec. It may never cross the requested cutoff.
    if execution_end > requested_end or provenance_end > requested_end:
        raise SystemExit(
            f"{window_id}: treino ultrapassa o fim solicitado: requested={requested_end.date()} "
            f"execution={execution_end.date()} provenance={provenance_end.date()}"
        )
    if abs((requested_end - execution_end).days) > 7:
        raise SystemExit(
            f"{window_id}: execution.end esta longe demais do corte solicitado: "
            f"{execution_end.date()} vs {requested_end.date()}"
        )
    if execution_end != provenance_end:
        raise SystemExit(
            f"{window_id}: fim do treino diverge entre execution e provenance: "
            f"{execution_end.date()} vs {provenance_end.date()}"
        )

    if requested_end >= oos_start:
        raise SystemExit(f"{window_id}: train_end solicitado precisa ser anterior a oos_start")
    if execution_end >= oos_start or provenance_end >= oos_start:
        raise SystemExit(f"{window_id}: treino comprovado invade o holdout")

    # Reuse the generic OOS fail-closed provenance gate so walk-forward cannot
    # silently become less strict than standalone OOS validation.
    oos_guard.verify_training_source(
        top_path,
        manifest_path,
        window["oos_start"],
        expected_config=expected_config,
    )

    top = pd.read_csv(top_path)
    required = ["gap_period", "signal_period", "momentum_period", "vol_period"]
    if top.empty or not set(required).issubset(top.columns):
        raise SystemExit(f"{window_id}: top_100 vazio ou schema invalido")
    if top[required].isna().any().any() or top.duplicated(required).any():
        raise SystemExit(f"{window_id}: top_100 contem parametros ausentes ou duplicados")
    return top.iloc[0], manifest


def main():
    args = parse_args()
    config.validate_run_config(
        start="2018-01-02",
        end=args.snapshot_end,
        gap_min=config.DEFAULT_GAP_MIN,
        gap_max=config.DEFAULT_GAP_MAX,
        signal_min=config.DEFAULT_SIGNAL_MIN,
        signal_max=config.DEFAULT_SIGNAL_MAX,
        momentum_min=config.DEFAULT_MOMENTUM_MIN,
        momentum_max=config.DEFAULT_MOMENTUM_MAX,
        vol_period=config.DEFAULT_VOL_PERIOD,
        initial_cash=args.initial_cash,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        odd_lot_extra_bps=args.odd_lot_extra_bps,
    )

    windows_payload = json.loads(args.windows.read_text(encoding="utf-8"))
    windows = list(windows_payload["windows"])
    if not windows:
        raise SystemExit("arquivo de janelas vazio")

    expected_training_config = {
        "initial_cash": args.initial_cash,
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
        "odd_lot_extra_bps": args.odd_lot_extra_bps,
    }
    capital = float(args.initial_cash)
    all_curves = []
    results = []
    snapshot_end = pd.Timestamp(args.snapshot_end).normalize()

    prior_oos_end: pd.Timestamp | None = None
    for window in windows:
        train_dir = args.training_root / window["id"]
        winner, manifest = verify_training(
            window,
            train_dir,
            expected_config=expected_training_config,
        )
        oos_start = pd.Timestamp(window["oos_start"]).normalize()
        configured_end = (
            pd.Timestamp(window["oos_end"]).normalize() if window.get("oos_end") else snapshot_end
        )
        oos_end = min(configured_end, snapshot_end)
        if oos_start > snapshot_end:
            continue
        if oos_end < oos_start:
            raise SystemExit(f"{window['id']}: OOS vazio")
        if prior_oos_end is not None and oos_start <= prior_oos_end:
            raise SystemExit(
                f"{window['id']}: janelas OOS sobrepostas: start={oos_start.date()} "
                f"prior_end={prior_oos_end.date()}"
            )

        g, s, m, v = (
            int(winner.gap_period),
            int(winner.signal_period),
            int(winner.momentum_period),
            int(winner.vol_period),
        )
        market = opt.load_market(
            args.data_root, oos_start.date().isoformat(), oos_end.date().isoformat()
        )
        summary, curve = red.detailed_backtest(
            market,
            g,
            s,
            m,
            v,
            initial_cash=capital,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            odd_lot_extra_bps=args.odd_lot_extra_bps,
        )
        start_capital = capital
        capital = float(summary["final_equity"])
        execution = manifest.get("execution") or {}
        window_result = {
            "window_id": window["id"],
            "train_start_requested": window["train_start"],
            "train_end_requested": window["train_end"],
            "train_start_proven": str(execution.get("start")),
            "train_end_proven": str(execution.get("end")),
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
            "oos_max_drawdown_close_to_close": float(summary["max_drawdown"]),
            "trades": int(summary["trades"]),
            "training_optimizer_sha": manifest.get("optimizer_sha", ""),
            "training_upstream_sha": manifest.get("upstream_sha", ""),
            "training_config_matches_walk_forward": True,
        }
        results.append(window_result)
        curve = curve.copy()
        curve.insert(0, "window_id", window["id"])
        curve.insert(1, "gap_period", g)
        curve.insert(2, "signal_period", s)
        curve.insert(3, "momentum_period", m)
        all_curves.append(curve)
        prior_oos_end = pd.Timestamp(summary["end"]).normalize()

    if not results:
        raise SystemExit("nenhuma janela OOS disponivel")

    combined = pd.concat(all_curves, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="raise")
    duplicate_dates = combined["date"].duplicated(keep=False)
    if duplicate_dates.any():
        examples = combined.loc[duplicate_dates, "date"].dt.date.astype(str).head(10).tolist()
        raise SystemExit(f"janelas OOS produziram datas duplicadas: {examples}")
    combined = combined.sort_values("date").reset_index(drop=True)

    risk = metrics.risk_metrics(combined[["date", "equity"]], args.initial_cash)
    annual = metrics.annual_metrics(combined[["date", "equity"]], args.initial_cash)
    parameter_changes = 0
    for previous, current in zip(results, results[1:]):
        a = (previous["gap_period"], previous["signal_period"], previous["momentum_period"])
        b = (current["gap_period"], current["signal_period"], current["momentum_period"])
        parameter_changes += int(a != b)

    payload = {
        "status": "PASS",
        "schema_version": 3,
        "method": "rolling_3y_train_then_next_year_oos_rank1_with_capital_carry",
        "selection_is_strictly_prior_to_each_holdout": True,
        "training_provenance_gate": "same_strict_gate_as_standalone_oos",
        "training_configuration_must_match_oos": True,
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
    pd.DataFrame(results).to_csv(
        args.output_dir / "WALK_FORWARD_WINDOWS.csv", index=False, float_format="%.12f"
    )
    combined.to_csv(
        args.output_dir / "WALK_FORWARD_EQUITY_DAILY.csv", index=False, float_format="%.12f"
    )

    lines = [
        "# Walk-forward — validacao sequencial fora da amostra",
        "",
        "Cada janela escolhe parametros apenas no treino anterior; o capital final OOS e carregado para a janela seguinte.",
        "Treinos passam pelo mesmo gate criptografico/configuracional usado na validacao OOS isolada.",
        "",
        "| Janela | Treino comprovado | OOS | Parametros | Retorno OOS | Capital final |",
        "|---|---|---|---|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['window_id']} | {item['train_start_proven']}..{item['train_end_proven']} | "
            f"{item['oos_start']}..{item['oos_end']} | "
            f"{item['gap_period']}/{item['signal_period']}/{item['momentum_period']} | "
            f"{item['oos_return'] * 100:.2f}% | R$ {item['end_equity']:.2f} |"
        )
    lines += [
        "",
        f"**Retorno OOS agregado:** {(capital / args.initial_cash - 1.0) * 100:.2f}%",
        f"**Capital final OOS:** R$ {capital:.2f}",
        f"**CAGR OOS agregado:** {float(risk['cagr']) * 100:.2f}%"
        if math.isfinite(float(risk["cagr"]))
        else "**CAGR OOS agregado:** N/D",
        f"**Max DD close-to-close OOS:** {float(risk['max_drawdown_close_to_close']) * 100:.2f}%",
        f"**Mudancas de parametros entre janelas:** {parameter_changes}",
    ]
    (args.output_dir / "WALK_FORWARD.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
