#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--curve", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--warning-sessions", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    if args.warning_sessions < 0:
        raise SystemExit("warning-sessions precisa ser >= 0")
    curve = pd.read_csv(args.curve)
    if not {"date", "holding"}.issubset(curve.columns):
        raise SystemExit("curve precisa conter date,holding")
    curve["date"] = pd.to_datetime(curve["date"], errors="raise")
    if curve.empty:
        raise SystemExit("curve vazia")

    universe = json.loads(
        (args.data_root / "data/universes/fixed_40_2018.json").read_text(encoding="utf-8")
    )
    tickers = [str(x).upper() for x in universe["tickers"]]
    sessions = set()
    ticker_dates = {}
    for ticker in tickers:
        path = args.data_root / "data/candles" / f"{ticker.lower()}_1d.csv"
        frame = pd.read_csv(path, usecols=["date"])
        dates = pd.DatetimeIndex(
            pd.to_datetime(frame["date"].astype(str).str[:10], errors="raise")
            .drop_duplicates()
            .sort_values()
        )
        ticker_dates[ticker] = dates
        sessions.update(dates.tolist())
    master = pd.DatetimeIndex(sorted(sessions))
    master_pos = {pd.Timestamp(day): i for i, day in enumerate(master)}

    warnings = []
    max_stale = 0
    last_invested_item = None
    terminal_row = curve.iloc[-1]
    terminal_holding = str(terminal_row["holding"]).upper()

    for row in curve.itertuples(index=False):
        holding = str(row.holding).upper()
        if holding == "CASH":
            continue
        if holding not in ticker_dates:
            raise SystemExit(f"holding fora do universo: {holding}")
        day = pd.Timestamp(row.date)
        dates = ticker_dates[holding]
        idx = dates.searchsorted(day, side="right") - 1
        if idx < 0:
            stale = None
            last = None
        else:
            last = pd.Timestamp(dates[idx])
            if day not in master_pos or last not in master_pos:
                stale = int((day - last).days)
            else:
                stale = int(master_pos[day] - master_pos[last])
            max_stale = max(max_stale, stale)
        item = {
            "date": day.date().isoformat(),
            "holding": holding,
            "last_trade_date": last.date().isoformat() if last is not None else None,
            "stale_master_sessions": stale,
        }
        if stale is None or stale > args.warning_sessions:
            warnings.append(item)
        last_invested_item = item

    if terminal_holding == "CASH":
        terminal = {
            "date": pd.Timestamp(terminal_row["date"]).date().isoformat(),
            "holding": "CASH",
            "last_trade_date": None,
            "stale_master_sessions": 0,
            "note": "terminal portfolio is cash; no terminal security price is used",
        }
    else:
        terminal = last_invested_item
        if terminal is None or terminal.get("holding") != terminal_holding:
            raise SystemExit("nao foi possivel reconciliar a posicao terminal da curva")

    payload = {
        "status": "PASS",
        "schema_version": 2,
        "warning_threshold_master_sessions": args.warning_sessions,
        "max_stale_master_sessions_while_invested": max_stale,
        "terminal_position_price_age": terminal,
        "warning_count": len(warnings),
        "warnings": warnings[:100],
        "method_note": (
            "master sessions are the union of observed snapshot sessions; warnings are diagnostic, "
            "not proof of delisting/suspension cause"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "max_stale_master_sessions_while_invested": max_stale,
                "terminal_holding": terminal_holding,
                "warning_count": len(warnings),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
