#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import required_warmup_sessions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--gap-max", required=True, type=int)
    parser.add_argument("--signal-max", required=True, type=int)
    parser.add_argument("--momentum-max", required=True, type=int)
    parser.add_argument("--vol-period", required=True, type=int)
    parser.add_argument("--archive-start-year", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    start = pd.Timestamp(args.start)
    required = required_warmup_sessions(args.gap_max, args.signal_max, args.momentum_max, args.vol_period)
    universe_path = args.data_root / "data/universes/fixed_40_2018.json"
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    tickers = [str(value).upper() for value in universe["tickers"]]
    records = []
    failures = []

    for ticker in tickers:
        candle_path = args.data_root / "data/candles" / f"{ticker.lower()}_1d.csv"
        if not candle_path.exists():
            failures.append({"ticker": ticker, "reason": "missing_file"})
            continue
        frame = pd.read_csv(candle_path, usecols=["date"])
        dates = pd.to_datetime(frame["date"].astype(str).str[:10], errors="raise").drop_duplicates().sort_values()
        before = dates[dates < start]
        count = int(len(before))
        first = dates.iloc[0] if len(dates) else None
        enough = count >= required
        structural_short_history = bool(first is not None and not enough and int(first.year) > args.archive_start_year)
        if not enough and not structural_short_history:
            failures.append({
                "ticker": ticker,
                "reason": "insufficient_snapshot_warmup",
                "sessions_before_start": count,
                "required_sessions": required,
                "first_date": first.date().isoformat() if first is not None else None,
            })
        records.append({
            "ticker": ticker,
            "sessions_before_start": count,
            "required_sessions": required,
            "enough_before_start": enough,
            "structural_short_history_candidate": structural_short_history,
            "first_date": first.date().isoformat() if first is not None else None,
        })

    payload = {
        "status": "PASS" if not failures else "FAIL",
        "start": args.start,
        "archive_start_year": args.archive_start_year,
        "required_warmup_sessions": required,
        "failures": failures,
        "tickers": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if failures:
        raise SystemExit("WARMUP AUDIT FAIL")


if __name__ == "__main__":
    main()
