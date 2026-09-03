#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--overrides", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--extreme-ratio", type=float, default=3.0)
    return p.parse_args()


def main():
    args = parse_args()
    if not math.isfinite(args.extreme_ratio) or args.extreme_ratio <= 1.0:
        raise SystemExit("extreme-ratio precisa ser > 1")
    payload = json.loads(args.overrides.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise SystemExit("override corporate-actions com schema_version invalida")
    events = list(payload.get("events") or [])
    required = {"ticker", "ex_date", "last_date_prior", "split_ratio", "event", "source_authority", "source_url"}
    keys = set()
    event_failures = []
    normalized_events = []
    for event in events:
        missing = sorted(required - set(event))
        if missing:
            event_failures.append({"event": event, "reason": f"missing_fields:{missing}"})
            continue
        ticker = str(event["ticker"]).upper()
        ex_date = str(event["ex_date"])
        key = (ticker, ex_date)
        if key in keys:
            event_failures.append({"ticker": ticker, "ex_date": ex_date, "reason": "duplicate_event_key"})
        keys.add(key)
        try:
            ratio = float(event["split_ratio"])
        except Exception:
            ratio = float("nan")
        if not math.isfinite(ratio) or ratio <= 0.0 or ratio == 1.0:
            event_failures.append({"ticker": ticker, "ex_date": ex_date, "reason": "invalid_split_ratio"})
        if not str(event["source_authority"]).strip() or not str(event["source_url"]).startswith("https://"):
            event_failures.append({"ticker": ticker, "ex_date": ex_date, "reason": "weak_or_missing_source"})
        if pd.Timestamp(event["last_date_prior"]) >= pd.Timestamp(ex_date):
            event_failures.append({"ticker": ticker, "ex_date": ex_date, "reason": "last_date_prior_not_before_ex_date"})
        normalized_events.append({"ticker": ticker, "ex_date": ex_date, "split_ratio": ratio})

    universe = json.loads(
        (args.data_root / "data/universes/fixed_40_2018.json").read_text(encoding="utf-8")
    )
    tickers = [str(x).upper() for x in universe["tickers"]]
    unknown_override_tickers = sorted({x["ticker"] for x in normalized_events} - set(tickers))
    if unknown_override_tickers:
        event_failures.append({"reason": "override_ticker_outside_universe", "tickers": unknown_override_tickers})

    extreme_moves = []
    for ticker in tickers:
        path = args.data_root / "data/candles" / f"{ticker.lower()}_1d.csv"
        frame = pd.read_csv(path, usecols=["date", "close"])
        frame["date"] = pd.to_datetime(frame["date"].astype(str).str[:10], errors="raise")
        frame["close"] = pd.to_numeric(frame["close"], errors="raise")
        ratio = frame["close"] / frame["close"].shift(1)
        mask = (ratio > args.extreme_ratio) | (ratio < 1.0 / args.extreme_ratio)
        for idx in frame.index[mask]:
            date = frame.loc[idx, "date"].date().isoformat()
            value = float(ratio.loc[idx])
            nearby_override = any(
                event["ticker"] == ticker
                and abs((pd.Timestamp(event["ex_date"]) - frame.loc[idx, "date"]).days) <= 5
                for event in normalized_events
            )
            extreme_moves.append({
                "ticker": ticker,
                "date": date,
                "close_ratio_vs_previous": value,
                "near_override_event": nearby_override,
            })

    result = {
        "status": "FAIL" if event_failures else "PASS",
        "schema_version": 1,
        "override_event_count": len(events),
        "override_schema_failures": event_failures,
        "extreme_normalized_close_moves": extreme_moves,
        "extreme_move_threshold_ratio": args.extreme_ratio,
        "note": (
            "Extreme normalized close moves are diagnostic because legitimate market moves can occur. "
            "Override schema/source failures are blocking."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "override_event_count": len(events),
        "extreme_normalized_close_moves": len(extreme_moves),
    }, ensure_ascii=False))
    if event_failures:
        raise SystemExit("CORPORATE ACTION AUDIT FAIL")


if __name__ == "__main__":
    main()
