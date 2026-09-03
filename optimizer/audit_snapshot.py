#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--upstream-sha", required=True)
    p.add_argument("--requested-end", required=True)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    args = parse_args()
    universe_path = args.root / "data/universes/fixed_40_2018.json"
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    tickers = [str(x).upper() for x in universe["tickers"]]
    if len(tickers) != 40 or len(set(tickers)) != 40:
        raise SystemExit(f"universo Pine precisa ter 40 tickers unicos, recebeu {len(tickers)}")
    if "PRIO3" not in tickers or "CVCB3" in tickers:
        raise SystemExit("snapshot nao corresponde ao Pine atual: PRIO3/CVCB3 divergente")

    hashes = {}
    sessions_by_ticker = {}
    first_by_ticker = {}
    end_by_ticker = {}
    for ticker in tickers:
        path = args.root / "data/candles" / f"{ticker.lower()}_1d.csv"
        if not path.exists():
            raise SystemExit(f"candle ausente: {path}")
        hashes[str(path.relative_to(args.root))] = sha256(path)
        df = pd.read_csv(path, usecols=["date", "open", "close"])
        dates = pd.to_datetime(df["date"].astype(str).str[:10], errors="raise")
        opens = pd.to_numeric(df["open"], errors="coerce")
        closes = pd.to_numeric(df["close"], errors="coerce")
        if dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise SystemExit(f"{ticker}: datas duplicadas ou fora de ordem")
        if len(dates) < 300:
            raise SystemExit(f"{ticker}: historico insuficiente ({len(dates)} linhas)")
        if opens.isna().any() or closes.isna().any() or (opens <= 0).any() or (closes <= 0).any():
            raise SystemExit(f"{ticker}: OHLC invalido")
        first_by_ticker[ticker] = dates.iloc[0].date()
        end_by_ticker[ticker] = dates.iloc[-1].date()
        sessions_by_ticker[ticker] = set(dates[dates >= pd.Timestamp("2018-01-02")].dt.date)

    master_sessions = set().union(*sessions_by_ticker.values())
    if not master_sessions:
        raise SystemExit("calendario mestre vazio")
    newest_end = max(end_by_ticker.values())
    requested_end = date.fromisoformat(args.requested_end)
    if (requested_end - newest_end).days > 4:
        raise SystemExit(f"snapshot desatualizado: requested={requested_end} newest={newest_end}")

    coverage = {}
    stale = {}
    for ticker in tickers:
        ratio = len(sessions_by_ticker[ticker]) / len(master_sessions)
        coverage[ticker] = ratio
        stale_days = (newest_end - end_by_ticker[ticker]).days
        stale[ticker] = stale_days
        if ratio < 0.95:
            raise SystemExit(f"{ticker}: cobertura de sessoes muito baixa ({ratio:.2%})")
        if stale_days > 10:
            raise SystemExit(f"{ticker}: ultimo candle esta {stale_days} dias atras do calendario mestre")

    payload = {
        "status": "PASS",
        "schema_version": 2,
        "upstream_sha": args.upstream_sha,
        "requested_end": args.requested_end,
        "actual_master_end": newest_end.isoformat(),
        "universe_count": 40,
        "universe_sha256": sha256(universe_path),
        "universe": tickers,
        "candle_sha256": hashes,
        "first_date_by_ticker": {k: v.isoformat() for k, v in first_by_ticker.items()},
        "session_coverage_ratio": coverage,
        "stale_calendar_days_at_end": stale,
        "minimum_session_coverage_ratio": min(coverage.values()),
        "pine_slot11": "PRIO3",
        "survivorship_safe": False,
        "purpose": "exact Pine Live retrospective optimization snapshot",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
