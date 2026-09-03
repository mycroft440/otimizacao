#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def audit_local_corporate_overrides(tickers: list[str]) -> tuple[list[dict[str, object]], str | None]:
    path = Path("optimizer/b3_strategy_live_corporate_action_overrides.json")
    if not path.exists():
        raise SystemExit("arquivo local de overrides corporativos ausente")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise SystemExit("schema de overrides corporativos invalido")
    required = {"ticker", "ex_date", "last_date_prior", "split_ratio", "event", "source_authority", "source_url"}
    seen = set()
    normalized = []
    for event in payload.get("events", []):
        missing = sorted(required - set(event))
        if missing:
            raise SystemExit(f"override corporativo sem campos obrigatorios: {missing}")
        ticker = str(event["ticker"]).upper()
        ex_date = str(event["ex_date"])
        key = (ticker, ex_date)
        if key in seen:
            raise SystemExit(f"override corporativo duplicado: {key}")
        seen.add(key)
        if ticker not in tickers:
            raise SystemExit(f"override corporativo fora do universo Pine: {ticker}")
        ratio = float(event["split_ratio"])
        if not math.isfinite(ratio) or ratio <= 0.0 or ratio == 1.0:
            raise SystemExit(f"split_ratio invalido em {key}: {ratio}")
        if pd.Timestamp(event["last_date_prior"]) >= pd.Timestamp(ex_date):
            raise SystemExit(f"last_date_prior nao antecede ex_date em {key}")
        if not str(event["source_authority"]).strip():
            raise SystemExit(f"override sem source_authority: {key}")
        if not str(event["source_url"]).startswith("https://"):
            raise SystemExit(f"override sem source_url HTTPS: {key}")
        normalized.append({"ticker": ticker, "ex_date": ex_date, "split_ratio": ratio})
    return normalized, sha256(path)


def main():
    args = parse_args()
    universe_path = args.root / "data/universes/fixed_40_2018.json"
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    tickers = [str(x).upper() for x in universe["tickers"]]
    if len(tickers) != 40 or len(set(tickers)) != 40:
        raise SystemExit(f"universo Pine precisa ter 40 tickers unicos, recebeu {len(tickers)}")
    if "PRIO3" not in tickers or "CVCB3" in tickers:
        raise SystemExit("snapshot nao corresponde ao Pine atual: PRIO3/CVCB3 divergente")

    corporate_overrides, corporate_override_sha = audit_local_corporate_overrides(tickers)
    hashes = {}
    sessions_by_ticker = {}
    first_by_ticker = {}
    end_by_ticker = {}
    extreme_normalized_moves = []
    extreme_ratio = 3.0
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
        close_ratio = closes / closes.shift(1)
        mask = (close_ratio > extreme_ratio) | (close_ratio < 1.0 / extreme_ratio)
        for idx in df.index[mask]:
            day = dates.iloc[idx]
            nearby_override = any(
                event["ticker"] == ticker
                and abs((pd.Timestamp(event["ex_date"]) - day).days) <= 5
                for event in corporate_overrides
            )
            extreme_normalized_moves.append(
                {
                    "ticker": ticker,
                    "date": day.date().isoformat(),
                    "close_ratio_vs_previous": float(close_ratio.iloc[idx]),
                    "near_local_override": nearby_override,
                }
            )
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
        "schema_version": 3,
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
        "corporate_action_overrides_sha256": corporate_override_sha,
        "corporate_action_override_count": len(corporate_overrides),
        "corporate_action_override_schema_and_sources": "PASS",
        "extreme_normalized_close_move_threshold": extreme_ratio,
        "extreme_normalized_close_moves": extreme_normalized_moves,
        "extreme_move_note": "diagnostic only; legitimate market moves can exceed threshold",
        "pine_slot11": "PRIO3",
        "survivorship_safe": False,
        "purpose": "exact Pine Live retrospective optimization snapshot",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "actual_master_end": payload["actual_master_end"],
        "minimum_session_coverage_ratio": payload["minimum_session_coverage_ratio"],
        "corporate_action_override_count": len(corporate_overrides),
        "extreme_normalized_close_moves": len(extreme_normalized_moves),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
