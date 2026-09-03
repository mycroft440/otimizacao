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


def load_official_calendar(root: Path, requested_end: date) -> tuple[pd.DatetimeIndex, dict[str, object], Path, Path]:
    calendar_path = root / "data/calendars/b3_sessions.csv"
    meta_path = root / "data/calendars/b3_sessions.json"
    if not calendar_path.exists() or not meta_path.exists():
        raise SystemExit("calendario oficial B3 COTAHIST ausente do snapshot")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "PASS" or meta.get("source") != "B3_COTAHIST_ANNUAL_ARCHIVES":
        raise SystemExit("metadata do calendario B3 invalida")
    if str(meta.get("requested_end")) != requested_end.isoformat():
        raise SystemExit(
            f"calendar requested_end divergente: {meta.get('requested_end')} != {requested_end.isoformat()}"
        )
    if str(meta.get("calendar_sha256")) != sha256(calendar_path):
        raise SystemExit("hash do calendario B3 nao confere")

    frame = pd.read_csv(calendar_path, usecols=["date"])
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"].astype(str).str[:10], errors="raise"))
    if dates.empty or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise SystemExit("calendario B3 vazio, duplicado ou fora de ordem")
    if dates.max().date() > requested_end:
        raise SystemExit("calendario B3 contem sessao posterior ao corte solicitado")
    if str(meta.get("first_session")) != dates.min().date().isoformat():
        raise SystemExit("first_session do calendario nao confere")
    if str(meta.get("last_session")) != dates.max().date().isoformat():
        raise SystemExit("last_session do calendario nao confere")
    if int(meta.get("session_count", -1)) != len(dates):
        raise SystemExit("session_count do calendario nao confere")
    return dates, meta, calendar_path, meta_path


def main():
    args = parse_args()
    requested_end = date.fromisoformat(args.requested_end)
    official_dates, calendar_meta, calendar_path, calendar_meta_path = load_official_calendar(
        args.root, requested_end
    )
    official_set = set(official_dates.tolist())
    official_backtest = official_dates[
        (official_dates >= pd.Timestamp("2018-01-02"))
        & (official_dates <= pd.Timestamp(requested_end))
    ]
    if official_backtest.empty:
        raise SystemExit("calendario oficial sem sessoes no periodo do backtest")
    official_end = official_backtest.max().date()
    if (requested_end - official_end).days > 4:
        raise SystemExit(
            f"calendario oficial desatualizado: requested={requested_end} official_end={official_end}"
        )

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
        outside_calendar = dates[~dates.isin(official_dates)]
        if len(outside_calendar):
            raise SystemExit(
                f"{ticker}: candle fora do calendario COTAHIST: "
                + ", ".join(day.date().isoformat() for day in outside_calendar[:10])
            )
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
        sessions_by_ticker[ticker] = set(
            dates[
                (dates >= pd.Timestamp("2018-01-02"))
                & (dates <= pd.Timestamp(official_end))
            ].dt.date
        )

    latest_observed_end = max(end_by_ticker.values())
    if (official_end - latest_observed_end).days > 4:
        raise SystemExit(
            f"candles desatualizados: official_end={official_end} newest_candle={latest_observed_end}"
        )

    official_session_set = set(official_backtest.date)
    coverage = {}
    stale = {}
    for ticker in tickers:
        ratio = len(sessions_by_ticker[ticker] & official_session_set) / len(official_session_set)
        coverage[ticker] = ratio
        stale_days = (official_end - end_by_ticker[ticker]).days
        stale[ticker] = stale_days
        if ratio < 0.95:
            raise SystemExit(f"{ticker}: cobertura de sessoes oficiais muito baixa ({ratio:.2%})")
        if stale_days > 10:
            raise SystemExit(f"{ticker}: ultimo candle esta {stale_days} dias atras do calendario oficial")

    payload = {
        "status": "PASS",
        "schema_version": 4,
        "upstream_sha": args.upstream_sha,
        "requested_end": args.requested_end,
        "actual_master_end": official_end.isoformat(),
        "calendar_source": "B3_COTAHIST_ANNUAL_ARCHIVES",
        "calendar_sha256": sha256(calendar_path),
        "calendar_meta_sha256": sha256(calendar_meta_path),
        "calendar_session_count_total": len(official_dates),
        "calendar_session_count_backtest": len(official_backtest),
        "calendar_archive_count": len(calendar_meta.get("archives", [])),
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
        "calendar_source": payload["calendar_source"],
        "minimum_session_coverage_ratio": payload["minimum_session_coverage_ratio"],
        "corporate_action_override_count": len(corporate_overrides),
        "extreme_normalized_close_moves": len(extreme_normalized_moves),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
