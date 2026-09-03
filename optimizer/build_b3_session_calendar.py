#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

# Same standard-equity BDI statuses accepted by b3-strategy-lab/cotahist.py.
STANDARD_EQUITY_BDI_CODES = {"02", "05", "06", "07", "08", "09", "11"}
STANDARD_LOT_MARKET_TYPE = "010"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--archives-dir", required=True, type=Path)
    p.add_argument("--start-year", required=True, type=int)
    p.add_argument("--end-year", required=True, type=int)
    p.add_argument("--end", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--meta", required=True, type=Path)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sessions_from_archive(path: Path) -> set[str]:
    sessions: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise SystemExit(f"{path}: esperado exatamente um TXT COTAHIST, recebeu {members}")
        with archive.open(members[0]) as handle:
            saw_header = False
            saw_trailer = False
            for line_number, raw in enumerate(handle, start=1):
                line = raw.decode("latin-1").rstrip("\r\n")
                if not line:
                    continue
                if len(line) < 245:
                    raise SystemExit(f"{path}: linha {line_number} truncada ({len(line)})")
                record_type = line[:2]
                if record_type == "00":
                    if saw_header:
                        raise SystemExit(f"{path}: mais de um cabecalho")
                    saw_header = True
                    continue
                if record_type == "99":
                    saw_trailer = True
                    continue
                if record_type != "01":
                    continue
                if line[10:12] not in STANDARD_EQUITY_BDI_CODES or line[24:27] != STANDARD_LOT_MARKET_TYPE:
                    continue
                raw_date = line[2:10]
                try:
                    parsed = date(int(raw_date[:4]), int(raw_date[4:6]), int(raw_date[6:8]))
                except Exception as exc:
                    raise SystemExit(f"{path}: data COTAHIST invalida {raw_date!r} na linha {line_number}") from exc
                sessions.add(parsed.isoformat())
            if not saw_header or not saw_trailer:
                raise SystemExit(f"{path}: envelope COTAHIST incompleto")
    if not sessions:
        raise SystemExit(f"{path}: nenhuma sessao de acoes padrao encontrada")
    return sessions


def build_calendar(
    *,
    archives_dir: Path,
    start_year: int,
    end_year: int,
    requested_end: date,
    output: Path,
    meta: Path,
) -> dict[str, object]:
    if start_year > end_year:
        raise SystemExit("start-year maior que end-year")

    all_sessions: set[str] = set()
    archives = []
    for year in range(start_year, end_year + 1):
        path = archives_dir / f"COTAHIST_A{year}.ZIP"
        if not path.exists():
            raise SystemExit(f"arquivo COTAHIST ausente: {path}")
        year_sessions = sessions_from_archive(path)
        all_sessions.update(day for day in year_sessions if date.fromisoformat(day) <= requested_end)
        archives.append({
            "year": year,
            "filename": path.name,
            "sha256": sha256(path),
            "session_count": len(year_sessions),
        })

    ordered = sorted(all_sessions)
    if not ordered:
        raise SystemExit("calendario oficial vazio")

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": ordered}).to_csv(output, index=False)
    payload = {
        "status": "PASS",
        "schema_version": 1,
        "source": "B3_COTAHIST_ANNUAL_ARCHIVES",
        "market_type": STANDARD_LOT_MARKET_TYPE,
        "bdi_codes": sorted(STANDARD_EQUITY_BDI_CODES),
        "requested_end": requested_end.isoformat(),
        "first_session": ordered[0],
        "last_session": ordered[-1],
        "session_count": len(ordered),
        "calendar_sha256": sha256(output),
        "archives": archives,
    }
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main():
    args = parse_args()
    payload = build_calendar(
        archives_dir=args.archives_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        requested_end=date.fromisoformat(args.end),
        output=args.output,
        meta=args.meta,
    )
    print(json.dumps({
        "status": payload["status"],
        "first_session": payload["first_session"],
        "last_session": payload["last_session"],
        "session_count": payload["session_count"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
