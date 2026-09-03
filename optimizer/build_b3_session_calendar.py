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


def main():
    args = parse_args()
    if args.start_year > args.end_year:
        raise SystemExit("start-year maior que end-year")
    requested_end = date.fromisoformat(args.end)

    all_sessions: set[str] = set()
    archives = []
    for year in range(args.start_year, args.end_year + 1):
        path = args.archives_dir / f"COTAHIST_A{year}.ZIP"
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
    if ordered != sorted(set(ordered)):
        raise SystemExit("calendario contem duplicatas")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": ordered}).to_csv(args.output, index=False)
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
        "calendar_sha256": sha256(args.output),
        "archives": archives,
    }
    args.meta.parent.mkdir(parents=True, exist_ok=True)
    args.meta.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "first_session": ordered[0],
        "last_session": ordered[-1],
        "session_count": len(ordered),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
