#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from metrics import annual_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--curve", required=True, type=Path)
    p.add_argument("--initial-cash", required=True, type=float)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--json", required=True, type=Path)
    p.add_argument("--md", required=True, type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    frame = pd.read_csv(args.curve)
    result = annual_metrics(frame, args.initial_cash)
    rows = list(result["years"])
    out = pd.DataFrame(rows)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.csv, index=False, float_format="%.12f")

    payload = {"status": "PASS", "schema_version": 2, **result}
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Retorno por ano — melhor combinacao",
        "",
        "| Ano | Retorno | Patrimonio final | Cobertura |",
        "|---:|---:|---:|---|",
    ]
    for row in rows:
        status = "completo" if row["complete_year"] else "parcial"
        lines.append(
            f"| {row['year']} | {float(row['return_pct']):.2f}% | R$ {float(row['end_equity']):.2f} | {status} |"
        )
    total = float(result["total_return_pct"])
    avg = float(result["average_complete_year_return_pct"])
    geo = float(result["geometric_mean_complete_year_return_pct"])
    lines += [
        "",
        f"**Retorno total:** {total:.2f}%",
        f"**Media aritmetica somente dos anos completos:** {avg:.2f}%" if math.isfinite(avg) else "**Media dos anos completos:** N/D",
        f"**Media geometrica somente dos anos completos:** {geo:.2f}%" if math.isfinite(geo) else "**Media geometrica dos anos completos:** N/D",
        "",
        "Anos iniciais ou terminais com cobertura parcial sao exibidos, mas nao entram nas medias de anos completos.",
    ]
    args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
