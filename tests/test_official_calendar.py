from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "optimizer"))
import build_b3_session_calendar as calendar_builder  # noqa: E402
import optimize_b3_pine as opt  # noqa: E402
import reference_engine as ref  # noqa: E402


def cotahist_line(*, record_type="01", day="20200102", bdi="02", ticker="TSTT3", market="010") -> str:
    chars = [" "] * 245
    chars[0:2] = list(record_type)
    if record_type == "01":
        chars[2:10] = list(day)
        chars[10:12] = list(bdi)
        chars[12:24] = list(ticker.ljust(12))
        chars[24:27] = list(market)
    return "".join(chars)


def make_cotahist_zip(path: Path, days: list[str]) -> None:
    lines = [cotahist_line(record_type="00")]
    lines.extend(cotahist_line(day=day) for day in days)
    lines.append(cotahist_line(record_type="99"))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("COTAHIST.TXT", "\n".join(lines) + "\n")


class OfficialCalendarBuilderTests(unittest.TestCase):
    def test_builds_calendar_from_cotahist_and_hashes_archive(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archives = root / "archives"
            archives.mkdir()
            source = archives / "COTAHIST_A2020.ZIP"
            make_cotahist_zip(source, ["20200102", "20200103", "20200106"])
            output = root / "calendar.csv"
            meta = root / "calendar.json"
            payload = calendar_builder.build_calendar(
                archives_dir=archives,
                start_year=2020,
                end_year=2020,
                requested_end=date(2020, 1, 5),
                output=output,
                meta=meta,
            )
            self.assertEqual(pd.read_csv(output)["date"].tolist(), ["2020-01-02", "2020-01-03"])
            self.assertEqual(payload["last_session"], "2020-01-03")
            self.assertEqual(payload["session_count"], 2)
            self.assertEqual(payload["archives"][0]["filename"], source.name)
            self.assertEqual(len(payload["archives"][0]["sha256"]), 64)


class OfficialCalendarMarketTests(unittest.TestCase):
    def _snapshot(self, root: Path, calendar_days: list[str], candle_days: list[str]):
        (root / "data/universes").mkdir(parents=True)
        (root / "data/candles").mkdir(parents=True)
        (root / "data/calendars").mkdir(parents=True)
        (root / "data/universes/fixed_40_2018.json").write_text(
            json.dumps({"tickers": ["AAA3", "BBB3"]}), encoding="utf-8"
        )
        pd.DataFrame({"date": calendar_days}).to_csv(root / "data/calendars/b3_sessions.csv", index=False)
        for ticker, base in [("AAA3", 10.0), ("BBB3", 20.0)]:
            pd.DataFrame(
                {
                    "date": candle_days,
                    "open": [base + i for i in range(len(candle_days))],
                    "close": [base + i + 0.5 for i in range(len(candle_days))],
                }
            ).to_csv(root / "data/candles" / f"{ticker.lower()}_1d.csv", index=False)

    def test_official_session_missing_from_all_tickers_still_controls_weekly_schedule(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Monday 6 Jan is an official session but deliberately absent from every ticker.
            self._snapshot(
                root,
                ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"],
                ["2020-01-02", "2020-01-03", "2020-01-07"],
            )
            market = opt.load_market(root, "2020-01-02", "2020-01-07")
            self.assertEqual([d.date().isoformat() for d in market.execution_dates], ["2020-01-06"])
            self.assertEqual([d.date().isoformat() for d in market.decision_dates], ["2020-01-03"])
            self.assertTrue(pd.isna(market.exec_open[0]).all())
            slow_exec, slow_decision = ref.weekly_schedule_reference(market.master_dates, market.start)
            self.assertEqual(list(slow_exec), list(market.execution_dates))
            self.assertEqual(list(slow_decision), list(market.decision_dates))

    def test_requested_weekend_end_becomes_last_actual_b3_session(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._snapshot(
                root,
                ["2020-01-02", "2020-01-03", "2020-01-06"],
                ["2020-01-02", "2020-01-03", "2020-01-06"],
            )
            market = opt.load_market(root, "2020-01-02", "2020-01-05")
            self.assertEqual(market.end, "2020-01-03")
            self.assertEqual(market.master_dates.max().date().isoformat(), "2020-01-03")

    def test_candle_outside_official_calendar_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._snapshot(
                root,
                ["2020-01-02", "2020-01-03"],
                ["2020-01-02", "2020-01-03", "2020-01-06"],
            )
            with self.assertRaises(RuntimeError):
                opt.load_market(root, "2020-01-02", "2020-01-06")


if __name__ == "__main__":
    unittest.main()
