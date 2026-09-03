from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "optimizer"))
import metrics  # noqa: E402


class CompleteYearTests(unittest.TestCase):
    def test_december_20_is_not_complete(self):
        dates = pd.Series(pd.to_datetime(["2020-01-02", "2020-12-20"]))
        self.assertFalse(metrics.year_is_complete(dates))

    def test_december_28_can_be_complete(self):
        dates = pd.Series(pd.to_datetime(["2020-01-02", "2020-12-28"]))
        self.assertTrue(metrics.year_is_complete(dates))

    def test_midyear_start_is_partial(self):
        dates = pd.Series(pd.to_datetime(["2020-07-01", "2020-12-30"]))
        self.assertFalse(metrics.year_is_complete(dates))


class StalePriceCashTests(unittest.TestCase):
    def test_terminal_cash_does_not_reuse_old_security_as_terminal_position(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_root = root / "snapshot"
            (data_root / "data/universes").mkdir(parents=True)
            (data_root / "data/candles").mkdir(parents=True)
            (data_root / "data/universes/fixed_40_2018.json").write_text(
                json.dumps({"tickers": ["TST1"]}), encoding="utf-8"
            )
            pd.DataFrame(
                {
                    "date": ["2020-01-02", "2020-01-03"],
                    "open": [10.0, 10.0],
                    "close": [10.0, 10.0],
                }
            ).to_csv(data_root / "data/candles/tst1_1d.csv", index=False)
            curve = root / "curve.csv"
            pd.DataFrame(
                {
                    "date": ["2020-01-02", "2020-01-03"],
                    "holding": ["TST1", "CASH"],
                    "equity": [1000.0, 1000.0],
                }
            ).to_csv(curve, index=False)
            output = root / "stale.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "optimizer/audit_stale_prices.py"),
                    "--data-root", str(data_root),
                    "--curve", str(curve),
                    "--output", str(output),
                ],
                check=True,
                cwd=ROOT,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["terminal_position_price_age"]["holding"], "CASH")
            self.assertEqual(payload["terminal_position_price_age"]["stale_master_sessions"], 0)


class ShardSetAuditTests(unittest.TestCase):
    def _write_shard(self, directory: Path, shard: int, fee: float = 3.25):
        csv = directory / f"shard_{shard}.csv"
        pd.DataFrame(
            {
                "gap_period": [5 + shard],
                "signal_period": [2],
                "momentum_period": [5],
                "vol_period": [21],
                "final_equity": [1000.0],
                "total_return": [0.0],
                "trades": [0],
                "skipped_executions": [0],
                "fees_paid": [0.0],
                "slippage_impact": [0.0],
                "final_holding": ["CASH"],
                "start": ["2020-01-02"],
                "end": ["2020-12-30"],
                "shard": [shard],
            }
        ).to_csv(csv, index=False)
        meta = {
            "shard": shard,
            "shards": 2,
            "gap_values": [5 + shard],
            "signal_min": 2,
            "signal_max": 2,
            "momentum_min": 5,
            "momentum_max": 5,
            "vol_period": 21,
            "rows": 1,
            "start": "2020-01-02",
            "end": "2020-12-30",
            "initial_cash": 1000.0,
            "fee_bps": fee,
            "slippage_bps": 10.0,
            "odd_lot_extra_bps": 5.0,
            "portfolio_policy": "pine_v17_hold_same_target_no_residual_reinvestment",
            "momentum_dtype": "float64",
        }
        csv.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")

    def _run(self, directory: Path, output: Path):
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "optimizer/audit_shard_set.py"),
                "--results-dir", str(directory),
                "--expected-shards", "2",
                "--initial-cash", "1000",
                "--fee-bps", "3.25",
                "--slippage-bps", "10",
                "--odd-lot-extra-bps", "5",
                "--output", str(output),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    def test_matching_csv_and_metadata_pass(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            self._write_shard(directory, 0)
            self._write_shard(directory, 1)
            result = self._run(directory, directory / "audit.json")
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_mixed_cost_metadata_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            self._write_shard(directory, 0)
            self._write_shard(directory, 1, fee=4.0)
            result = self._run(directory, directory / "audit.json")
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
