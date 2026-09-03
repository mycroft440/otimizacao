from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "optimizer"))
import metrics  # noqa: E402
import optimize_b3_pine_fast as fast  # noqa: E402


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
                    "--data-root",
                    str(data_root),
                    "--curve",
                    str(curve),
                    "--output",
                    str(output),
                ],
                check=True,
                cwd=ROOT,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["terminal_position_price_age"]["holding"], "CASH")
            self.assertEqual(payload["terminal_position_price_age"]["stale_master_sessions"], 0)


class EmptyShardTests(unittest.TestCase):
    def test_empty_shard_has_full_schema_hash_and_actual_snapshot_end(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_root = root / "snapshot"
            data_root.mkdir()
            (data_root / "SNAPSHOT_META.json").write_text(
                json.dumps(
                    {
                        "upstream_sha": "a" * 40,
                        "universe_sha256": "b" * 64,
                        "requested_end": "2020-12-27",
                        "actual_master_end": "2020-12-23",
                    }
                ),
                encoding="utf-8",
            )
            output = root / "shard_19.csv"
            args = SimpleNamespace(
                output=output,
                data_root=data_root,
                shard_id=19,
                shards=20,
                signal_min=2,
                signal_max=60,
                momentum_min=5,
                momentum_max=252,
                vol_period=21,
                start="2020-01-02",
                end="2020-12-27",
                initial_cash=1000.0,
                fee_bps=3.25,
                slippage_bps=10.0,
                odd_lot_extra_bps=5.0,
            )
            fast._write_empty_shard(args)
            frame = pd.read_csv(output)
            meta = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertTrue(frame.empty)
            self.assertIn("skipped_executions", frame.columns)
            self.assertEqual(meta["schema_version"], 2)
            self.assertEqual(meta["end"], "2020-12-23")
            self.assertEqual(meta["gap_values"], [])
            self.assertEqual(meta["csv_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())


class ShardSetAuditTests(unittest.TestCase):
    def _write_shard(
        self,
        directory: Path,
        shard: int,
        fee: float = 3.25,
        run_id: str = "123",
        optimizer_sha: str | None = None,
    ):
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
        csv_hash = hashlib.sha256(csv.read_bytes()).hexdigest()
        meta = {
            "schema_version": 2,
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
            "csv_sha256": csv_hash,
            "github_run_id": run_id,
            "optimizer_sha": optimizer_sha or "a" * 40,
            "github_repository": "mycroft440/otimizacao",
            "snapshot_upstream_sha": "b" * 40,
            "snapshot_universe_sha256": "c" * 64,
            "snapshot_requested_end": "2020-12-30",
        }
        csv.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")

    def _run(self, directory: Path, output: Path):
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "optimizer/audit_shard_set.py"),
                "--results-dir",
                str(directory),
                "--expected-shards",
                "2",
                "--initial-cash",
                "1000",
                "--fee-bps",
                "3.25",
                "--slippage-bps",
                "10",
                "--odd-lot-extra-bps",
                "5",
                "--output",
                str(output),
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

    def test_mixed_run_ids_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            self._write_shard(directory, 0, run_id="123")
            self._write_shard(directory, 1, run_id="999")
            result = self._run(directory, directory / "audit.json")
            self.assertNotEqual(result.returncode, 0)

    def test_csv_tampering_breaks_hash(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            self._write_shard(directory, 0)
            self._write_shard(directory, 1)
            with (directory / "shard_1.csv").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            result = self._run(directory, directory / "audit.json")
            self.assertNotEqual(result.returncode, 0)

    def test_invalid_or_empty_optimizer_sha_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            self._write_shard(directory, 0, optimizer_sha="abc")
            self._write_shard(directory, 1, optimizer_sha="abc")
            result = self._run(directory, directory / "audit.json")
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
