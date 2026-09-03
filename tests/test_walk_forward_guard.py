from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "optimizer"))
import walk_forward_validate as wf  # noqa: E402


class WalkForwardTrainingBoundaryTests(unittest.TestCase):
    def _training_dir(
        self,
        root: Path,
        *,
        execution_end: str,
        provenance_end: str | None = None,
        fee_bps: float = 3.25,
    ) -> Path:
        directory = root / "w1"
        directory.mkdir(parents=True)
        top = directory / "top_100.csv"
        pd.DataFrame(
            {
                "gap_period": [41],
                "signal_period": [15],
                "momentum_period": [49],
                "vol_period": [21],
            }
        ).to_csv(top, index=False)
        top_hash = hashlib.sha256(top.read_bytes()).hexdigest()
        manifest = {
            "schema_version": 2,
            "optimizer_sha": "a" * 40,
            "upstream_sha": "b" * 40,
            "optimizer_source_sha256": {
                "optimizer/optimize_b3_pine.py": "c" * 64,
            },
            "execution": {
                "start": "2018-01-02",
                "end": execution_end,
                "initial_cash": 1000.0,
                "fee_bps_per_side": fee_bps,
                "slippage_bps_per_side": 10.0,
                "odd_lot_extra_bps_weighted": 5.0,
            },
            "selection_provenance": {
                "mode": "training_only",
                "training_start": "2018-01-02",
                "training_end": provenance_end or execution_end,
                "top_100_sha256": top_hash,
            },
        }
        (directory / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        return directory

    @staticmethod
    def _window() -> dict[str, str]:
        return {
            "id": "w1",
            "train_start": "2018-01-02",
            "train_end": "2020-12-31",
            "oos_start": "2021-01-01",
            "oos_end": "2021-12-31",
        }

    @staticmethod
    def _config() -> dict[str, float]:
        return {
            "initial_cash": 1000.0,
            "fee_bps": 3.25,
            "slippage_bps": 10.0,
            "odd_lot_extra_bps": 5.0,
        }

    def test_last_real_session_before_calendar_cutoff_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            directory = self._training_dir(Path(td), execution_end="2020-12-30")
            winner, _manifest = wf.verify_training(
                self._window(), directory, expected_config=self._config()
            )
            self.assertEqual(int(winner.gap_period), 41)

    def test_provenance_end_must_match_execution_end(self):
        with tempfile.TemporaryDirectory() as td:
            directory = self._training_dir(
                Path(td), execution_end="2020-12-30", provenance_end="2020-12-29"
            )
            with self.assertRaises(SystemExit):
                wf.verify_training(self._window(), directory, expected_config=self._config())

    def test_training_cannot_cross_requested_cutoff_or_holdout(self):
        with tempfile.TemporaryDirectory() as td:
            directory = self._training_dir(Path(td), execution_end="2021-01-01")
            with self.assertRaises(SystemExit):
                wf.verify_training(self._window(), directory, expected_config=self._config())

    def test_training_end_cannot_be_implausibly_far_before_cutoff(self):
        with tempfile.TemporaryDirectory() as td:
            directory = self._training_dir(Path(td), execution_end="2020-12-20")
            with self.assertRaises(SystemExit):
                wf.verify_training(self._window(), directory, expected_config=self._config())

    def test_training_costs_must_match_walk_forward(self):
        with tempfile.TemporaryDirectory() as td:
            directory = self._training_dir(
                Path(td), execution_end="2020-12-30", fee_bps=4.0
            )
            with self.assertRaises(SystemExit):
                wf.verify_training(self._window(), directory, expected_config=self._config())

    def test_missing_source_hashes_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            directory = self._training_dir(Path(td), execution_end="2020-12-30")
            manifest_path = directory / "MANIFEST.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload.pop("optimizer_source_sha256")
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(SystemExit):
                wf.verify_training(self._window(), directory, expected_config=self._config())


if __name__ == "__main__":
    unittest.main()
